#!/usr/bin/env python3
# main_gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pdk import DummyPDK
from src.forward import estimate_capacitance, estimate_capacitance_grid
from src.optimize import design_from_C
from src.surrogate_sklearn import load_model as load_model_sklearn
from src.surrogate_torch import load_model_torch


# GUI scaffolding
root = tk.Tk()
root.title("2-Port Capacitor Designer")
root.geometry("950x600")

frame_input = ttk.LabelFrame(root, text="Input")
frame_input.pack(side="left", fill="y", padx=10, pady=10)

frame_output = ttk.LabelFrame(root, text="Results")
frame_output.pack(side="right", fill="both", expand=True, padx=10, pady=10)

ttk.Label(frame_input, text="Mode:").pack(pady=5)
mode_var = tk.StringVar(value="Physics")
mode_menu = ttk.Combobox(
    frame_input,
    textvariable=mode_var,
    values=["Physics", "Sklearn", "Torch"],
    state="readonly",
)
mode_menu.pack(pady=5)

ttk.Label(frame_input, text="Target C [pF]:").pack(pady=5)
entry_C = ttk.Entry(frame_input)
entry_C.insert(0, "1.0")
entry_C.pack(pady=5)

fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=frame_output)
canvas.get_tk_widget().pack(fill="both", expand=True)

result_text = tk.StringVar()
ttk.Label(frame_input, textvariable=result_text, wraplength=220, justify="left").pack(pady=10)


def plot_results(freqs, S11, S21):
    ax.clear()
    ax.plot(freqs / 1e9, np.abs(S11), label="|S11|")
    ax.plot(freqs / 1e9, np.abs(S21), label="|S21|")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Magnitude")
    ax.set_title("S-parameters")
    ax.legend()
    canvas.draw()


def _coarse_to_fine_search_for_C(target_C, pdk, Nx_range=(1, 200), Ny_range=(1, 200)):
    # Coarse step search, then local refine around the best
    Nx_lo, Nx_hi = Nx_range
    Ny_lo, Ny_hi = Ny_range

    def scan(step):
        best = (float("inf"), None)
        for Nx in range(Nx_lo, Nx_hi + 1, step):
            for Ny in range(Ny_lo, Ny_hi + 1, step):
                C = estimate_capacitance_grid(Nx, Ny, pdk)
                err = abs(C - target_C)
                if err < best[0]:
                    best = (err, (Nx, Ny, C))
        return best

    _, (bx, by, bC) = scan(step=10)

    # refine window
    Nx_r = range(max(Nx_lo, bx - 9), min(Nx_hi, bx + 9) + 1)
    Ny_r = range(max(Ny_lo, by - 9), min(Ny_hi, by + 9) + 1)

    best = (float("inf"), None)
    for Nx in Nx_r:
        for Ny in Ny_r:
            C = estimate_capacitance_grid(Nx, Ny, pdk)
            err = abs(C - target_C)
            if err < best[0]:
                best = (err, (Nx, Ny, C))
    return best[1]  # (Nx, Ny, C)


def run_design():
    try:
        mode = mode_var.get()
        C_target_pf = float(entry_C.get())
        C_target = C_target_pf * 1e-12
        pdk = DummyPDK()

        if mode == "Physics":
            area = design_from_C(C_target, pdk)
            C_check = estimate_capacitance(area, pdk)
            result_text.set(
                f"[Physics]\n"
                f"Target C={C_target_pf:.2f} pF\n"
                f"Required area={area*1e12:.2f} um^2\n"
                f"Check C={C_check*1e12:.2f} pF"
            )
            freqs = np.linspace(1e9, 20e9, 201)
            plot_results(freqs, np.zeros_like(freqs), np.zeros_like(freqs))

        elif mode == "Sklearn":
            try:
                model = load_model_sklearn("models/mlp_wideband.joblib")
                _ = model  # placeholder, prediction flow TBD
                freqs = np.load("data/freqs.npy")
                result_text.set(
                    "[Sklearn]\nSurrogate loaded (demo placeholder)."
                )
                plot_results(freqs, np.zeros_like(freqs), np.zeros_like(freqs))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load sklearn model: {e}")

        elif mode == "Torch":
            try:
                model = load_model_torch("models/mlp_wideband.pt", in_dim=2, out_dim=804)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load torch model: {e}")
                return

            try:
                freqs = np.load("data/freqs.npy")
            except Exception:
                freqs = np.linspace(1e9, 20e9, 201)

            # choose (Nx,Ny) to match target C using physics, then get S-params via surrogate
            Nx, Ny, C_sel = _coarse_to_fine_search_for_C(C_target, pdk)

            inp = np.array([[Nx, Ny]], dtype=np.float32)
            with torch.no_grad():
                pred = model(torch.tensor(inp)).cpu().numpy()[0]

            k = len(freqs)
            S11 = pred[0 : 2 * k].reshape(-1, 2)
            S21 = pred[2 * k : 4 * k].reshape(-1, 2)
            S11 = S11[:, 0] + 1j * S11[:, 1]
            S21 = S21[:, 0] + 1j * S21[:, 1]

            result_text.set(
                f"[Torch Surrogate]\n"
                f"Target C={C_target_pf:.2f} pF\n"
                f"Selected (Nx,Ny)=({Nx},{Ny})\n"
                f"Computed C={C_sel*1e12:.2f} pF"
            )
            plot_results(freqs, S11, S21)

    except Exception as e:
        messagebox.showerror("Error", str(e))


ttk.Button(frame_input, text="Run", command=run_design).pack(pady=10)

root.mainloop()

