#!/usr/bin/env python3
from src.pdk import DummyPDK
from src.optimize import design_from_C
from src.forward import estimate_capacitance

if __name__ == "__main__":
    pdk = DummyPDK()
    targetC = 1e-12  # 1 pF

    area = design_from_C(targetC, pdk)
    C_check = estimate_capacitance(area, pdk)

    print(f"Target C = {targetC*1e12:.2f} pF")
    # Avoid non-ASCII micro symbol for Windows console
    print(f"Calculated area = {area*1e12:.2f} um^2")
    print(f"Back-estimated C = {C_check*1e12:.2f} pF")

