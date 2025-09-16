# src/synth_teacher_wideband.py
import numpy as np
from src.forward import estimate_capacitance_grid   # 면적→C 계산
from src.synth_teacher import rlc_teacher           # 간단 S-파라미터 근사 (재사용)

def make_wideband_dataset(pdk, Nx_max=500, Ny_max=500, f_low=1e9, f_high=20e9, k=201, step=1):
    freqs = np.linspace(f_low, f_high, k)
    X, Y = [], []
    for Nx in range(1, Nx_max+1, step):   # Nx loop (step 샘플링)
        for Ny in range(1, Ny_max+1, step):  # Ny loop (step 샘플링)
            # 1) 면적 기반 캐패시턴스 계산
            C = estimate_capacitance_grid(Nx, Ny, pdk)

            # 2) 주파수별 S-파라미터 근사
            S11, S21 = rlc_teacher(C, pdk, freqs)

            # 3) 입력과 출력 저장
            feat = [Nx, Ny]   # 입력 = grid
            targ = np.concatenate([S11.real, S11.imag, S21.real, S21.imag])  # 출력 = S-param
            X.append(feat)
            Y.append(targ)

    return freqs, np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
