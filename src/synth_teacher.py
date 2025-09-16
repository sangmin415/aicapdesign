# src/synth_teacher.py
import numpy as np

def rlc_teacher(C, pdk, freqs):
    """
    간단한 RLC 근사 모델로 S11, S21을 생성
    C: capacitance [F]
    pdk: PDK 객체 (metal_r, via_l 등 사용 가능)
    freqs: 주파수 축 [Hz]
    """
    R = getattr(pdk, "metal_r", 0.05)   # series resistance
    L = getattr(pdk, "via_l", 5e-12)   # parasitic inductance

    w = 2 * np.pi * freqs
    Zc = 1 / (1j * w * C)               # capacitor impedance
    Zl = 1j * w * L
    Z = R + Zl + Zc                     # 총 임피던스

    Z0 = 50                             # reference impedance
    S11 = (Z - Z0) / (Z + Z0)
    S21 = np.ones_like(S11) - np.abs(S11)  # 간단한 근사

    return S11, S21
