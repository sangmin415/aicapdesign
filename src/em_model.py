# src/em_model.py
import numpy as np

def cap_from_grid(Nx, Ny, pdk):
    area = (Nx * pdk.unit_size) * (Ny * pdk.unit_size)  # [m^2]
    EPS0 = 8.854e-12
    C = pdk.er * EPS0 * area / pdk.d                    # [F]
    return C

def parasitics_from_grid(Nx, Ny, pdk):
    # 매우 단순 근사: Nx+Ny에 비례해 ESR/ESL 변화 (원하면 정교화)
    # 면적↑ → ESR↓, ESL↑ 경향을 가볍게 반영
    base_R = pdk.sheet_R
    R = base_R / max(Nx*Ny, 1) ** 0.4
    L = pdk.via_L * max(Nx+Ny, 1) ** 0.3
    return R, L

def sparams_pi_2port(C, R, L, freqs, Z0=50.0):
    """
    대칭 π 네트워크:
      Shunt C/2 양단 + 중앙에 Series (R + jωL)
    """
    w = 2*np.pi*freqs
    j = 1j

    Zseries = R + j*w*L
    Ysh = j*w*C/2  # 각 포트 shunt

    # ABCD 행렬 (Port1 shunt Y, series Z, Port2 shunt Y)
    # Shunt: A=1, B=0, C=Y, D=1
    # Series: A=1, B=Z, C=0, D=1
    # 전체 ABCD = Shunt * Series * Shunt
    A = 1 + 0* w
    B = 0 + 0* w
    Cc= 0 + 0* w
    D = 1 + 0* w

    # 첫 shunt
    A1 = A
    B1 = B
    C1 = Cc + Ysh
    D1 = D

    # series
    A2 = A1
    B2 = B1 + Zseries
    C2 = C1
    D2 = D1

    # 둘째 shunt
    A3 = A2
    B3 = B2
    C3 = C2 + Ysh
    D3 = D2

    # ABCD→S 변환
    # S11 = (A+B/Z0-C*Z0-D)/Δ, S21 = 2/Δ, Δ=(A+B/Z0+C*Z0+D)
    Den = (A3 + B3/Z0 + C3*Z0 + D3)
    S11 = (A3 + B3/Z0 - C3*Z0 - D3) / Den
    S21 = 2.0 / Den
    S22 = (-A3 + B3/Z0 - C3*Z0 + D3) / Den
    S12 = (2.0*(A3*D3 - B3*C3)/Den) / Den  # 대칭 근사에선 S12≈S21

    return S11, S21, S12, S22
