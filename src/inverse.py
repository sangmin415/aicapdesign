# src/inverse.py
import numpy as np

def _unpack(y, k):
    # y = [C_pf, ReS11_k,ImS11_k, ReS21_k,ImS21_k]
    C_pf = y[0]
    r = y[1:]
    ReS11, ImS11 = r[0:2*k:2], r[1:2*k:2]  # 잘못될 수 있어 안전한 방식:
    # 안전: 재구성
    S11_re = r[0:2*k:2]; S11_im = r[1:2*k:2]
    S21_re = r[2*k:4*k:2]; S21_im = r[2*k+1:4*k:2]  # 이 인덱싱은 혼동 여지↑

    # 더 명확하게:
    ReS11 = r[0:2*k][0:k]; ImS11 = r[0:2*k][k:2*k]
    ReS21 = r[2*k:4*k][0:k]; ImS21 = r[2*k:4*k][k:2*k]

    S11 = ReS11 + 1j*ImS11
    S21 = ReS21 + 1j*ImS21
    return C_pf, S11, S21

def inverse_from_specs(model, freqs, C_target_pf,
                       band=(8e9,12e9), RL_target_db=-10.0, IL_target_db=-3.0,
                       Nx_max=200, Ny_max=200, wC=1.0, wRL=1.0, wIL=1.0):
    k = len(freqs)
    fmask = (freqs>=band[0]) & (freqs<=band[1])

    best = None
    best_loss = 1e99
    for Nx in range(1, Nx_max+1):
        for Ny in range(1, Ny_max+1):
            y = model.predict([[Nx, Ny]])[0]
            C_pf, S11, S21 = _unpack(y, k)

            # 손실 성분
            loss_C  = (C_pf - C_target_pf)**2
            RL_db   = 20*np.log10(np.abs(S11[fmask]) + 1e-12)
            IL_db   = 20*np.log10(np.abs(S21[fmask]) + 1e-12)
            loss_RL = np.mean(np.maximum(RL_db - RL_target_db, 0.0)**2)  # -10dB 이하 유도
            loss_IL = np.mean((IL_db - IL_target_db)**2)                  # 예: -3dB 근접

            loss = wC*loss_C + wRL*loss_RL + wIL*loss_IL
            if loss < best_loss:
                best_loss = loss
                best = (Nx, Ny, C_pf, np.mean(RL_db), np.mean(IL_db), best_loss)
    return best  # Nx,Ny,Cpf, avgRL, avgIL, loss
