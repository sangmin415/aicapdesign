# main_train.py
import os
from src.pdk import DummyPDK
from src.synth_teacher_wideband import make_wideband_dataset
from src.surrogate_torch import train_and_save_torch   # torch 버전 학습 불러오기
import numpy as np

if __name__ == "__main__":
    # 무리한 데이터 크기를 피하기 위해 기본 그리드 축소 + 샘플링 간격 추가
    Nx_max, Ny_max = 200, 200
    f_low, f_high, k = 1e9, 20e9, 201

    pdk = DummyPDK()
    print(f"▶ dataset: Nx={Nx_max}, Ny={Ny_max}, f=[{f_low/1e9:.1f}..{f_high/1e9:.1f}]GHz, k={k}")
    # step을 사용해 샘플 수를 줄이고 학습 시간을 단축
    freqs, X, Y = make_wideband_dataset(pdk, Nx_max, Ny_max, f_low, f_high, k, step=5)
    print("X:", X.shape, "Y:", Y.shape)

    os.makedirs("models", exist_ok=True)
    path = train_and_save_torch(X, Y, "models/mlp_wideband.pt", epochs=30, batch_size=256, lr=1e-3)
    np.save("data/freqs.npy", freqs)

    print("saved:", path, "and data/freqs.npy")

