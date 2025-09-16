# src/pdk.py
class DummyPDK:
    def __init__(self):
        # 공정/소재
        self.er = 3.9           # 상대 유전율 (예: SiO2)
        self.d  = 1.0e-6        # 유전체 두께 [m]

        # 유닛셀 (격자 해상도) — 필요하면 0.5e-6로 올리면 더 큰 C 가능
        self.unit_size = 1.0e-6  # 1 µm x 1 µm

        # 금속/비아의 단순 기생 근사
        self.sheet_R   = 10e-3   # [Ω/□] 근사 (ESR 산정용)
        self.via_L     = 10e-12  # [H] (ESL 산정용, Nx,Ny에 따라 스케일)
        self.pad_Cscale = 0.0    # 필요시 패드 기생 C 보정용 (기본 0)
