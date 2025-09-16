# src/optimize.py
from .forward import estimate_capacitance, EPS0

def design_from_C(C_target, pdk):
    """목표 캐패시턴스 → 필요한 면적 계산"""
    area = C_target * pdk.d / (pdk.er * EPS0)
    return area
