# src/forward.py
import numpy as np
from .pdk import DummyPDK

EPS0 = 8.854e-12  # [F/m]

def estimate_capacitance(area_m2, pdk):
    return pdk.er * EPS0 * area_m2 / pdk.d

def estimate_capacitance_grid(Nx, Ny, pdk):
    """
    Nx, Ny (정수) → C 계산
    """
    area_m2 = (Nx * pdk.unit_size) * (Ny * pdk.unit_size)
    return estimate_capacitance(area_m2, pdk)
