"""
Float64 FSOT scalar engine — mirrors archive vendor/fsot_compute.py.

Authority remains compute_authority.py (mpmath). This module is the hot path
for API latency. Tests enforce agreement with mpmath on domain scalars.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ── Seeds ──────────────────────────────────────────────────────────────────
PI = math.pi
E = math.e
PHI = (1.0 + math.sqrt(5.0)) / 2.0
GAMMA = 0.5772156649015328606
G_CAT = 0.9159655941772190150

# ── Layer 1 ────────────────────────────────────────────────────────────────
ALPHA = math.log(PI) / (E * PHI**13)
PSI_CON = 1.0 - math.exp(-1.0)
ETA_EFF = 1.0 / (PI - 1.0)
BETA = 1.0 / math.exp(PI**PI + (E - 1.0))
GAMMA_C = -math.log(2.0) / PHI
OMEGA = math.sin(PI / E) * math.sqrt(2.0)
THETA_S = math.sin(PSI_CON * ETA_EFF)
POOF = math.exp((-math.log(PI) / E) / (ETA_EFF * math.log(PHI)))

# ── Layer 2 ────────────────────────────────────────────────────────────────
C_EFF = (1.0 - POOF * math.sin(THETA_S)) * (1.0 + 0.01 * G_CAT / (PI * PHI))
A_BLEED = math.sin(PI / E) * PHI / math.sqrt(2.0)
P_VAR = -math.cos(THETA_S + PI)
B_IN = C_EFF * (1.0 - math.sin(THETA_S) / PHI)
A_IN = A_BLEED * (1.0 + math.cos(THETA_S) / PHI)
SUCTION = POOF * (-math.cos(THETA_S - PI))
CHAOS = GAMMA_C / OMEGA
P_BASE = GAMMA / E
P_NEW = P_BASE * math.sqrt(2.0)
C_FACTOR = C_EFF * P_NEW
K = PHI * (GAMMA / E) * math.sqrt(2.0) / math.log(PI) * 0.99
C_COSM = 1.0 / (PHI * 10.0)


@dataclass
class ScalarInputF:
    N: float = 1.0
    P: float = 1.0
    D_eff: float = 25.0
    psi_con: float = PSI_CON
    delta_psi: float = 1.0
    recent_hits: float = 0.0
    rho: float = 1.0
    B_in: float = B_IN
    C_eff: float = C_EFF
    P_new: float = P_NEW
    observed: bool = False
    beta: float = BETA
    chaos: float = CHAOS
    poof: float = POOF
    suction: float = SUCTION
    theta_s: float = THETA_S
    delta_theta: float = 1.0
    A_bleed: float = A_BLEED
    A_in: float = A_IN
    P_var: float = P_VAR
    scale: float = 1.0
    amplitude: float = 1.0
    trend_bias: float = 0.0
    alpha: float = ALPHA


def compute_scalar_terms_fast(s: ScalarInputF) -> dict[str, float]:
    """Return T1, T2, T3 and S = K*(T1+T2+T3)."""
    N = max(s.N, 1e-12)
    P = s.P
    D = max(s.D_eff, 1e-12)
    dp = s.delta_psi
    dt = s.delta_theta
    hits = s.recent_hits

    growth = math.exp(s.alpha * (1.0 - hits / N) * GAMMA / PHI)
    base = (
        (N * P / math.sqrt(D))
        * math.cos((s.psi_con + dp) / ETA_EFF)
        * math.exp(-s.alpha * hits / N + s.rho + s.B_in * dp)
        * (1.0 + growth * s.C_eff)
    )
    T1 = base * (1.0 + s.P_new * math.log(D / 25.0))
    if s.observed:
        T1 = T1 * math.exp(C_FACTOR * s.P_var) * math.cos(dp + s.P_var)

    T2 = s.scale * s.amplitude + s.trend_bias

    valve = (
        s.beta
        * math.cos(dp)
        * (N * P / math.sqrt(D))
        * (1.0 + s.chaos * (D - 25.0) / 25.0)
        * (1.0 + s.poof * math.cos(s.theta_s + PI) + s.suction * math.sin(s.theta_s))
    )
    acoustic = (
        1.0
        + (s.A_bleed * math.sin(dt) ** 2) / PHI
        + (s.A_in * math.cos(dt) ** 2) / PHI
    )
    phase = 1.0 + s.B_in * s.P_var
    T3 = valve * acoustic * phase

    S = K * (T1 + T2 + T3)
    return {
        "T1": float(T1),
        "T2": float(T2),
        "T3": float(T3),
        "S": float(S),
        "K": float(K),
    }


def compute_scalar_fast(s: ScalarInputF | dict[str, Any]) -> float:
    if isinstance(s, dict):
        s = ScalarInputF(**{k: v for k, v in s.items() if k in ScalarInputF.__dataclass_fields__})
    return compute_scalar_terms_fast(s)["S"]
