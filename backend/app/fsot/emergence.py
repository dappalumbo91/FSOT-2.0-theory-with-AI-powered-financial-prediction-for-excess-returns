"""
FSOT financial emergence / entropy fields.

Theory alignment (FSOT 2.1 layman + Mathematical Key):
  - Positive S / rising S  → emergence (structure forming)
  - Negative / falling S   → dispersal / entropy rise
  - Observer flag couples news/sentiment into δψ (quirk_mod path)

CRITICAL: Do NOT use (S − Economics_S0) residual as a return forecast.
That almost always collapses to ~50% directional accuracy because S0 is a
domain constant and residual is dominated by scale, not direction.

Correct causal signal stack:
  1. S_t from Economics-domain ScalarInput with market-modulated phases
  2. dS_t = S_t − S_{t−1}  (emergence pulse)
  3. entropy_t = normalized return entropy / vol (dispersal pressure)
  4. emergence_score = z(dS) − z(entropy) + observer_boost
  5. signal = sign(emergence_score) with deadzone on |score|
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .domain import ECONOMICS_DOMAIN
from .fast import (
    C_FACTOR,
    K,
    P_BASE,
    PHI,
    PSI_CON,
    ScalarInputF,
    compute_scalar_terms_fast,
)


def shannon_entropy_returns(rets: np.ndarray, bins: int = 10) -> float:
    """Normalized Shannon entropy of return distribution in [0, 1]."""
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return 0.5
    hist, _ = np.histogram(r, bins=bins, density=False)
    p = hist.astype(float)
    s = p.sum()
    if s <= 0:
        return 0.5
    p = p / s
    p = p[p > 0]
    h = float(-(p * np.log(p)).sum())
    h_max = math.log(bins)
    return float(np.clip(h / h_max if h_max > 0 else 0.5, 0.0, 1.0))


def map_bar_to_scalar(
    close: float,
    ret: float,
    vol: float,
    rel_volume: float,
    rsi: float,
    autocorr: float,
    trend: float,
    up_frac: float,
    entropy: float,
    observer_mod: float = 0.0,
) -> ScalarInputF:
    """
    Map a single bar's state → ScalarInput.

    Keep N≈1, P≈1 near Economics domain (archive domain_scalar convention).
    Put market structure into hits, phases, amplitude, trend_bias, rho.
    """
    base_hits = float(ECONOMICS_DOMAIN["hits"])
    base_dp = float(ECONOMICS_DOMAIN["delta_psi"])

    # mild intensity — never explode P/N (that produced BTC S~10)
    P = 1.0 + 0.15 * math.tanh(math.log(max(rel_volume, 1e-6)))
    N = 1.0

    # hits: activity of positive structure (up-fraction * scale)
    recent_hits = float(np.clip(base_hits + 4.0 * (up_frac - 0.5) + 1.5 * entropy, 0.0, 12.0))

    # observer-coupled phase: RSI + external news observer_mod ∈ [-1,1]
    rsi_n = (rsi - 50.0) / 50.0
    delta_psi = base_dp + 0.45 * math.tanh(vol * 25.0) + 0.25 * rsi_n + 0.35 * observer_mod

    # directional phase from return
    delta_theta = math.atan(ret * 80.0) + 0.2 * math.atan(trend * 10.0)

    # coherence: high autocorr + low entropy → ordered market
    rho = 1.0 + 0.6 * autocorr - 0.4 * (entropy - 0.5)

    amplitude = float(np.clip(0.2 + 2.5 * vol + 0.8 * entropy, 0.05, 2.5))
    scale = 1.0
    trend_bias = float(trend * P_BASE * 3.0)

    return ScalarInputF(
        N=N,
        P=P,
        D_eff=float(ECONOMICS_DOMAIN["D_eff"]),
        psi_con=PSI_CON,
        delta_psi=float(delta_psi),
        delta_theta=float(delta_theta),
        recent_hits=recent_hits,
        rho=float(np.clip(rho, 0.2, 2.0)),
        observed=True,  # markets are observed systems
        scale=scale,
        amplitude=amplitude,
        trend_bias=trend_bias,
    )


def compute_emergence_frame(
    df: pd.DataFrame,
    window: int = 30,
    observer_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Causal FSOT emergence frame for OHLCV.

    Columns: S, T1, T2, T3, dS, entropy, emergence_score, signal_raw, ...
    """
    if df is None or len(df) < window + 2:
        return pd.DataFrame()

    out = df.copy().reset_index(drop=True)
    close = out["close"].astype(float)
    rets = close.pct_change().fillna(0.0)
    vol_s = out["volume"].astype(float) if "volume" in out.columns else pd.Series(1.0, index=out.index)
    med_vol = vol_s.rolling(window, min_periods=5).median().replace(0, np.nan)
    rel_vol = (vol_s / med_vol).fillna(1.0).clip(0.05, 20.0)

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50.0)

    rows: list[dict[str, Any]] = []
    for i in range(len(out)):
        if i < 5:
            rows.append(
                {
                    "S": np.nan,
                    "T1": np.nan,
                    "T2": np.nan,
                    "T3": np.nan,
                    "dS": 0.0,
                    "entropy": 0.5,
                    "emergence_score": 0.0,
                    "observer_mod": 0.0,
                }
            )
            continue

        lo = max(0, i - window + 1)
        r_win = rets.iloc[lo : i + 1].values
        entropy = shannon_entropy_returns(r_win)
        vol = float(np.std(r_win)) if len(r_win) > 1 else 0.01
        if not np.isfinite(vol) or vol <= 0:
            vol = 0.01
        up_frac = float((r_win > 0).mean()) if len(r_win) else 0.5
        ac = pd.Series(r_win).autocorr(lag=1)
        autocorr = float(ac) if ac is not None and np.isfinite(ac) else 0.0
        c_win = close.iloc[lo : i + 1]
        trend = float((c_win.iloc[-1] - c_win.mean()) / (c_win.mean() or 1.0))
        ret = float(rets.iloc[i])
        obs = 0.0
        if observer_series is not None and i < len(observer_series):
            try:
                obs = float(observer_series.iloc[i])
            except Exception:
                obs = 0.0
            if not np.isfinite(obs):
                obs = 0.0

        si = map_bar_to_scalar(
            close=float(close.iloc[i]),
            ret=ret,
            vol=vol,
            rel_volume=float(rel_vol.iloc[i]),
            rsi=float(rsi.iloc[i]),
            autocorr=autocorr,
            trend=trend,
            up_frac=up_frac,
            entropy=entropy,
            observer_mod=obs,
        )
        terms = compute_scalar_terms_fast(si)
        rows.append(
            {
                "S": terms["S"],
                "T1": terms["T1"],
                "T2": terms["T2"],
                "T3": terms["T3"],
                "dS": 0.0,  # filled below
                "entropy": entropy,
                "emergence_score": 0.0,
                "observer_mod": obs,
                "vol": vol,
                "up_frac": up_frac,
                "params_delta_psi": si.delta_psi,
                "params_delta_theta": si.delta_theta,
                "params_hits": si.recent_hits,
                "params_rho": si.rho,
            }
        )

    frame = pd.DataFrame(rows)
    for c in frame.columns:
        out[c] = frame[c]

    out["dS"] = out["S"].diff().fillna(0.0)

    # z-scores (expanding, causal)
    dS = out["dS"]
    ent = out["entropy"]
    dS_mu = dS.expanding(min_periods=20).mean()
    dS_sd = dS.expanding(min_periods=20).std().replace(0, np.nan)
    ent_mu = ent.expanding(min_periods=20).mean()
    ent_sd = ent.expanding(min_periods=20).std().replace(0, np.nan)

    z_dS = ((dS - dS_mu) / dS_sd).fillna(0.0).clip(-4, 4)
    z_ent = ((ent - ent_mu) / ent_sd).fillna(0.0).clip(-4, 4)

    # Emergence pulse vs entropy pressure + light observer
    out["emergence_score"] = (
        z_dS - 0.65 * z_ent + 0.35 * out["observer_mod"].fillna(0.0)
    ).astype(float)

    # Smooth score (causal EMA)
    out["emergence_ema"] = out["emergence_score"].ewm(span=5, adjust=False).mean()

    # S level regime (positive emergence continuum)
    out["regime"] = np.where(out["dS"] >= 0, "emergence", "dispersal")
    return out


def score_to_signal(score: float, deadzone: float = 0.25) -> str:
    if score > deadzone:
        return "LONG"
    if score < -deadzone:
        return "SHORT"
    return "FLAT"


def theory_constants() -> dict[str, Any]:
    return {
        "formula": "S = K * (T1 + T2 + T3)",
        "K": K,
        "C_factor": C_FACTOR,
        "phi": PHI,
        "economics": ECONOMICS_DOMAIN,
        "signal_rule": "sign(z(dS) - 0.65*z(entropy) + 0.35*observer_mod)",
        "note": "Not residual vs domain S0 — that mapping is invalid for direction.",
    }
