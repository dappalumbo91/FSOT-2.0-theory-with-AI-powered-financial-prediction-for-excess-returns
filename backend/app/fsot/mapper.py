"""
Map OHLCV market features → FSOT ScalarInputF.

Preserves spirit of legacy finance mapper (hits from up-days, δψ from vol/RSI)
but feeds the canonical additive scalar engine (Economics base + dynamics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .domain import domain_base_params
from .fast import P_BASE, ScalarInputF


@dataclass
class MarketFeatures:
    n_bars: int
    relative_volume: float
    up_day_count: float
    realized_vol: float
    rsi: float
    mean_return: float
    autocorr: float
    trend_slope: float
    atr_pct: float


def _rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    val = float(rsi.iloc[-1])
    return 50.0 if np.isnan(val) else val


def extract_features(df: pd.DataFrame, window: int = 30) -> MarketFeatures:
    """Extract features from OHLCV dataframe (columns: open, high, low, close, volume)."""
    if df is None or len(df) < 2:
        return MarketFeatures(
            n_bars=1,
            relative_volume=1.0,
            up_day_count=0.0,
            realized_vol=0.01,
            rsi=50.0,
            mean_return=0.0,
            autocorr=0.0,
            trend_slope=0.0,
            atr_pct=0.02,
        )

    w = min(window, len(df))
    tail = df.iloc[-w:].copy()
    close = tail["close"].astype(float)
    rets = close.pct_change().fillna(0.0)

    vol_series = tail["volume"].astype(float) if "volume" in tail.columns else pd.Series(np.ones(w))
    med_vol = float(vol_series.median()) or 1.0
    rel_vol = float(np.clip(vol_series.iloc[-1] / med_vol, 0.05, 20.0))

    up_days = float((rets > 0).sum())
    realized_vol = float(rets.std()) if len(rets) > 1 else 0.01
    if np.isnan(realized_vol) or realized_vol <= 0:
        realized_vol = 0.01

    mean_ret = float(rets.mean())
    if np.isnan(mean_ret):
        mean_ret = 0.0

    if len(rets) > 2:
        ac = rets.autocorr(lag=1)
        autocorr = float(ac) if ac is not None and not np.isnan(ac) else 0.0
    else:
        autocorr = 0.0

    # EMA-like slope: last vs mean
    c_mean = float(close.mean()) or 1.0
    trend_slope = float((close.iloc[-1] - c_mean) / c_mean)

    if {"high", "low"}.issubset(tail.columns):
        high = tail["high"].astype(float)
        low = tail["low"].astype(float)
        tr = (high - low).abs()
        atr = float(tr.mean())
        atr_pct = atr / (float(close.iloc[-1]) or 1.0)
    else:
        atr_pct = realized_vol

    rsi = _rsi(df["close"].astype(float), 14)

    return MarketFeatures(
        n_bars=w,
        relative_volume=rel_vol,
        up_day_count=up_days,
        realized_vol=realized_vol,
        rsi=rsi,
        mean_return=mean_ret,
        autocorr=autocorr,
        trend_slope=trend_slope,
        atr_pct=float(max(atr_pct, 1e-6)),
    )


def map_ohlcv_to_scalar_input(
    df: pd.DataFrame,
    window: int = 30,
    domain: str = "Economics",
) -> tuple[ScalarInputF, MarketFeatures, dict[str, Any]]:
    """
    Build ScalarInputF from OHLCV.

    Dynamic fields ride on Economics domain base (D_eff=20, observed=True).
    """
    base = domain_base_params(domain)
    feat = extract_features(df, window=window)

    # N: sample depth in window (normalized lightly so huge N doesn't explode)
    N = float(max(1.0, min(feat.n_bars, 252.0)))
    # P: relative volume as probability/intensity proxy
    P = float(feat.relative_volume)

    # recent_hits: up-day count in window (market activity of positive moves)
    recent_hits = float(np.clip(feat.up_day_count, 0.0, N))

    # delta_psi: base economics 1.5 ± vol/RSI modulation
    rsi_factor = (feat.rsi - 50.0) / 50.0
    vol_mod = float(np.tanh(feat.realized_vol * 20.0 - 0.2 + rsi_factor * 0.5))
    delta_psi = float(base["delta_psi"] + 0.35 * vol_mod)

    # delta_theta: direction angle from mean return
    delta_theta = float(np.arctan(feat.mean_return * 100.0))

    # rho: regime coherence from autocorr (map [-1,1] → [0.2, 1.8])
    rho = float(1.0 + 0.8 * np.clip(feat.autocorr, -1.0, 1.0))

    # scale / amplitude from volatility (unit conversion, not free fit)
    scale = 1.0
    amplitude = float(np.clip(feat.atr_pct * 10.0, 0.05, 3.0))

    # trend_bias scaled by seed P_base
    trend_bias = float(feat.trend_slope * P_BASE * 2.0)

    si = ScalarInputF(
        N=N,
        P=P,
        D_eff=float(base["D_eff"]),
        delta_psi=delta_psi,
        delta_theta=delta_theta,
        recent_hits=recent_hits,
        rho=rho,
        observed=True,
        scale=scale,
        amplitude=amplitude,
        trend_bias=trend_bias,
    )

    meta = {
        "domain": base["domain"],
        "window": window,
        "features": {
            "n_bars": feat.n_bars,
            "relative_volume": feat.relative_volume,
            "up_day_count": feat.up_day_count,
            "realized_vol": feat.realized_vol,
            "rsi": feat.rsi,
            "mean_return": feat.mean_return,
            "autocorr": feat.autocorr,
            "trend_slope": feat.trend_slope,
            "atr_pct": feat.atr_pct,
        },
        "params": {
            "N": si.N,
            "P": si.P,
            "D_eff": si.D_eff,
            "recent_hits": si.recent_hits,
            "delta_psi": si.delta_psi,
            "delta_theta": si.delta_theta,
            "rho": si.rho,
            "scale": si.scale,
            "amplitude": si.amplitude,
            "trend_bias": si.trend_bias,
            "observed": si.observed,
        },
    }
    return si, feat, meta
