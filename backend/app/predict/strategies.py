"""
FSOT-aligned trading strategies.

1d raw sign(dS) is near-random on liquid markets (efficiency). Theory-aligned
uses of the scalar field are:

  A) REGIME FILTER on momentum (emergence → ride trend; high entropy → flat/defensive)
  B) Multi-horizon (5d / 20d) — structure emerges slower than one session
  C) Cross-sectional ranking (relative emergence across assets)

These are not free-parameter least-squares fits; thresholds use FSOT-native
z-scores and deadzones already defined in emergence.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.fsot.emergence import compute_emergence_frame


def evaluate_strategies(df: pd.DataFrame, window: int = 30) -> dict[str, Any]:
    if df is None or len(df) < window + 60:
        return {"error": "insufficient_data"}

    frame = compute_emergence_frame(df, window=window)
    frame = frame.dropna(subset=["S", "emergence_ema"]).copy()
    close = frame["close"].astype(float)
    ret1 = close.pct_change()
    # forward returns (causal: measured after signal at t)
    frame["fwd_1"] = ret1.shift(-1)
    frame["fwd_5"] = close.pct_change(5).shift(-5)
    frame["fwd_20"] = close.pct_change(20).shift(-20)

    # medium trend: 20d momentum known at t (no lookahead)
    frame["mom20"] = close.pct_change(20)
    frame = frame.dropna(subset=["fwd_1", "mom20"])

    score = frame["emergence_ema"].astype(float)
    ent = frame["entropy"].astype(float)
    mom = frame["mom20"].astype(float)

    # --- Strategy A: raw emergence sign (1d) ---
    pos_a = np.sign(score.replace(0, np.nan)).fillna(0.0)
    # --- Strategy B: regime-filtered momentum ---
    # Long when emergence > 0 AND entropy below expanding median AND mom20 > 0
    ent_med = ent.expanding(60).median()
    emerge_ok = score > 0.15
    calm = ent < ent_med
    pos_b = np.where(emerge_ok & calm & (mom > 0), 1.0, np.where((score < -0.15) & calm & (mom < 0), -1.0, 0.0))
    # --- Strategy C: entropy-fade (high entropy → fade 1d mom) ---
    ret_1d = ret1.reindex(frame.index).fillna(0.0)
    high_ent = ent > ent.expanding(60).quantile(0.75)
    pos_c = np.where(high_ent, -np.sign(ret_1d), 0.0)

    def pack(name: str, pos: np.ndarray, horizon: str) -> dict[str, Any]:
        fwd = frame[horizon].values.astype(float)
        p = np.asarray(pos, dtype=float)
        # align lengths
        n = min(len(p), len(fwd))
        p, fwd = p[:n], fwd[:n]
        m = np.isfinite(fwd) & np.isfinite(p)
        p, fwd = p[m], fwd[m]
        active = p != 0
        if active.sum() < 30:
            return {"name": name, "horizon": horizon, "error": "too_few_active", "n_active": int(active.sum())}
        hit = float((np.sign(p[active]) == np.sign(fwd[active])).mean())
        strat = p * fwd
        # only count days in market for return; flat earns 0
        cum = float(np.nanprod(1.0 + strat) - 1.0)
        if strat.std() > 1e-12:
            sharpe = float(strat.mean() / strat.std() * np.sqrt(252 / max(1, int(horizon.split("_")[1]) if "_" in horizon else 1)))
        else:
            sharpe = 0.0
        # buy hold same horizon
        bh = float(np.nanprod(1.0 + fwd) - 1.0)
        return {
            "name": name,
            "horizon": horizon,
            "n": int(len(fwd)),
            "n_active": int(active.sum()),
            "pct_in_market": float(active.mean()),
            "directional_accuracy_active": hit,
            "strategy_return": cum,
            "buy_hold_return": bh,
            "sharpe_approx": sharpe,
            "mean_active_return": float(strat[active].mean()),
        }

    results = {
        "emergence_1d": pack("emergence_sign", pos_a.values, "fwd_1"),
        "emergence_5d": pack("emergence_sign", pos_a.values, "fwd_5"),
        "emergence_20d": pack("emergence_sign", pos_a.values, "fwd_20"),
        "regime_mom_1d": pack("regime_filtered_momentum", pos_b, "fwd_1"),
        "regime_mom_5d": pack("regime_filtered_momentum", pos_b, "fwd_5"),
        "regime_mom_20d": pack("regime_filtered_momentum", pos_b, "fwd_20"),
        "entropy_fade_1d": pack("entropy_fade", pos_c, "fwd_1"),
    }

    # Regime bucket hit rates at 5d (structure timescale)
    bins = [-np.inf, -0.5, -0.15, 0.15, 0.5, np.inf]
    labels = ["strong_dispersal", "dispersal", "neutral", "emergence", "strong_emergence"]
    frame["bin"] = pd.cut(score, bins=bins, labels=labels)
    buckets = []
    for lab in labels:
        sub = frame[frame["bin"] == lab]
        if len(sub) < 40:
            continue
        buckets.append(
            {
                "regime": lab,
                "n": int(len(sub)),
                "hit_1d": float((sub["fwd_1"] > 0).mean()),
                "hit_5d": float((sub["fwd_5"].dropna() > 0).mean()) if sub["fwd_5"].notna().any() else None,
                "mean_fwd_5": float(sub["fwd_5"].mean()) if sub["fwd_5"].notna().any() else None,
                "mean_fwd_20": float(sub["fwd_20"].mean()) if sub["fwd_20"].notna().any() else None,
            }
        )

    # Pick best non-error strategy by active accuracy then sharpe
    ranked = []
    for k, v in results.items():
        if v.get("error"):
            continue
        ranked.append((k, v))
    ranked.sort(key=lambda x: (x[1]["directional_accuracy_active"], x[1]["sharpe_approx"]), reverse=True)
    best = ranked[0] if ranked else (None, None)

    return {
        "error": None,
        "strategies": results,
        "regime_buckets": buckets,
        "best_strategy": best[0],
        "best_metrics": best[1],
        "n_bars": int(len(frame)),
        "note": (
            "1d raw FSOT sign is near-efficient (~50%). Prefer regime-filtered momentum "
            "and multi-day horizons. FSOT provides the emergence/entropy field; it is not "
            "a free-parameter price regressor."
        ),
    }
