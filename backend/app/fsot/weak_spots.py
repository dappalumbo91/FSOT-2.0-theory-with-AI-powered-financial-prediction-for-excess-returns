"""
FSOT weak-spot diagnostics from historical OHLCV.

Finds where emergence/entropy mapping loses directional edge:
  - asset class (inverse vol, alt-crypto, commodity)
  - entropy quintiles
  - |emergence| strength
  - vol regime
  - decade
  - conflict: emergence vs momentum disagreement
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.fsot.emergence import compute_emergence_frame


def _hit(pos: np.ndarray, fwd: np.ndarray) -> float:
    m = (pos != 0) & np.isfinite(fwd) & np.isfinite(pos)
    if m.sum() < 20:
        return float("nan")
    return float((np.sign(pos[m]) == np.sign(fwd[m])).mean())


def diagnose_symbol(df: pd.DataFrame, symbol: str = "", window: int = 30) -> dict[str, Any]:
    if df is None or len(df) < window + 80:
        return {"symbol": symbol, "error": "insufficient_data"}

    frame = compute_emergence_frame(df, window=window)
    frame = frame.dropna(subset=["S", "emergence_ema"]).copy()
    close = frame["close"].astype(float)
    frame["fwd_1"] = close.pct_change().shift(-1)
    frame["fwd_5"] = close.pct_change(5).shift(-5)
    frame["fwd_20"] = close.pct_change(20).shift(-20)
    frame["mom20"] = close.pct_change(20)
    frame["vol20"] = close.pct_change().rolling(20).std()
    frame["d_ent"] = frame["entropy"].diff()
    frame = frame.dropna(subset=["fwd_1", "mom20", "vol20"])

    score = frame["emergence_ema"].astype(float)
    ent = frame["entropy"].astype(float)
    mom = frame["mom20"].astype(float)
    fwd1 = frame["fwd_1"].values
    fwd5 = frame["fwd_5"].values
    fwd20 = frame["fwd_20"].values

    # baseline positions
    ent_med = ent.expanding(60).median()
    calm = ent < ent_med
    pos_regime = np.where(
        (score > 0.15) & calm & (mom > 0),
        1.0,
        np.where((score < -0.15) & calm & (mom < 0), -1.0, 0.0),
    )

    # Conflict: score and mom disagree
    conflict = np.sign(score) * np.sign(mom) < 0
    agree = np.sign(score) * np.sign(mom) > 0

    # Entropy rising (dispersal accelerating)
    dent = frame["d_ent"].fillna(0.0)
    ent_rising = dent > dent.expanding(60).quantile(0.7)

    # Vol regime
    vol = frame["vol20"]
    vol_hi = vol > vol.expanding(60).quantile(0.75)
    vol_lo = vol < vol.expanding(60).quantile(0.4)

    slices = {
        "all_regime_active": np.asarray(pos_regime != 0, dtype=bool),
        "agree_score_mom": np.asarray(agree & (pos_regime != 0), dtype=bool),
        "conflict_score_mom": np.asarray(conflict & (np.abs(score) > 0.15), dtype=bool),
        "entropy_rising": np.asarray(ent_rising & (pos_regime != 0), dtype=bool),
        "entropy_falling": np.asarray((~ent_rising) & (pos_regime != 0), dtype=bool),
        "vol_high": np.asarray(vol_hi & (pos_regime != 0), dtype=bool),
        "vol_low": np.asarray(vol_lo & (pos_regime != 0), dtype=bool),
        "strong_score": np.asarray((np.abs(score) > 0.5) & (pos_regime != 0), dtype=bool),
        "weak_score": np.asarray(
            (np.abs(score) <= 0.5) & (np.abs(score) > 0.15) & (pos_regime != 0), dtype=bool
        ),
        "calm": np.asarray(calm & (pos_regime != 0), dtype=bool),
        "turbulent": np.asarray((~calm) & (np.abs(score) > 0.15), dtype=bool),
    }

    slice_hits: dict[str, Any] = {}
    for name, m in slices.items():
        if int(m.sum()) < 20:
            slice_hits[name] = {"n": int(m.sum()), "hit_1d": None, "hit_5d": None, "hit_20d": None}
            continue
        p = pos_regime[m]
        slice_hits[name] = {
            "n": int(m.sum()),
            "hit_1d": _hit(p, fwd1[m]),
            "hit_5d": _hit(p, fwd5[m]),
            "hit_20d": _hit(p, fwd20[m]),
        }

    # Rank weakest slices by hit_1d
    ranked = sorted(
        [(k, v) for k, v in slice_hits.items() if v.get("hit_1d") is not None],
        key=lambda x: x[1]["hit_1d"],
    )

    # Decade
    decades = {}
    if "time" in frame.columns:
        years = pd.to_datetime(frame["time"]).dt.year
        for y0 in range(2005, 2030, 5):
            m = ((years >= y0) & (years < y0 + 5)).values & (pos_regime != 0)
            if m.sum() < 40:
                continue
            decades[f"{y0}-{y0+4}"] = {
                "n": int(m.sum()),
                "hit_1d": _hit(pos_regime[m], fwd1[m]),
                "hit_20d": _hit(pos_regime[m], fwd20[m]),
            }

    # Weak-spot flags for this symbol
    flags = []
    base = slice_hits.get("all_regime_active", {}).get("hit_1d")
    conf = slice_hits.get("conflict_score_mom", {}).get("hit_1d")
    if conf is not None and base is not None and conf < base - 0.03:
        flags.append("conflict_score_momentum_hurts")
    er = slice_hits.get("entropy_rising", {}).get("hit_1d")
    if er is not None and base is not None and er < base - 0.02:
        flags.append("entropy_rising_hurts")
    vh = slice_hits.get("vol_high", {}).get("hit_1d")
    if vh is not None and base is not None and vh < base - 0.02:
        flags.append("high_vol_hurts")
    if base is not None and base < 0.50:
        flags.append("below_coin_flip_regime")

    # Asset class heuristic
    sym = symbol.upper().replace("^", "")
    if sym in ("VIX",):
        asset_class = "inverse_vol"
    elif sym in ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC"):
        asset_class = "crypto"
    elif sym in ("XOM", "XLE", "GLD", "TLT", "HYG"):
        asset_class = "macro_proxy"
    else:
        asset_class = "equity_index"

    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "n_bars": int(len(frame)),
        "baseline_regime_hit_1d": base,
        "baseline_regime_hit_20d": slice_hits.get("all_regime_active", {}).get("hit_20d"),
        "slices": slice_hits,
        "weakest_slices": [{"name": k, **v} for k, v in ranked[:5]],
        "strongest_slices": [{"name": k, **v} for k, v in ranked[-5:][::-1]],
        "decades": decades,
        "flags": flags,
        "remedies": _remedies(flags, asset_class),
    }


def _remedies(flags: list[str], asset_class: str) -> list[str]:
    r = []
    if "conflict_score_momentum_hurts" in flags:
        r.append("require_score_momentum_agreement")
    if "entropy_rising_hurts" in flags:
        r.append("block_when_entropy_rising")
    if "high_vol_hurts" in flags:
        r.append("widen_deadzone_in_high_vol")
    if "below_coin_flip_regime" in flags:
        r.append("raise_selectivity_or_invert_for_inverse_assets")
    if asset_class == "inverse_vol":
        r.append("invert_position_for_vix_class")
    if asset_class == "crypto":
        r.append("stricter_gates_and_prefer_20d_horizon")
    if asset_class == "macro_proxy":
        r.append("require_strong_score_and_calm")
    return r


def aggregate_weak_spots(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-asset weak-spot summary."""
    ok = [r for r in reports if not r.get("error")]
    flag_counts: dict[str, int] = {}
    remedy_counts: dict[str, int] = {}
    class_acc: dict[str, list[float]] = {}
    for r in ok:
        for f in r.get("flags") or []:
            flag_counts[f] = flag_counts.get(f, 0) + 1
        for rem in r.get("remedies") or []:
            remedy_counts[rem] = remedy_counts.get(rem, 0) + 1
        ac = r.get("asset_class", "unknown")
        h = r.get("baseline_regime_hit_1d")
        if h is not None:
            class_acc.setdefault(ac, []).append(h)

    # Average slice hits across assets
    slice_pool: dict[str, list[float]] = {}
    for r in ok:
        for name, s in (r.get("slices") or {}).items():
            if s.get("hit_1d") is not None:
                slice_pool.setdefault(name, []).append(s["hit_1d"])

    slice_means = {
        k: {"mean_hit_1d": float(np.mean(v)), "n_assets": len(v)}
        for k, v in slice_pool.items()
    }
    ranked_slices = sorted(slice_means.items(), key=lambda x: x[1]["mean_hit_1d"])

    return {
        "n_symbols": len(ok),
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda x: -x[1])),
        "remedy_counts": dict(sorted(remedy_counts.items(), key=lambda x: -x[1])),
        "accuracy_by_asset_class": {
            k: {"mean_hit_1d": float(np.mean(v)), "n": len(v)} for k, v in class_acc.items()
        },
        "slice_means_weakest": [{"name": k, **v} for k, v in ranked_slices[:6]],
        "slice_means_strongest": [{"name": k, **v} for k, v in ranked_slices[-6:][::-1]],
        "global_remedies_priority": [
            "require_score_momentum_agreement",
            "block_when_entropy_rising",
            "widen_deadzone_in_high_vol",
            "invert_position_for_vix_class",
            "stricter_gates_and_prefer_20d_horizon",
            "cross_sectional_relative_to_spy",
            "cost_aware_evaluation",
        ],
    }
