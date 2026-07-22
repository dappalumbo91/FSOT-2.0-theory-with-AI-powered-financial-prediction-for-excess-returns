"""Walk-forward evaluation — intrinsic FSOT only."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.fsot.intrinsic import evaluate_intrinsic_walkforward, spine_scalars


def walk_forward_backtest(
    df: pd.DataFrame,
    window: int = 21,
    domain: str = "Economics",
    observer_mod: float = 0.0,
    symbol: str = "",
    market_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    r = evaluate_intrinsic_walkforward(df, window=window, sentiment=float(observer_mod))
    if r.get("error"):
        return r

    spine = spine_scalars()
    return {
        "error": None,
        "method": "fsot_full_engine_econophysics",
        "free_parameters": 0,
        "n_bars": r["n_bars"],
        "directional_accuracy": r.get("directional_accuracy_1d"),
        "directional_accuracy_active": r.get("directional_accuracy_1d"),
        "directional_accuracy_5d": r.get("directional_accuracy_5d"),
        "directional_accuracy_20d": r.get("directional_accuracy_20d"),
        "raw_emergence_1d_accuracy": r.get("directional_accuracy_1d"),
        "boosted_1d_accuracy": r.get("directional_accuracy_1d"),
        "boosted_20d_accuracy": r.get("directional_accuracy_20d"),
        "lift_1d_acc": None,
        "lift_20d_acc": None,
        "information_coefficient": None,
        "sharpe": r.get("sharpe"),
        "max_drawdown": None,
        "buy_hold_return": None,
        "strategy_return": None,
        "buy_hold_max_drawdown": None,
        "hit_rate_long": None,
        "hit_rate_short": None,
        "avg_pred_return": None,
        "avg_actual_return": None,
        "pct_time_in_market": r.get("pct_in_market"),
        "window": r.get("fib_window", window),
        "deadzone": None,
        "S_econ": spine["Economics"]["S"],
        "S_finance": spine["Finance_Markets"]["S"],
        "S_econophysics": spine["Econophysics"]["S"],
        "consciousness_factor": r.get("consciousness_factor"),
        "note": r.get("note"),
        "vol_persistence": r.get("vol_persistence"),
        "kelly_f": r.get("kelly_f"),
        "poof": r.get("poof"),
        "directional_accuracy_1d_ungated": r.get("directional_accuracy_1d_ungated"),
        "remedies_applied": [
            "growth_term_engine",
            "quirk_mod_consciousness_factor",
            "market_fluctuation_sentiment_to_delta_psi",
            "D_eff_scale_ladder",
            "vol_persistence_beta_equals_gamma",
            "phi_ewma_drift",
            "mu_K_C_growth_d_observer",
            "poof_decoherence_extreme",
            "kelly_gate_edge_ge_gamma",
        ],
    }
