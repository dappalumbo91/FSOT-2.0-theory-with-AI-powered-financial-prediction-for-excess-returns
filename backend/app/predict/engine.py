"""FSOT prediction engine — intrinsic zero-free path + Monte Carlo collapse ensemble."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.fsot.intrinsic import (
    compute_intrinsic_frame,
    evaluate_intrinsic_walkforward,
    latest_intrinsic,
    spine_scalars,
)
from app.fsot.monte_carlo import (
    DEFAULT_HORIZON,
    DEFAULT_N_PATHS,
    mc_walkforward_hit,
    run_dynamic_fsot_monte_carlo,
    run_fsot_monte_carlo,
)


class PredictEngine:
    def predict_frame(
        self,
        df: pd.DataFrame,
        window: int = 21,
        domain: str = "Economics",
        observer_mod: float = 0.0,
        symbol: str = "",
        market_df: pd.DataFrame | None = None,
        boosted: bool = False,
    ) -> pd.DataFrame:
        return compute_intrinsic_frame(df, window=window, sentiment=float(observer_mod))

    def latest_prediction(
        self,
        df: pd.DataFrame,
        window: int = 21,
        horizons: list[int] | None = None,
        domain: str = "Economics",
        observer_mod: float = 0.0,
        symbol: str = "",
        market_df: pd.DataFrame | None = None,
        boosted: bool = False,
        include_monte_carlo: bool = False,
        mc_horizon: int = DEFAULT_HORIZON,
        mc_n_paths: int = DEFAULT_N_PATHS,
    ) -> dict[str, Any]:
        horizons = horizons or [1, 5, 20]
        sent = float(observer_mod)
        result = latest_intrinsic(df, window=window, symbol=symbol, sentiment=sent)
        if result.get("error"):
            return result

        # Multi-horizon: Fib windows → different preregistered D_eff folds
        fib_for = {1: 8, 5: 21, 20: 55}
        horizon_preds = {}
        for h in horizons:
            w = fib_for.get(h, 21)
            r_h = latest_intrinsic(df, window=w, symbol=symbol, sentiment=sent)
            if r_h.get("error"):
                continue
            horizon_preds[str(h)] = {
                "horizon_days": h,
                "window": w,
                "D_eff": r_h.get("D_eff"),
                "route_name": r_h.get("route_name"),
                "quirk_mod": r_h.get("quirk_mod"),
                "consciousness_factor": r_h.get("consciousness_factor"),
                "emergence_score": r_h.get("score"),
                "S": r_h.get("S_live"),
                "pred_return": r_h.get("pred_return"),
                "signal": r_h.get("signal"),
            }
        result["horizons"] = horizon_preds

        if include_monte_carlo:
            mc = run_dynamic_fsot_monte_carlo(
                df,
                horizon=mc_horizon,
                n_paths=mc_n_paths,
                sentiment=sent,
                window=window,
                symbol=symbol,
                persist=True,
            )
            result["monte_carlo"] = mc
            if not mc.get("error"):
                result["mc_signal"] = mc.get("signal")
                result["mc_confidence"] = mc.get("confidence")
                result["mc_p_up"] = mc.get("ensemble", {}).get("p_up_observed_branch")
                result["mc_pattern"] = mc.get("pattern")
        return result

    def monte_carlo(
        self,
        df: pd.DataFrame,
        *,
        horizon: int = DEFAULT_HORIZON,
        n_paths: int = DEFAULT_N_PATHS,
        observer_mod: float = 0.0,
        window: int = 21,
        symbol: str = "",
        seed: int | None = None,
        dynamic: bool = True,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Multi-path FSOT simulation; dynamic=True trains/solidifies patterns on history."""
        if dynamic:
            return run_dynamic_fsot_monte_carlo(
                df,
                horizon=horizon,
                n_paths=n_paths,
                sentiment=float(observer_mod),
                window=window,
                symbol=symbol,
                seed=seed,
                persist=persist,
            )
        return run_fsot_monte_carlo(
            df,
            horizon=horizon,
            n_paths=n_paths,
            sentiment=float(observer_mod),
            window=window,
            symbol=symbol,
            seed=seed,
            dynamic=False,
        )

    def monte_carlo_walkforward(
        self,
        df: pd.DataFrame,
        *,
        horizon: int = 5,
        n_paths: int = 128,
        window: int = 21,
        step: int = 5,
        observer_mod: float = 0.0,
        dynamic: bool = True,
        symbol: str = "",
    ) -> dict[str, Any]:
        return mc_walkforward_hit(
            df,
            horizon=horizon,
            n_paths=n_paths,
            window=window,
            step=step,
            sentiment=float(observer_mod),
            dynamic=dynamic,
            symbol=symbol,
        )

    def walkforward(
        self, df: pd.DataFrame, window: int = 21, sentiment: float = 0.0
    ) -> dict[str, Any]:
        return evaluate_intrinsic_walkforward(df, window=window, sentiment=sentiment)


def theory_constants() -> dict[str, Any]:
    spine = spine_scalars()
    return {
        "formula": "S = K * (T1 + T2 + T3)",
        "free_parameters": 0,
        "Economics_S": spine["Economics"]["S"],
        "Finance_Markets_S": spine["Finance_Markets"]["S"],
        "authority": "I:/FSOT-Physical-Archive/02_FSOT-2.1-Lean-Full/vendor/fsot_compute.py",
        "monte_carlo": "fsot_dynamic_monte_carlo_pattern_collapse",
    }
