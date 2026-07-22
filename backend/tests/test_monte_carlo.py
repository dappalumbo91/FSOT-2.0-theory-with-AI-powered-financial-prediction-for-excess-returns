"""FSOT Monte Carlo: free_params=0, collapse bounds, pattern solidification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.fsot.monte_carlo import (
    collapse_probability,
    run_dynamic_fsot_monte_carlo,
    run_fsot_monte_carlo,
    train_pattern_memory,
)
from app.fsot.pattern_memory import PatternMemory, SOLIDIFY_ACC
from app.fsot.intrinsic import SEED_C, SEED_POOF


def _synthetic_df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0002, 0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + r)
    return pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=n, freq="B"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(1e6, 2e6, size=n),
        }
    )


def test_collapse_probability_bounded():
    p = collapse_probability(1.5, 1.5, 0.0, SEED_C)
    assert SEED_POOF <= p <= 1.0 - SEED_POOF
    p_hi = collapse_probability(3.0, 1.5, 0.8, SEED_C)
    p_lo = collapse_probability(1.5, 1.5, -0.5, SEED_C)
    assert p_hi >= p_lo


def test_run_mc_ensemble_shape():
    df = _synthetic_df()
    mc = run_fsot_monte_carlo(
        df, horizon=13, n_paths=64, sentiment=0.1, symbol="TEST", seed=1
    )
    assert mc.get("error") is None
    assert mc["free_parameters"] == 0
    assert mc["method"] == "fsot_monte_carlo_observer_collapse"
    assert mc["horizon"] in (5, 8, 13, 21, 34, 55)
    ens = mc["ensemble"]
    assert 0.0 <= ens["p_up"] <= 1.0
    assert abs(ens["p_up"] + ens["p_down"] - 1.0) < 1e-9
    assert mc["signal"] in ("LONG", "SHORT", "FLAT")
    assert ens["quantiles_price"]["p10"] <= ens["quantiles_price"]["p50"]
    assert ens["quantiles_price"]["p50"] <= ens["quantiles_price"]["p90"]
    assert len(mc["fan_chart"]) >= 2


def test_mc_insufficient_data():
    df = _synthetic_df(n=20)
    mc = run_fsot_monte_carlo(df, horizon=5, n_paths=32)
    assert mc.get("error") == "insufficient_data"


def test_pattern_memory_solidify_seed_gate():
    mem = PatternMemory(symbol="X")
    # Force a pattern to solidify with consistent hits
    key = "D19|mu1|d1|f1|s0|q1|p0|a1|imid"
    for i in range(12):
        mem.observe(key, pred_dir=1, realized_dir=1, bar_index=i)
    rec = mem.get(key)
    assert rec.trials >= 8
    assert rec.acc_phi >= SOLIDIFY_ACC or rec.solidified
    assert rec.solidified is True
    assert rec.preferred_dir == 1
    bias = mem.bias_for(key)
    assert bias["solidified"] == 1.0
    assert bias["dir"] == 1.0
    assert bias["mu_scale"] >= 1.0


def test_train_and_dynamic_mc():
    # mild trend so some signatures can solidify
    rng = np.random.default_rng(2)
    n = 200
    r = rng.normal(0.0015, 0.008, size=n)
    close = 100.0 * np.cumprod(1.0 + r)
    df = pd.DataFrame(
        {
            "time": pd.date_range("2018-01-01", periods=n, freq="B"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(1e6, 2e6, size=n),
        }
    )
    mem, diag = train_pattern_memory(df, window=21, symbol="T", step=1)
    assert diag.get("error") is None
    assert diag["free_parameters"] == 0
    assert mem.n_updates > 0
    mc = run_dynamic_fsot_monte_carlo(
        df, horizon=8, n_paths=48, symbol="T", seed=3, persist=False
    )
    assert mc.get("error") is None
    assert mc["free_parameters"] == 0
    assert mc["dynamic"] is True
    assert "pattern" in mc
    assert mc["method"] == "fsot_dynamic_monte_carlo_pattern_collapse"
