"""Synthetic USD paper portfolio — causal, free_params=0."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.paper_portfolio import run_paper_portfolio


def _df(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + r)
    return pd.DataFrame(
        {
            "time": pd.date_range("2019-01-01", periods=n, freq="B"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(1e6, 2e6, size=n),
        }
    )


def test_paper_capital_scales():
    df = _df()
    a = run_paper_portfolio(df, capital=10_000, mode="buy_hold", symbol="T")
    b = run_paper_portfolio(df, capital=50_000, mode="buy_hold", symbol="T")
    assert a.get("error") is None
    assert b.get("error") is None
    assert a["free_parameters"] == 0
    # Returns should match; P&L scales with capital
    assert abs(a["total_return"] - b["total_return"]) < 1e-9
    assert abs(b["total_pnl"] / a["total_pnl"] - 5.0) < 1e-6 or abs(a["total_pnl"]) < 1e-6


def test_paper_modes():
    df = _df(n=250)
    for mode in ("always_in", "solid_gated", "long_only", "buy_hold"):
        r = run_paper_portfolio(df, capital=5000, mode=mode, symbol="T")
        assert r.get("error") is None
        assert r["capital_start"] == 5000
        assert r["capital_end"] > 0
        assert len(r["equity_curve"]) >= 2


def test_bhs_mode():
    df = _df(n=400, seed=3)
    r = run_paper_portfolio(df, capital=10_000, mode="bhs", symbol="T")
    assert r.get("error") is None
    assert r["method"] == "fsot_bhs_buy_hold_sell"
    assert r["free_parameters"] == 0
    assert "commit_directional_accuracy" in r
    assert "progress_to_70_80" in r
    assert r["mode"] == "bhs"

