"""Intrinsic frame: no free-param explosion; finite seed scores."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.fsot.intrinsic import compute_intrinsic_frame, latest_intrinsic  # noqa: E402


def _synthetic_ohlcv(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.015, n)
    close = 100 * np.exp(np.cumsum(rets))
    times = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(1e6, 5e6, n),
        }
    )


def test_intrinsic_frame_finite():
    df = _synthetic_ohlcv()
    frame = compute_intrinsic_frame(df, window=21)
    assert len(frame) == len(df)
    assert np.isfinite(frame["S"].iloc[-1])
    assert np.isfinite(frame["quirk_mod"].iloc[-1])
    assert np.isfinite(frame["consciousness_factor"].iloc[-1])
    assert frame["D_eff"].iloc[-1] > 0
    assert frame["signal"].iloc[-1] in ("LONG", "SHORT", "FLAT")


def test_latest_reports_zero_free():
    df = _synthetic_ohlcv()
    r = latest_intrinsic(df, symbol="TEST")
    assert r["free_parameters"] == 0
    assert r["method"] == "fsot_full_engine_econophysics"
    assert "growth" in r
    assert "mu" in r
    assert "sig_pred" in r
    assert "quirk_mod" in r
    assert "consciousness_factor" in r
    assert "S_econ" in r
    assert "S_finance" in r
