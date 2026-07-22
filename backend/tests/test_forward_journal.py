"""Forward journal records and resolves predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.monitor.forward_journal import ForwardJournal


def _df(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.001, 0.02, size=n)
    close = 100.0 * np.cumprod(1.0 + r)
    return pd.DataFrame(
        {
            "time": pd.date_range("2022-01-01", periods=n, freq="D"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(1e6, 2e6, size=n),
        }
    )


def test_record_and_summary(tmp_path):
    path = tmp_path / "fwd.json"
    j = ForwardJournal(path)
    df = _df()
    out = j.record_from_market(df, symbol="BTC", horizon=5)
    assert out.get("error") is None
    assert out["entry"]["resolved"] is False
    assert out["entry"]["price_at_prediction"] > 0
    s = j.summary()
    assert s["n_open"] >= 1
    assert s["free_parameters"] == 0
