#!/usr/bin/env python3
"""
Build FSOT emergence/entropy historical pattern ledger from downloaded OHLCV.

Writes to: D:\\training data\\FSOT-Market-History\\patterns\\
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backtest.walk_forward import walk_forward_backtest  # noqa: E402
from app.fsot.emergence import compute_emergence_frame  # noqa: E402
from app.predict.engine import PredictEngine  # noqa: E402

HISTORY = Path(r"D:\training data\FSOT-Market-History")
OHLCV = HISTORY / "ohlcv"
PATTERNS = HISTORY / "patterns"
PATTERNS.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "close" not in df.columns:
        raise ValueError(f"no close in {path}")
    out = df.reset_index()
    first = out.columns[0]
    out = out.rename(columns={first: "time"})
    for c in ("open", "high", "low", "close", "volume"):
        if c not in out.columns:
            if c == "volume":
                out[c] = 0.0
            else:
                out[c] = out["close"]
    return out


def analyze_symbol(path: Path) -> dict:
    sym = path.stem
    df = load_csv(path)
    if len(df) < 100:
        return {"symbol": sym, "error": "too_short", "bars": len(df)}

    engine = PredictEngine()
    frame = engine.predict_frame(df, window=30)
    frame = frame.dropna(subset=["S", "emergence_ema"]).copy()
    frame["fwd_1"] = frame["close"].pct_change().shift(-1)
    frame["fwd_5"] = frame["close"].pct_change(5).shift(-5)
    frame = frame.dropna(subset=["fwd_1"])

    score = frame["emergence_ema"]
    # Pattern bins
    bins = [-np.inf, -0.75, -0.25, 0.25, 0.75, np.inf]
    labels = ["strong_dispersal", "dispersal", "neutral", "emergence", "strong_emergence"]
    frame["regime_bin"] = pd.cut(score, bins=bins, labels=labels)

    pattern_rows = []
    for lab in labels:
        sub = frame[frame["regime_bin"] == lab]
        if len(sub) < 20:
            continue
        pattern_rows.append(
            {
                "regime": lab,
                "n": int(len(sub)),
                "mean_fwd_1": float(sub["fwd_1"].mean()),
                "mean_fwd_5": float(sub["fwd_5"].dropna().mean()) if sub["fwd_5"].notna().any() else None,
                "hit_rate_1d": float((sub["fwd_1"] > 0).mean()),
                "mean_entropy": float(sub["entropy"].mean()),
                "mean_dS": float(sub["dS"].mean()),
                "mean_S": float(sub["S"].mean()),
            }
        )

    bt = walk_forward_backtest(df, window=30)

    # Decade splits
    decades = {}
    if "time" in frame.columns:
        years = pd.to_datetime(frame["time"]).dt.year
        for decade_start in range(2005, 2030, 5):
            mask = (years >= decade_start) & (years < decade_start + 5)
            sub = frame.loc[mask]
            if len(sub) < 80:
                continue
            sc = sub["emergence_ema"].values
            ac = sub["fwd_1"].values
            active = np.abs(sc) > 0.25
            if active.sum() < 20:
                continue
            acc = float((np.sign(sc[active]) == np.sign(ac[active])).mean())
            decades[f"{decade_start}-{decade_start+4}"] = {
                "n": int(active.sum()),
                "directional_accuracy_active": acc,
            }

    # Save per-symbol emergence series (parquet-friendly csv)
    series_path = PATTERNS / f"{sym}_emergence_series.csv"
    cols = [c for c in ["time", "close", "S", "dS", "entropy", "emergence_ema", "pred_return", "regime"] if c in frame.columns]
    frame[cols].to_csv(series_path, index=False)

    return {
        "symbol": sym,
        "bars": int(len(df)),
        "first": str(pd.to_datetime(df["time"].iloc[0]).date()) if "time" in df.columns else None,
        "last": str(pd.to_datetime(df["time"].iloc[-1]).date()) if "time" in df.columns else None,
        "patterns": pattern_rows,
        "backtest": bt,
        "decades": decades,
        "series_path": str(series_path),
    }


def main() -> int:
    files = sorted(OHLCV.glob("*.csv"))
    if not files:
        print(f"No OHLCV in {OHLCV} — run download_history.py first")
        return 1

    results = []
    print(f"Analyzing {len(files)} symbols from {OHLCV}")
    for p in files:
        print(f"  {p.stem}...", end=" ", flush=True)
        try:
            r = analyze_symbol(p)
            acc = r.get("backtest", {}).get("directional_accuracy_active")
            print(f"acc={acc}")
            results.append(r)
        except Exception as e:
            print(f"ERR {e}")
            results.append({"symbol": p.stem, "error": str(e)})

    # Cross-asset summary
    ok = [r for r in results if r.get("backtest") and not r["backtest"].get("error")]
    if ok:
        mean_acc = float(np.mean([r["backtest"]["directional_accuracy_active"] for r in ok]))
        mean_ic = float(np.mean([r["backtest"].get("information_coefficient", 0) for r in ok]))
    else:
        mean_acc = mean_ic = 0.0

    # Aggregate regime hit rates across assets
    regime_pool: dict[str, list[float]] = {}
    for r in results:
        for pat in r.get("patterns") or []:
            regime_pool.setdefault(pat["regime"], []).append(pat["hit_rate_1d"])
    regime_summary = {
        k: {"mean_hit_rate_1d": float(np.mean(v)), "n_assets": len(v)}
        for k, v in regime_pool.items()
    }

    ledger = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "emergence_entropy_v2",
        "n_symbols": len(results),
        "mean_directional_accuracy_active": mean_acc,
        "mean_information_coefficient": mean_ic,
        "regime_summary": regime_summary,
        "assets": results,
        "interpretation": {
            "strong_emergence": "Expect higher P(next ret > 0) if mapping is correct",
            "strong_dispersal": "Expect higher P(next ret < 0) / risk-off",
            "neutral": "FLAT / no edge",
        },
    }
    out = PATTERNS / "emergence_entropy_ledger.json"
    out.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"\nLedger → {out}")
    print(f"Mean active directional accuracy: {mean_acc:.3f}")
    print(f"Mean IC: {mean_ic:.3f}")
    print("Regime summary:", json.dumps(regime_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
