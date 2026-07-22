#!/usr/bin/env python3
"""Smoke: FSOT Monte Carlo observer-collapse ensemble on historical OHLCV."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from app.fsot.monte_carlo import mc_walkforward_hit, run_fsot_monte_carlo


def load(sym: str) -> pd.DataFrame:
    p = Path(r"D:\training data\FSOT-Market-History\ohlcv") / f"{sym}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df = df.rename(columns={c: str(c).lower() for c in df.columns}).reset_index()
    df = df.rename(columns={df.columns[0]: "time"})
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            df[c] = df["close"] if c != "volume" else 0
    return df


def main() -> None:
    symbols = ["SPY", "QQQ", "AAPL"]
    for sym in symbols:
        df = load(sym)
        # Use last ~2y for speed if very long
        if len(df) > 600:
            df = df.iloc[-600:].copy()

        print(f"\n=== {sym}  n_bars={len(df)}  last_close={float(df['close'].iloc[-1]):.2f} ===")
        mc = run_fsot_monte_carlo(
            df,
            horizon=21,
            n_paths=256,
            sentiment=0.0,
            window=21,
            symbol=sym,
            seed=42,
        )
        if mc.get("error"):
            print("ERROR", mc["error"])
            continue
        ens = mc["ensemble"]
        print(
            f"  signal={mc['signal']:5} conf={mc['confidence']:.3f} "
            f"p_up={ens['p_up']:.3f} p_up_obs={ens['p_up_observed_branch']:.3f}"
        )
        print(
            f"  E[r]={ens['expected_return']:+.4f} med={ens['median_return']:+.4f} "
            f"mode_bin=[{ens['most_probable_return_bin']['low']:+.4f},"
            f"{ens['most_probable_return_bin']['high']:+.4f}]"
        )
        print(
            f"  collapse_true_frac={ens['mean_collapse_true_fraction']:.3f} "
            f"terminal p50={ens['quantiles_price']['p50']:.2f} "
            f"p10={ens['quantiles_price']['p10']:.2f} p90={ens['quantiles_price']['p90']:.2f}"
        )
        print(f"  free_params={mc['free_parameters']} method={mc['method']}")

        # Causal walk-forward (lighter)
        wf = mc_walkforward_hit(
            df, horizon=5, n_paths=64, window=21, step=15, sentiment=0.0
        )
        if not wf.get("error"):
            print(
                f"  MC walkforward h=5: acc={wf.get('directional_accuracy')} "
                f"acc_conf={wf.get('directional_accuracy_confident')} "
                f"n={wf.get('n_eval')} n_conf={wf.get('n_confident')}"
            )
        else:
            print("  MC walkforward:", wf.get("error"))

    # Dump one full JSON sample for inspection
    df = load("SPY")
    if len(df) > 600:
        df = df.iloc[-600:].copy()
    sample = run_fsot_monte_carlo(
        df, horizon=13, n_paths=128, sentiment=0.15, symbol="SPY", seed=7
    )
    out = ROOT / "scripts" / "_mc_sample_spy.json"
    # strip large fan if needed — keep fan
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)
    print(f"\nWrote sample → {out}")


if __name__ == "__main__":
    main()
