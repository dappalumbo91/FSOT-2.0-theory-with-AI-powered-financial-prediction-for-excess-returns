#!/usr/bin/env python3
"""
Evaluate Buy/Hold/Sell progress toward 70–80% commit accuracy + paper profit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from app.fsot.bhs_engine import run_bhs_backtest

HISTORY = Path(r"D:\training data\FSOT-Market-History\ohlcv")
OUT = Path(r"D:\training data\FSOT-Market-History\verification\bhs_target_eval.json")
SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "BTC", "ETH", "IWM"]


def load(sym: str, max_bars: int = 2500) -> pd.DataFrame | None:
    p = HISTORY / f"{sym}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df = df.rename(columns={c: str(c).lower() for c in df.columns}).reset_index()
    df = df.rename(columns={df.columns[0]: "time"})
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            df[c] = df["close"] if c != "volume" else 0
    if len(df) > max_bars:
        df = df.iloc[-max_bars:].copy()
    return df


def main() -> None:
    capital = 10_000.0
    rows = []
    print("=" * 64)
    print("FSOT BUY/HOLD/SELL → target commit accuracy 70–80%")
    print(f"capital=${capital:,.0f}  free_parameters=0")
    print("=" * 64)

    for sym in SYMBOLS:
        df = load(sym)
        if df is None:
            print(f"{sym}: missing")
            continue
        r = run_bhs_backtest(df, capital=capital, symbol=sym)
        if r.get("error"):
            print(f"{sym}: {r['error']}")
            continue
        rows.append(r)
        acc = r.get("commit_directional_accuracy")
        acc_s = "  n/a" if acc is None else f"{acc*100:5.1f}%"
        print(
            f"{sym:5} acc={acc_s:>7}  "
            f"prog={r.get('progress_to_70_80', 0)*100:5.1f}%  "
            f"pnl=${r['total_pnl']:+9.2f}  ret={r['total_return']*100:+6.2f}%  "
            f"trades={r['trades']:3d}  hold%={r['pct_hold']*100:5.1f}%  "
            f"sharpe={r['sharpe']:+.2f}  solid={r.get('pattern_memory',{}).get('n_solidified',0)}"
        )

    # Primary aggregate: symbols with enough commits (Fib 8) so noise doesn't dominate
    ok = [
        r
        for r in rows
        if r.get("commit_directional_accuracy") is not None and r.get("trades", 0) >= 8
    ]
    ok_all = [r for r in rows if r.get("commit_directional_accuracy") is not None]
    agg = {}
    if ok or ok_all:
        import numpy as np

        use = ok if ok else ok_all
        accs = [r["commit_directional_accuracy"] for r in use]
        pnls = [r["total_pnl"] for r in use]
        progs = [r["progress_to_70_80"] for r in use]
        holds = [r["pct_hold"] for r in use]
        # Trade-weighted accuracy (honest overall hit rate)
        tw_num = sum(r["commit_directional_accuracy"] * r["trades"] for r in use)
        tw_den = sum(r["trades"] for r in use)
        agg = {
            "n_symbols": len(use),
            "n_symbols_all_with_trades": len(ok_all),
            "min_trades_filter": 8 if ok else 0,
            "mean_commit_accuracy": float(np.mean(accs)),
            "median_commit_accuracy": float(np.median(accs)),
            "trade_weighted_accuracy": float(tw_num / tw_den) if tw_den else None,
            "mean_progress_to_70_80": float(np.mean(progs)),
            "mean_pnl": float(np.mean(pnls)),
            "mean_pct_hold": float(np.mean(holds)),
            "n_above_60": int(sum(1 for a in accs if a >= 0.60)),
            "n_above_70": int(sum(1 for a in accs if a >= 0.70)),
            "symbols_above_70": [
                r["symbol"] for r in use if r["commit_directional_accuracy"] >= 0.70
            ],
        }
        print("\nAGGREGATE (min 8 commits)")
        print(json.dumps(agg, indent=2))
        if agg["mean_commit_accuracy"] >= 0.70:
            print("\n>>> TARGET BAND REACHED on mean commit accuracy")
        else:
            gap = 0.70 - agg["mean_commit_accuracy"]
            print(f"\n>>> Gap to 70%: {gap*100:.1f} pts · keep refining gates / history")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"aggregate": agg, "symbols": rows}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
