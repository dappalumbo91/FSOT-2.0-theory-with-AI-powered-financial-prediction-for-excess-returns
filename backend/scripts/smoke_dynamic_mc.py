#!/usr/bin/env python3
"""Smoke: intelligent dynamic FSOT Monte Carlo with pattern solidification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from app.fsot.monte_carlo import run_dynamic_fsot_monte_carlo, train_pattern_memory
from app.fsot.pattern_memory import PatternMemory, SOLIDIFY_ACC


def load(sym: str) -> pd.DataFrame:
    p = Path(r"D:\training data\FSOT-Market-History\ohlcv") / f"{sym}.csv"
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df = df.rename(columns={c: str(c).lower() for c in df.columns}).reset_index()
    df = df.rename(columns={df.columns[0]: "time"})
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            df[c] = df["close"] if c != "volume" else 0
    return df


def main() -> None:
    print(f"SOLIDIFY_ACC (0.5+Poof) = {SOLIDIFY_ACC:.4f}")
    print(f"free_parameters = 0\n")

    for sym in ["SPY", "QQQ", "AAPL"]:
        df = load(sym)
        if len(df) > 1200:
            df = df.iloc[-1200:].copy()
        print(f"=== {sym} bars={len(df)} ===")

        mem, diag = train_pattern_memory(
            df, window=21, symbol=sym, feedback_horizon=1, step=1
        )
        print(
            f"  train: patterns={diag['memory']['n_patterns']} "
            f"solid={diag['memory']['n_solidified']} "
            f"raw_acc={diag.get('raw_directional_accuracy')} "
            f"anchored_acc={diag.get('anchored_directional_accuracy')}"
        )
        if diag.get("refinement_lift") is not None:
            print(
                f"  refine: early={diag['anchored_early']:.3f} "
                f"late={diag['anchored_late']:.3f} "
                f"lift={diag['refinement_lift']:+.3f}"
            )
        tops = diag["memory"].get("top_patterns") or []
        for t in tops[:3]:
            print(
                f"    · {t['key'][:48]}… solid={t['solidified']} "
                f"accφ={t['acc_phi']} n={t['trials']} dir={t['preferred_dir']}"
            )

        mc = run_dynamic_fsot_monte_carlo(
            df,
            horizon=21,
            n_paths=256,
            symbol=sym,
            seed=42,
            persist=True,
            max_train_bars=800,
        )
        ens = mc["ensemble"]
        pat = mc.get("pattern") or {}
        bias = pat.get("bias") or {}
        print(
            f"  MC: signal={mc['signal']} conf={mc['confidence']:.3f} "
            f"p_up_obs={ens['p_up_observed_branch']:.3f} "
            f"solid={bias.get('solidified')} strength={bias.get('strength', 0):.3f}"
        )
        print(f"  method={mc['method']} free={mc['free_parameters']}")
        print()

    # sample dump
    df = load("SPY")
    if len(df) > 800:
        df = df.iloc[-800:].copy()
    sample = run_dynamic_fsot_monte_carlo(
        df, horizon=13, n_paths=128, symbol="SPY", seed=7, persist=True
    )
    # trim for json
    slim = {
        k: sample[k]
        for k in (
            "method",
            "free_parameters",
            "signal",
            "confidence",
            "ensemble",
            "pattern",
            "training",
            "dynamic",
            "note",
        )
        if k in sample
    }
    out = ROOT / "scripts" / "_dynamic_mc_spy.json"
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
