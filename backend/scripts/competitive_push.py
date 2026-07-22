#!/usr/bin/env python3
"""
Competitive push: single-name + cross-sectional FSOT ranking.
Writes results to D:\\training data\\FSOT-Market-History\\patterns\\competitive_push.json
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

from app.fsot.intrinsic import compute_intrinsic_frame, evaluate_intrinsic_walkforward, spine_scalars

OHLCV = Path(r"D:\training data\FSOT-Market-History\ohlcv")
OUT = Path(r"D:\training data\FSOT-Market-History\patterns")
OUT.mkdir(parents=True, exist_ok=True)

EQUITIES = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "JPM", "XOM", "XLF", "TSLA", "JNJ",
]


def load(sym: str) -> pd.DataFrame:
    p = OHLCV / f"{sym}.csv"
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df = df.rename(columns={c: str(c).lower() for c in df.columns}).reset_index()
    df = df.rename(columns={df.columns[0]: "time"})
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            df[c] = df["close"] if c != "volume" else 0
    return df


def main() -> int:
    spine = spine_scalars()
    single = {}
    frames = []

    for s in EQUITIES:
        if not (OHLCV / f"{s}.csv").exists():
            continue
        print(f"  {s}…", flush=True)
        df = load(s)
        w = evaluate_intrinsic_walkforward(df, window=21)
        single[s] = {
            "acc_1d": w.get("directional_accuracy_1d"),
            "acc_1d_all": w.get("directional_accuracy_1d_ungated"),
            "acc_5d": w.get("directional_accuracy_5d"),
            "acc_20d": w.get("directional_accuracy_20d"),
            "sharpe": w.get("sharpe"),
            "pct_in_market": w.get("pct_in_market"),
            "acc_field": w.get("acc_field_gated"),
        }
        f = compute_intrinsic_frame(df, window=21)
        f["date"] = pd.to_datetime(f["time"]).dt.tz_localize(None).dt.normalize()
        f["sym"] = s
        f["fwd_1"] = f["close"].pct_change().shift(-1)
        f["fwd_5"] = f["close"].pct_change(5).shift(-5)
        f["fwd_20"] = f["close"].pct_change(20).shift(-20)
        frames.append(
            f[["date", "sym", "score", "mu", "S_live", "d_observer", "fwd_1", "fwd_5", "fwd_20"]]
        )

    panel = pd.concat(frames, ignore_index=True).dropna(subset=["score", "fwd_1"])

    def xs_hit(fwd_col: str, top_n: int = 3) -> dict:
        hits = []
        rets = []
        for _, g in panel.dropna(subset=[fwd_col]).groupby("date"):
            if len(g) < top_n * 2 + 2:
                continue
            g = g.sort_values("score")
            bot, top = g.head(top_n), g.tail(top_n)
            ls = float(top[fwd_col].mean() - bot[fwd_col].mean())
            rets.append(ls)
            hits.append(1.0 if ls > 0 else 0.0)
        if not hits:
            return {"hit": None, "n": 0, "mean_ls": None, "sharpe": None}
        r = np.array(rets)
        # annualize roughly by horizon
        horizon = int(fwd_col.split("_")[1])
        sh = float(r.mean() / r.std() * np.sqrt(252 / horizon)) if r.std() > 0 else 0.0
        return {
            "hit": float(np.mean(hits)),
            "n": len(hits),
            "mean_ls": float(r.mean()),
            "sharpe": sh,
        }

    xs = {
        "1d": xs_hit("fwd_1"),
        "5d": xs_hit("fwd_5"),
        "20d": xs_hit("fwd_20"),
    }

    acc20 = [v["acc_20d"] for v in single.values() if v.get("acc_20d") is not None]
    acc1 = [v["acc_1d"] for v in single.values() if v.get("acc_1d") is not None]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "fsot_full_engine_econophysics",
        "free_parameters": 0,
        "spine": {k: v["S"] for k, v in spine.items()},
        "single_name": single,
        "single_mean_1d": float(np.mean(acc1)) if acc1 else None,
        "single_mean_20d": float(np.mean(acc20)) if acc20 else None,
        "cross_section_long_short_top3": xs,
        "competitor_baselines": {
            "random_direction": 0.50,
            "always_long_up_day_rate": 0.53,
            "careful_ml_1d_walkforward": [0.50, 0.54],
            "selective_trend_multiday": [0.55, 0.65],
            "factor_model_typical_level_error_pct": 10.0,
        },
        "note": (
            "Cross-section: each day long top-3 FSOT score, short bottom-3. "
            "Hit = top basket beats bottom basket over horizon."
        ),
    }

    path = OUT / "competitive_push.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "single_mean_1d": report["single_mean_1d"],
        "single_mean_20d": report["single_mean_20d"],
        "xs_1d": xs["1d"],
        "xs_5d": xs["5d"],
        "xs_20d": xs["20d"],
        "best_single_20d": max(
            ((k, v["acc_20d"]) for k, v in single.items() if v.get("acc_20d")),
            key=lambda x: x[1],
        ),
    }, indent=2))
    print("Wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
