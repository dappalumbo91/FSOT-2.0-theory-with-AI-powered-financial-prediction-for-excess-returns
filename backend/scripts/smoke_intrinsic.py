#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.fsot.intrinsic import (
    consciousness_factor,
    evaluate_intrinsic_walkforward,
    latest_intrinsic,
    quirk_mod,
    spine_scalars,
    growth_term,
    SEED_POOF,
    VOL_PERSISTENCE,
    KELLY_F,
)


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
    print("C (consciousness) =", consciousness_factor())
    print("quirk_mod(1.5)    =", quirk_mod(1.5, True))
    print("growth(hits=3,N=1)=", growth_term(3, 1))
    print("vol_persist γ     =", VOL_PERSISTENCE)
    print("Kelly f*          =", KELLY_F)
    print("Poof              =", SEED_POOF)
    print("SPINE", {k: round(v["S"], 6) for k, v in spine_scalars().items()})
    print()
    for sym in ["SPY", "GSPC", "BTC", "AAPL", "IXIC", "QQQ", "NVDA", "MSFT"]:
        df = load(sym)
        r = latest_intrinsic(df, symbol=sym, sentiment=0.0)
        w = evaluate_intrinsic_walkforward(df, window=21)
        print(
            f"{sym:5} sig={r['signal']:5} "
            f"1d={w.get('directional_accuracy_1d')} "
            f"ag1={w.get('acc_scale_agree_1d')} "
            f"ag5={w.get('acc_scale_agree_5d')} "
            f"ag20={w.get('acc_scale_agree_20d')} "
            f"20d={w.get('directional_accuracy_20d')} "
            f"sh={w.get('sharpe')} "
            f"in={w.get('pct_in_market')} ag%={w.get('pct_scale_agree')}"
        )


if __name__ == "__main__":
    main()
