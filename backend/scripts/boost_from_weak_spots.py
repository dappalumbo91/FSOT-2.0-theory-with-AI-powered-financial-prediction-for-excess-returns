#!/usr/bin/env python3
"""
1) Audit weak spots across historical OHLCV
2) Run FSOT boosted v3 vs v2 baseline
3) Write gap-fill report to game drive
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

from app.fsot.weak_spots import aggregate_weak_spots, diagnose_symbol  # noqa: E402
from app.predict.boosted import evaluate_boosted  # noqa: E402

HISTORY = Path(r"D:\training data\FSOT-Market-History")
OHLCV = HISTORY / "ohlcv"
OUT = HISTORY / "patterns"
OUT.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    out = df.reset_index()
    out = out.rename(columns={out.columns[0]: "time"})
    for c in ("open", "high", "low", "close", "volume"):
        if c not in out.columns:
            out[c] = out["close"] if c != "volume" else 0.0
    return out


def main() -> int:
    files = sorted(OHLCV.glob("*.csv"))
    if not files:
        print("No OHLCV — run download_history.py")
        return 1

    # Market benchmark for cross-section
    spy_path = OHLCV / "SPY.csv"
    market_df = load_csv(spy_path) if spy_path.exists() else None

    weak_reports = []
    boost_rows = []
    print(f"Auditing + boosting {len(files)} symbols…")

    for p in files:
        sym = p.stem
        df = load_csv(p)
        print(f"  {sym}…", end=" ", flush=True)
        try:
            w = diagnose_symbol(df, symbol=sym)
            weak_reports.append(w)
            use_mkt = market_df if sym.upper() not in ("SPY", "GSPC") else None
            b = evaluate_boosted(df, symbol=sym, market_df=use_mkt)
            boost_rows.append(b)
            v3 = (b.get("v3") or {}).get("1d") or {}
            v2 = (b.get("v2_baseline") or {}).get("1d") or {}
            v3_20 = (b.get("v3") or {}).get("20d") or {}
            print(
                f"v2={_pct(v2.get('directional_accuracy_active'))} "
                f"v3={_pct(v3.get('directional_accuracy_active'))} "
                f"v3_20={_pct(v3_20.get('directional_accuracy_active'))} "
                f"lift={_pp(b.get('lift_1d_acc'))} "
                f"flags={w.get('flags')}"
            )
        except Exception as e:
            print(f"ERR {e}")
            weak_reports.append({"symbol": sym, "error": str(e)})
            boost_rows.append({"symbol": sym, "error": str(e)})

    agg = aggregate_weak_spots(weak_reports)

    def mean_metric(rows, path_keys, key="directional_accuracy_active"):
        vals = []
        for r in rows:
            cur = r
            for k in path_keys:
                cur = (cur or {}).get(k) if isinstance(cur, dict) else None
            if isinstance(cur, dict) and cur.get(key) is not None and not cur.get("error"):
                vals.append(cur[key])
        return float(np.mean(vals)) if vals else None

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_symbols": len(boost_rows),
        "mean_v2_1d_acc": mean_metric(boost_rows, ["v2_baseline", "1d"]),
        "mean_v3_1d_acc": mean_metric(boost_rows, ["v3", "1d"]),
        "mean_v2_20d_acc": mean_metric(boost_rows, ["v2_baseline", "20d"]),
        "mean_v3_20d_acc": mean_metric(boost_rows, ["v3", "20d"]),
        "mean_v3_1d_sharpe_net": mean_metric(boost_rows, ["v3", "1d"], key="sharpe_net"),
        "mean_v3_1d_pct_in_market": mean_metric(boost_rows, ["v3", "1d"], key="pct_in_market"),
        "mean_lift_1d": float(
            np.mean([r["lift_1d_acc"] for r in boost_rows if r.get("lift_1d_acc") is not None])
        )
        if any(r.get("lift_1d_acc") is not None for r in boost_rows)
        else None,
        "mean_lift_20d": float(
            np.mean([r["lift_20d_acc"] for r in boost_rows if r.get("lift_20d_acc") is not None])
        )
        if any(r.get("lift_20d_acc") is not None for r in boost_rows)
        else None,
    }

    # Flagship table
    flagship = {}
    for r in boost_rows:
        if r.get("symbol") in ("SPY", "GSPC", "BTC", "IXIC", "QQQ", "VIX", "AAPL", "NVDA"):
            flagship[r["symbol"]] = {
                "v2_1d": (r.get("v2_baseline") or {}).get("1d"),
                "v3_1d": (r.get("v3") or {}).get("1d"),
                "v3_20d": (r.get("v3") or {}).get("20d"),
                "lift_1d": r.get("lift_1d_acc"),
                "lift_20d": r.get("lift_20d_acc"),
                "latest": r.get("latest"),
            }

    report = {
        "summary": summary,
        "weak_spot_aggregate": agg,
        "per_symbol_weak": weak_reports,
        "per_symbol_boost": boost_rows,
        "flagship": flagship,
        "interpretation": {
            "weak_spots": "Historical slices where regime signal loses edge (conflict, entropy rising, high vol, inverse assets).",
            "boost": "v3 applies FSOT multi-field composite + gates derived from those weak spots; costs applied for net Sharpe.",
            "not_magic": "1d markets remain near-efficient; lift comes from selectivity and multi-day structure.",
        },
    }

    out_json = OUT / "weak_spots_and_boost_v3.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Markdown summary
    md = []
    md.append("# FSOT Gap Fill — Weak Spots & Boost v3\n")
    md.append(f"Generated: {summary['generated_at']}\n")
    md.append("## Weak spots (cross-asset)\n")
    md.append("### Flags\n")
    for k, v in (agg.get("flag_counts") or {}).items():
        md.append(f"- `{k}`: {v} assets\n")
    md.append("\n### Slice accuracy (mean hit 1d)\n")
    md.append("| Slice | Mean hit 1d | n assets |\n|---|---|---|\n")
    for row in (agg.get("slice_means_weakest") or []) + (agg.get("slice_means_strongest") or []):
        # de-dupe later visually
        md.append(
            f"| {row['name']} | {row['mean_hit_1d']*100:.1f}% | {row['n_assets']} |\n"
        )
    md.append("\n### By asset class\n")
    for k, v in (agg.get("accuracy_by_asset_class") or {}).items():
        md.append(f"- **{k}**: {v['mean_hit_1d']*100:.1f}% (n={v['n']})\n")

    md.append("\n## Before / After\n")
    md.append("| Metric | v2 | v3 boosted |\n|---|---|---|\n")
    md.append(
        f"| Mean 1d active acc | {_pct(summary['mean_v2_1d_acc'])} | **{_pct(summary['mean_v3_1d_acc'])}** |\n"
    )
    md.append(
        f"| Mean 20d active acc | {_pct(summary['mean_v2_20d_acc'])} | **{_pct(summary['mean_v3_20d_acc'])}** |\n"
    )
    md.append(f"| Mean lift 1d | — | {_pp(summary['mean_lift_1d'])} |\n")
    md.append(f"| Mean lift 20d | — | {_pp(summary['mean_lift_20d'])} |\n")
    md.append(
        f"| Mean 1d Sharpe (net of costs) | — | {summary['mean_v3_1d_sharpe_net']:.3f} |\n"
        if summary.get("mean_v3_1d_sharpe_net") is not None
        else ""
    )
    md.append(
        f"| Mean % time in market (v3 1d) | — | {_pct(summary['mean_v3_1d_pct_in_market'])} |\n"
    )

    md.append("\n## Flagship\n")
    md.append("| Symbol | v2 1d | v3 1d | v3 20d | lift 1d |\n|---|---|---|---|---|\n")
    for sym, f in flagship.items():
        md.append(
            f"| {sym} | {_pct((f.get('v2_1d') or {}).get('directional_accuracy_active'))} | "
            f"{_pct((f.get('v3_1d') or {}).get('directional_accuracy_active'))} | "
            f"{_pct((f.get('v3_20d') or {}).get('directional_accuracy_active'))} | "
            f"{_pp(f.get('lift_1d'))} |\n"
        )

    md.append("\n## Remedies applied\n")
    md.append(
        "1. Score × momentum agreement\n"
        "2. Block when entropy rising\n"
        "3. Wider deadzone in high vol\n"
        "4. Invert VIX class\n"
        "5. Stricter crypto gates\n"
        "6. Multi-horizon FSOT coherence (fast/slow emergence)\n"
        "7. Composite field: emergence + z(dS) + z(T1) − z(entropy)\n"
        "8. Cross-sectional vs SPY\n"
        "9. Cost-aware net Sharpe\n"
    )
    md_path = OUT / "GAP_FILL_BOOST_V3.md"
    md_path.write_text("".join(md), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {md_path}")
    return 0


def _pct(x):
    if x is None:
        return "—"
    return f"{100*float(x):.1f}%"


def _pp(x):
    if x is None:
        return "—"
    return f"{100*float(x):+.2f}pp"


if __name__ == "__main__":
    raise SystemExit(main())
