#!/usr/bin/env python3
"""
Long-term evaluation: does intelligent FSOT Monte Carlo accuracy rise over time?

Also scores the MC layer as a standalone *pattern-intelligence* system
(for possible extraction later): solidification growth, refinement lift,
solid-only hit rate, early vs late epochs.

Uses only history on D:\\training data\\FSOT-Market-History\\ohlcv
and FSOT seeds (free_parameters=0).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from app.fsot.monte_carlo import run_fsot_monte_carlo, train_pattern_memory
from app.fsot.pattern_memory import PatternMemory, SOLIDIFY_ACC, extract_state

HISTORY = Path(r"D:\training data\FSOT-Market-History")
OUT_DIR = HISTORY / "verification"
OHLCV = HISTORY / "ohlcv"

# Evaluation symbols spanning indices / mega-cap / crypto
DEFAULT_SYMBOLS = ["SPY", "QQQ", "GSPC", "AAPL", "MSFT", "NVDA", "BTC", "ETH"]


def load(sym: str) -> pd.DataFrame | None:
    p = OHLCV / f"{sym}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df = df.rename(columns={c: str(c).lower() for c in df.columns}).reset_index()
    df = df.rename(columns={df.columns[0]: "time"})
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            df[c] = df["close"] if c != "volume" else 0
    return df


def epoch_slice_accuracies(
    hits: list[float],
    solid_flags: list[bool],
    n_epochs: int = 5,
) -> list[dict]:
    """Split hit series into chronological epochs; report all vs solid-only acc."""
    n = len(hits)
    if n < n_epochs * 5:
        n_epochs = max(2, n // 20) if n >= 40 else 1
    out = []
    for e in range(n_epochs):
        lo = e * n // n_epochs
        hi = (e + 1) * n // n_epochs
        chunk = hits[lo:hi]
        sflags = solid_flags[lo:hi]
        solid_hits = [h for h, s in zip(chunk, sflags) if s]
        out.append(
            {
                "epoch": e + 1,
                "n": len(chunk),
                "accuracy": float(np.mean(chunk)) if chunk else None,
                "n_solid_calls": len(solid_hits),
                "solid_accuracy": float(np.mean(solid_hits)) if solid_hits else None,
                "solid_share": float(np.mean(sflags)) if sflags else 0.0,
            }
        )
    return out


def evaluate_symbol(
    sym: str,
    *,
    window: int = 21,
    horizon: int = 5,
    step: int = 5,
    n_paths: int = 48,
    max_bars: int | None = None,
) -> dict:
    """
    Expanding causal walk:
      - grow PatternMemory only on past bars
      - at each forecast bar: lightweight anchored dir + optional MC ensemble
      - score vs realized horizon return
    Intelligence metrics accumulate over the full series.
    """
    df = load(sym)
    if df is None or len(df) < window + horizon + 100:
        return {"symbol": sym, "error": "insufficient_or_missing_data"}

    if max_bars and len(df) > max_bars:
        df = df.iloc[-max_bars:].copy()

    close = df["close"].astype(float).values
    rets = pd.Series(close).pct_change().fillna(0.0).values
    vols = df["volume"].astype(float).values if "volume" in df.columns else None

    mem = PatternMemory(symbol=sym)
    hits_raw: list[float] = []
    hits_anchored: list[float] = []
    hits_mc: list[float] = []
    solid_flags: list[bool] = []
    solid_count_series: list[int] = []
    last_trained = window + 5

    # Forecast points
    i = window + 30
    end = len(df) - horizon - 1
    t0 = time.perf_counter()
    n_mc_calls = 0

    while i < end:
        # Incremental train through known outcomes (1d feedback, causal)
        j = last_trained
        while j < i - 1:
            block = rets[j - window + 1 : j + 1]
            vblock = vols[j - window + 1 : j + 1] if vols is not None else None
            key, _ev, pred_dir = extract_state(block, vblock, 0.0, window)
            r1 = float(close[j + 1] / close[j] - 1.0)
            real_d = 1 if r1 > 0 else (-1 if r1 < 0 else 0)
            mem.observe(key, pred_dir, real_d, bar_index=j, used_observed_branch=True)
            j += 1
        last_trained = i - 1

        # Current state
        block = rets[i - window + 1 : i + 1]
        vblock = vols[i - window + 1 : i + 1] if vols is not None else None
        key, ev, pred_dir = extract_state(block, vblock, 0.0, window)
        bias = mem.bias_for(key)
        solid = bool(bias["solidified"] > 0 and bias["dir"] != 0)

        r_fwd = float(close[i + horizon] / close[i] - 1.0)
        real_up = r_fwd > 0
        real_dir = 1 if r_fwd > 0 else (-1 if r_fwd < 0 else 0)

        # Always-in raw FSOT (no intelligence gate)
        if pred_dir != 0 and real_dir != 0:
            hits_raw.append(1.0 if (pred_dir > 0) == real_up else 0.0)

        # Intelligence gate: only commit when pattern is SOLID
        # unsolidified → skip (FLAT) — not scored as a directional call
        if solid and pred_dir != 0 and real_dir != 0:
            hit = 1.0 if (pred_dir > 0) == real_up else 0.0
            hits_anchored.append(hit)
            solid_flags.append(True)
            solid_count_series.append(sum(1 for p in mem.patterns.values() if p.solidified))
        elif real_dir != 0:
            # Track coverage: non-solid bars (no directional commit)
            solid_flags.append(False)

        # Dynamic MC (gated signal) — sample regularly + always when solid
        run_mc = solid or (n_mc_calls % 2 == 0)
        if run_mc and real_dir != 0:
            sub = df.iloc[: i + 1]
            mc = run_fsot_monte_carlo(
                sub,
                horizon=horizon,
                n_paths=n_paths,
                window=window,
                seed=i,
                symbol=sym,
                memory=mem,
                dynamic=True,
            )
            n_mc_calls += 1
            if not mc.get("error"):
                sig = str(mc.get("signal", "FLAT")).upper()
                if sig in ("LONG", "SHORT"):
                    pred_up = sig == "LONG"
                    hits_mc.append(1.0 if pred_up == real_up else 0.0)

        i += step

    elapsed = time.perf_counter() - t0
    summ = mem.summary()
    # solid_flags now marks every forecast bar; hits_anchored only solid commits
    n_forecast_bars = len(solid_flags)
    coverage = float(np.mean(solid_flags)) if solid_flags else 0.0
    solid_acc = float(np.mean(hits_anchored)) if hits_anchored else None

    # Epoch refinement on solid-gated hits only (chronological chunks of commits)
    # Also track solid share per epoch from solid_flags timeline
    epochs = []
    n_ep = 5
    if hits_anchored:
        for e in range(n_ep):
            lo = e * len(hits_anchored) // n_ep
            hi = (e + 1) * len(hits_anchored) // n_ep
            chunk = hits_anchored[lo:hi]
            # solid share of timeline in matching fraction of solid_flags
            flo = e * n_forecast_bars // n_ep
            fhi = (e + 1) * n_forecast_bars // n_ep
            share = float(np.mean(solid_flags[flo:fhi])) if fhi > flo else 0.0
            epochs.append(
                {
                    "epoch": e + 1,
                    "n_commits": len(chunk),
                    "accuracy": float(np.mean(chunk)) if chunk else None,
                    "timeline_solid_share": share,
                }
            )

    late = epochs[-1]["accuracy"] if epochs and epochs[-1]["accuracy"] is not None else 0.5
    early = epochs[0]["accuracy"] if epochs and epochs[0]["accuracy"] is not None else 0.5
    lift = late - early
    n_solid = summ["n_solidified"]
    c_lift = float(np.clip(0.5 + lift, 0.0, 1.0))
    c_solid_acc = float(np.clip(solid_acc if solid_acc is not None else 0.5, 0.0, 1.0))
    # Coverage sweet-spot: too rare or always-on both weak — peak near φ-1
    from app.fsot.intrinsic import SEED_C, SEED_PHI, SEED_POOF

    ideal_cov = 1.0 / SEED_PHI  # ≈ 0.618 of bars would be too high; use Poof..φ band
    ideal_cov = SEED_POOF + (1.0 / SEED_PHI) * (0.5 - SEED_POOF)  # ~0.28
    c_share = float(np.clip(1.0 - abs(coverage - ideal_cov) / max(ideal_cov, 0.1), 0.0, 1.0))
    c_bank = float(np.clip(n_solid / 20.0, 0.0, 1.0))
    # edge over coin-flip on solid commits
    edge = (solid_acc - 0.5) if solid_acc is not None else 0.0
    c_edge = float(np.clip(0.5 + edge * 2.0, 0.0, 1.0))

    w_c, w_phi, w_p = SEED_C, 1.0 / SEED_PHI, SEED_POOF
    w_sum = w_c + w_phi + w_p + SEED_C
    intelligence = 100.0 * (
        w_c * c_solid_acc + w_phi * c_lift + w_p * c_share + SEED_C * c_edge
    ) / w_sum

    result = {
        "symbol": sym,
        "error": None,
        "n_bars": len(df),
        "horizon": horizon,
        "step": step,
        "n_forecast_bars": n_forecast_bars,
        "n_solid_commits": len(hits_anchored),
        "n_eval_mc_commits": len(hits_mc),
        "coverage_solid": coverage,
        "elapsed_sec": round(elapsed, 2),
        "accuracy": {
            "raw_fsot_always_in": float(np.mean(hits_raw)) if hits_raw else None,
            "solid_gated_commits": solid_acc,
            "monte_carlo_gated_commits": float(np.mean(hits_mc)) if hits_mc else None,
            "n_solid_commits": len(hits_anchored),
            "n_mc_commits": len(hits_mc),
        },
        "refinement": {
            "early_solid_accuracy": early,
            "late_solid_accuracy": late,
            "lift": lift,
            "epochs": epochs,
        },
        "memory": {
            "n_patterns": summ["n_patterns"],
            "n_solidified": n_solid,
            "n_updates": summ["n_updates"],
            "n_solidify_events": summ["n_solidify_events"],
            "n_soften_events": summ["n_soften_events"],
            "solidify_threshold": SOLIDIFY_ACC,
            "top_patterns": summ.get("top_patterns", [])[:8],
            "memory_refinement": summ.get("refinement"),
        },
        "intelligence": {
            "score_0_100": round(intelligence, 2),
            "components": {
                "solid_accuracy": round(c_solid_acc, 4),
                "refinement_lift_mapped": round(c_lift, 4),
                "coverage_fit": round(c_share, 4),
                "edge_over_coinflip": round(c_edge, 4),
                "pattern_bank": round(c_bank, 4),
            },
            "note": (
                "Standalone MC intelligence: only SOLID patterns commit (gate). "
                "Score = seed blend of solid_acc, time lift, coverage fit, edge. free_params=0."
            ),
            "adoptable_as": "pattern_intelligence_core",
        },
        "free_parameters": 0,
    }

    # Persist ledger for this symbol
    try:
        from app.fsot.pattern_memory import default_ledger_path

        mem.save(default_ledger_path(sym))
        result["ledger"] = str(default_ledger_path(sym))
    except Exception as e:
        result["ledger_error"] = str(e)

    return result


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Long-term FSOT MC intelligence eval")
    ap.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--n-paths", type=int, default=48)
    ap.add_argument("--max-bars", type=int, default=2500, help="Cap history length per symbol")
    ap.add_argument("--quick", action="store_true", help="Faster: fewer symbols, larger step")
    args = ap.parse_args()

    symbols = args.symbols
    step = args.step
    max_bars = args.max_bars
    n_paths = args.n_paths
    if args.quick:
        symbols = ["SPY", "QQQ", "AAPL", "BTC"]
        step = 10
        max_bars = 1500
        n_paths = 32

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print("FSOT DYNAMIC MONTE CARLO — LONG-TERM ACCURACY + INTELLIGENCE")
    print(f"solidify_acc = {SOLIDIFY_ACC:.4f}  free_parameters = 0")
    print(f"horizon={args.horizon}d  step={step}  n_paths={n_paths}  max_bars={max_bars}")
    print("=" * 64)

    rows = []
    for sym in symbols:
        print(f"\n>>> {sym} …", flush=True)
        r = evaluate_symbol(
            sym,
            horizon=args.horizon,
            step=step,
            n_paths=n_paths,
            max_bars=max_bars,
        )
        rows.append(r)
        if r.get("error"):
            print(f"  ERROR: {r['error']}")
            continue
        acc = r["accuracy"]
        ref = r["refinement"]
        intel = r["intelligence"]
        print(
            f"  bars={r['n_bars']} forecasts={r['n_forecast_bars']} "
            f"solid_commits={r['n_solid_commits']} coverage={r['coverage_solid']:.2f} "
            f"patterns_solid={r['memory']['n_solidified']} elapsed={r['elapsed_sec']}s"
        )
        print(
            f"  acc raw(always-in)={acc['raw_fsot_always_in']}  "
            f"solid_gated={acc['solid_gated_commits']}  "
            f"MC_gated={acc['monte_carlo_gated_commits']}"
        )
        print(
            f"  refine solid early={ref['early_solid_accuracy']:.3f} "
            f"late={ref['late_solid_accuracy']:.3f} lift={ref['lift']:+.3f}"
        )
        print(f"  INTEL score={intel['score_0_100']}/100  {intel['components']}")
        for ep in ref["epochs"]:
            print(
                f"    epoch {ep['epoch']}: solid_acc={ep['accuracy']}  "
                f"n={ep['n_commits']}  timeline_solid_share={ep['timeline_solid_share']:.2f}"
            )

    # Aggregate
    ok = [r for r in rows if not r.get("error")]
    summary = {
        "method": "fsot_dynamic_monte_carlo_longterm_eval",
        "free_parameters": 0,
        "solidify_threshold": SOLIDIFY_ACC,
        "config": {
            "horizon": args.horizon,
            "step": step,
            "n_paths": n_paths,
            "max_bars": max_bars,
            "symbols": symbols,
        },
        "symbols": rows,
        "aggregate": {},
    }
    if ok:
        def mean_key(path):
            vals = []
            for r in ok:
                cur = r
                for p in path.split("."):
                    cur = cur.get(p) if isinstance(cur, dict) else None
                if isinstance(cur, (int, float)):
                    vals.append(float(cur))
            return float(np.mean(vals)) if vals else None

        summary["aggregate"] = {
            "n_symbols": len(ok),
            "mean_raw_always_in": mean_key("accuracy.raw_fsot_always_in"),
            "mean_solid_gated": mean_key("accuracy.solid_gated_commits"),
            "mean_mc_gated": mean_key("accuracy.monte_carlo_gated_commits"),
            "mean_coverage": mean_key("coverage_solid"),
            "mean_refinement_lift": mean_key("refinement.lift"),
            "mean_intelligence_score": mean_key("intelligence.score_0_100"),
            "mean_n_solidified": mean_key("memory.n_solidified"),
            "verdict": None,
        }
        lift = summary["aggregate"]["mean_refinement_lift"]
        solid = summary["aggregate"]["mean_solid_gated"]
        if lift is not None and solid is not None:
            if lift > 0 and solid > 0.52:
                verdict = "POSITIVE: solid-gated accuracy rises over time and beats coin-flip"
            elif solid > 0.52:
                verdict = "MIXED: solid commits beat coin-flip but lift not sustained"
            elif lift > 0:
                verdict = "MIXED: refinement lift positive but solid-gated still near random"
            else:
                verdict = "WEAK: no clear long-term intelligence edge on this sample"
            summary["aggregate"]["verdict"] = verdict

    out_path = OUT_DIR / "mc_longterm_intelligence_eval.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 64)
    print("AGGREGATE")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"\nWrote {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
