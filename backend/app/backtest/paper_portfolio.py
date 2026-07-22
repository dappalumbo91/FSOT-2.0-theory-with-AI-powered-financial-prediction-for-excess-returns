"""
Synthetic dollar paper portfolio — theoretical P&L on real OHLCV.

Adjustable starting capital. Causal walk on real prices.
Modes:
  bhs           — Buy/Hold/Sell engine (recommended): multi-gate, Fib hold
  bhs_long_only — BHS but never short
  always_in     — trade every FSOT non-FLAT signal (raw engine)
  solid_gated   — only commit when pattern memory solidified (1d)
  long_only     — solid_gated but never short
  buy_hold      — 100% long benchmark

Sizing uses Kelly f* = 1/e scaled by |μ|/σ edge (seed), capped at Kelly.
No free LSQ parameters. Research only — not financial advice.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.fsot.bhs_engine import HOLD_HORIZON, run_bhs_backtest
from app.fsot.intrinsic import (
    KELLY_F,
    SEED_C,
    SEED_PHI,
    SEED_POOF,
    evaluate_market_bar,
)
from app.fsot.pattern_memory import PatternMemory, extract_state


def _equity_stats(equity: np.ndarray, capital0: float) -> dict[str, float]:
    if len(equity) < 2:
        return {
            "final_equity": float(equity[-1]) if len(equity) else capital0,
            "total_return": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_dollars": 0.0,
            "sharpe": 0.0,
        }
    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    max_dd = float(dd.min())
    max_dd_dollars = float((equity - peak).min())
    mu = float(np.mean(rets))
    sig = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (mu / sig) * math.sqrt(252.0) if sig > 1e-12 else 0.0
    return {
        "final_equity": float(equity[-1]),
        "total_return": float(equity[-1] / capital0 - 1.0),
        "total_pnl": float(equity[-1] - capital0),
        "max_drawdown": max_dd,
        "max_drawdown_dollars": max_dd_dollars,
        "sharpe": sharpe,
    }


def run_paper_portfolio(
    df: pd.DataFrame,
    *,
    capital: float = 10_000.0,
    window: int = 21,
    mode: str = "bhs",
    symbol: str = "",
    sentiment: float = 0.0,
    step: int = 1,
    store_curve: bool = True,
    curve_max_points: int = 400,
    hold_horizon: int = HOLD_HORIZON,
) -> dict[str, Any]:
    """
    Simulate synthetic USD equity path from FSOT signals on real prices.

    capital: starting paper dollars (user-adjustable)
    mode: bhs | bhs_long_only | always_in | solid_gated | long_only | buy_hold
    """
    if df is None or len(df) < window + 30:
        return {"error": "insufficient_data"}

    capital0 = float(max(capital, 1.0))
    mode = (mode or "bhs").lower().strip()
    if mode in ("bhs", "buy_hold_sell", "bhs_mc"):
        return run_bhs_backtest(
            df,
            capital=capital0,
            window=window,
            hold_horizon=hold_horizon,
            symbol=symbol,
            sentiment=sentiment,
            long_only=False,
            store_curve=store_curve,
            curve_max_points=curve_max_points,
        )
    if mode in ("bhs_long_only", "bhs_long"):
        return run_bhs_backtest(
            df,
            capital=capital0,
            window=window,
            hold_horizon=hold_horizon,
            symbol=symbol,
            sentiment=sentiment,
            long_only=True,
            store_curve=store_curve,
            curve_max_points=curve_max_points,
        )
    if mode not in ("always_in", "solid_gated", "long_only", "buy_hold"):
        mode = "bhs"
        return run_bhs_backtest(
            df,
            capital=capital0,
            window=window,
            hold_horizon=hold_horizon,
            symbol=symbol,
            sentiment=sentiment,
            long_only=False,
            store_curve=store_curve,
            curve_max_points=curve_max_points,
        )

    close = df["close"].astype(float).values
    rets = pd.Series(close).pct_change().fillna(0.0).values
    vols = df["volume"].astype(float).values if "volume" in df.columns else None
    times = df["time"].astype(str).values if "time" in df.columns else np.arange(len(df))

    equity = capital0
    cash = capital0
    position = 0.0  # +1 long, -1 short, 0 flat (fraction of equity at risk)
    equity_path: list[float] = [capital0]
    time_path: list[str] = [str(times[window])]
    pos_path: list[float] = [0.0]
    trades = 0
    wins = 0
    losses = 0
    n_long = 0
    n_short = 0
    n_flat = 0
    daily_pnls: list[float] = []

    mem = PatternMemory(symbol=symbol)
    last_trained = window + 5

    i = window + 10
    end = len(df) - 2
    while i < end:
        # Causal pattern train (1d feedback known for j < i)
        if mode in ("solid_gated", "long_only"):
            j = last_trained
            while j < i - 1:
                block = rets[j - window + 1 : j + 1]
                vblock = vols[j - window + 1 : j + 1] if vols is not None else None
                key, _ev, pred_dir = extract_state(block, vblock, sentiment, window)
                r1 = float(close[j + 1] / close[j] - 1.0)
                real_d = 1 if r1 > 0 else (-1 if r1 < 0 else 0)
                mem.observe(key, pred_dir, real_d, bar_index=j, used_observed_branch=True)
                j += 1
            last_trained = i - 1

        block = rets[i - window + 1 : i + 1]
        vblock = vols[i - window + 1 : i + 1] if vols is not None else None
        key, ev, pred_dir = extract_state(block, vblock, sentiment, window)
        bias = mem.bias_for(key)
        solid = bool(bias.get("solidified", 0) > 0)

        # Target position in {-1, 0, +1} then sized by Kelly edge
        if mode == "buy_hold":
            target_side = 1
        elif mode == "always_in":
            target_side = pred_dir
        elif mode == "solid_gated":
            target_side = pred_dir if solid else 0
        else:  # long_only
            target_side = pred_dir if (solid and pred_dir > 0) else 0

        # Size: Kelly * edge, strength boost if solid
        sig = float(max(ev.get("sig_pred", 0.0), 1e-8))
        mu = abs(float(ev.get("mu", 0.0)))
        edge = mu / sig
        size = float(min(KELLY_F, max(edge * KELLY_F, 0.0)))
        if solid:
            size = min(KELLY_F, size * (1.0 + SEED_C * float(bias.get("strength", 0.0))))
        if target_side == 0:
            size = 0.0
            n_flat += 1
        elif target_side > 0:
            n_long += 1
        else:
            n_short += 1

        # Apply position over next `step` bars (default 1)
        r_next = float(close[min(i + step, len(close) - 1)] / close[i] - 1.0)
        # P&L on risked fraction of current equity
        risked = equity * size
        pnl = risked * r_next * float(target_side if target_side != 0 else 0)
        if target_side != 0:
            trades += 1
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

        equity = equity + pnl
        # Floor: cannot go below Poof * capital0 (decoherence residual cash notion)
        equity = max(equity, capital0 * SEED_POOF * 0.01)
        daily_pnls.append(pnl)
        equity_path.append(equity)
        time_path.append(str(times[min(i + step, len(times) - 1)]))
        pos_path.append(float(target_side) * size)

        i += step

    eq = np.asarray(equity_path, dtype=float)
    stats = _equity_stats(eq, capital0)

    # Buy & hold comparison on same window
    bh0 = float(close[window + 10])
    bh1 = float(close[min(end, len(close) - 1)])
    bh_ret = bh1 / bh0 - 1.0 if bh0 > 0 else 0.0
    bh_final = capital0 * (1.0 + bh_ret)

    # Downsample equity curve for UI
    curve = []
    if store_curve and len(equity_path) > 1:
        n = len(equity_path)
        if n <= curve_max_points:
            idxs = range(n)
        else:
            idxs = np.linspace(0, n - 1, curve_max_points, dtype=int)
        for k in idxs:
            curve.append(
                {
                    "t": time_path[k] if k < len(time_path) else str(k),
                    "equity": round(float(equity_path[k]), 2),
                    "position": round(float(pos_path[k]), 4) if k < len(pos_path) else 0.0,
                }
            )

    win_rate = wins / trades if trades else None
    mem_sum = mem.summary() if mode in ("solid_gated", "long_only") else None

    return {
        "error": None,
        "method": "fsot_paper_portfolio_synthetic_usd",
        "free_parameters": 0,
        "symbol": symbol,
        "mode": mode,
        "capital_start": capital0,
        "capital_end": stats["final_equity"],
        "total_pnl": stats["total_pnl"],
        "total_return": stats["total_return"],
        "max_drawdown": stats["max_drawdown"],
        "max_drawdown_dollars": stats["max_drawdown_dollars"],
        "sharpe": stats["sharpe"],
        "buy_hold_return": bh_ret,
        "buy_hold_final": bh_final,
        "buy_hold_pnl": bh_final - capital0,
        "vs_buy_hold_pnl": stats["total_pnl"] - (bh_final - capital0),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "n_long_bars": n_long,
        "n_short_bars": n_short,
        "n_flat_bars": n_flat,
        "pct_time_in_market": (n_long + n_short) / max(n_long + n_short + n_flat, 1),
        "kelly_f_cap": KELLY_F,
        "window": window,
        "step": step,
        "n_bars_sim": len(equity_path),
        "equity_curve": curve,
        "pattern_memory": mem_sum,
        "note": (
            "Synthetic USD only — research / paper trading. Signals causal on real OHLCV. "
            "solid_gated commits only after FSOT pattern solidifies (acc_φ > 0.5+Poof). "
            "Not financial advice; past paper P&L ≠ future results."
        ),
    }
