"""
FSOT Buy / Hold / Sell engine — quality over frequency.

Philosophy (user target ~70–80% when committed, not always-in):
  HOLD is the default. BUY or SELL only when multiple FSOT gates lock:
    1) Dual-scale agreement (As Above So Below)
    2) Pattern memory solidified on *horizon-aligned* history
    3) Edge |μ|/σ and field consciousness gate
    4) Optional MC lean agrees with direction (when ensemble available)

Training feedback horizon = Fib 5 (hold period matches learning period).
Solidify bar stays seed-based: acc_φ ≥ 0.5+Poof, trials ≥ Fib(13), raw acc same.
No free LSQ. Research only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.fsot.intrinsic import (
    KELLY_F,
    SEED_C,
    SEED_GAMMA,
    SEED_K,
    SEED_PHI,
    SEED_POOF,
    evaluate_market_bar,
    growth_term,
    phi_ewma,
)
from app.fsot.pattern_memory import (
    SOLIDIFY_ACC,
    PatternMemory,
    _sign_bin,
    pred_dir_from_ev,
)

# Hold / feedback horizon — Fib ladder (not free)
HOLD_HORIZON = 5  # Fib
_MIN_TRAIN_BARS = 55  # slow scale warm-up


def dual_scale_state(
    rets: np.ndarray,
    volumes: np.ndarray | None,
    sentiment: float = 0.0,
    window: int = 21,
) -> dict[str, Any]:
    """Mid + fast + slow FSOT state; scale_agree ∈ {-1,0,+1}."""
    w = window
    w_fast, w_slow = 8, 55
    n = len(rets)
    if n < w_slow:
        ev = evaluate_market_bar(rets[-max(w, 5) :], None, sentiment, window=max(w, 5))
        return {
            **ev,
            "mu_final": float(ev["mu"]),
            "scale_agree": 0.0,
            "pred_dir": pred_dir_from_ev(ev),
        }

    block = rets[-w:]
    vblock = volumes[-w:] if volumes is not None else None
    vf = volumes[-w_fast:] if volumes is not None else None
    vs = volumes[-w_slow:] if volumes is not None else None

    ev = evaluate_market_bar(block, vblock, sentiment, window=w)
    ev_f = evaluate_market_bar(rets[-w_fast:], vf, sentiment, window=w_fast)
    ev_s = evaluate_market_bar(rets[-w_slow:], vs, sentiment, window=w_slow)

    cross = float(ev_f["S_live"] - ev_s["S_live"])
    mu_cross = float(math.tanh(cross) * ev["sig_pred"] * SEED_GAMMA)
    inv_phi = 1.0 / SEED_PHI
    mu_mid = float(ev["mu"])
    mu_final = (1.0 - inv_phi) * mu_cross + inv_phi * mu_mid
    if mu_cross * mu_mid > 0:
        mu_final = mu_final * (1.0 + SEED_C)
        agree = 1.0
    elif mu_cross * mu_mid < 0:
        mu_final = mu_final * SEED_POOF
        agree = -1.0
    else:
        agree = 0.0

    ev_out = dict(ev)
    ev_out["mu"] = float(mu_final)
    ev_out["mu_cross"] = mu_cross
    ev_out["mu_mid"] = mu_mid
    ev_out["scale_agree"] = agree
    ev_out["S_fast"] = float(ev_f["S_live"])
    ev_out["S_slow"] = float(ev_s["S_live"])
    ev_out["pred_dir"] = _sign_bin(mu_final)
    return ev_out


def bhs_signature(ev: dict[str, Any], window: int = 21) -> str:
    """Signature includes dual-scale agree bin for higher-quality solidification."""
    from app.fsot.pattern_memory import PatternMemory as PM

    # Ensure scale_agree present for fingerprint
    key = PM.signature_from_eval(ev, window=window)
    return key


def field_and_edge(ev: dict[str, Any]) -> tuple[float, float]:
    g = float(ev.get("growth", 1.0))
    d_obs = float(ev.get("d_observer", 0.0))
    field = SEED_K * SEED_C * g * d_obs
    sig = float(max(ev.get("sig_pred", 1e-8), 1e-8))
    mu = abs(float(ev.get("mu", 0.0)))
    edge = mu / sig
    return field, edge


def gates_pass(
    ev: dict[str, Any],
    bias: dict[str, float],
    *,
    recent_rets: np.ndarray | None = None,
) -> bool:
    """All commit gates must pass — otherwise HOLD.

    Core stack (best empirical toward 70%):
      solid + scale_agree + field/edge + strength
    Plus quality soft-check: if pattern has enough trials, require
      accuracy still ≥ 0.5+Poof (decayed solids blocked).
    """
    solid = bool(bias.get("solidified", 0) > 0)
    strength = float(bias.get("strength", 0.0))
    if not solid:
        return False
    if float(ev.get("scale_agree", 0.0)) <= 0:
        return False
    if int(ev.get("pred_dir", 0)) == 0 and int(bias.get("dir", 0)) == 0:
        return False
    field, edge = field_and_edge(ev)
    if not ((abs(field) >= SEED_C * SEED_C) or (edge >= SEED_GAMMA * SEED_POOF)):
        return False
    if strength < SEED_C * SEED_POOF * SEED_POOF:
        return False
    # Decayed solids: if trials mature and raw acc slipped under soften bar, HOLD
    trials = float(bias.get("trials", 0.0))
    raw_acc = float(bias.get("accuracy", 0.5))
    if trials >= 13 and raw_acc < (0.5 + SEED_POOF * 0.5):  # midway: 0.5 + Poof/2
        return False
    # φ short-memory veto only on strong opposite momentum
    pref = int(bias.get("dir", 0)) or int(ev.get("pred_dir", 0))
    if recent_rets is not None and len(recent_rets) >= 5 and pref != 0:
        mom = phi_ewma(np.asarray(recent_rets[-8:], dtype=float))
        sig = float(ev.get("sig_pred", 1e-8)) or 1e-8
        if abs(mom) >= sig * SEED_GAMMA and _sign_bin(mom) == -pref:
            return False
    return True


def decide_bhs(
    ev: dict[str, Any],
    bias: dict[str, float],
    *,
    long_only: bool = False,
    recent_rets: np.ndarray | None = None,
) -> str:
    """
    Returns BUY | HOLD | SELL.
    Default HOLD; BUY/SELL only when gates pass.
    Live μ direction; history preferred must not conflict.
    """
    if not gates_pass(ev, bias, recent_rets=recent_rets):
        return "HOLD"
    d = int(ev.get("pred_dir", 0))
    pref = int(bias.get("dir", 0))
    if pref != 0 and d != 0 and pref != d:
        return "HOLD"
    if d == 0:
        d = pref
    if d > 0:
        return "BUY"
    if d < 0:
        return "HOLD" if long_only else "SELL"
    return "HOLD"


def run_bhs_backtest(
    df: pd.DataFrame,
    *,
    capital: float = 10_000.0,
    window: int = 21,
    hold_horizon: int = HOLD_HORIZON,
    symbol: str = "",
    sentiment: float = 0.0,
    long_only: bool = False,
    store_curve: bool = True,
    curve_max_points: int = 400,
) -> dict[str, Any]:
    """
    Causal Buy/Hold/Sell paper path on real OHLCV.

    Patterns train on hold_horizon-forward returns (aligned with trade duration).
    Position held for hold_horizon bars when BUY/SELL; flat on HOLD.
    """
    if df is None or len(df) < _MIN_TRAIN_BARS + hold_horizon + 30:
        return {"error": "insufficient_data"}

    capital0 = float(max(capital, 1.0))
    H = int(hold_horizon)
    close = df["close"].astype(float).values
    rets = pd.Series(close).pct_change().fillna(0.0).values
    vols = df["volume"].astype(float).values if "volume" in df.columns else None
    times = df["time"].astype(str).values if "time" in df.columns else np.arange(len(df))

    mem = PatternMemory(symbol=symbol, strict=True)

    equity = capital0
    equity_path: list[float] = [capital0]
    time_path: list[str] = [str(times[min(_MIN_TRAIN_BARS, len(times) - 1)])]
    pos_path: list[float] = [0.0]
    action_path: list[str] = ["HOLD"]

    trades = 0
    wins = 0
    losses = 0
    n_buy = n_sell = n_hold = 0
    commit_hits: list[float] = []
    last_trained = _MIN_TRAIN_BARS

    i = _MIN_TRAIN_BARS + 5
    end = len(df) - H - 1

    while i < end:
        # Incremental train with H-day feedback (causal: j+H < i)
        j = last_trained
        train_upto = i - H
        while j < train_upto:
            if j < _MIN_TRAIN_BARS:
                j += 1
                continue
            sub_r = rets[: j + 1]
            sub_v = vols[: j + 1] if vols is not None else None
            st = dual_scale_state(sub_r, sub_v, sentiment, window)
            key = bhs_signature(st, window)
            pred = int(st["pred_dir"])
            r_h = float(close[j + H] / close[j] - 1.0)
            real_d = 1 if r_h > 0 else (-1 if r_h < 0 else 0)
            # Meaningful-move filter: ignore noise days (seed: |r| ≥ Poof·σ)
            sig_j = float(max(st.get("sig_pred", 1e-8), 1e-8))
            meaningful = abs(r_h) >= SEED_POOF * sig_j * math.sqrt(max(H, 1))
            # Only learn when scale agreed + material move (pattern quality)
            if float(st.get("scale_agree", 0)) > 0 and pred != 0 and meaningful:
                mem.observe(key, pred, real_d, bar_index=j, used_observed_branch=True)
            j += 1
        last_trained = max(last_trained, train_upto)

        # Decision at i
        sub_r = rets[: i + 1]
        sub_v = vols[: i + 1] if vols is not None else None
        st = dual_scale_state(sub_r, sub_v, sentiment, window)
        key = bhs_signature(st, window)
        bias = mem.bias_for(key)
        recent = rets[max(0, i - 8) : i + 1]
        action = decide_bhs(st, bias, long_only=long_only, recent_rets=recent)

        if action == "BUY":
            side = 1
            n_buy += 1
        elif action == "SELL":
            side = -1
            n_sell += 1
        else:
            side = 0
            n_hold += 1

        # Size
        _, edge = field_and_edge(st)
        size = float(min(KELLY_F, max(edge * KELLY_F, 0.0)))
        if side != 0:
            size = min(KELLY_F, size * (1.0 + SEED_C * float(bias.get("strength", 0.0))))
        else:
            size = 0.0

        r_fwd = float(close[i + H] / close[i] - 1.0)
        risked = equity * size
        pnl = risked * r_fwd * float(side)

        if side != 0:
            trades += 1
            hit = 1.0 if (side > 0 and r_fwd > 0) or (side < 0 and r_fwd < 0) else 0.0
            commit_hits.append(hit)
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

        equity = max(equity + pnl, capital0 * SEED_POOF * 0.01)
        equity_path.append(equity)
        time_path.append(str(times[min(i + H, len(times) - 1)]))
        pos_path.append(float(side) * size)
        action_path.append(action)

        i += H  # non-overlapping holds (honest, no stacked leverage)

    eq = np.asarray(equity_path, dtype=float)
    rets_e = np.diff(eq) / np.maximum(eq[:-1], 1e-12) if len(eq) > 1 else np.array([0.0])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-12)
    max_dd = float(dd.min()) if len(dd) else 0.0
    max_dd_d = float((eq - peak).min()) if len(eq) else 0.0
    sig = float(np.std(rets_e, ddof=1)) if len(rets_e) > 1 else 0.0
    sharpe = float(np.mean(rets_e) / sig * math.sqrt(252.0 / max(H, 1))) if sig > 1e-12 else 0.0

    bh0 = float(close[_MIN_TRAIN_BARS + 5])
    bh1 = float(close[min(end, len(close) - 1)])
    bh_ret = bh1 / bh0 - 1.0 if bh0 > 0 else 0.0
    bh_final = capital0 * (1.0 + bh_ret)

    commit_acc = float(np.mean(commit_hits)) if commit_hits else None
    # Toward 70–80% target
    target_lo, target_hi = 0.70, 0.80
    if commit_acc is None:
        progress = 0.0
    elif commit_acc >= target_hi:
        progress = 1.0
    elif commit_acc >= target_lo:
        progress = 0.5 + 0.5 * (commit_acc - target_lo) / (target_hi - target_lo)
    else:
        progress = max(0.0, commit_acc / target_lo) * 0.5

    curve = []
    if store_curve and len(equity_path) > 1:
        n = len(equity_path)
        idxs = (
            range(n)
            if n <= curve_max_points
            else np.linspace(0, n - 1, curve_max_points, dtype=int)
        )
        for k in idxs:
            curve.append(
                {
                    "t": time_path[k] if k < len(time_path) else str(k),
                    "equity": round(float(equity_path[k]), 2),
                    "position": round(float(pos_path[k]), 4) if k < len(pos_path) else 0.0,
                    "action": action_path[k] if k < len(action_path) else "HOLD",
                }
            )

    n_decisions = n_buy + n_sell + n_hold
    return {
        "error": None,
        "method": "fsot_bhs_buy_hold_sell",
        "free_parameters": 0,
        "symbol": symbol,
        "mode": "bhs" if not long_only else "bhs_long_only",
        "hold_horizon": H,
        "capital_start": capital0,
        "capital_end": float(eq[-1]),
        "total_pnl": float(eq[-1] - capital0),
        "total_return": float(eq[-1] / capital0 - 1.0),
        "max_drawdown": max_dd,
        "max_drawdown_dollars": max_dd_d,
        "sharpe": sharpe,
        "buy_hold_return": bh_ret,
        "buy_hold_final": bh_final,
        "buy_hold_pnl": bh_final - capital0,
        "vs_buy_hold_pnl": float(eq[-1] - capital0) - (bh_final - capital0),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / trades) if trades else None,
        "commit_directional_accuracy": commit_acc,
        "n_commit_hits": len(commit_hits),
        "n_buy": n_buy,
        "n_sell": n_sell,
        "n_hold": n_hold,
        "n_long_bars": n_buy,
        "n_short_bars": n_sell,
        "n_flat_bars": n_hold,
        "pct_time_in_market": (n_buy + n_sell) / max(n_decisions, 1),
        "pct_hold": n_hold / max(n_decisions, 1),
        "target_accuracy_band": [target_lo, target_hi],
        "progress_to_70_80": round(progress, 4),
        "kelly_f_cap": KELLY_F,
        "solidify_threshold": SOLIDIFY_ACC,
        "pattern_memory": mem.summary(),
        "equity_curve": curve,
        "gates": [
            "dual_scale_agree",
            "pattern_solidified_horizon_aligned",
            "live_quality_acc_and_acc_phi",
            "meaningful_move_training_filter",
            "field_or_edge_consciousness",
            "phi_momentum_not_opposed",
            "preferred_dir_primary_live_confirm",
        ],
        "note": (
            "Buy/Hold/Sell v2: HOLD default. BUY/SELL when dual-scale agrees, "
            f"pattern solid+quality on {H}d material moves, edge/field + φ-mom gates. "
            f"Target commit accuracy 70–80% (progress={progress:.0%}). "
            "Synthetic USD paper on real OHLCV · not financial advice · free_parameters=0."
        ),
    }
