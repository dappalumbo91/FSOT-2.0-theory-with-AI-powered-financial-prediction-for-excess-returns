"""
FSOT Monte Carlo path ensemble with observer collapse + intelligent pattern memory.

Ontology:
  - Many futures exist as fluid possibilities until observation couples.
  - quirk_mod + consciousness_factor set collapse strength (observed True vs False).
  - Market value (price path) + fluctuation + sentiment update δψ each step.
  - Collapse TRUE  → path locks to FSOT μ branch (engine + γ-vol).
  - Collapse FALSE → unobserved/chaotic branch (noise scaled by Poof; weaker drift).

Intelligent / dynamic layer:
  - Causal walk over history builds PatternMemory (FSOT discrete signatures).
  - When a signature's φ-EWMA accuracy clears 0.5+Poof for ≥Fib(8) trials → solidify.
  - Subsequent MC paths bias μ, collapse, and bootstrap shocks from solidified anchors.
  - Refinement is online: accuracy patterns lock; misses soften via φ/Poof — no free fits.

Seeds only (no free LSQ):
  γ vol persistence, K, C, Poof, φ, ψ_con, Finance/Economics D_eff folds.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.fsot.fast import C_FACTOR, GAMMA, K, PHI, POOF, PSI_CON, P_VAR
from app.fsot.intrinsic import (
    SEED_ALPHA,
    SEED_C,
    SEED_GAMMA,
    SEED_K,
    SEED_ONE_OVER_E,
    SEED_PHI,
    SEED_POOF,
    VOL_PERSISTENCE,
    evaluate_market_bar,
    growth_term,
    phi_ewma,
    quirk_mod,
    route_for_window,
    route_scalar,
    vol_forecast_gamma,
)
from app.fsot.pattern_memory import (
    PatternMemory,
    default_ledger_path,
    extract_state,
)

# Default ensemble size — large enough for stable quantiles; seed-based not free
DEFAULT_N_PATHS = 512
DEFAULT_HORIZON = 21  # Fib ~21
_FIB = (5, 8, 13, 21, 34, 55)


def collapse_probability(
    delta_psi: float,
    base_delta_psi: float,
    sentiment: float,
    consciousness: float = SEED_C,
) -> float:
    """
    Probability path collapses to observed=True branch this step.

    Strength of observation = consciousness × |δψ − base| × (1 + sentiment coupling).
    Logistic uses only seed C and φ; Poof floors residual unobserved mass.
    """
    # Observation intensity from phase departure + sentiment (measured)
    intensity = consciousness * SEED_PHI * abs(delta_psi - base_delta_psi)
    intensity = intensity + consciousness * math.atan(sentiment)
    # logistic
    p = 1.0 / (1.0 + math.exp(-intensity))
    # Unobserved residual cannot vanish below Poof (decoherence shell always possible)
    p = float(np.clip(p, SEED_POOF, 1.0 - SEED_POOF))
    return p


def _hist_innovations(df: pd.DataFrame, window: int = 21) -> np.ndarray:
    """Empirical shocks: r − φ-EWMA(r), for bootstrap noise (measurement residuals)."""
    r = df["close"].astype(float).pct_change().dropna().values
    if len(r) < window + 5:
        return r - np.mean(r) if len(r) else np.array([0.0])
    innov = []
    for i in range(window, len(r)):
        mu = phi_ewma(r[i - window : i])
        innov.append(r[i] - mu)
    arr = np.asarray(innov, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 10:
        return r - np.mean(r)
    return arr


def train_pattern_memory(
    df: pd.DataFrame,
    *,
    window: int = 21,
    sentiment: float = 0.0,
    symbol: str = "",
    feedback_horizon: int = 1,
    step: int = 1,
    memory: PatternMemory | None = None,
    max_bars: int | None = None,
    resume: bool = True,
) -> tuple[PatternMemory, dict[str, Any]]:
    """
    Causal walk: at each bar extract FSOT signature, predict dir, score on future return.
    Solidifies accurate patterns; returns memory + refinement diagnostics.

    If resume=True and memory has prior last_update_i values, only train bars after
    the max index already observed (no double-count when reloading ledgers).
    """
    if df is None or len(df) < window + feedback_horizon + 20:
        mem = memory or PatternMemory(symbol=symbol)
        return mem, {"error": "insufficient_data"}

    mem = memory or PatternMemory(symbol=symbol)
    mem.symbol = symbol or mem.symbol

    close = df["close"].astype(float).values
    rets = pd.Series(close).pct_change().fillna(0.0).values
    vols = df["volume"].astype(float).values if "volume" in df.columns else None

    residual_bank: dict[str, list[float]] = getattr(mem, "_residual_bank", None) or {}
    if not isinstance(residual_bank, dict):
        residual_bank = {}

    start = window + 5
    end = len(df) - feedback_horizon - 1
    if max_bars is not None and end - start > max_bars:
        start = end - max_bars

    # Resume past last trained bar (ledger warm-start without double count)
    if resume and mem.patterns:
        last_i = max((p.last_update_i for p in mem.patterns.values()), default=-1)
        if last_i >= start:
            start = last_i + step

    raw_hits: list[float] = []
    anchored_hits: list[float] = []
    solid_at: list[dict[str, Any]] = []
    n_solid_before = mem.n_solidify_events

    i = start
    while i < end:
        block = rets[i - window + 1 : i + 1]
        vblock = vols[i - window + 1 : i + 1] if vols is not None else None
        key, ev, pred_dir = extract_state(block, vblock, sentiment, window)

        # If already solidified, prefer anchored direction for "intelligent" call
        bias = mem.bias_for(key)
        if bias["solidified"] > 0 and bias["dir"] != 0:
            live_dir = int(bias["dir"])
        else:
            live_dir = pred_dir

        r_fwd = float(close[i + feedback_horizon] / close[i] - 1.0)
        real_dir = 1 if r_fwd > 0 else (-1 if r_fwd < 0 else 0)

        mu = float(ev["mu"])
        resid = float(rets[i] - mu) if i < len(rets) else 0.0
        bank = residual_bank.setdefault(key, [])
        bank.append(resid)
        if len(bank) > 256:
            residual_bank[key] = bank[-256:]

        before_solid = mem.get(key).solidified
        mem.observe(
            key,
            pred_dir,
            real_dir,
            bar_index=i,
            used_observed_branch=True,
        )
        after = mem.get(key)
        if after.solidified and not before_solid:
            solid_at.append(
                {"bar": i, "key": key, "acc_phi": after.acc_phi, "strength": after.strength}
            )

        if pred_dir != 0 and real_dir != 0:
            raw_hits.append(1.0 if pred_dir == real_dir else 0.0)
        if live_dir != 0 and real_dir != 0:
            anchored_hits.append(1.0 if live_dir == real_dir else 0.0)

        i += step

    mem._residual_bank = residual_bank  # type: ignore[attr-defined]

    diag = {
        "error": None,
        "bars_trained": len(raw_hits),
        "feedback_horizon": feedback_horizon,
        "train_start": start,
        "train_end": end,
        "raw_directional_accuracy": float(np.mean(raw_hits)) if raw_hits else None,
        "anchored_directional_accuracy": float(np.mean(anchored_hits)) if anchored_hits else None,
        "new_solidifications": mem.n_solidify_events - n_solid_before,
        "solidification_events": solid_at[-20:],
        "memory": mem.summary(),
        "free_parameters": 0,
    }
    if len(anchored_hits) >= 30:
        third = len(anchored_hits) // 3
        diag["anchored_early"] = float(np.mean(anchored_hits[:third]))
        diag["anchored_late"] = float(np.mean(anchored_hits[-third:]))
        diag["refinement_lift"] = diag["anchored_late"] - diag["anchored_early"]
    # Also use memory global rolling refinement if present
    summ = mem.summary()
    if summ.get("refinement"):
        diag["memory_refinement"] = summ["refinement"]
    return mem, diag


def run_fsot_monte_carlo(
    df: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    n_paths: int = DEFAULT_N_PATHS,
    sentiment: float = 0.0,
    window: int = 21,
    seed: int | None = None,
    symbol: str = "",
    memory: PatternMemory | None = None,
    dynamic: bool = False,
) -> dict[str, Any]:
    """
    Simulate n_paths of length horizon with FSOT observer collapse each day.
    If dynamic=True and memory provided (or trained), paths use solidified anchors.
    """
    if df is None or len(df) < window + 10:
        return {"error": "insufficient_data"}

    horizon = int(min(_FIB, key=lambda x: abs(x - max(horizon, 5))))
    rng = np.random.default_rng(seed)

    close = df["close"].astype(float).values
    rets = pd.Series(close).pct_change().fillna(0.0).values
    vols = df["volume"].astype(float).values if "volume" in df.columns else None
    price0 = float(close[-1])

    # Current FSOT state
    block = rets[-window:]
    vblock = vols[-window:] if vols is not None else None
    ev = evaluate_market_bar(block, vblock, sentiment, window=window)
    route = route_for_window(window)
    base_dp = float(route["delta_psi"])

    mu0 = float(ev["mu"])
    sig0 = float(max(ev["sig_pred"], 1e-8))
    var0 = sig0 * sig0
    qm0 = float(ev["quirk_mod"])
    g0 = float(ev["growth"])
    d_obs0 = float(ev["d_observer"])
    S_live0 = float(ev["S_live"])
    S_base0 = float(ev["S_base"])

    # Pattern bias at t0
    sig_key = PatternMemory.signature_from_eval(ev, window=window)
    bias0 = (
        memory.bias_for(sig_key)
        if (dynamic and memory is not None)
        else {
            "solidified": 0.0,
            "strength": 0.0,
            "dir": 0.0,
            "mu_scale": 1.0,
            "collapse_boost": 0.0,
            "path_weight": 1.0,
            "acc_phi": 0.5,
            "trials": 0.0,
        }
    )

    innov = _hist_innovations(df, window=window)
    # Pattern-conditional residuals if available
    residual_bank = getattr(memory, "_residual_bank", None) if memory is not None else None
    if residual_bank and sig_key in residual_bank and len(residual_bank[sig_key]) >= 10:
        cond = np.asarray(residual_bank[sig_key], dtype=float)
        cond = cond[np.isfinite(cond)]
        if len(cond) >= 10:
            innov = cond

    innov_std = float(np.std(innov)) if len(innov) > 1 else 1.0
    if innov_std < 1e-12:
        innov_std = 1.0
    innov_unit = innov / innov_std

    # Path arrays
    terminal = np.zeros(n_paths)
    path_up = np.zeros(n_paths, dtype=bool)
    collapse_true_frac = np.zeros(n_paths)
    path_weights = np.ones(n_paths)
    n_store = min(n_paths, 64)
    stored = np.zeros((n_store, horizon + 1))
    stored[:, 0] = price0

    for p in range(n_paths):
        price = price0
        var_t = var0
        r_last = float(rets[-1])
        buf = list(block.astype(float))
        sent = float(sentiment)
        n_collapse = 0
        # Path inherits anchor weight when starting pattern is solid
        path_weights[p] = float(bias0.get("path_weight", 1.0))

        for t in range(horizon):
            sig_t = math.sqrt(max(var_t, 1e-18))
            arr = np.asarray(buf[-window:], dtype=float)
            mu_w = float(np.mean(arr)) if len(arr) else 0.0
            sig_w = float(np.std(arr, ddof=1)) if len(arr) > 1 else sig_t
            fluc = mu_w / (sig_w * SEED_GAMMA) if sig_w > 1e-12 else 0.0
            shock = r_last / (sig_t * SEED_GAMMA) if sig_t > 1e-12 else 0.0
            delta_psi = base_dp + math.atan(fluc) + math.atan(sent) + math.atan(shock)
            qm = quirk_mod(delta_psi, observed=True)

            d_obs = d_obs0 + SEED_K * math.atan(sum(buf[-8:]) if len(buf) >= 8 else mu_w)
            hits = float(np.sum(arr > 0)) if len(arr) else 3.0
            N = max(len(arr) / SEED_PHI, 1.0)
            g = growth_term(hits, N)
            field = SEED_K * SEED_C * g * d_obs
            mu_t = float(math.tanh(field / SEED_C) * sig_t * SEED_GAMMA)
            mu_phi = phi_ewma(arr) if len(arr) else 0.0
            inv_phi = 1.0 / SEED_PHI
            mu_t = (1.0 - inv_phi) * mu_t + inv_phi * mu_phi

            # Dynamic: re-fingerprint path state and pull live bias (anchors evolve along path)
            step_bias = bias0
            if dynamic and memory is not None:
                # lightweight signature from running buffer (same bins as memory)
                step_ev = {
                    "mu": mu_t,
                    "d_observer": d_obs,
                    "fluctuation": fluc,
                    "sentiment": sent,
                    "quirk_mod": qm,
                    "poof_extreme": abs(r_last) > sig_t / math.sqrt(max(SEED_POOF, 1e-6)),
                    "delta_psi": delta_psi,
                    "D_eff": ev["D_eff"],
                    "scale_agree": 1 if mu_t * d_obs > 0 else (-1 if mu_t * d_obs < 0 else 0),
                }
                step_key = PatternMemory.signature_from_eval(step_ev, window=window)
                step_bias = memory.bias_for(step_key)

            # Anchor-aware μ: scale + blend preferred direction when solidified
            if step_bias.get("solidified", 0) > 0 and step_bias.get("dir", 0) != 0:
                s = float(step_bias["strength"])
                pref = float(step_bias["dir"])
                mu_t = float(step_bias["mu_scale"]) * mu_t
                # Blend direction toward historical winning dir (measurement-anchored)
                mu_mag = abs(mu_t)
                mu_anchored = pref * mu_mag
                mu_t = (1.0 - s) * mu_t + s * mu_anchored

            # Collapse probability + pattern boost
            p_col = collapse_probability(delta_psi, base_dp, sent, SEED_C)
            if step_bias.get("solidified", 0) > 0:
                p_col = float(np.clip(p_col + float(step_bias["collapse_boost"]), SEED_POOF, 1.0 - SEED_POOF))

            collapse_true = bool(rng.random() < p_col)
            if collapse_true:
                n_collapse += 1
                z = float(rng.choice(innov_unit)) if len(innov_unit) else float(rng.standard_normal())
                r = mu_t + sig_t * z
            else:
                z = float(rng.choice(innov_unit)) if len(innov_unit) else float(rng.standard_normal())
                r = -mu_t * SEED_POOF + sig_t * (1.0 + SEED_POOF) * z

            var_t = vol_forecast_gamma(var_t, r)
            r_last = r
            price = price * (1.0 + r)
            buf.append(r)
            if len(buf) > max(window * 2, 55):
                buf = buf[-max(window * 2, 55) :]
            sent = sent * (1.0 - 1.0 / SEED_PHI)

            if p < n_store:
                stored[p, t + 1] = price

        terminal[p] = price
        path_up[p] = price > price0
        collapse_true_frac[p] = n_collapse / horizon

    # Weighted ensemble (solidified patterns up-weight their paths)
    w = path_weights / max(path_weights.sum(), 1e-12)
    rets_term = terminal / price0 - 1.0
    p_up = float(np.sum(w[path_up]))
    p_down = 1.0 - p_up
    # Weighted quantiles via sorted sample
    order = np.argsort(terminal)
    cdf = np.cumsum(w[order])
    qs = []
    for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        idx = int(np.searchsorted(cdf, q, side="left"))
        idx = min(max(idx, 0), len(order) - 1)
        qs.append(float(terminal[order[idx]]))
    qs = np.asarray(qs)

    hist, edges = np.histogram(rets_term, bins=21, weights=w)
    mode_i = int(np.argmax(hist))
    mode_lo, mode_hi = float(edges[mode_i]), float(edges[mode_i + 1])
    mode_center = 0.5 * (mode_lo + mode_hi)

    mean_collapse = float(np.average(collapse_true_frac, weights=w))
    e_ret = float(np.average(rets_term, weights=w))
    med_ret = float(qs[3] / price0 - 1.0)

    obs_mask = collapse_true_frac >= float(PSI_CON)
    if obs_mask.sum() >= max(10, n_paths // 20):
        w_obs = w[obs_mask]
        w_obs = w_obs / max(w_obs.sum(), 1e-12)
        e_ret_obs = float(np.average(rets_term[obs_mask], weights=w_obs))
        p_up_obs = float(np.average(path_up[obs_mask].astype(float), weights=w_obs))
    else:
        e_ret_obs = e_ret
        p_up_obs = p_up

    # If pattern solidified with preferred dir, align ensemble lean when strong
    if bias0.get("solidified", 0) > 0 and bias0.get("strength", 0) >= SEED_C:
        if bias0["dir"] > 0 and p_up_obs >= 0.5 - SEED_POOF * SEED_C:
            p_up_obs = max(p_up_obs, 0.5 + SEED_POOF * bias0["strength"])
        elif bias0["dir"] < 0 and p_up_obs <= 0.5 + SEED_POOF * SEED_C:
            p_up_obs = min(p_up_obs, 0.5 - SEED_POOF * bias0["strength"])

    if p_up_obs > 0.5 + SEED_POOF:
        signal = "LONG"
    elif p_up_obs < 0.5 - SEED_POOF:
        signal = "SHORT"
    else:
        signal = "FLAT"

    conf_base = abs(p_up_obs - 0.5) * 2.0 * (0.5 + 0.5 * mean_collapse)
    conf_boost = float(bias0.get("strength", 0.0)) * float(bias0.get("solidified", 0.0))
    confidence = float(min(0.99, conf_base * (1.0 + SEED_C * conf_boost)))

    # Intelligence gate: unsolidified signatures stay fluid → FLAT commit
    # (ensemble still returned for analysis; signal is what systems should trade)
    gated = False
    if dynamic:
        if bias0.get("solidified", 0) <= 0:
            signal = "FLAT"
            confidence = float(min(confidence, SEED_POOF))
            gated = True
        else:
            # Solid pattern: require ensemble agree with preferred dir, else FLAT
            pref = int(bias0.get("dir", 0))
            if pref > 0 and signal != "LONG":
                signal = "FLAT"
                confidence = float(confidence * SEED_POOF)
                gated = True
            elif pref < 0 and signal != "SHORT":
                signal = "FLAT"
                confidence = float(confidence * SEED_POOF)
                gated = True

    fan = []
    if n_store > 5:
        for t in range(horizon + 1):
            col = stored[:, t]
            col = col[col > 0]
            if len(col) < 3:
                continue
            fan.append(
                {
                    "step": t,
                    "p10": float(np.quantile(col, 0.10)),
                    "p50": float(np.quantile(col, 0.50)),
                    "p90": float(np.quantile(col, 0.90)),
                }
            )

    method = (
        "fsot_dynamic_monte_carlo_pattern_collapse"
        if dynamic
        else "fsot_monte_carlo_observer_collapse"
    )

    return {
        "error": None,
        "method": method,
        "free_parameters": 0,
        "symbol": symbol,
        "price0": price0,
        "horizon": horizon,
        "n_paths": n_paths,
        "sentiment": float(sentiment),
        "dynamic": bool(dynamic),
        "pattern": {
            "key": sig_key,
            "bias": bias0,
            "memory_summary": memory.summary() if memory is not None else None,
        },
        "state0": {
            "mu": mu0,
            "sig": sig0,
            "S_live": S_live0,
            "S_base": S_base0,
            "d_observer": d_obs0,
            "growth": g0,
            "quirk_mod": qm0,
            "consciousness_factor": SEED_C,
            "D_eff": float(ev["D_eff"]),
            "route_name": ev["route_name"],
            "delta_psi": float(ev["delta_psi"]),
        },
        "seeds": {
            "gamma_vol_persistence": VOL_PERSISTENCE,
            "consciousness_factor": SEED_C,
            "poof": SEED_POOF,
            "phi": SEED_PHI,
            "K": SEED_K,
            "kelly_f": SEED_ONE_OVER_E,
            "psi_con": float(PSI_CON),
        },
        "ensemble": {
            "p_up": p_up,
            "p_down": p_down,
            "p_up_observed_branch": p_up_obs,
            "expected_return": e_ret,
            "expected_return_observed_branch": e_ret_obs,
            "median_return": med_ret,
            "mean_terminal_price": float(np.average(terminal, weights=w)),
            "median_terminal_price": float(qs[3]),
            "quantiles_price": {
                "p05": float(qs[0]),
                "p10": float(qs[1]),
                "p25": float(qs[2]),
                "p50": float(qs[3]),
                "p75": float(qs[4]),
                "p90": float(qs[5]),
                "p95": float(qs[6]),
            },
            "most_probable_return_bin": {
                "low": mode_lo,
                "high": mode_hi,
                "center": mode_center,
                "count": int(hist[mode_i] * n_paths) if hist[mode_i] <= 1 else int(hist[mode_i]),
            },
            "mean_collapse_true_fraction": mean_collapse,
            "observed_branch_path_fraction": float(obs_mask.mean()) if n_paths else 0.0,
        },
        "signal": signal,
        "confidence": confidence,
        "gated_unsolidified": gated,
        "fan_chart": fan,
        "note": (
            "Dynamic FSOT MC intelligence: train signatures on history; solidify when "
            "acc_φ > 0.5+Poof (≥Fib8). Only SOLID patterns may commit LONG/SHORT; "
            "unsolidified → FLAT (fluid possibilities). Paths bias μ/collapse from anchors. "
            "free_parameters=0."
            if dynamic
            else (
                "Each day: p_collapse from consciousness×observer phase; "
                "TRUE → FSOT μ + γ-vol path; FALSE → Poof-scaled chaotic branch."
            )
        ),
    }


def run_dynamic_fsot_monte_carlo(
    df: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    n_paths: int = DEFAULT_N_PATHS,
    sentiment: float = 0.0,
    window: int = 21,
    seed: int | None = None,
    symbol: str = "",
    feedback_horizon: int = 1,
    train_step: int = 1,
    max_train_bars: int | None = None,
    persist: bool = True,
    memory: PatternMemory | None = None,
) -> dict[str, Any]:
    """
    Full intelligent pipeline:
      1) Causal pattern training on history (solidify accurate FSOT signatures)
      2) Forward multipath MC biased by solidified anchors
      3) Optional persist ledger under D:\\training data\\FSOT-Market-History\\patterns
    """
    if df is None or len(df) < window + 30:
        return {"error": "insufficient_data"}

    # Fresh causal train on this series (indices are relative to `df` slice).
    # Persist saves the resulting ledger; do not warm-merge across different
    # range windows (bar indices would misalign). Always rebuild from df.
    mem = memory if memory is not None else PatternMemory(symbol=symbol)
    mem.symbol = symbol or mem.symbol

    mem, train_diag = train_pattern_memory(
        df,
        window=window,
        sentiment=sentiment,
        symbol=symbol,
        feedback_horizon=feedback_horizon,
        step=train_step,
        memory=mem,
        max_bars=max_train_bars,
        resume=False,  # full causal pass on provided history
    )
    if train_diag.get("error") and train_diag["error"] != "insufficient_data":
        return {"error": train_diag["error"]}

    if persist and symbol:
        try:
            mem.save(default_ledger_path(symbol))
        except Exception:
            pass

    mc = run_fsot_monte_carlo(
        df,
        horizon=horizon,
        n_paths=n_paths,
        sentiment=sentiment,
        window=window,
        seed=seed,
        symbol=symbol,
        memory=mem,
        dynamic=True,
    )
    if mc.get("error"):
        return mc
    mc["training"] = train_diag
    mc["method"] = "fsot_dynamic_monte_carlo_pattern_collapse"
    return mc


def mc_walkforward_hit(
    df: pd.DataFrame,
    *,
    horizon: int = 5,
    n_paths: int = 128,
    window: int = 21,
    step: int = 5,
    sentiment: float = 0.0,
    dynamic: bool = True,
    symbol: str = "",
) -> dict[str, Any]:
    """
    Causal MC evaluation with growing PatternMemory (incremental, not full retrain).
    At each i: score MC vs future; then observe 1d FSOT signature outcomes for bars
    in (last_i, i] so memory solidifies only on past-known results.
    """
    if df is None or len(df) < window + horizon + 50:
        return {"error": "insufficient_data"}

    close = df["close"].astype(float).values
    rets = pd.Series(close).pct_change().fillna(0.0).values
    vols = df["volume"].astype(float).values if "volume" in df.columns else None
    hits: list[float] = []
    hits_obs: list[float] = []
    hits_solid: list[float] = []
    mem = PatternMemory(symbol=symbol) if dynamic else None
    last_trained = window + 5
    i = window + 10
    while i < len(df) - horizon - 1:
        # Incremental causal train: only bars whose 1d outcome is known before forecast i
        if dynamic and mem is not None:
            train_upto = i - 1  # outcome of bar j uses close[j+1], need j+1 <= i
            j = last_trained
            while j < train_upto:
                block = rets[j - window + 1 : j + 1]
                vblock = vols[j - window + 1 : j + 1] if vols is not None else None
                key, _ev, pred_dir = extract_state(block, vblock, sentiment, window)
                r_1 = float(close[j + 1] / close[j] - 1.0)
                real_dir = 1 if r_1 > 0 else (-1 if r_1 < 0 else 0)
                mem.observe(key, pred_dir, real_dir, bar_index=j, used_observed_branch=True)
                j += 1
            last_trained = train_upto

        sub = df.iloc[: i + 1]
        if dynamic and mem is not None:
            mc = run_fsot_monte_carlo(
                sub,
                horizon=horizon,
                n_paths=n_paths,
                sentiment=sentiment,
                window=window,
                seed=i,
                symbol=symbol,
                memory=mem,
                dynamic=True,
            )
        else:
            mc = run_fsot_monte_carlo(
                sub,
                horizon=horizon,
                n_paths=n_paths,
                sentiment=sentiment,
                window=window,
                seed=i,
            )
        if mc.get("error"):
            i += step
            continue
        pred_up = mc["ensemble"]["p_up_observed_branch"] >= 0.5
        r_real = float(close[i + horizon] / close[i] - 1.0)
        real_up = r_real > 0
        hit = 1.0 if pred_up == real_up else 0.0
        hits.append(hit)
        if abs(mc["ensemble"]["p_up_observed_branch"] - 0.5) > SEED_POOF:
            hits_obs.append(hit)
        pat = mc.get("pattern") or {}
        if (pat.get("bias") or {}).get("solidified", 0) > 0:
            hits_solid.append(hit)
        i += step

    out: dict[str, Any] = {
        "error": None,
        "method": (
            "fsot_dynamic_monte_carlo_pattern_collapse"
            if dynamic
            else "fsot_monte_carlo_observer_collapse"
        ),
        "horizon": horizon,
        "n_paths": n_paths,
        "n_eval": len(hits),
        "directional_accuracy": float(np.mean(hits)) if hits else None,
        "directional_accuracy_confident": float(np.mean(hits_obs)) if hits_obs else None,
        "n_confident": len(hits_obs),
        "directional_accuracy_solidified": float(np.mean(hits_solid)) if hits_solid else None,
        "n_solidified_calls": len(hits_solid),
        "free_parameters": 0,
    }
    if dynamic and mem is not None:
        out["memory"] = mem.summary()
        if len(hits) >= 12:
            third = len(hits) // 3
            out["early_acc"] = float(np.mean(hits[:third]))
            out["late_acc"] = float(np.mean(hits[-third:]))
            out["refinement_lift"] = out["late_acc"] - out["early_acc"]
    return out
