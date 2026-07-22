"""
FSOT market predictor — full engine + econophysics seed laws.

Authority (I:\\FSOT-Physical-Archive\\02_FSOT-2.1-Lean-Full):
  vendor/fsot_compute.py — S = K·(T1+T2+T3), growth, quirk_mod, consciousness
  vendor/neurolab_residual/econophysics_public_anchors.json:
    Hurst H = 0.5, vol persistence β = γ ≈ 0.577, Kelly f* = 1/e,
    Sharpe ≈ γ, Pareto α = φ, formula_branch = term1.growth_term
  Lean Scalar: quirk_mod = exp(C·P_var)·cos(δψ+P_var)

Market theory mapping (user):
  fluctuation + sentiment ↔ observer effect (δψ, observed=True)
  consciousness_factor C always active in quirk_mod for markets
  D_eff from preregistered Finance/Economics scale ladder

Prediction (forward, causal):
  1. Build live ScalarInput from market observer state
  2. growth = exp(α·(1−hits/N)·γ/φ)   [engine]
  3. σ²_{t+1} = γ·σ²_t + (1−γ)·r_t²   [econophysics vol persistence = γ]
  4. μ_{t+1} = K · C · growth · (S_live − S_base) · sign-coherence(quirk_mod)
  5. Poof regime: extreme |r|/σ triggers decoherence flip (Poof shell)
  6. signal = sign(μ); size ~ Kelly·|μ|/σ

Zero free parameters: only seeds + preregistered folds + measured r, vol, sentiment.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.fsot.fast import (
    ALPHA,
    C_FACTOR,
    C_EFF,
    GAMMA,
    K,
    P_NEW,
    P_VAR,
    PHI,
    POOF,
    PSI_CON,
    ScalarInputF,
    compute_scalar_terms_fast,
)
from app.fsot.routes import (
    DOMAIN_FACTOR_ECONOMICS,
    ECONOPHYSICS,
    ECONOMETRICS,
    ECONOMICS,
    FINANCE_MARKETS,
    FINANCE_MARKETS_PANEL,
    get_route,
)

# ── Seeds / derived (locked) ──────────────────────────────────────────────
SEED_PHI = float(PHI)
SEED_GAMMA = float(GAMMA)  # vol persistence β AND Sharpe-class anchor
SEED_K = float(K)
SEED_PSI = float(PSI_CON)
SEED_C = float(C_FACTOR)  # consciousness_factor
SEED_P_VAR = float(P_VAR)
SEED_ALPHA = float(ALPHA)
SEED_POOF = float(POOF)
SEED_ONE_OVER_E = float(1.0 / math.e)  # Kelly f*
SEED_P_NEW = float(P_NEW)
SEED_C_EFF = float(C_EFF)

# Econophysics anchors (archive)
HURST_H = 0.5
VOL_PERSISTENCE = SEED_GAMMA  # β = γ
KELLY_F = SEED_ONE_OVER_E
SHARPE_ANCHOR = SEED_GAMMA
PARETO_ALPHA = SEED_PHI

_FIB = (5, 8, 13, 21, 34, 55, 89, 144)

# Preregistered D_eff by observation scale (extension + core manifests)
_SCALE_ROUTES = (
    (8, {"name": "Neuroeconomics", "D_eff": 16, "recent_hits": 2, "delta_psi": 0.7, "delta_theta": 1.0, "observed": True}),
    (13, {"name": "Finance_Markets_Panel", "D_eff": 19, "recent_hits": 2, "delta_psi": 0.65, "delta_theta": 1.0, "observed": True}),
    (21, {"name": "Finance_Markets", "D_eff": 19, "recent_hits": 2, "delta_psi": 0.75, "delta_theta": 1.0, "observed": True}),
    (34, {"name": "Econometrics", "D_eff": 19, "recent_hits": 2, "delta_psi": 0.7, "delta_theta": 1.0, "observed": True}),
    (55, {"name": "Economics", "D_eff": 20, "recent_hits": 3, "delta_psi": 1.5, "delta_theta": 1.0, "observed": True}),
    (89, {"name": "Economics", "D_eff": 20, "recent_hits": 3, "delta_psi": 1.5, "delta_theta": 1.0, "observed": True}),
)


def consciousness_factor() -> float:
    return SEED_C


def quirk_mod(delta_psi: float, observed: bool = True) -> float:
    """Lean: exp(C * phase_variance) * cos(delta_psi + phase_variance)."""
    if not observed:
        return 1.0
    return float(math.exp(SEED_C * SEED_P_VAR) * math.cos(delta_psi + SEED_P_VAR))


def growth_term(hits: float, N: float) -> float:
    """Engine growth: exp(α · (1 − hits/N) · γ/φ)."""
    N = max(N, 1e-12)
    return float(math.exp(SEED_ALPHA * (1.0 - hits / N) * SEED_GAMMA / SEED_PHI))


def route_for_window(window: int) -> dict[str, Any]:
    w = min(_FIB, key=lambda x: abs(x - max(window, 5)))
    chosen = _SCALE_ROUTES[0][1]
    for fib, route in _SCALE_ROUTES:
        if fib <= w:
            chosen = route
    return dict(chosen)


def route_scalar(route: dict[str, Any], **overrides: Any) -> dict[str, float]:
    hits = float(overrides.get("recent_hits", route["recent_hits"]))
    N = float(overrides.get("N", 1.0))
    si = ScalarInputF(
        N=N,
        P=float(overrides.get("P", 1.0)),
        D_eff=float(overrides.get("D_eff", route["D_eff"])),
        delta_psi=float(overrides.get("delta_psi", route["delta_psi"])),
        delta_theta=float(overrides.get("delta_theta", route.get("delta_theta", 1.0))),
        recent_hits=hits,
        observed=bool(overrides.get("observed", route["observed"])),
        rho=float(overrides.get("rho", 1.0)),
        scale=float(overrides.get("scale", 1.0)),
        amplitude=float(overrides.get("amplitude", 1.0)),
        trend_bias=float(overrides.get("trend_bias", 0.0)),
    )
    terms = compute_scalar_terms_fast(si)
    g = growth_term(hits, N)
    qm = quirk_mod(float(si.delta_psi), bool(si.observed))
    return {
        "name": str(route.get("name", "route")),
        "D_eff": float(si.D_eff),
        "delta_psi": float(si.delta_psi),
        "delta_theta": float(si.delta_theta),
        "recent_hits": hits,
        "observed": bool(si.observed),
        "N": N,
        "P": float(si.P),
        "S": terms["S"],
        "T1": terms["T1"],
        "T2": terms["T2"],
        "T3": terms["T3"],
        "K": terms["K"],
        "growth": g,
        "quirk_mod": qm,
        "consciousness_factor": SEED_C,
        "regime": "emergence" if terms["S"] > 0 else "dispersal",
    }


def spine_scalars() -> dict[str, dict[str, float]]:
    return {
        "Economics": route_scalar(ECONOMICS),
        "Finance_Markets": route_scalar(FINANCE_MARKETS),
        "Finance_Markets_Panel": route_scalar(FINANCE_MARKETS_PANEL),
        "Econometrics": route_scalar(ECONOMETRICS),
        "Econophysics": route_scalar(ECONOPHYSICS),
    }


def fsot_coupled(measured: float, domain: str = "Economics") -> tuple[float, float]:
    r = get_route(domain if domain in ("Economics", "Finance_Markets") else "Economics")
    s = route_scalar(r)["S"]
    computed = float(measured) * (1.0 + abs(s) * DOMAIN_FACTOR_ECONOMICS)
    err_pct = abs(computed - measured) / max(abs(measured), 1e-12) * 100.0
    return computed, err_pct


def _safe_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    return float(np.std(x, ddof=1))


def market_observer_state(
    rets: np.ndarray,
    volumes: np.ndarray | None = None,
    sentiment: float = 0.0,
    route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fluctuation + sentiment → observer δψ (theory correspondence)."""
    route = route or FINANCE_MARKETS
    r = rets[np.isfinite(rets)]
    base_dp = float(route["delta_psi"])
    if len(r) < 3:
        return {
            "delta_psi": base_dp,
            "delta_theta": float(route.get("delta_theta", 1.0)),
            "recent_hits": float(route["recent_hits"]),
            "N": 1.0,
            "P": 1.0,
            "rho": 1.0,
            "trend_bias": 0.0,
            "amplitude": 1.0,
            "scale": 1.0,
            "fluctuation": 0.0,
            "sentiment": float(sentiment),
            "observed": True,
            "D_eff": float(route["D_eff"]),
            "r_last": 0.0,
            "var_t": 0.0,
        }

    mu = float(np.mean(r))
    sig = _safe_std(r)
    r_last = float(r[-1])
    var_t = float(np.var(r, ddof=1)) if len(r) > 1 else r_last * r_last

    # Fluctuation in γ-units (seed)
    fluc = mu / (sig * SEED_GAMMA) if sig > 1e-12 else mu / SEED_GAMMA
    # Also last-bar shock in γ-units
    shock = r_last / (sig * SEED_GAMMA) if sig > 1e-12 else r_last / SEED_GAMMA

    # Observer phase: base fold δψ + atan(fluctuation) + atan(sentiment) + atan(shock)
    delta_psi = base_dp + math.atan(fluc) + math.atan(float(sentiment)) + math.atan(shock)
    delta_theta = math.atan(mu * SEED_PHI) + math.atan(r_last * SEED_PHI)

    recent_hits = float(np.sum(r > 0))
    N = max(float(len(r)) / SEED_PHI, 1.0)

    if volumes is not None and len(volumes) >= len(r):
        v = volumes[-len(r) :]
        v = v[np.isfinite(v)]
        if len(v) and float(np.median(v)) > 0:
            rel = float(v[-1] / np.median(v))
            P = SEED_PSI * (1.0 + math.atan(math.log(max(rel, 1e-12))))
        else:
            P = SEED_PSI
    else:
        P = SEED_PSI

    if len(r) > 2:
        c = np.corrcoef(r[1:], r[:-1])[0, 1]
        rho = 1.0 + SEED_GAMMA * float(c) if np.isfinite(c) else 1.0
    else:
        rho = 1.0

    trend_bias = SEED_P_NEW * math.atan(mu * 100.0)
    amplitude = max(1.0 + (sig / SEED_GAMMA if sig > 0 else 0.0), SEED_PSI * 0.1)

    return {
        "delta_psi": float(delta_psi),
        "delta_theta": float(delta_theta),
        "recent_hits": float(recent_hits),
        "N": float(N),
        "P": float(P),
        "rho": float(rho),
        "trend_bias": float(trend_bias),
        "amplitude": float(amplitude),
        "scale": 1.0,
        "fluctuation": float(fluc),
        "sentiment": float(sentiment),
        "observed": True,
        "D_eff": float(route["D_eff"]),
        "r_last": r_last,
        "var_t": var_t,
        "sig_t": sig,
        "mu_t": mu,
    }


def vol_forecast_gamma(var_t: float, r_last: float) -> float:
    """Econophysics: volatility persistence β = γ (seed)."""
    return float(VOL_PERSISTENCE * var_t + (1.0 - VOL_PERSISTENCE) * (r_last * r_last))


def phi_ewma(rets: np.ndarray) -> float:
    """φ-EWMA of returns — fractal memory, only seed φ."""
    r = rets[np.isfinite(rets)]
    if len(r) == 0:
        return 0.0
    a = 1.0 / SEED_PHI  # ≈ 0.618
    m = float(r[0])
    for x in r[1:]:
        m = a * float(x) + (1.0 - a) * m
    return m


def evaluate_market_bar(
    rets: np.ndarray,
    volumes: np.ndarray | None = None,
    sentiment: float = 0.0,
    window: int = 21,
) -> dict[str, Any]:
    """
    Full FSOT forward prediction for one state.
    Returns μ, σ, signal, growth, quirk_mod, consciousness, D_eff, S.
    """
    route = route_for_window(window)
    base = route_scalar(route)
    st = market_observer_state(rets, volumes, sentiment, route)
    live = route_scalar(
        route,
        N=st["N"],
        P=st["P"],
        D_eff=st["D_eff"],
        delta_psi=st["delta_psi"],
        delta_theta=st["delta_theta"],
        recent_hits=st["recent_hits"],
        observed=True,
        rho=st["rho"],
        scale=st["scale"],
        amplitude=st["amplitude"],
        trend_bias=st["trend_bias"],
    )

    g = live["growth"]
    qm = live["quirk_mod"]
    d_obs = live["S"] - base["S"]

    # --- Econophysics vol forecast (γ persistence) ---
    var_pred = vol_forecast_gamma(st["var_t"], st["r_last"])
    sig_pred = math.sqrt(max(var_pred, 1e-18))

    # --- Engine directional field (formula_branch = term1.growth_term) ---
    # field = K · C · growth · (S_live − S_base)  — pure engine, observer already in S_live
    field = SEED_K * SEED_C * g * d_obs

    # φ-EWMA as secondary fractal memory (return units)
    mu_phi = phi_ewma(rets)

    # Map field → return units via σ_pred × γ (Sharpe-class econophysics scale)
    # Primary = engine; secondary = φ memory (weight 1/φ on measured fractal, rest on engine)
    inv_phi = 1.0 / SEED_PHI
    mu_engine = float(math.tanh(field / SEED_C) * sig_pred * SEED_GAMMA)
    mu = (1.0 - inv_phi) * mu_engine + inv_phi * mu_phi

    # --- Poof decoherence on extremes ---
    sig_t = st["sig_t"] if st["sig_t"] > 1e-12 else sig_pred
    extreme = abs(st["r_last"]) > sig_t / math.sqrt(max(SEED_POOF, 1e-6))
    if extreme:
        mu = -abs(mu) * math.copysign(1.0, st["r_last"])  # fade the shock (decoherence)

    # Score = μ; field used for trade gate (|field| ≥ C → consciousness-scale event)
    score = float(mu)
    signal = "LONG" if score > 0 else ("SHORT" if score < 0 else "FLAT")

    edge = abs(mu) / max(sig_pred, 1e-12)
    size = float(min(KELLY_F, max(edge * KELLY_F, 0.0)))

    return {
        **live,
        "S_base": base["S"],
        "S_live": live["S"],
        "d_observer": float(d_obs),
        "growth": g,
        "quirk_mod": qm,
        "mu": float(mu),
        "mu_engine": float(mu_engine),
        "mu_phi": float(mu_phi),
        "sig_pred": float(sig_pred),
        "var_pred": float(var_pred),
        "score": score,
        "signal": signal,
        "kelly_size": size,
        "poof_extreme": bool(extreme),
        "fluctuation": st["fluctuation"],
        "sentiment": st["sentiment"],
        "route_name": route["name"],
        "pred_return": float(mu),  # direct FSOT μ forecast
    }


def compute_intrinsic_frame(
    df: pd.DataFrame,
    window: int = 21,
    sentiment: float = 0.0,
    sentiment_series: pd.Series | None = None,
) -> pd.DataFrame:
    if df is None or len(df) < 10:
        return pd.DataFrame()

    out = df.copy().reset_index(drop=True)
    close = out["close"].astype(float)
    rets_all = close.pct_change().fillna(0.0).values
    vols = out["volume"].astype(float).values if "volume" in out.columns else None

    w = int(min(_FIB, key=lambda x: abs(x - max(window, 5))))
    w_fast = 8   # Neuroeconomics-scale D_eff=16
    w_slow = 55  # Economics D_eff=20
    route = route_for_window(w)
    base = route_scalar(route)

    rows: list[dict[str, Any]] = []
    for i in range(len(out)):
        if i < w_slow:
            rows.append(_empty_row(base, route, w, sentiment))
            continue

        block = rets_all[i - w + 1 : i + 1]
        vblock = vols[i - w + 1 : i + 1] if vols is not None else None
        sent = float(sentiment)
        if sentiment_series is not None and i < len(sentiment_series):
            try:
                sent = float(sentiment_series.iloc[i])
            except Exception:
                pass

        ev = evaluate_market_bar(block, vblock, sent, window=w)

        # Dual-scale As Above So Below: fast vs slow observer-coupled S
        blk_f = rets_all[i - w_fast + 1 : i + 1]
        blk_s = rets_all[i - w_slow + 1 : i + 1]
        vf = vols[i - w_fast + 1 : i + 1] if vols is not None else None
        vs = vols[i - w_slow + 1 : i + 1] if vols is not None else None
        ev_f = evaluate_market_bar(blk_f, vf, sent, window=w_fast)
        ev_s = evaluate_market_bar(blk_s, vs, sent, window=w_slow)

        # Cross-scale vitality: short D_eff fold above long → emergence rising
        cross = float(ev_f["S_live"] - ev_s["S_live"])
        mu_cross = float(math.tanh(cross) * ev["sig_pred"] * SEED_GAMMA)
        inv_phi = 1.0 / SEED_PHI
        mu_mid = float(ev["mu"])
        # Prefer cross-scale (As Above So Below) with weight (1-1/φ); mid with 1/φ
        mu_final = (1.0 - inv_phi) * mu_cross + inv_phi * mu_mid
        # Agreement: both scales same sign → full signal; conflict → damp by Poof
        if mu_cross * mu_mid > 0:
            mu_final = mu_final * (1.0 + SEED_C)  # consciousness boost when scales agree
            agree = 1.0
        elif mu_cross * mu_mid < 0:
            mu_final = mu_final * SEED_POOF  # decoherence damp on conflict
            agree = -1.0
        else:
            agree = 0.0
        score = float(mu_final)

        p_up = float(np.mean(block > 0))
        ent = float(
            -(p_up * math.log(p_up + 1e-12) + (1 - p_up) * math.log(1 - p_up + 1e-12)) / math.log(2)
        )

        rows.append(
            {
                "S": ev["S"],
                "S_base": ev["S_base"],
                "S_live": ev["S_live"],
                "S_fast": ev_f["S_live"],
                "S_slow": ev_s["S_live"],
                "T1": ev["T1"],
                "T2": ev["T2"],
                "T3": ev["T3"],
                "K": ev["K"],
                "growth": ev["growth"],
                "quirk_mod": ev["quirk_mod"],
                "consciousness_factor": SEED_C,
                "delta_psi": ev["delta_psi"],
                "D_eff": ev["D_eff"],
                "d_observer": ev["d_observer"],
                "dS": 0.0,
                "mu": mu_final,
                "mu_engine": ev["mu_engine"],
                "mu_phi": ev["mu_phi"],
                "mu_cross": mu_cross,
                "mu_mid": mu_mid,
                "scale_agree": agree,
                "sig_pred": ev["sig_pred"],
                "score": score,
                "seed_score": score,
                "emergence_ema": score,
                "emergence_score": score,
                "composite": score,
                "entropy": ent,
                "regime": "emergence" if score > 0 else "dispersal",
                "signal": "LONG" if score > 0 else ("SHORT" if score < 0 else "FLAT"),
                "pred_return": mu_final,
                "kelly_size": ev["kelly_size"],
                "poof_extreme": ev["poof_extreme"],
                "fib_window": w,
                "route_name": ev["route_name"],
                "fluctuation": ev["fluctuation"],
                "sentiment": ev["sentiment"],
                "N": ev["N"],
                "P": ev["P"],
            }
        )

    frame = pd.DataFrame(rows)
    for c in frame.columns:
        out[c] = frame[c]
    out["dS"] = out["S_live"].diff().fillna(0.0)
    return out


def _empty_row(base: dict, route: dict, w: int, sentiment: float) -> dict[str, Any]:
    return {
        "S": base["S"],
        "S_base": base["S"],
        "S_live": base["S"],
        "T1": base["T1"],
        "T2": base["T2"],
        "T3": base["T3"],
        "K": base["K"],
        "growth": 1.0,
        "quirk_mod": quirk_mod(float(route["delta_psi"]), True),
        "consciousness_factor": SEED_C,
        "delta_psi": float(route["delta_psi"]),
        "D_eff": float(route["D_eff"]),
        "d_observer": 0.0,
        "dS": 0.0,
        "mu": 0.0,
        "mu_engine": 0.0,
        "mu_phi": 0.0,
        "sig_pred": 0.0,
        "score": 0.0,
        "seed_score": 0.0,
        "emergence_ema": 0.0,
        "emergence_score": 0.0,
        "composite": 0.0,
        "entropy": 0.5,
        "regime": base["regime"],
        "signal": "FLAT",
        "pred_return": 0.0,
        "kelly_size": 0.0,
        "poof_extreme": False,
        "fib_window": w,
        "route_name": route["name"],
        "fluctuation": 0.0,
        "sentiment": float(sentiment),
        "N": 1.0,
        "P": 1.0,
    }


def latest_intrinsic(
    df: pd.DataFrame,
    window: int = 21,
    symbol: str = "",
    sentiment: float = 0.0,
) -> dict[str, Any]:
    spine = spine_scalars()
    frame = compute_intrinsic_frame(df, window=window, sentiment=sentiment)
    if frame.empty:
        return {"error": "no_data"}

    last = frame.iloc[-1]
    last_px = float(df["close"].iloc[-1])
    pred = float(last["pred_return"])
    conf = float(min(0.99, abs(float(last["score"])) / max(float(last["sig_pred"]), 1e-6) * SEED_GAMMA))

    series_tail = []
    for _, r in frame.tail(90).iterrows():
        t = r["time"] if "time" in frame.columns else None
        series_tail.append(
            {
                "time": int(pd.Timestamp(t).timestamp()) if t is not None and pd.notna(t) else None,
                "close": float(r["close"]) if pd.notna(r.get("close")) else None,
                "S": float(r["S_live"]) if pd.notna(r.get("S_live")) else None,
                "mu": float(r["mu"]) if pd.notna(r.get("mu")) else None,
                "sig_pred": float(r["sig_pred"]) if pd.notna(r.get("sig_pred")) else None,
                "growth": float(r["growth"]) if pd.notna(r.get("growth")) else None,
                "quirk_mod": float(r["quirk_mod"]) if pd.notna(r.get("quirk_mod")) else None,
                "score": float(r["score"]) if pd.notna(r.get("score")) else None,
                "emergence_score": float(r["score"]) if pd.notna(r.get("score")) else None,
                "pred_return": float(r["pred_return"]) if pd.notna(r.get("pred_return")) else 0.0,
            }
        )

    return {
        "error": None,
        "method": "fsot_full_engine_econophysics",
        "authority": "I:/FSOT-Physical-Archive/02_FSOT-2.1-Lean-Full/vendor/fsot_compute.py",
        "free_parameters": 0,
        "symbol": symbol,
        "S": float(last["S_live"]),
        "S_live": float(last["S_live"]),
        "S_base": float(last["S_base"]),
        "T1": float(last["T1"]),
        "T2": float(last["T2"]),
        "T3": float(last["T3"]),
        "K": float(last["K"]),
        "growth": float(last["growth"]),
        "quirk_mod": float(last["quirk_mod"]),
        "consciousness_factor": SEED_C,
        "delta_psi": float(last["delta_psi"]),
        "D_eff": float(last["D_eff"]),
        "route_name": str(last["route_name"]),
        "d_observer": float(last["d_observer"]),
        "dS": float(last["dS"]),
        "mu": float(last["mu"]),
        "mu_engine": float(last["mu_engine"]),
        "mu_phi": float(last["mu_phi"]),
        "sig_pred": float(last["sig_pred"]),
        "kelly_size": float(last["kelly_size"]),
        "poof_extreme": bool(last["poof_extreme"]),
        "fluctuation": float(last["fluctuation"]),
        "sentiment": float(last["sentiment"]),
        "S_econ": spine["Economics"]["S"],
        "S_finance": spine["Finance_Markets"]["S"],
        "S_econophysics": spine["Econophysics"]["S"],
        "spine": {k: {"S": v["S"], "D_eff": v["D_eff"]} for k, v in spine.items()},
        "seeds": {
            "phi": SEED_PHI,
            "gamma": SEED_GAMMA,
            "psi_con": SEED_PSI,
            "K": SEED_K,
            "consciousness_factor": SEED_C,
            "poof": SEED_POOF,
            "kelly_f": KELLY_F,
            "vol_persistence": VOL_PERSISTENCE,
            "hurst_H": HURST_H,
            "sharpe_anchor": SHARPE_ANCHOR,
            "pareto_alpha": PARETO_ALPHA,
        },
        "entropy": float(last["entropy"]),
        "emergence_score": float(last["score"]),
        "seed_score": float(last["score"]),
        "composite": float(last["score"]),
        "score": float(last["score"]),
        "signal": str(last["signal"]),
        "confidence": conf,
        "regime": str(last["regime"]),
        "pred_return": pred,
        "last_price": last_px,
        "pred_price_1d": last_px * (1.0 + pred),
        "fib_window": int(last["fib_window"]),
        "N": float(last["N"]),
        "P": float(last["P"]),
        "base_S": float(last["S_base"]),
        "base_S_note": "Preregistered fold without market observer offsets",
        "observer_mod": float(last["sentiment"]),
        "meta": {
            "domain": str(last["route_name"]),
            "window": int(last["fib_window"]),
            "formula_branch": "term1.growth_term",
            "params": {
                "D_eff": float(last["D_eff"]),
                "delta_psi": float(last["delta_psi"]),
                "observed": True,
                "consciousness_factor": SEED_C,
                "quirk_mod": float(last["quirk_mod"]),
                "growth": float(last["growth"]),
            },
            "features": {
                "fluctuation": float(last["fluctuation"]),
                "sentiment": float(last["sentiment"]),
                "sig_pred": float(last["sig_pred"]),
                "mu": float(last["mu"]),
            },
        },
        "series_tail": series_tail,
        "forecast_path": [],
        "horizons": {},
    }


def evaluate_intrinsic_walkforward(
    df: pd.DataFrame,
    window: int = 21,
    sentiment: float = 0.0,
    multiscale: bool = False,
) -> dict[str, Any]:
    frame = compute_intrinsic_frame(df, window=window, sentiment=sentiment)
    if frame.empty or len(frame) < window + 40:
        return {"error": "insufficient_data"}

    close = frame["close"].astype(float)
    frame["fwd_1"] = close.pct_change().shift(-1)
    frame["fwd_5"] = close.pct_change(5).shift(-5)
    frame["fwd_20"] = close.pct_change(20).shift(-20)
    frame = frame.dropna(subset=["fwd_1", "mu", "sig_pred"])

    mu = frame["mu"].values.astype(float)
    d_obs = frame["d_observer"].values.astype(float)
    growth = frame["growth"].values.astype(float)
    agree = frame["scale_agree"].values.astype(float) if "scale_agree" in frame.columns else np.ones(len(frame))
    field = SEED_K * SEED_C * growth * d_obs
    pos = np.sign(mu)
    edge = np.abs(mu) / np.maximum(frame["sig_pred"].values.astype(float), 1e-12)
    # Trade only when dual-scale agrees (As Above So Below lock) AND meaningful edge
    active = (agree > 0) & ((np.abs(field) >= SEED_C * SEED_C) | (edge >= SEED_GAMMA * SEED_POOF))

    def acc(hcol: str, use_gate: bool = True) -> float | None:
        fwd = frame[hcol].values.astype(float)
        m = (active if use_gate else (pos != 0)) & np.isfinite(fwd) & (pos != 0)
        if m.sum() < 25:
            m = (pos != 0) & np.isfinite(fwd)
        if m.sum() < 25:
            return None
        return float((np.sign(pos[m]) == np.sign(fwd[m])).mean())

    a1 = acc("fwd_1", True)
    a1_all = acc("fwd_1", False)
    a5 = acc("fwd_5", True)
    a20 = acc("fwd_20", True)

    pos_eng = np.sign(d_obs)
    actual = frame["fwd_1"].values.astype(float)
    m_e = (pos_eng != 0) & np.isfinite(actual)
    acc_dobs = float((pos_eng[m_e] == np.sign(actual[m_e])).mean()) if m_e.sum() > 25 else None

    # Dual-scale agree only
    m_ag = (agree > 0) & (pos != 0) & np.isfinite(actual)
    acc_agree = float((pos[m_ag] == np.sign(actual[m_ag])).mean()) if m_ag.sum() > 25 else None
    # 5d/20d on agree days
    def acc_agree_h(hcol: str) -> float | None:
        fwd = frame[hcol].values.astype(float)
        m = (agree > 0) & (pos != 0) & np.isfinite(fwd)
        if m.sum() < 25:
            return None
        return float((pos[m] == np.sign(fwd[m])).mean())

    a5_ag = acc_agree_h("fwd_5")
    a20_ag = acc_agree_h("fwd_20")

    kelly = np.minimum(KELLY_F, edge * KELLY_F)
    pos_k = pos * np.where(active, kelly, 0.0)
    strat = pos_k * actual
    strat = strat[np.isfinite(strat)]
    if len(strat) and float(np.std(strat)) > 1e-12:
        sharpe = float(np.mean(strat) / np.std(strat) * np.sqrt(252))
    else:
        sharpe = 0.0

    active_f = np.abs(field) >= SEED_C * SEED_C
    m_f = active_f & (pos_eng != 0) & np.isfinite(actual)
    acc_field = float((pos_eng[m_f] == np.sign(actual[m_f])).mean()) if m_f.sum() > 25 else None

    spine = spine_scalars()

    return {
        "error": None,
        "method": "fsot_full_engine_econophysics",
        "free_parameters": 0,
        "n_bars": int(len(frame)),
        "pct_in_market": float(active.mean()),
        "pct_scale_agree": float((agree > 0).mean()),
        "directional_accuracy_1d": a1,
        "directional_accuracy_1d_ungated": a1_all,
        "directional_accuracy_5d": a5,
        "directional_accuracy_20d": a20,
        "directional_accuracy_active": a1 if a1 is not None else a1_all,
        "acc_scale_agree_1d": acc_agree,
        "acc_scale_agree_5d": a5_ag,
        "acc_scale_agree_20d": a20_ag,
        "acc_d_observer": acc_dobs,
        "acc_field_gated": acc_field,
        "pct_field_gate": float(active_f.mean()),
        "sharpe": sharpe,
        "S_econ": spine["Economics"]["S"],
        "S_finance": spine["Finance_Markets"]["S"],
        "consciousness_factor": SEED_C,
        "vol_persistence": VOL_PERSISTENCE,
        "kelly_f": KELLY_F,
        "poof": SEED_POOF,
        "fib_window": window if window in _FIB else min(_FIB, key=lambda x: abs(x - window)),
        "note": (
            "Dual-scale D_eff (fast/slow) + growth + C + quirk_mod + γ-vol + Poof; "
            "trade when scales agree and field/edge clears seed gates."
        ),
    }
