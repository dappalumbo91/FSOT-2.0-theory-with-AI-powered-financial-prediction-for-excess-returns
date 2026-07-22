"""
FSOT pattern memory for intelligent Monte Carlo.

Patterns are discrete FSOT state signatures (seeds + preregistered folds only).
When a signature shows sustained directional accuracy on real history, it
solidifies and becomes an anchor for subsequent Monte Carlo paths.

No free LSQ / invented thresholds:
  - Solidify when φ-EWMA accuracy > 0.5 + Poof and trials ≥ Fib(8)=8
  - Soften on misses with φ memory decay
  - Strength = C · (acc − 0.5) / (0.5 − Poof) clipped to [0,1]
  - Preferred direction from realized hit mass (measurement), not fit coeffs
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.fsot.intrinsic import (
    SEED_C,
    SEED_GAMMA,
    SEED_PHI,
    SEED_POOF,
    SEED_PSI,
    evaluate_market_bar,
    route_for_window,
)

# Fib solidification horizon (preregistered ladder, not free)
_MIN_TRIALS = 8  # Fib
_CONFIRM_WINDOW = 13  # Fib — recent window for re-check
# Accuracy bar: coin-flip + Poof residual (must clear decoherence shell)
SOLIDIFY_ACC = 0.5 + SEED_POOF  # ≈ 0.653
# Soften if recent φ-window drops below fair + ψ_con·Poof
SOFTEN_ACC = 0.5 + SEED_POOF * SEED_PSI


def _sign_bin(x: float, eps: float = 0.0) -> int:
    """Mathematical sign for pattern bins — no free dead-zone on returns.

    eps=0: strict sign (correct for μ / returns in daily units).
    Optional tiny eps only for pure float noise if caller passes it.
    """
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def pred_dir_from_ev(ev: dict[str, Any]) -> int:
    """FSOT directional call for learning: μ first, then d_observer, then quirk."""
    for key in ("mu", "d_observer", "quirk_mod", "fluctuation"):
        d = _sign_bin(float(ev.get(key, 0.0)))
        if d != 0:
            return d
    return 0


def _intensity_bin(delta_psi: float, base_dp: float) -> str:
    """Partition observer intensity by seed γ (not free quantiles)."""
    d = abs(float(delta_psi) - float(base_dp))
    if d < SEED_GAMMA * SEED_C:
        return "low"
    if d < SEED_GAMMA:
        return "mid"
    return "high"


@dataclass
class PatternRecord:
    key: str
    trials: int = 0
    hits: int = 0
    # φ-EWMA of hit indicators (online accuracy)
    acc_phi: float = 0.5
    # preferred realized direction when correct: sum of realized signs on hits
    dir_mass: float = 0.0
    # when prediction matched, was collapse/obs branch favored?
    collapse_true_hits: int = 0
    collapse_true_trials: int = 0
    solidified: bool = False
    strength: float = 0.0
    preferred_dir: int = 0  # -1 short, +1 long, 0 unknown
    last_update_i: int = -1
    soft_fails: int = 0

    def accuracy(self) -> float:
        if self.trials <= 0:
            return 0.5
        return self.hits / self.trials

    def preferred_collapse_true(self) -> float:
        """Fraction of successful trials that used observed-branch lean."""
        if self.collapse_true_trials <= 0:
            return 0.5
        return self.collapse_true_hits / self.collapse_true_trials


class PatternMemory:
    """
    Online causal ledger of FSOT signatures → accuracy → solidification.
    """

    def __init__(self, symbol: str = "") -> None:
        self.symbol = symbol
        self.patterns: dict[str, PatternRecord] = {}
        self.n_updates = 0
        self.n_solidify_events = 0
        self.n_soften_events = 0
        self.rolling_hits: list[float] = []  # global post-solidify refinement curve
        self.free_parameters = 0

    # ── Signature (FSOT discrete state) ─────────────────────────────────

    @staticmethod
    def signature_from_eval(ev: dict[str, Any], window: int = 21) -> str:
        route = route_for_window(window)
        base_dp = float(route["delta_psi"])
        mu = float(ev.get("mu", 0.0))
        d_obs = float(ev.get("d_observer", 0.0))
        fluc = float(ev.get("fluctuation", 0.0))
        sent = float(ev.get("sentiment", 0.0))
        qm = float(ev.get("quirk_mod", 0.0))
        poof = 1 if ev.get("poof_extreme") else 0
        scale_agree = int(ev.get("scale_agree", _sign_bin(mu * d_obs)))
        # D_eff is preregistered fold id
        d_eff = int(round(float(ev.get("D_eff", route["D_eff"]))))
        inten = _intensity_bin(float(ev.get("delta_psi", base_dp)), base_dp)
        parts = [
            f"D{d_eff}",
            f"mu{_sign_bin(mu)}",
            f"d{_sign_bin(d_obs)}",
            f"f{_sign_bin(fluc)}",
            f"s{_sign_bin(sent)}",
            f"q{_sign_bin(qm)}",
            f"p{poof}",
            f"a{scale_agree}",
            f"i{inten}",
        ]
        return "|".join(parts)

    def get(self, key: str) -> PatternRecord:
        if key not in self.patterns:
            self.patterns[key] = PatternRecord(key=key)
        return self.patterns[key]

    # ── Online update (causal: only past outcomes) ──────────────────────

    def observe(
        self,
        key: str,
        pred_dir: int,
        realized_dir: int,
        *,
        bar_index: int = -1,
        used_observed_branch: bool = True,
    ) -> PatternRecord:
        """
        Record one causal outcome for a pattern key.
        pred_dir / realized_dir ∈ {-1, 0, +1}.
        """
        rec = self.get(key)
        if pred_dir == 0 and realized_dir == 0:
            hit = 0.5
            correct = False
            skip_dir_mass = True
        elif pred_dir == 0:
            # Engine FLAT: count as miss vs market move (pattern was uninformative)
            hit = 0.0
            correct = False
            skip_dir_mass = True
        elif realized_dir == 0:
            hit = 0.5
            correct = False
            skip_dir_mass = True
        else:
            correct = pred_dir == realized_dir
            hit = 1.0 if correct else 0.0
            skip_dir_mass = False

        rec.trials += 1
        if correct:
            rec.hits += 1
            if not skip_dir_mass:
                rec.dir_mass += float(realized_dir)
            rec.soft_fails = 0
        else:
            rec.soft_fails += 1
            # Track realized mass even on miss for preferred_dir recovery after softens
            if not skip_dir_mass:
                rec.dir_mass += float(realized_dir) * SEED_POOF  # weak pull toward reality

        # φ-EWMA accuracy (fractal memory of recognition)
        a = 1.0 / SEED_PHI
        rec.acc_phi = a * hit + (1.0 - a) * rec.acc_phi

        if used_observed_branch:
            rec.collapse_true_trials += 1
            if correct:
                rec.collapse_true_hits += 1

        rec.last_update_i = bar_index
        self.n_updates += 1
        self.rolling_hits.append(hit)

        # Preferred direction from hit mass (measurement)
        if abs(rec.dir_mass) > SEED_POOF:
            rec.preferred_dir = 1 if rec.dir_mass > 0 else -1

        # Strength from consciousness × accuracy margin
        margin = rec.acc_phi - 0.5
        denom = max(0.5 - SEED_POOF, 1e-9)
        rec.strength = float(np.clip(SEED_C * margin / denom, 0.0, 1.0))

        # Solidify / soften (seed gates only)
        if (
            not rec.solidified
            and rec.trials >= _MIN_TRIALS
            and rec.acc_phi >= SOLIDIFY_ACC
            and rec.preferred_dir != 0
        ):
            rec.solidified = True
            self.n_solidify_events += 1
        elif rec.solidified and rec.acc_phi < SOFTEN_ACC and rec.soft_fails >= 3:
            # decoherence: pattern loses lock
            rec.solidified = False
            rec.strength *= SEED_POOF
            self.n_soften_events += 1

        return rec

    def bias_for(self, key: str) -> dict[str, float]:
        """
        Monte Carlo control biases for this signature.
        All derived from solidified accuracy + seeds.
        """
        rec = self.get(key)
        if not rec.solidified or rec.strength <= 0:
            return {
                "solidified": 0.0,
                "strength": 0.0,
                "dir": 0.0,
                "mu_scale": 1.0,
                "collapse_boost": 0.0,
                "path_weight": 1.0,
                "acc_phi": rec.acc_phi,
                "trials": float(rec.trials),
            }
        # Collapse toward observed True when that branch was historically accurate
        p_obs = rec.preferred_collapse_true()
        collapse_boost = rec.strength * (2.0 * p_obs - 1.0)  # ∈ [-s, s]
        # Amplify |μ| when pattern strong; direction lock via preferred_dir
        mu_scale = 1.0 + rec.strength * SEED_PHI * SEED_C
        path_weight = 1.0 + rec.strength / SEED_PHI
        return {
            "solidified": 1.0,
            "strength": rec.strength,
            "dir": float(rec.preferred_dir),
            "mu_scale": float(mu_scale),
            "collapse_boost": float(collapse_boost),
            "path_weight": float(path_weight),
            "acc_phi": rec.acc_phi,
            "trials": float(rec.trials),
        }

    def summary(self) -> dict[str, Any]:
        solid = [p for p in self.patterns.values() if p.solidified]
        # Top patterns by strength
        ranked = sorted(
            self.patterns.values(),
            key=lambda p: (p.solidified, p.strength, p.trials),
            reverse=True,
        )[:12]
        # Refinement curve: φ-EWMA of global hits in chunks
        refine = None
        if len(self.rolling_hits) >= _MIN_TRIALS:
            a = 1.0 / SEED_PHI
            m = 0.5
            series = []
            for h in self.rolling_hits:
                m = a * h + (1.0 - a) * m
                series.append(m)
            # first third vs last third
            n = len(series)
            t1 = n // 3
            t3 = 2 * n // 3
            refine = {
                "early_acc_phi": float(np.mean(series[: max(t1, 1)])),
                "late_acc_phi": float(np.mean(series[t3:])) if t3 < n else float(series[-1]),
                "final_acc_phi": float(series[-1]),
                "lift": float(series[-1] - series[max(t1 - 1, 0)]),
            }
        return {
            "symbol": self.symbol,
            "n_patterns": len(self.patterns),
            "n_solidified": len(solid),
            "n_updates": self.n_updates,
            "n_solidify_events": self.n_solidify_events,
            "n_soften_events": self.n_soften_events,
            "solidify_threshold": SOLIDIFY_ACC,
            "soften_threshold": SOFTEN_ACC,
            "min_trials": _MIN_TRIALS,
            "free_parameters": 0,
            "refinement": refine,
            "top_patterns": [
                {
                    "key": p.key,
                    "solidified": p.solidified,
                    "strength": round(p.strength, 4),
                    "acc_phi": round(p.acc_phi, 4),
                    "accuracy": round(p.accuracy(), 4),
                    "trials": p.trials,
                    "preferred_dir": p.preferred_dir,
                }
                for p in ranked
                if p.trials > 0
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_updates": self.n_updates,
            "n_solidify_events": self.n_solidify_events,
            "n_soften_events": self.n_soften_events,
            "patterns": {k: asdict(v) for k, v in self.patterns.items()},
            "rolling_hits": self.rolling_hits[-5000:],  # cap
            "free_parameters": 0,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatternMemory":
        mem = cls(symbol=str(data.get("symbol", "")))
        mem.n_updates = int(data.get("n_updates", 0))
        mem.n_solidify_events = int(data.get("n_solidify_events", 0))
        mem.n_soften_events = int(data.get("n_soften_events", 0))
        mem.rolling_hits = list(data.get("rolling_hits", []))
        for k, v in (data.get("patterns") or {}).items():
            mem.patterns[k] = PatternRecord(**v)
        return mem

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "PatternMemory":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def default_ledger_path(symbol: str) -> Path:
    base = Path(r"D:\training data\FSOT-Market-History\patterns")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in symbol.upper())
    return base / f"{safe}_mc_pattern_ledger.json"


def extract_state(
    rets: np.ndarray,
    volumes: np.ndarray | None,
    sentiment: float,
    window: int,
) -> tuple[str, dict[str, Any], int]:
    """Evaluate FSOT bar → signature + pred_dir."""
    ev = evaluate_market_bar(rets, volumes, sentiment, window=window)
    key = PatternMemory.signature_from_eval(ev, window=window)
    pred_dir = pred_dir_from_ev(ev)
    return key, ev, pred_dir
