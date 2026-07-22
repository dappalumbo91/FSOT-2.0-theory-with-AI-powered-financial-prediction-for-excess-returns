#!/usr/bin/env python3
"""
Verify Market Monitor FSOT pin against archive mpmath authority (FSOT 2.1).

This does NOT re-run Lean 4 obligations (those live in FSOT-2.1-Lean).
It certifies that the app's float path matches vendor/fsot_compute.py domain scalars.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app" / "fsot"))

from app.fsot.domain import ECONOMICS_DOMAIN  # noqa: E402
from app.fsot.fast import ScalarInputF, compute_scalar_fast  # noqa: E402

OUT = Path(r"D:\training data\FSOT-Market-History\verification")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    from compute_authority import domain_scalar, DOMAINS  # type: ignore

    rows = []
    all_ok = True
    for name in sorted(DOMAINS.keys()):
        auth = float(domain_scalar(name))
        d = DOMAINS[name]
        si = ScalarInputF(
            N=1.0,
            P=1.0,
            D_eff=float(d.D_eff),
            delta_psi=float(d.delta_psi),
            delta_theta=float(d.delta_theta),
            recent_hits=float(d.hits),
            observed=bool(d.observed),
            rho=1.0,
            scale=1.0,
            amplitude=1.0,
            trend_bias=0.0,
        )
        fast = compute_scalar_fast(si)
        rel = abs(fast - auth) / max(abs(auth), 1e-12)
        ok = rel < 1e-8
        all_ok = all_ok and ok
        rows.append(
            {
                "domain": name,
                "authority_S": auth,
                "fast_S": fast,
                "rel_err": rel,
                "ok": ok,
            }
        )

    econ_auth = float(domain_scalar("Economics"))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lean_verification_integrated": False,
        "lean_note": (
            "Lean/Coq/Isabelle/F*/Rust obligation gauntlet is NOT executed inside "
            "FSOT-Market-Monitor. Authority is pinned compute from "
            "I:\\FSOT-Physical-Archive\\02_FSOT-2.1-Lean-Full\\vendor\\fsot_compute.py "
            "matching FSOT-2.1-Lean. Run scripts/run_publication_verification_bundle.py "
            "in the Lean repo for formal obligations."
        ),
        "old_finance_repo_updated": False,
        "old_finance_note": (
            "The Sep-2025 finance GitHub repo still has multiplicative S_D_chaotic. "
            "This app does NOT silently claim that repo is Lean-updated; it is reference-only under legacy_ref/."
        ),
        "formula": "S = K * (T1 + T2 + T3)",
        "economics_domain": ECONOMICS_DOMAIN,
        "economics_S_authority": econ_auth,
        "domains_checked": len(rows),
        "all_ok": all_ok,
        "max_rel_err": max(r["rel_err"] for r in rows),
        "rows": rows,
        "prediction_method": "emergence_entropy_v2",
        "prediction_note": (
            "Direction uses z(dS)-0.65*z(entropy)+observer_mod. "
            "Does NOT use residual S-S0 (that produced ~50% accuracy)."
        ),
    }
    path = OUT / "fsot_pin_verification.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "rows"}, indent=2))
    print(f"Wrote {path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
