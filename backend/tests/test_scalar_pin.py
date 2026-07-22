"""Verify intrinsic engine uses preregistered Economics fold from archive authority."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app" / "fsot"))

from app.fsot.intrinsic import route_scalar, spine_scalars  # noqa: E402
from app.fsot.routes import ECONOMICS, FINANCE_MARKETS  # noqa: E402


def test_economics_matches_mpmath_authority():
    from compute_authority import domain_scalar  # type: ignore

    auth = float(domain_scalar("Economics"))
    spine = spine_scalars()
    fast = spine["Economics"]["S"]
    rel = abs(fast - auth) / max(abs(auth), 1e-12)
    assert rel < 1e-9, f"fast={fast} auth={auth} rel={rel}"


def test_finance_markets_preregistered_fold():
    r = route_scalar(FINANCE_MARKETS)
    assert r["D_eff"] == 19
    assert r["recent_hits"] == 2
    assert abs(r["delta_psi"] - 0.75) < 1e-12
    assert r["S"] != 0


def test_economics_route_locked():
    assert ECONOMICS["D_eff"] == 20
    assert ECONOMICS["recent_hits"] == 3
    assert ECONOMICS["delta_psi"] == 1.5
    assert ECONOMICS["observed"] is True
