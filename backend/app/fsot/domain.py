"""Preregistered FSOT domain routes — re-export from routes.py."""

from __future__ import annotations

from typing import Any

from app.fsot.routes import ECONOMICS, FINANCE_MARKETS, get_route

ECONOMICS_DOMAIN = {
    "name": ECONOMICS["name"],
    "D_eff": ECONOMICS["D_eff"],
    "hits": ECONOMICS["recent_hits"],
    "delta_psi": ECONOMICS["delta_psi"],
    "delta_theta": ECONOMICS["delta_theta"],
    "observed": ECONOMICS["observed"],
}

FINANCE_MARKETS_PANEL = {
    "name": FINANCE_MARKETS["name"],
    "D_eff": FINANCE_MARKETS["D_eff"],
    "hits": FINANCE_MARKETS["recent_hits"],
    "delta_psi": FINANCE_MARKETS["delta_psi"],
    "delta_theta": FINANCE_MARKETS["delta_theta"],
    "observed": FINANCE_MARKETS["observed"],
}


def domain_base_params(domain: str = "Economics") -> dict[str, Any]:
    r = get_route(domain)
    return {
        "D_eff": float(r["D_eff"]),
        "recent_hits": float(r["recent_hits"]),
        "delta_psi": float(r["delta_psi"]),
        "delta_theta": float(r.get("delta_theta", 1.0)),
        "observed": bool(r["observed"]),
        "domain": r["name"],
    }
