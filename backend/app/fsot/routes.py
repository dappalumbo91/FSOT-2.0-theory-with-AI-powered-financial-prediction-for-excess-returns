"""
Preregistered FSOT domain routes — authority from I:\\FSOT-Physical-Archive.

These are fractal coordinates of one seed engine, NOT free fit parameters.
Sources:
  - vendor/fsot_compute.py (core 35, Economics)
  - data/extension_domains_manifest.yaml (Finance_Markets, Econometrics, Econophysics)
"""

from __future__ import annotations

from typing import Any

# Core 35 — Mathematical Key §5 / fsot_compute.py
ECONOMICS: dict[str, Any] = {
    "name": "Economics",
    "D_eff": 20,
    "recent_hits": 3,
    "delta_psi": 1.5,
    "delta_theta": 1.0,
    "observed": True,
    "source": "vendor/fsot_compute.py DomainConfig Economics",
}

# Extension folds — extension_domains_manifest.yaml (preregistered)
FINANCE_MARKETS: dict[str, Any] = {
    "name": "Finance_Markets",
    "D_eff": 19,
    "recent_hits": 2,
    "delta_psi": 0.75,
    "delta_theta": 1.0,
    "observed": True,
    "source": "extension_domains_manifest.yaml Finance_Markets",
}

FINANCE_MARKETS_PANEL: dict[str, Any] = {
    "name": "Finance_Markets_Panel",
    "D_eff": 19,
    "recent_hits": 2,
    "delta_psi": 0.65,
    "delta_theta": 1.0,
    "observed": True,
    "source": "extension_domains_manifest.yaml Finance_Markets_Panel",
}

ECONOMETRICS: dict[str, Any] = {
    "name": "Econometrics",
    "D_eff": 19,
    "recent_hits": 2,
    "delta_psi": 0.7,
    "delta_theta": 1.0,
    "observed": True,
    "source": "extension_domains_manifest.yaml Econometrics",
}

ECONOPHYSICS: dict[str, Any] = {
    "name": "Econophysics",
    "D_eff": 20,
    "recent_hits": 3,
    "delta_psi": 1.5,
    "delta_theta": 1.0,
    "observed": True,
    "source": "extension_domains_manifest.yaml Econophysics",
}

# Declared Economics coupling factor from scripts/fsot_api_predict_lib.py DOMAIN_FACTORS
# Used the same way archive finance/economics panels attach scalar to measured authority.
DOMAIN_FACTOR_ECONOMICS = 0.0004

ROUTES = {
    "Economics": ECONOMICS,
    "Finance_Markets": FINANCE_MARKETS,
    "Finance_Markets_Panel": FINANCE_MARKETS_PANEL,
    "Econometrics": ECONOMETRICS,
    "Econophysics": ECONOPHYSICS,
}


def get_route(name: str = "Economics") -> dict[str, Any]:
    key = name if name in ROUTES else "Economics"
    return dict(ROUTES[key])
