from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from app import __version__
from app.fsot.intrinsic import spine_scalars
from app.fsot.routes import DOMAIN_FACTOR_ECONOMICS, ECONOMICS, FINANCE_MARKETS

router = APIRouter(tags=["health"])

PIN_PATH = Path(__file__).resolve().parents[1] / "fsot" / "PIN.json"


@router.get("/api/health")
def health():
    pin = {}
    if PIN_PATH.exists():
        pin = json.loads(PIN_PATH.read_text(encoding="utf-8-sig"))

    spine = spine_scalars()
    return {
        "status": "ok",
        "version": __version__,
        "fsot": {
            "formula": "S = K * (T1 + T2 + T3)",
            "free_parameters": 0,
            "method": "fsot_intrinsic_zero_free",
            "authority": "I:/FSOT-Physical-Archive/02_FSOT-2.1-Lean-Full/vendor/fsot_compute.py",
            "K": spine["Economics"]["K"],
            "economics_S": spine["Economics"]["S"],
            "finance_markets_S": spine["Finance_Markets"]["S"],
            "econophysics_S": spine["Econophysics"]["S"],
            "domain": ECONOMICS,
            "finance_markets_route": FINANCE_MARKETS,
            "domain_factor_economics": DOMAIN_FACTOR_ECONOMICS,
            "pin": pin,
            "spine": {k: v["S"] for k, v in spine.items()},
        },
    }
