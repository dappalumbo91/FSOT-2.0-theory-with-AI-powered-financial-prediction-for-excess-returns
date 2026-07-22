from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

HISTORY = Path(r"D:\training data\FSOT-Market-History")

router = APIRouter(tags=["history"])


@router.get("/api/history/status")
def history_status():
    ohlcv = HISTORY / "ohlcv"
    patterns = HISTORY / "patterns" / "emergence_entropy_ledger.json"
    verify = HISTORY / "verification" / "fsot_pin_verification.json"
    manifest = HISTORY / "manifests" / "download_manifest.json"
    files = list(ohlcv.glob("*.csv")) if ohlcv.exists() else []
    return {
        "history_root": str(HISTORY),
        "ohlcv_files": len(files),
        "symbols": sorted(p.stem for p in files),
        "has_pattern_ledger": patterns.exists(),
        "has_verification": verify.exists(),
        "has_manifest": manifest.exists(),
        "paths": {
            "ohlcv": str(ohlcv),
            "patterns": str(patterns),
            "verification": str(verify),
            "manifest": str(manifest),
        },
    }


@router.get("/api/history/patterns")
def history_patterns():
    path = HISTORY / "patterns" / "emergence_entropy_ledger.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Pattern ledger not built yet. Run scripts/build_historical_patterns.py",
        )
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/api/history/verification")
def history_verification():
    path = HISTORY / "verification" / "fsot_pin_verification.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Run scripts/verify_fsot_pin.py first",
        )
    return json.loads(path.read_text(encoding="utf-8"))
