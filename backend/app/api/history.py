from __future__ import annotations

import os as _os
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[3]


def _fsot_local_path(env, default):
    """Local path outside the repo: $env if set, else a path relative to the repo root."""
    _v = _os.environ.get(env, "").strip()
    return _Path(_v) if _v else _REPO_ROOT / default


import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

HISTORY = _fsot_local_path('FSOT_TRAINING_DATA', 'data_external/training_data') / 'FSOT-Market-History'

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
