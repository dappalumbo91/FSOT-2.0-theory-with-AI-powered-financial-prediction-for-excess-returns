"""Broker API — dry-run Robinhood crypto by default."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.broker.robinhood_crypto import OrderIntent, RobinhoodCryptoClient
from app.market.providers import get_market_service
from app.fsot.bhs_engine import run_bhs_backtest

router = APIRouter(tags=["broker"])


class PreviewBody(BaseModel):
    symbol: str = "BTC"
    side: str = Field("buy", description="buy | sell")
    notional_usd: float = Field(25.0, ge=1.0, le=100_000.0)
    source_signal: str = ""


@router.get("/api/broker/status")
def broker_status():
    """Show broker wiring status. Live trading is off unless dual env flags set."""
    client = RobinhoodCryptoClient()
    return client.status()


@router.post("/api/broker/preview")
def broker_preview(body: PreviewBody):
    """
    Dry-run only: build signed order preview for Robinhood Crypto.
    Never submits a real order from this endpoint.
    """
    client = RobinhoodCryptoClient()
    intent = OrderIntent(
        symbol=body.symbol,
        side=body.side,
        notional_usd=body.notional_usd,
        source_signal=body.source_signal or body.side.upper(),
        note="API preview",
    )
    result = client.preview_order(intent)
    return result.to_dict()


@router.post("/api/broker/preview-from-signal")
def broker_preview_from_signal(
    symbol: str = Query("BTC"),
    notional_usd: float = Query(25.0, ge=1.0, le=10_000.0),
    range: str = Query("1y", alias="range"),
):
    """
    Run BHS on real OHLCV, map BUY/SELL → dry-run Robinhood order preview.
    HOLD → no order. Still never live-submits.
    """
    svc = get_market_service()
    df = svc.get_ohlcv(symbol, range_=range)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    # Use latest BHS decision via short backtest tail (last decision)
    paper = run_bhs_backtest(df, capital=10_000.0, symbol=symbol, store_curve=True)
    if paper.get("error"):
        raise HTTPException(status_code=400, detail=paper["error"])

    # Infer last action from equity curve or recompute — use summary fields
    # Last non-hold from curve if present
    action = "HOLD"
    curve = paper.get("equity_curve") or []
    for pt in reversed(curve):
        a = str(pt.get("action") or "").upper()
        if a in ("BUY", "SELL"):
            action = a
            break
    # If curve has no action field, use hold-heavy default
    if action == "HOLD" and paper.get("n_buy", 0) + paper.get("n_sell", 0) == 0:
        action = "HOLD"

    client = RobinhoodCryptoClient()
    intent = client.signal_to_intent(symbol, action, notional_usd=notional_usd)
    if intent is None:
        return {
            "action": action,
            "order": None,
            "message": "HOLD — no dry-run order (stay in synthetic cash)",
            "bhs_summary": {
                "commit_directional_accuracy": paper.get("commit_directional_accuracy"),
                "total_pnl": paper.get("total_pnl"),
                "pct_hold": paper.get("pct_hold"),
            },
            "broker": client.status(),
        }

    result = client.preview_order(intent)
    return {
        "action": action,
        "order": result.to_dict(),
        "bhs_summary": {
            "commit_directional_accuracy": paper.get("commit_directional_accuracy"),
            "total_pnl": paper.get("total_pnl"),
            "pct_hold": paper.get("pct_hold"),
            "progress_to_70_80": paper.get("progress_to_70_80"),
        },
        "broker": client.status(),
    }
