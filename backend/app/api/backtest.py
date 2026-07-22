from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.backtest.walk_forward import walk_forward_backtest
from app.market.providers import get_market_service

router = APIRouter(tags=["backtest"])


@router.get("/api/backtest/{symbol}")
def backtest(
    symbol: str,
    range: str = Query("2y", alias="range"),
    window: int = Query(30, ge=5, le=252),
    domain: str = Query("Economics"),
):
    svc = get_market_service()
    df = svc.get_ohlcv(symbol, range_=range)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    mkt = None
    if str(symbol).upper().replace("^", "") not in ("SPY", "GSPC"):
        try:
            mkt = svc.get_ohlcv("SPY", range_=range)
        except Exception:
            mkt = None
    result = walk_forward_backtest(
        df, window=window, domain=domain, symbol=symbol, market_df=mkt
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    meta = svc.resolve(symbol)
    return {
        "symbol": meta["symbol"],
        "name": meta.get("name"),
        "range": range,
        "backtest": result,
    }
