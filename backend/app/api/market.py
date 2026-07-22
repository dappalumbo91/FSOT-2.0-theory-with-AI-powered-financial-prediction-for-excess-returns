from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.market.providers import get_market_service

router = APIRouter(tags=["market"])


@router.get("/api/watchlist")
def watchlist():
    return get_market_service().list_watchlist()


@router.get("/api/market/{symbol}/ohlcv")
def ohlcv(
    symbol: str,
    range: str = Query("1y", alias="range"),
    interval: str = Query("1d"),
):
    svc = get_market_service()
    records = svc.ohlcv_records(symbol, range_=range, interval=interval)
    if not records:
        raise HTTPException(status_code=404, detail=f"No OHLCV data for {symbol}")
    meta = svc.resolve(symbol)
    return {
        "symbol": meta["symbol"],
        "name": meta.get("name"),
        "class": meta.get("class") or meta.get("section"),
        "range": range,
        "interval": interval,
        "bars": records,
    }


@router.get("/api/market/{symbol}/quote")
def quote(symbol: str):
    q = get_market_service().get_quote(symbol)
    if q.get("price") is None:
        raise HTTPException(status_code=404, detail=f"No quote for {symbol}")
    return q
