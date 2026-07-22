"""Forward prediction monitor — record now, score when future arrives."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.market.providers import get_market_service
from app.monitor.forward_journal import ForwardJournal

router = APIRouter(tags=["monitor"])


def _journal() -> ForwardJournal:
    return ForwardJournal()


@router.get("/api/monitor/forward")
def forward_summary():
    """Summary of open + resolved forward predictions (true future test)."""
    return _journal().summary()


@router.post("/api/monitor/forward/record")
def forward_record(
    symbol: str = Query("BTC"),
    horizon: int = Query(5, ge=1, le=21),
    range: str = Query("2y", alias="range"),
):
    """
    Record a forward prediction at the latest bar.
    Resolve later with POST /api/monitor/forward/resolve after horizon days.
    """
    svc = get_market_service()
    df = svc.get_ohlcv(symbol, range_=range)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    out = _journal().record_from_market(
        df, symbol=symbol, horizon=horizon, source="api_monitor"
    )
    if out.get("error"):
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@router.post("/api/monitor/forward/resolve")
def forward_resolve(symbol: str | None = Query(None)):
    """Resolve pending predictions using latest real OHLCV (future now known)."""
    svc = get_market_service()

    def get_ohlcv(sym: str):
        return svc.get_ohlcv(sym, range_="2y")

    return _journal().resolve_pending(get_ohlcv, symbol=symbol)


@router.post("/api/monitor/forward/record-crypto-watchlist")
def forward_record_crypto(
    horizon: int = Query(5, ge=1, le=21),
    range: str = Query("1y", alias="range"),
):
    """Record forward predictions for crypto watchlist (BTC, ETH, …)."""
    svc = get_market_service()
    wl = svc.list_watchlist()
    cryptos = [c["symbol"] for c in (wl.get("crypto") or [])]
    if not cryptos:
        cryptos = ["BTC", "ETH", "SOL"]
    j = _journal()
    results = []
    for sym in cryptos:
        try:
            df = svc.get_ohlcv(sym, range_=range)
            if df is None or df.empty:
                results.append({"symbol": sym, "error": "no_data"})
                continue
            r = j.record_from_market(df, symbol=sym, horizon=horizon, source="crypto_watchlist")
            results.append({"symbol": sym, "entry": r.get("entry"), "error": r.get("error")})
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})
    return {"count": len(results), "items": results, "summary": j.summary()}
