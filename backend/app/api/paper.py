"""Synthetic dollar paper portfolio API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.backtest.paper_portfolio import run_paper_portfolio
from app.market.providers import get_market_service
from app.news.feeds import get_news_service

router = APIRouter(tags=["paper"])


@router.get("/api/paper/{symbol}")
def paper_portfolio(
    symbol: str,
    range: str = Query("2y", alias="range"),
    capital: float = Query(10_000.0, ge=100.0, le=100_000_000.0, description="Starting synthetic USD"),
    window: int = Query(21, ge=5, le=252),
    mode: str = Query(
        "bhs",
        description="bhs | bhs_long_only | always_in | solid_gated | long_only | buy_hold",
    ),
    hold_horizon: int = Query(5, ge=1, le=21, description="Fib hold days for BHS mode"),
    use_news: bool = Query(True),
    step: int = Query(1, ge=1, le=10),
):
    """
    Theoretical money: Buy/Hold/Sell (default) or other modes on real OHLCV.
    Adjust capital to see synthetic P&L before live markets.
    """
    svc = get_market_service()
    df = svc.get_ohlcv(symbol, range_=range)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    obs = 0.0
    if use_news:
        try:
            obs = float(get_news_service().observer_mod(symbol).get("observer_mod", 0.0))
        except Exception:
            obs = 0.0

    result = run_paper_portfolio(
        df,
        capital=capital,
        window=window,
        mode=mode,
        symbol=symbol,
        sentiment=obs,
        step=step,
        hold_horizon=hold_horizon,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    meta = svc.resolve(symbol)
    return {
        "symbol": meta["symbol"],
        "name": meta.get("name"),
        "range": range,
        "observer_mod": obs,
        "paper": result,
    }
