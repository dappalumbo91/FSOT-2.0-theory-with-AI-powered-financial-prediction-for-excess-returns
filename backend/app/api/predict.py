from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.market.providers import get_market_service
from app.news.feeds import get_news_service
from app.predict.engine import PredictEngine

router = APIRouter(tags=["predict"])
engine = PredictEngine()


def _observer_for(symbol: str | None = None) -> float:
    try:
        return float(get_news_service().observer_mod(symbol).get("observer_mod", 0.0))
    except Exception:
        return 0.0


@router.get("/api/predict/batch")
def predict_batch(
    range: str = Query("6mo", alias="range"),
    window: int = Query(30, ge=5, le=252),
    section: str | None = Query(None, description="indices|stocks|crypto or omit for all"),
    use_news: bool = Query(True),
):
    """Must be registered before /api/predict/{symbol} so 'batch' is not captured as a symbol."""
    svc = get_market_service()
    wl = svc.list_watchlist()
    symbols: list[str] = []
    sections = [section] if section else ["indices", "stocks", "crypto"]
    for sec in sections:
        for item in wl.get(sec, []) or []:
            symbols.append(item["symbol"])

    # Global market observer once (fast); per-symbol optional later
    global_obs = _observer_for(None) if use_news else 0.0
    results: list[dict[str, Any]] = []

    def _one(sym: str) -> dict[str, Any]:
        try:
            df = svc.get_ohlcv(sym, range_=range)
            if df is None or df.empty:
                return {"symbol": sym, "error": "no_data"}
            obs = _observer_for(sym) if use_news else global_obs
            # blend symbol-specific with global
            if use_news:
                obs = 0.6 * obs + 0.4 * global_obs
            pred = engine.latest_prediction(
                df, window=window, observer_mod=obs, symbol=sym, boosted=False
            )
            # ensure sentiment/observer reached intrinsic (engine passes observer_mod)
            quote = svc.get_quote(sym)
            meta = svc.resolve(sym)
            return {
                "symbol": meta["symbol"],
                "name": meta.get("name"),
                "class": meta.get("class") or meta.get("section"),
                "price": quote.get("price"),
                "change_pct": quote.get("change_pct"),
                "signal": pred.get("signal"),
                "confidence": pred.get("confidence"),
                "S": pred.get("S"),
                "dS": pred.get("dS"),
                "entropy": pred.get("entropy"),
                "emergence_score": pred.get("emergence_score"),
                "composite": pred.get("composite"),
                "pred_return": pred.get("pred_return"),
                "regime": pred.get("regime"),
                "observer_mod": obs,
                "method": pred.get("method"),
                "error": None,
            }
        except Exception as e:
            return {"symbol": sym, "error": str(e)}

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_one, s): s for s in symbols}
        for fut in as_completed(futs):
            results.append(fut.result())

    order = {s: i for i, s in enumerate(symbols)}
    results.sort(key=lambda r: order.get(r.get("symbol", ""), 999))
    return {"count": len(results), "items": results, "global_observer_mod": global_obs}


@router.get("/api/predict/{symbol}/montecarlo")
def predict_montecarlo(
    symbol: str,
    range: str = Query("2y", alias="range"),
    window: int = Query(21, ge=5, le=252),
    horizon: int = Query(21, ge=5, le=55, description="Fib-snapped path length (days)"),
    n_paths: int = Query(512, ge=32, le=2048, description="Ensemble size"),
    use_news: bool = Query(True),
    seed: int | None = Query(None, description="RNG seed for reproducibility"),
    dynamic: bool = Query(
        True,
        description="Train pattern memory on history, solidify accurate FSOT signatures, bias paths",
    ),
    persist: bool = Query(True, description="Save pattern ledger to history drive"),
    walkforward: bool = Query(False, description="Also run causal MC hit-rate eval"),
    wf_horizon: int = Query(5, ge=1, le=21),
    wf_n_paths: int = Query(64, ge=16, le=256),
    wf_step: int = Query(10, ge=1, le=21),
):
    """
    Intelligent FSOT Monte Carlo (dynamic pattern collapse).

    1) Causal walk: FSOT signatures scored on real forward returns.
    2) Solidify when φ-EWMA accuracy > 0.5+Poof and trials ≥ Fib(8).
    3) Multipath futures: collapse TRUE→μ / FALSE→Poof, biased by solid anchors.
    Zero free parameters (seeds + preregistered folds + measured data only).
    """
    svc = get_market_service()
    df = svc.get_ohlcv(symbol, range_=range)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    obs = 0.0
    news_payload = None
    if use_news:
        news_payload = get_news_service().observer_mod(symbol)
        obs = float(news_payload.get("observer_mod", 0.0))

    mc = engine.monte_carlo(
        df,
        horizon=horizon,
        n_paths=n_paths,
        observer_mod=obs,
        window=window,
        symbol=symbol,
        seed=seed,
        dynamic=dynamic,
        persist=persist,
    )
    if mc.get("error"):
        raise HTTPException(status_code=400, detail=mc["error"])

    payload: dict[str, Any] = {
        "symbol": svc.resolve(symbol)["symbol"],
        "name": svc.resolve(symbol).get("name"),
        "quote": svc.get_quote(symbol),
        "observer_mod": obs,
        "news_observer": news_payload,
        "monte_carlo": mc,
    }
    if walkforward:
        payload["walkforward"] = engine.monte_carlo_walkforward(
            df,
            horizon=wf_horizon,
            n_paths=wf_n_paths,
            window=window,
            step=wf_step,
            observer_mod=obs,
            dynamic=dynamic,
            symbol=symbol,
        )
    return payload


@router.get("/api/predict/{symbol}")
def predict(
    symbol: str,
    range: str = Query("1y", alias="range"),
    window: int = Query(30, ge=5, le=252),
    domain: str = Query("Economics"),
    use_news: bool = Query(True),
    include_monte_carlo: bool = Query(False, description="Attach MC ensemble summary"),
    mc_horizon: int = Query(21, ge=5, le=55),
    mc_n_paths: int = Query(256, ge=32, le=1024),
):
    svc = get_market_service()
    df = svc.get_ohlcv(symbol, range_=range)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    obs = 0.0
    news_payload = None
    if use_news:
        news_payload = get_news_service().observer_mod(symbol)
        obs = float(news_payload.get("observer_mod", 0.0))
    result = engine.latest_prediction(
        df,
        window=window,
        domain=domain,
        observer_mod=obs,
        symbol=symbol,
        boosted=False,
        include_monte_carlo=include_monte_carlo,
        mc_horizon=mc_horizon,
        mc_n_paths=mc_n_paths,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    meta = svc.resolve(symbol)
    quote = svc.get_quote(symbol)
    return {
        "symbol": meta["symbol"],
        "name": meta.get("name"),
        "class": meta.get("class") or meta.get("section"),
        "quote": quote,
        "prediction": result,
        "news_observer": news_payload,
    }
