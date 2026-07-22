from __future__ import annotations

from fastapi import APIRouter, Query

from app.news.feeds import get_news_service

router = APIRouter(tags=["news"])


@router.get("/api/news")
def news(limit: int = Query(40, ge=1, le=200)):
    items = get_news_service().fetch_all()[:limit]
    return {"count": len(items), "items": items}


@router.get("/api/news/observer")
def observer(symbol: str | None = Query(None)):
    return get_news_service().observer_mod(symbol)
