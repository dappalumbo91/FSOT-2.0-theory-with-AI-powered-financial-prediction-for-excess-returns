"""
Free financial news / observer feeds (no API keys).

Sources:
  - Yahoo Finance RSS
  - CNBC / MarketWatch / Reuters world / CoinDesk RSS
  - SEC EDGAR recent filings RSS
  - Federal Reserve press RSS
  - Google News finance RSS (query)

Sentiment is a lightweight lexicon observer_mod in [-1, 1] for FSOT δψ coupling.
Not a commercial NLP model — intentional zero-credential design.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from email.utils import parsedate_to_datetime

import requests

from app.market.cache import cache

# Bullish / bearish keyword lexicons (finance observer proxies)
BULL = {
    "surge", "soar", "rally", "gain", "gains", "beat", "beats", "record", "high",
    "growth", "upgrade", "bull", "bullish", "optimism", "strong", "outperform",
    "profit", "profits", "recovery", "breakout", "expansion", "buy", "all-time",
}
BEAR = {
    "crash", "plunge", "fall", "falls", "drop", "drops", "miss", "misses", "low",
    "recession", "downgrade", "bear", "bearish", "fear", "weak", "underperform",
    "loss", "losses", "default", "layoff", "bankruptcy", "selloff", "sell-off",
    "inflation", "tariff", "war", "sanction", "fraud", "probe", "investigation",
}

FEEDS = [
    {"id": "yahoo_finance", "name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rssindex"},
    {"id": "cnbc_top", "name": "CNBC Top News", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"},
    {"id": "cnbc_finance", "name": "CNBC Finance", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"},
    {"id": "marketwatch", "name": "MarketWatch Top", "url": "https://feeds.marketwatch.com/marketwatch/topstories/"},
    {"id": "reuters_business", "name": "Reuters Business", "url": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best"},
    {"id": "coindesk", "name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
    {"id": "cointelegraph", "name": "Cointelegraph", "url": "https://cointelegraph.com/rss"},
    {"id": "fed_press", "name": "Federal Reserve Press", "url": "https://www.federalreserve.gov/feeds/press_all.xml"},
    {"id": "sec_edgar", "name": "SEC EDGAR", "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=include&count=40&output=atom"},
    {
        "id": "google_markets",
        "name": "Google News Markets",
        "url": "https://news.google.com/rss/search?q=stock+market+OR+S%26P+500+OR+bitcoin&hl=en-US&gl=US&ceid=US:en",
    },
    {
        "id": "google_fed",
        "name": "Google News Fed",
        "url": "https://news.google.com/rss/search?q=Federal+Reserve+OR+interest+rates&hl=en-US&gl=US&ceid=US:en",
    },
]


@dataclass
class NewsItem:
    source: str
    source_id: str
    title: str
    link: str
    published: str | None
    summary: str
    sentiment: float
    tokens_bull: int
    tokens_bear: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "title": self.title,
            "link": self.link,
            "published": self.published,
            "summary": self.summary[:400],
            "sentiment": self.sentiment,
            "tokens_bull": self.tokens_bull,
            "tokens_bear": self.tokens_bear,
        }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\\-]+", text.lower())


def score_text(text: str) -> tuple[float, int, int]:
    toks = _tokenize(text)
    if not toks:
        return 0.0, 0, 0
    b = sum(1 for t in toks if t in BULL)
    r = sum(1 for t in toks if t in BEAR)
    raw = b - r
    # normalize by mention density
    dens = (b + r) / max(len(toks), 1)
    score = float(np_tanh(raw * 0.35 + dens * 2.0 * (1 if raw >= 0 else -1 if raw < 0 else 0)))
    if b + r == 0:
        score = 0.0
    else:
        score = float(max(-1.0, min(1.0, (b - r) / (b + r) * min(1.0, 0.3 + 0.1 * (b + r)))))
    return score, b, r


def np_tanh(x: float) -> float:
    import math
    return math.tanh(x)


def _parse_rss(xml_text: str, source_id: str, source_name: str) -> list[NewsItem]:
    items: list[NewsItem] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    # RSS 2.0
    channels = root.findall("channel")
    if channels:
        for item in channels[0].findall("item")[:40]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = (item.findtext("description") or "").strip()
            pub = item.findtext("pubDate")
            published = None
            if pub:
                try:
                    published = parsedate_to_datetime(pub).astimezone(timezone.utc).isoformat()
                except Exception:
                    published = pub
            text = f"{title} {re.sub('<[^>]+>', ' ', desc)}"
            sent, tb, tr = score_text(text)
            if title:
                items.append(
                    NewsItem(source_name, source_id, title, link, published, desc, sent, tb, tr)
                )
        return items

    # Atom
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("a:entry", ns)[:40]:
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        link_el = entry.find("a:link", ns)
        link = link_el.get("href") if link_el is not None else ""
        summary = entry.findtext("a:summary", default="", namespaces=ns) or entry.findtext(
            "a:content", default="", namespaces=ns
        ) or ""
        updated = entry.findtext("a:updated", default="", namespaces=ns)
        text = f"{title} {re.sub('<[^>]+>', ' ', summary)}"
        sent, tb, tr = score_text(text)
        if title:
            items.append(
                NewsItem(source_name, source_id, title, link or "", updated or None, summary, sent, tb, tr)
            )
    return items


class NewsService:
    def fetch_feed(self, feed: dict[str, str], timeout: float = 12.0) -> list[NewsItem]:
        cache_key = f"news:{feed['id']}"
        cached = cache.get(cache_key)
        if cached is not None:
            return [NewsItem(**x) if isinstance(x, dict) else x for x in cached]

        headers = {
            "User-Agent": "FSOT-Market-Monitor/1.0 (research; +https://github.com/dappalumbo91)",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        }
        try:
            r = requests.get(feed["url"], headers=headers, timeout=timeout)
            r.raise_for_status()
            items = _parse_rss(r.text, feed["id"], feed["name"])
            cache.set(cache_key, [i.to_dict() for i in items], ttl_seconds=300)
            return items
        except Exception:
            return []

    def fetch_all(self, max_per_feed: int = 15) -> list[dict[str, Any]]:
        all_items: list[NewsItem] = []
        for feed in FEEDS:
            items = self.fetch_feed(feed)[:max_per_feed]
            all_items.extend(items)
            time.sleep(0.05)
        # sort by published if possible
        def key(it: NewsItem):
            return it.published or ""

        all_items.sort(key=key, reverse=True)
        return [i.to_dict() for i in all_items]

    def observer_mod(self, symbol: str | None = None) -> dict[str, Any]:
        """
        Aggregate observer_mod in [-1, 1] for FSOT δψ coupling.
        Optional symbol filters titles containing ticker keywords.
        """
        items = self.fetch_all()
        if symbol:
            sym = symbol.upper().replace("^", "")
            aliases = {sym, sym.lower()}
            if sym == "BTC":
                aliases |= {"bitcoin", "btc"}
            if sym == "ETH":
                aliases |= {"ethereum", "eth"}
            if sym in ("GSPC", "SPY"):
                aliases |= {"s&p", "s&p 500", "sp500", "spy", "wall street", "stocks"}
            filtered = []
            for it in items:
                t = (it["title"] + " " + it.get("summary", "")).lower()
                if any(a in t for a in aliases):
                    filtered.append(it)
            use = filtered if filtered else items[:30]
        else:
            use = items[:40]

        if not use:
            return {
                "observer_mod": 0.0,
                "n_headlines": 0,
                "mean_sentiment": 0.0,
                "bull_share": 0.0,
                "bear_share": 0.0,
                "top": [],
                "as_of": datetime.now(timezone.utc).isoformat(),
            }

        sents = [float(x["sentiment"]) for x in use]
        mean = float(sum(sents) / len(sents))
        # concentration toward extremes
        bull_share = sum(1 for s in sents if s > 0.15) / len(sents)
        bear_share = sum(1 for s in sents if s < -0.15) / len(sents)
        observer = float(max(-1.0, min(1.0, mean * (0.6 + 0.4 * abs(bull_share - bear_share)))))

        top = sorted(use, key=lambda x: abs(x["sentiment"]), reverse=True)[:12]
        return {
            "observer_mod": observer,
            "n_headlines": len(use),
            "mean_sentiment": mean,
            "bull_share": bull_share,
            "bear_share": bear_share,
            "top": top,
            "feeds": [f["id"] for f in FEEDS],
            "as_of": datetime.now(timezone.utc).isoformat(),
            "symbol_filter": symbol,
        }


@lru_cache(maxsize=1)
def get_news_service() -> NewsService:
    return NewsService()
