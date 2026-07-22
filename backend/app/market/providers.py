"""Market data providers: equities (Yahoo) + crypto (CoinGecko / Binance public)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
import yfinance as yf

from .cache import cache

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
WATCHLIST_PATH = ROOT / "config" / "watchlist.yaml"
HISTORY_OHLCV = Path(r"D:\training data\FSOT-Market-History\ohlcv")

COINGECKO = "https://api.coingecko.com/api/v3"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


def load_watchlist() -> dict[str, Any]:
    with open(WATCHLIST_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    out = df.copy()
    # Flatten multiindex columns from yfinance if present
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "Datetime": "time",
        "Date": "time",
    }
    out = out.rename(columns=rename)
    if "time" not in out.columns:
        out = out.reset_index()
        first = out.columns[0]
        out = out.rename(columns={first: "time"})

    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            out[col] = 0.0 if col == "volume" else out.get("close", 0.0)

    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    out = out.dropna(subset=["close"])
    return out[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _range_to_period(range_: str) -> str:
    m = {
        "1mo": "1mo",
        "3mo": "3mo",
        "6mo": "6mo",
        "1y": "1y",
        "2y": "2y",
        "5y": "5y",
        "10y": "10y",
        "20y": "max",
        "max": "max",
    }
    return m.get(range_, "1y")


def _days_from_range(range_: str) -> int:
    m = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
        "20y": 7300,
        "max": 10000,
    }
    return m.get(range_, 365)


class MarketService:
    def __init__(self) -> None:
        self.watchlist = load_watchlist()
        self._symbol_index = self._build_index()

    def _build_index(self) -> dict[str, dict[str, Any]]:
        idx: dict[str, dict[str, Any]] = {}
        for section in ("indices", "stocks", "crypto"):
            for item in self.watchlist.get(section, []) or []:
                sym = str(item["symbol"]).upper()
                idx[sym] = {**item, "section": section}
                # also index without ^
                if sym.startswith("^"):
                    idx[sym[1:]] = idx[sym]
        return idx

    def resolve(self, symbol: str) -> dict[str, Any]:
        key = symbol.upper()
        if key in self._symbol_index:
            return self._symbol_index[key]
        # ad-hoc equity
        if not key.startswith("BTC") and len(key) <= 6 and key.isalpha():
            return {"symbol": key, "name": key, "class": "equity", "yahoo": key, "section": "stocks"}
        # ad-hoc crypto guess
        return {
            "symbol": key,
            "name": key,
            "class": "crypto",
            "coingecko": key.lower(),
            "section": "crypto",
        }

    def list_watchlist(self) -> dict[str, Any]:
        return {
            "indices": self.watchlist.get("indices", []),
            "stocks": self.watchlist.get("stocks", []),
            "crypto": self.watchlist.get("crypto", []),
            "defaults": self.watchlist.get("defaults", {}),
        }

    def _load_history_csv(self, symbol: str) -> pd.DataFrame | None:
        """Prefer 20y archive on game drive when present."""
        safe = str(symbol).upper().replace("^", "")
        path = HISTORY_OHLCV / f"{safe}.csv"
        if not path.exists():
            # try raw symbol
            path = HISTORY_OHLCV / f"{str(symbol).upper()}.csv"
        if not path.exists():
            return None
        try:
            raw = pd.read_csv(path, index_col=0, parse_dates=True)
            raw = raw.sort_index()
            raw = raw.rename(columns={c: str(c).lower() for c in raw.columns})
            raw = raw.reset_index()
            first = raw.columns[0]
            raw = raw.rename(columns={first: "time"})
            return _normalize_ohlcv(raw)
        except Exception as e:
            log.warning("history csv load failed %s: %s", path, e)
            return None

    def get_ohlcv(
        self,
        symbol: str,
        range_: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        meta = self.resolve(symbol)
        cache_key = f"ohlcv:{meta['symbol']}:{range_}:{interval}"
        defaults = self.watchlist.get("defaults", {})
        ttl = float(defaults.get("ohlcv_cache_seconds", 300))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        # Long ranges: prefer local training archive
        hist = None
        if range_ in ("2y", "5y", "10y", "max", "20y") or interval == "1d":
            hist = self._load_history_csv(meta["symbol"])

        if hist is not None and not hist.empty:
            df = hist
            # trim to requested range
            days = _days_from_range(range_)
            if range_ not in ("max", "20y"):
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
                df = df[df["time"] >= cutoff].reset_index(drop=True)
            if range_ == "20y":
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365 * 20)
                df = df[df["time"] >= cutoff].reset_index(drop=True)
        else:
            asset_class = meta.get("class") or meta.get("section")
            if asset_class == "crypto" or meta.get("coingecko"):
                df = self._crypto_ohlcv(meta, range_)
            else:
                df = self._equity_ohlcv(meta, range_, interval)
            df = _normalize_ohlcv(df)

        cache.set(cache_key, df, ttl)
        return df

    def _equity_ohlcv(self, meta: dict[str, Any], range_: str, interval: str) -> pd.DataFrame:
        ticker = meta.get("yahoo") or meta["symbol"]
        period = _range_to_period(range_)
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval, auto_adjust=True)
            if df is None or df.empty:
                # fallback download
                df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            return df
        except Exception as e:
            log.warning("yfinance failed for %s: %s", ticker, e)
            return pd.DataFrame()

    def _crypto_ohlcv(self, meta: dict[str, Any], range_: str) -> pd.DataFrame:
        days = _days_from_range(range_)
        cg_id = meta.get("coingecko") or str(meta["symbol"]).lower()
        # CoinGecko
        try:
            r = requests.get(
                f"{COINGECKO}/coins/{cg_id}/market_chart",
                params={"vs_currency": "usd", "days": days},
                timeout=20,
            )
            if r.status_code == 200:
                data = r.json()
                prices = data.get("prices") or []
                volumes = data.get("total_volumes") or []
                rows = []
                vol_map = {int(v[0]): v[1] for v in volumes}
                for ts, price in prices:
                    ts_i = int(ts)
                    rows.append(
                        {
                            "time": datetime.fromtimestamp(ts_i / 1000, tz=timezone.utc),
                            "open": price,
                            "high": price,
                            "low": price,
                            "close": price,
                            "volume": vol_map.get(ts_i, 0.0),
                        }
                    )
                df = pd.DataFrame(rows)
                if len(df) > 1:
                    # reconstruct rough OHLC from consecutive closes
                    df["open"] = df["close"].shift(1).fillna(df["close"])
                    df["high"] = df[["open", "close"]].max(axis=1)
                    df["low"] = df[["open", "close"]].min(axis=1)
                return df
        except Exception as e:
            log.warning("CoinGecko failed for %s: %s", cg_id, e)

        # Binance public klines fallback
        pair = (meta.get("binance") or f"{meta['symbol']}/USDT").replace("/", "")
        try:
            limit = min(1000, max(30, days))
            r = requests.get(
                BINANCE_KLINES,
                params={"symbol": pair, "interval": "1d", "limit": limit},
                timeout=20,
            )
            r.raise_for_status()
            rows = []
            for k in r.json():
                rows.append(
                    {
                        "time": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    }
                )
            return pd.DataFrame(rows)
        except Exception as e:
            log.warning("Binance failed for %s: %s", pair, e)
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> dict[str, Any]:
        meta = self.resolve(symbol)
        cache_key = f"quote:{meta['symbol']}"
        defaults = self.watchlist.get("defaults", {})
        ttl = float(defaults.get("quote_cache_seconds", 60))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        df = self.get_ohlcv(symbol, range_="3mo", interval="1d")
        if df is None or df.empty:
            quote = {
                "symbol": meta["symbol"],
                "name": meta.get("name", meta["symbol"]),
                "class": meta.get("class") or meta.get("section"),
                "price": None,
                "change_pct": None,
                "volume": None,
                "high": None,
                "low": None,
                "as_of": None,
                "error": "no_data",
            }
            cache.set(cache_key, quote, ttl)
            return quote

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        price = float(last["close"])
        prev_close = float(prev["close"]) or price
        change_pct = (price - prev_close) / prev_close * 100.0 if prev_close else 0.0
        # 24h-ish range from last bar
        quote = {
            "symbol": meta["symbol"],
            "name": meta.get("name", meta["symbol"]),
            "class": meta.get("class") or meta.get("section"),
            "price": price,
            "change_pct": change_pct,
            "volume": float(last.get("volume") or 0.0),
            "high": float(last.get("high") or price),
            "low": float(last.get("low") or price),
            "as_of": pd.Timestamp(last["time"]).isoformat(),
            "error": None,
        }
        cache.set(cache_key, quote, ttl)
        return quote

    def ohlcv_records(self, symbol: str, range_: str = "1y", interval: str = "1d") -> list[dict[str, Any]]:
        df = self.get_ohlcv(symbol, range_=range_, interval=interval)
        if df is None or df.empty:
            return []
        records = []
        for _, row in df.iterrows():
            ts = pd.Timestamp(row["time"])
            records.append(
                {
                    "time": int(ts.timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"] or 0.0),
                }
            )
        return records


@lru_cache(maxsize=1)
def get_market_service() -> MarketService:
    return MarketService()
