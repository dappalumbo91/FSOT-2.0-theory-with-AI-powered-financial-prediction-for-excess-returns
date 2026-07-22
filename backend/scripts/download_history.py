#!/usr/bin/env python3
"""
Download ~20 years of OHLCV for FSOT Market Monitor watchlist.

Target: D:\\training data\\FSOT-Market-History\\ohlcv
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
import yaml

ROOT = Path(__file__).resolve().parents[2]
HISTORY_ROOT = Path(r"D:\training data\FSOT-Market-History")
OHLCV_DIR = HISTORY_ROOT / "ohlcv"
MANIFEST_DIR = HISTORY_ROOT / "manifests"
WATCHLIST = ROOT / "config" / "watchlist.yaml"

START = "2005-01-01"  # ~20y+ where available
END = datetime.now(timezone.utc).strftime("%Y-%m-%d")

COINGECKO = "https://api.coingecko.com/api/v3"
BINANCE = "https://api.binance.com/api/v3/klines"


def load_assets() -> list[dict]:
    wl = yaml.safe_load(WATCHLIST.read_text(encoding="utf-8"))
    assets = []
    for section in ("indices", "stocks", "crypto"):
        for item in wl.get(section, []) or []:
            assets.append({**item, "section": section})
    # Extra long-history anchors for emergence/entropy study
    extras = [
        {"symbol": "VIX", "name": "VIX", "class": "index", "yahoo": "^VIX", "section": "indices"},
        {"symbol": "GLD", "name": "Gold ETF", "class": "equity", "yahoo": "GLD", "section": "stocks"},
        {"symbol": "TLT", "name": "20Y Treasury ETF", "class": "equity", "yahoo": "TLT", "section": "stocks"},
        {"symbol": "HYG", "name": "High Yield Corp Bond", "class": "equity", "yahoo": "HYG", "section": "stocks"},
        {"symbol": "QQQ", "name": "Nasdaq 100 ETF", "class": "equity", "yahoo": "QQQ", "section": "stocks"},
        {"symbol": "IWM", "name": "Russell 2000 ETF", "class": "equity", "yahoo": "IWM", "section": "stocks"},
        {"symbol": "XLF", "name": "Financials ETF", "class": "equity", "yahoo": "XLF", "section": "stocks"},
        {"symbol": "XLE", "name": "Energy ETF", "class": "equity", "yahoo": "XLE", "section": "stocks"},
    ]
    seen = {a["symbol"].upper() for a in assets}
    for e in extras:
        if e["symbol"].upper() not in seen:
            assets.append(e)
    return assets


def save_ohlcv(symbol: str, df: pd.DataFrame) -> Path | None:
    if df is None or df.empty:
        return None
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "time" in out.columns:
            out = out.set_index("time")
        elif "Date" in out.columns:
            out = out.set_index("Date")
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    cols = {}
    for c in out.columns:
        cl = str(c).lower().replace(" ", "_")
        cols[c] = cl
    out = out.rename(columns=cols)
    # standardize
    rename_map = {
        "adj_close": "adj_close",
        "adjclose": "adj_close",
    }
    out = out.rename(columns=rename_map)
    keep = [c for c in ["open", "high", "low", "close", "volume", "adj_close"] if c in out.columns]
    if "close" not in keep and "adj_close" in out.columns:
        out["close"] = out["adj_close"]
        keep = [c for c in ["open", "high", "low", "close", "volume", "adj_close"] if c in out.columns]
    out = out[keep].dropna(subset=["close"])
    path = OHLCV_DIR / f"{symbol.replace('^', '')}.csv"
    out.to_csv(path, date_format="%Y-%m-%d")
    return path


def download_equity(meta: dict) -> tuple[pd.DataFrame | None, str]:
    ticker = meta.get("yahoo") or meta["symbol"]
    try:
        df = yf.download(
            ticker,
            start=START,
            end=END,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            t = yf.Ticker(ticker)
            df = t.history(start=START, end=END, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df, "yfinance"
    except Exception as e:
        return None, f"yfinance_error:{e}"


def download_crypto_coingecko(meta: dict) -> tuple[pd.DataFrame | None, str]:
    cg = meta.get("coingecko") or meta["symbol"].lower()
    # max free daily history ~ max days; use market_chart/range if available
    try:
        # days=max for free API often capped; try range timestamps
        start_ts = int(pd.Timestamp(START, tz="UTC").timestamp())
        end_ts = int(pd.Timestamp(END, tz="UTC").timestamp())
        url = f"{COINGECKO}/coins/{cg}/market_chart/range"
        r = requests.get(
            url,
            params={"vs_currency": "usd", "from": start_ts, "to": end_ts},
            timeout=60,
        )
        if r.status_code != 200:
            # fallback days=max
            r = requests.get(
                f"{COINGECKO}/coins/{cg}/market_chart",
                params={"vs_currency": "usd", "days": "max"},
                timeout=60,
            )
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices") or []
        volumes = data.get("total_volumes") or []
        vol_map = {int(v[0]): v[1] for v in volumes}
        rows = []
        for ts, price in prices:
            ts_i = int(ts)
            rows.append(
                {
                    "Date": pd.to_datetime(ts_i, unit="ms", utc=True),
                    "Open": price,
                    "High": price,
                    "Low": price,
                    "Close": price,
                    "Volume": vol_map.get(ts_i, 0.0),
                }
            )
        if not rows:
            return None, "coingecko_empty"
        df = pd.DataFrame(rows).set_index("Date")
        # daily resample for long history
        ohlc = df["Close"].resample("1D").ohlc()
        ohlc.columns = ["Open", "High", "Low", "Close"]
        vol = df["Volume"].resample("1D").sum()
        ohlc["Volume"] = vol
        ohlc = ohlc.dropna(subset=["Close"])
        return ohlc, "coingecko"
    except Exception as e:
        return None, f"coingecko_error:{e}"


def download_crypto_binance(meta: dict) -> tuple[pd.DataFrame | None, str]:
    pair = (meta.get("binance") or f"{meta['symbol']}/USDT").replace("/", "")
    # Binance only goes back so far; paginate
    all_rows = []
    start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(END, tz="UTC").timestamp() * 1000)
    cursor = start_ms
    try:
        while cursor < end_ms:
            r = requests.get(
                BINANCE,
                params={
                    "symbol": pair,
                    "interval": "1d",
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
                timeout=30,
            )
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            all_rows.extend(batch)
            last = batch[-1][0]
            nxt = last + 86_400_000
            if nxt <= cursor:
                break
            cursor = nxt
            time.sleep(0.15)
        if not all_rows:
            return None, "binance_empty"
        rows = []
        for k in all_rows:
            rows.append(
                {
                    "Date": pd.to_datetime(k[0], unit="ms", utc=True),
                    "Open": float(k[1]),
                    "High": float(k[2]),
                    "Low": float(k[3]),
                    "Close": float(k[4]),
                    "Volume": float(k[5]),
                }
            )
        df = pd.DataFrame(rows).drop_duplicates("Date").set_index("Date").sort_index()
        return df, "binance"
    except Exception as e:
        return None, f"binance_error:{e}"


def main() -> int:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    assets = load_assets()
    results = []
    print(f"Downloading {len(assets)} assets → {OHLCV_DIR}")
    print(f"Range: {START} → {END}")

    for i, meta in enumerate(assets, 1):
        sym = str(meta["symbol"]).upper()
        safe = sym.replace("^", "")
        print(f"[{i}/{len(assets)}] {sym} ...", end=" ", flush=True)
        df = None
        source = ""
        section = meta.get("section") or meta.get("class")
        if section == "crypto" or meta.get("coingecko"):
            # Yahoo crypto USD pairs have long free history (preferred)
            yahoo_crypto = {
                "BTC": "BTC-USD",
                "ETH": "ETH-USD",
                "SOL": "SOL-USD",
                "BNB": "BNB-USD",
                "XRP": "XRP-USD",
                "ADA": "ADA-USD",
                "DOGE": "DOGE-USD",
                "AVAX": "AVAX-USD",
                "LINK": "LINK-USD",
                "LTC": "LTC-USD",
            }
            ysym = yahoo_crypto.get(sym, f"{sym}-USD")
            df, source = download_equity({**meta, "yahoo": ysym})
            if df is None or df.empty:
                df, source = download_crypto_binance(meta)
            if df is None or df.empty:
                time.sleep(1.2)
                df, source = download_crypto_coingecko(meta)
        else:
            df, source = download_equity(meta)

        path = save_ohlcv(safe, df) if df is not None else None
        n = 0 if df is None or df.empty else len(df)
        first = last = None
        if path and n:
            idx = pd.read_csv(path, index_col=0, parse_dates=True).index
            first = str(idx.min().date())
            last = str(idx.max().date())
            print(f"OK {n} bars {first}→{last} via {source}")
        else:
            print(f"FAIL {source}")
        results.append(
            {
                "symbol": sym,
                "safe": safe,
                "name": meta.get("name"),
                "section": section,
                "source": source,
                "bars": n,
                "first": first,
                "last": last,
                "path": str(path) if path else None,
                "ok": bool(path and n),
            }
        )
        time.sleep(0.25)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start_requested": START,
        "end_requested": END,
        "history_root": str(HISTORY_ROOT),
        "assets": results,
        "ok_count": sum(1 for r in results if r["ok"]),
        "fail_count": sum(1 for r in results if not r["ok"]),
    }
    man_path = MANIFEST_DIR / "download_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone: {manifest['ok_count']} ok, {manifest['fail_count']} fail")
    print(f"Manifest: {man_path}")
    return 0 if manifest["ok_count"] else 1


if __name__ == "__main__":
    sys.exit(main())
