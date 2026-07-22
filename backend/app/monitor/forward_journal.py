"""
Forward prediction journal.

Purpose: test whether FSOT predicts *future* outcomes, not only historical
walk-forwards. At time T we record a signal + horizon; at T+H we resolve
against real prices that did not exist at prediction time.

Storage: local JSON ledger (default under D:\\training data or ./data).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.fsot.bhs_engine import HOLD_HORIZON, bhs_signature, decide_bhs, dual_scale_state
from app.fsot.pattern_memory import PatternMemory


def _default_path() -> Path:
    pref = Path(r"D:\training data\FSOT-Market-History\monitor")
    try:
        pref.mkdir(parents=True, exist_ok=True)
        return pref / "forward_journal.json"
    except Exception:
        local = Path(__file__).resolve().parents[2] / "data"
        local.mkdir(parents=True, exist_ok=True)
        return local / "forward_journal.json"


class ForwardJournal:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "version": 1,
            "method": "fsot_forward_prediction_journal",
            "free_parameters": 0,
            "entries": [],
        }

    def save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def record_from_market(
        self,
        df: pd.DataFrame,
        *,
        symbol: str,
        horizon: int = HOLD_HORIZON,
        sentiment: float = 0.0,
        window: int = 21,
        capital_note: float = 10_000.0,
        source: str = "live_monitor",
    ) -> dict[str, Any]:
        """
        Snapshot current FSOT BHS decision as a forward prediction.
        Outcome unknown until resolve() after horizon bars/days.
        """
        if df is None or len(df) < window + 60:
            return {"error": "insufficient_data"}

        close = df["close"].astype(float).values
        rets = pd.Series(close).pct_change().fillna(0.0).values
        vols = df["volume"].astype(float).values if "volume" in df.columns else None
        price = float(close[-1])
        t_last = str(df["time"].iloc[-1]) if "time" in df.columns else datetime.now(timezone.utc).isoformat()

        # Lightweight memory train on past (causal) for solid gate
        mem = PatternMemory(symbol=symbol, strict=True)
        H = int(horizon)
        start = 60
        end = len(df) - H - 1
        j = start
        while j < end:
            sub_r = rets[: j + 1]
            sub_v = vols[: j + 1] if vols is not None else None
            st = dual_scale_state(sub_r, sub_v, sentiment, window)
            key = bhs_signature(st, window)
            pred = int(st.get("pred_dir", 0))
            r_h = float(close[j + H] / close[j] - 1.0)
            real_d = 1 if r_h > 0 else (-1 if r_h < 0 else 0)
            sig_j = float(max(st.get("sig_pred", 1e-8), 1e-8))
            import math
            from app.fsot.intrinsic import SEED_POOF

            meaningful = abs(r_h) >= SEED_POOF * sig_j * math.sqrt(max(H, 1))
            if float(st.get("scale_agree", 0)) > 0 and pred != 0 and meaningful:
                mem.observe(key, pred, real_d, bar_index=j, used_observed_branch=True)
            j += max(1, H // 2)

        st_now = dual_scale_state(rets, vols, sentiment, window)
        key_now = bhs_signature(st_now, window)
        bias = mem.bias_for(key_now)
        recent = rets[-8:]
        action = decide_bhs(st_now, bias, recent_rets=recent)

        entry = {
            "id": str(uuid.uuid4()),
            "symbol": symbol.upper(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "bar_time": t_last,
            "price_at_prediction": price,
            "horizon_days": H,
            "action": action,
            "pred_dir": int(st_now.get("pred_dir", 0)),
            "mu": float(st_now.get("mu", 0.0)),
            "scale_agree": float(st_now.get("scale_agree", 0.0)),
            "pattern_key": key_now,
            "pattern_solid": bool(bias.get("solidified", 0) > 0),
            "pattern_strength": float(bias.get("strength", 0.0)),
            "pattern_accuracy": float(bias.get("accuracy", 0.5)),
            "source": source,
            "capital_context_usd": capital_note,
            "resolved": False,
            "resolved_at": None,
            "price_at_resolve": None,
            "realized_return": None,
            "hit": None,
            "note": "Forward prediction — score after horizon with real future prices",
        }
        self._data["entries"].append(entry)
        self.save()
        return {"error": None, "entry": entry}

    def resolve_pending(
        self,
        get_ohlcv_fn,
        *,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """
        Resolve entries whose horizon has elapsed using fresh OHLCV.
        get_ohlcv_fn(symbol) -> DataFrame with time, close
        """
        resolved = []
        still_open = 0
        now = datetime.now(timezone.utc)

        for e in self._data["entries"]:
            if e.get("resolved"):
                continue
            if symbol and e.get("symbol", "").upper() != symbol.upper():
                still_open += 1
                continue

            sym = e["symbol"]
            try:
                df = get_ohlcv_fn(sym)
            except Exception:
                still_open += 1
                continue
            if df is None or df.empty:
                still_open += 1
                continue

            close = df["close"].astype(float).values
            times = pd.to_datetime(df["time"], utc=True) if "time" in df.columns else None
            price0 = float(e["price_at_prediction"])
            H = int(e.get("horizon_days", HOLD_HORIZON))

            # Find bar at/after prediction bar_time, then +H trading bars
            pred_t = e.get("bar_time")
            idx0 = len(close) - 1
            if times is not None and pred_t:
                try:
                    pt = pd.Timestamp(pred_t)
                    if pt.tzinfo is None:
                        pt = pt.tz_localize("UTC")
                    # last bar at or before pred time
                    for i in range(len(times) - 1, -1, -1):
                        if times.iloc[i] <= pt:
                            idx0 = i
                            break
                except Exception:
                    pass

            idx1 = idx0 + H
            if idx1 >= len(close):
                still_open += 1
                continue

            price1 = float(close[idx1])
            r = price1 / price0 - 1.0 if price0 > 0 else 0.0
            action = str(e.get("action", "HOLD")).upper()
            if action == "HOLD" or e.get("pred_dir", 0) == 0:
                hit = None  # no directional commit
            elif action == "BUY":
                hit = 1.0 if r > 0 else 0.0
            elif action == "SELL":
                hit = 1.0 if r < 0 else 0.0
            else:
                hit = None

            e["resolved"] = True
            e["resolved_at"] = now.isoformat()
            e["price_at_resolve"] = price1
            e["realized_return"] = r
            e["hit"] = hit
            resolved.append(e)

        self.save()
        return {
            "error": None,
            "n_resolved": len(resolved),
            "n_still_open": still_open,
            "resolved": resolved[-20:],
        }

    def summary(self) -> dict[str, Any]:
        entries = self._data.get("entries") or []
        open_e = [e for e in entries if not e.get("resolved")]
        done = [e for e in entries if e.get("resolved")]
        commits = [e for e in done if e.get("hit") is not None]
        hits = [float(e["hit"]) for e in commits]
        by_sym: dict[str, list[float]] = {}
        for e in commits:
            by_sym.setdefault(e["symbol"], []).append(float(e["hit"]))

        return {
            "path": str(self.path),
            "n_total": len(entries),
            "n_open": len(open_e),
            "n_resolved": len(done),
            "n_directional_commits": len(commits),
            "forward_commit_accuracy": float(np.mean(hits)) if hits else None,
            "by_symbol": {
                s: {"n": len(v), "accuracy": float(np.mean(v))} for s, v in by_sym.items()
            },
            "recent_open": open_e[-10:],
            "recent_resolved": done[-10:],
            "note": (
                "Forward journal scores predictions only after real future bars arrive. "
                "This is the live test of predictive power beyond historical backtests."
            ),
            "free_parameters": 0,
        }
