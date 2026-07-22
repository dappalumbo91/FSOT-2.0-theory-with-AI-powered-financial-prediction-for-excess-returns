"""
Broker configuration.

Safety model:
  - dry_run=True always unless BOTH env flags set intentionally
  - live_trading_enabled requires FSOT_BROKER_LIVE=1 AND FSOT_BROKER_I_UNDERSTAND=YES
  - Keys only from environment / local files never committed to git
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class BrokerSettings:
    provider: str = "robinhood_crypto"
    # Default SAFE: never touch real money
    dry_run: bool = True
    live_trading_enabled: bool = False
    api_key: str = ""
    # Path to Ed25519 private key PEM or base64 seed (user-managed)
    private_key_path: str = ""
    private_key_b64: str = ""
    base_url: str = "https://trading.robinhood.com"
    # Max notional per dry-run / future live order (USD) — seed-friendly default
    max_order_usd: float = 100.0
    # Allowed symbols for crypto adapter (subset of watchlist)
    allowed_symbols: tuple[str, ...] = ("BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "AVAX", "LINK")

    @property
    def configured(self) -> bool:
        return bool(self.api_key) and (
            bool(self.private_key_path) or bool(self.private_key_b64)
        )

    @property
    def can_place_live(self) -> bool:
        return (
            self.live_trading_enabled
            and not self.dry_run
            and self.configured
        )


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


@lru_cache(maxsize=1)
def get_broker_settings() -> BrokerSettings:
    live_flag = _truthy(os.environ.get("FSOT_BROKER_LIVE"))
    understand = os.environ.get("FSOT_BROKER_I_UNDERSTAND", "").strip() == "YES"
    dry_want_off = os.environ.get("FSOT_BROKER_DRY_RUN", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    )
    # Live only if ALL three intentional; otherwise force dry_run
    live_ok = live_flag and understand and dry_want_off
    dry_run = not live_ok

    max_usd = float(os.environ.get("FSOT_BROKER_MAX_ORDER_USD", "100") or 100)
    key_path = os.environ.get("FSOT_RH_PRIVATE_KEY_PATH", "").strip()
    # default local path outside repo if exists
    if not key_path:
        cand = Path.home() / ".fsot" / "robinhood_crypto_private.pem"
        if cand.exists():
            key_path = str(cand)

    return BrokerSettings(
        dry_run=dry_run,
        live_trading_enabled=live_ok and not dry_run,
        api_key=os.environ.get("FSOT_RH_API_KEY", "").strip(),
        private_key_path=key_path,
        private_key_b64=os.environ.get("FSOT_RH_PRIVATE_KEY_B64", "").strip(),
        base_url=os.environ.get(
            "FSOT_RH_BASE_URL", "https://trading.robinhood.com"
        ).strip(),
        max_order_usd=max(1.0, max_usd),
    )


def reset_broker_settings_cache() -> None:
    get_broker_settings.cache_clear()
