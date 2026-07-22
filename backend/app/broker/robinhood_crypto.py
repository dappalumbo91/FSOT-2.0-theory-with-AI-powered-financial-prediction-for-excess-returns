"""
Robinhood Crypto Trading API adapter.

Default mode: DRY RUN — builds signed request payloads and logs intents,
never submits live orders unless dual live flags are set (see config).

Auth sketch (official Crypto Trading API):
  - API key from https://robinhood.com/account/crypto
  - Ed25519 key pair; requests signed with private key
  - Headers: x-api-key, x-signature, x-timestamp

Docs: https://docs.robinhood.com/
Help: https://robinhood.com/us/en/support/articles/crypto-api/

This module does NOT enable real trading by default. Safe for monitoring builds.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.broker.config import BrokerSettings, get_broker_settings

log = logging.getLogger("fsot.broker.rh_crypto")

# Symbol map FSOT watchlist → common RH crypto product codes (adjust as needed)
_SYMBOL_MAP = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "DOGE": "DOGE-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
    "AVAX": "AVAX-USD",
    "LINK": "LINK-USD",
}


@dataclass
class OrderIntent:
    symbol: str
    side: str  # buy | sell
    notional_usd: float | None = None
    quantity: float | None = None
    order_type: str = "market"
    client_order_id: str = ""
    source_signal: str = ""  # BUY | SELL from BHS
    note: str = ""

    def normalized_side(self) -> str:
        s = self.side.lower().strip()
        if s in ("buy", "long", "b"):
            return "buy"
        if s in ("sell", "short", "s"):
            return "sell"
        raise ValueError(f"invalid side: {self.side}")


@dataclass
class OrderResult:
    ok: bool
    dry_run: bool
    status: str
    intent: dict[str, Any]
    message: str
    broker: str = "robinhood_crypto"
    request_preview: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] | None = None
    as_of: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RobinhoodCryptoClient:
    """
    Crypto broker client with forced dry-run safety.

    Usage now: preview_order / place_order → always dry unless settings.can_place_live.
    """

    def __init__(self, settings: BrokerSettings | None = None) -> None:
        self.settings = settings or get_broker_settings()
        self._private_key = None  # lazy load cryptography key

    # ── Key material ────────────────────────────────────────────────────

    def _load_private_key(self):
        if self._private_key is not None:
            return self._private_key
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError as e:
            raise RuntimeError(
                "cryptography package required for Robinhood signing: pip install cryptography"
            ) from e

        raw = None
        if self.settings.private_key_path:
            p = Path(self.settings.private_key_path)
            if p.exists():
                raw = p.read_bytes()
        if raw is None and self.settings.private_key_b64:
            raw = base64.b64decode(self.settings.private_key_b64)

        if raw is None:
            return None

        # PEM or raw 32-byte seed
        try:
            if b"BEGIN" in raw:
                self._private_key = serialization.load_pem_private_key(raw, password=None)
            else:
                seed = raw[:32] if len(raw) >= 32 else raw
                self._private_key = Ed25519PrivateKey.from_private_bytes(seed)
        except Exception as e:
            log.warning("failed to load private key: %s", e)
            return None
        return self._private_key

    def signing_ready(self) -> bool:
        return self._load_private_key() is not None and bool(self.settings.api_key)

    def _sign(self, message: str) -> str:
        key = self._load_private_key()
        if key is None:
            # Deterministic placeholder signature for dry-run without keys
            digest = hashlib.sha256(message.encode("utf-8")).digest()
            return base64.b64encode(digest).decode("ascii")
        sig = key.sign(message.encode("utf-8"))
        return base64.b64encode(sig).decode("ascii")

    def _auth_headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        ts = str(int(time.time()))
        # Message format aligned with RH-style: timestamp + method + path + body
        msg = f"{self.settings.api_key}{ts}{method.upper()}{path}{body}"
        return {
            "x-api-key": self.settings.api_key or "DRY_RUN_NO_KEY",
            "x-timestamp": ts,
            "x-signature": self._sign(msg),
            "Content-Type": "application/json",
        }

    # ── Status ──────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        s = self.settings
        return {
            "provider": s.provider,
            "dry_run": s.dry_run,
            "live_trading_enabled": s.live_trading_enabled,
            "can_place_live": s.can_place_live,
            "credentials_configured": s.configured,
            "signing_ready": self.signing_ready(),
            "api_key_present": bool(s.api_key),
            "max_order_usd": s.max_order_usd,
            "allowed_symbols": list(s.allowed_symbols),
            "base_url": s.base_url,
            "how_to_enable_later": {
                "1": "Create API key at https://robinhood.com/account/crypto (desktop)",
                "2": "Set FSOT_RH_API_KEY and FSOT_RH_PRIVATE_KEY_PATH (or B64)",
                "3": "Leave dry_run on (default) — use POST /api/broker/preview",
                "4": "Live only when ready: FSOT_BROKER_LIVE=1 FSOT_BROKER_I_UNDERSTAND=YES FSOT_BROKER_DRY_RUN=0",
            },
            "safety": "Real orders blocked unless dual live flags set. Synthetic $ remains primary.",
        }

    def map_symbol(self, symbol: str) -> str:
        sym = symbol.upper().replace("USD", "").replace("-", "")
        if sym in _SYMBOL_MAP:
            return _SYMBOL_MAP[sym]
        if symbol.upper() in _SYMBOL_MAP.values():
            return symbol.upper()
        return f"{sym}-USD"

    def _clamp_notional(self, usd: float) -> float:
        return float(min(max(usd, 1.0), self.settings.max_order_usd))

    # ── Orders ──────────────────────────────────────────────────────────

    def preview_order(self, intent: OrderIntent) -> OrderResult:
        """Build what would be sent — never hits network for submit."""
        return self.place_order(intent, force_dry_run=True)

    def place_order(
        self, intent: OrderIntent, *, force_dry_run: bool | None = None
    ) -> OrderResult:
        dry = self.settings.dry_run if force_dry_run is None else force_dry_run
        if force_dry_run is True:
            dry = True
        # Absolute safety: live only if settings allow
        if not dry and not self.settings.can_place_live:
            dry = True

        as_of = datetime.now(timezone.utc).isoformat()
        try:
            side = intent.normalized_side()
        except ValueError as e:
            return OrderResult(
                ok=False,
                dry_run=True,
                status="rejected",
                intent=asdict(intent),
                message=str(e),
                as_of=as_of,
            )

        sym = intent.symbol.upper()
        if sym not in self.settings.allowed_symbols and not sym.endswith("-USD"):
            # allow BTC-USD form
            base = sym.split("-")[0]
            if base not in self.settings.allowed_symbols:
                return OrderResult(
                    ok=False,
                    dry_run=True,
                    status="rejected",
                    intent=asdict(intent),
                    message=f"symbol {sym} not in allowed crypto list",
                    as_of=as_of,
                )

        product = self.map_symbol(sym)
        notional = intent.notional_usd
        if notional is None and intent.quantity is None:
            notional = min(25.0, self.settings.max_order_usd)
        if notional is not None:
            notional = self._clamp_notional(float(notional))

        path = "/api/v1/crypto/trading/orders/"  # path used for signing preview
        body_obj: dict[str, Any] = {
            "symbol": product,
            "side": side,
            "type": intent.order_type or "market",
            "client_order_id": intent.client_order_id
            or f"fsot-{int(time.time() * 1000)}",
        }
        if notional is not None:
            body_obj["notional"] = f"{notional:.2f}"
        if intent.quantity is not None:
            body_obj["quantity"] = str(intent.quantity)

        body = json.dumps(body_obj, separators=(",", ":"))
        headers = self._auth_headers("POST", path, body)
        preview = {
            "method": "POST",
            "url": f"{self.settings.base_url.rstrip('/')}{path}",
            "headers_redacted": {
                **{k: v for k, v in headers.items() if k != "x-signature"},
                "x-signature": headers["x-signature"][:12] + "…",
            },
            "body": body_obj,
        }

        if dry:
            log.info(
                "DRY-RUN order %s %s notional=%s signal=%s",
                side,
                product,
                notional,
                intent.source_signal,
            )
            return OrderResult(
                ok=True,
                dry_run=True,
                status="dry_run_accepted",
                intent={**asdict(intent), "product": product, "notional_usd": notional},
                message=(
                    "Dry-run only — order NOT submitted to Robinhood. "
                    "Synthetic monitoring continues; enable live only when ready."
                ),
                request_preview=preview,
                response={"simulated": True, "would_submit": body_obj},
                as_of=as_of,
            )

        # Live path — still double-checked
        if not self.settings.can_place_live:
            return OrderResult(
                ok=False,
                dry_run=True,
                status="blocked",
                intent=asdict(intent),
                message="Live trading blocked by safety flags",
                request_preview=preview,
                as_of=as_of,
            )

        url = preview["url"]
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(url, headers=headers, content=body)
            data = None
            try:
                data = r.json()
            except Exception:
                data = {"text": r.text[:500]}
            ok = 200 <= r.status_code < 300
            return OrderResult(
                ok=ok,
                dry_run=False,
                status="submitted" if ok else f"http_{r.status_code}",
                intent={**asdict(intent), "product": product},
                message="Live order response" if ok else "Live order failed",
                request_preview=preview,
                response=data if isinstance(data, dict) else {"raw": data},
                as_of=as_of,
            )
        except Exception as e:
            return OrderResult(
                ok=False,
                dry_run=False,
                status="error",
                intent=asdict(intent),
                message=str(e),
                request_preview=preview,
                as_of=as_of,
            )

    def signal_to_intent(
        self,
        symbol: str,
        action: str,
        *,
        notional_usd: float = 25.0,
    ) -> OrderIntent | None:
        """Map BHS BUY/HOLD/SELL → order intent (HOLD → None)."""
        a = (action or "HOLD").upper()
        if a in ("BUY", "LONG"):
            return OrderIntent(
                symbol=symbol,
                side="buy",
                notional_usd=notional_usd,
                source_signal="BUY",
                note="from FSOT BHS",
            )
        if a in ("SELL", "SHORT"):
            return OrderIntent(
                symbol=symbol,
                side="sell",
                notional_usd=notional_usd,
                source_signal="SELL",
                note="from FSOT BHS",
            )
        return None
