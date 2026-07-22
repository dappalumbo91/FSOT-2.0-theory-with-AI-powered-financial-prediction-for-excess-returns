"""Broker adapter must stay dry-run by default."""

from __future__ import annotations

import os

from app.broker.config import get_broker_settings, reset_broker_settings_cache
from app.broker.robinhood_crypto import OrderIntent, RobinhoodCryptoClient


def test_default_is_dry_run():
    reset_broker_settings_cache()
    # Clear live flags
    os.environ.pop("FSOT_BROKER_LIVE", None)
    os.environ.pop("FSOT_BROKER_I_UNDERSTAND", None)
    os.environ["FSOT_BROKER_DRY_RUN"] = "1"
    reset_broker_settings_cache()
    s = get_broker_settings()
    assert s.dry_run is True
    assert s.can_place_live is False


def test_preview_never_live():
    reset_broker_settings_cache()
    client = RobinhoodCryptoClient()
    r = client.preview_order(
        OrderIntent(symbol="BTC", side="buy", notional_usd=25.0, source_signal="BUY")
    )
    assert r.ok is True
    assert r.dry_run is True
    assert r.status == "dry_run_accepted"
    assert r.response is not None
    assert r.response.get("simulated") is True


def test_signal_hold_no_intent():
    client = RobinhoodCryptoClient()
    assert client.signal_to_intent("BTC", "HOLD") is None
    assert client.signal_to_intent("BTC", "BUY") is not None
