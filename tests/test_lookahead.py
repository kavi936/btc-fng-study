"""Look-ahead bias checks for entry timing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.events import attach_entry_prices, detect_episodes


def _toy_daily() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    # F&G: high, then drops to extreme fear for a few days, then recovers
    fng = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 10, 8, 12, 20, 30, 40, 40, 40, 40, 40]
    close = [100 + i for i in range(20)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": [c - 0.5 for c in close],
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": 1.0,
            "fng": fng,
            "fng_class": ["x"] * 20,
            "price_source": "test",
            "sma_200": float("nan"),
            "above_sma_200": False,
        }
    )


def test_next_open_entry_after_signal():
    daily = _toy_daily()
    ep = detect_episodes(daily, threshold=12, cooldown=7)
    assert len(ep) >= 1
    attached = attach_entry_prices(ep, daily, timing="next_open")
    assert len(attached) >= 1
    for _, row in attached.iterrows():
        assert row["entry_idx"] > row["signal_idx"], "Base case must enter after signal day"
        # Entry price should be next day's open
        assert row["entry_price"] == pytest.approx(
            float(daily.loc[row["entry_idx"], "open"])
        )


def test_signal_close_entry_same_day():
    daily = _toy_daily()
    ep = detect_episodes(daily, threshold=12, cooldown=7)
    attached = attach_entry_prices(ep, daily, timing="signal_close")
    for _, row in attached.iterrows():
        assert row["entry_idx"] == row["signal_idx"]
        assert row["entry_price"] == pytest.approx(
            float(daily.loc[row["entry_idx"], "close"])
        )
