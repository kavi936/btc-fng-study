"""Episode detection on a synthetic F&G series."""

from __future__ import annotations

import pandas as pd

from src.events import detect_episodes, detect_signal_days


def _synth() -> pd.DataFrame:
    # 40 days: above, dip below 12 for 3 days, recover, short dip (insufficient cooldown), deep dip
    fng = (
        [40] * 14  # days 0-13 above
        + [10, 8, 11]  # days 14-16 episode 1
        + [40] * 5  # days 17-21 above (only 5 days — not enough for cooldown=14)
        + [9, 9]  # days 22-23 — should NOT start episode with N=14
        + [40] * 14  # days 24-37 cooldown satisfied
        + [5, 6, 7, 20]  # days 38-40 — but we only have 40 days: need to extend
    )
    # Fix length carefully
    fng = (
        [40] * 14
        + [10, 8, 11]
        + [40] * 5
        + [9, 9]
        + [40] * 14
        + [5, 6, 7]
        + [25]
    )
    n = len(fng)
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "fng": fng,
        }
    )


def test_signal_days_count():
    daily = _synth()
    sig = detect_signal_days(daily, threshold=12)
    assert len(sig) == (daily["fng"] <= 12).sum()


def test_episodes_respect_cooldown():
    daily = _synth()
    ep14 = detect_episodes(daily, threshold=12, cooldown=14)
    # First episode at day 14 (after 14 days above from start... actually start has 14 days above
    # before first dip, so eligible). Second deep dip after 14 days above should qualify.
    # The short dip after only 5 days above should NOT.
    assert len(ep14) == 2
    assert ep14.iloc[0]["min_fng"] == 8
    assert ep14.iloc[1]["min_fng"] == 5
    assert ep14.iloc[0]["duration_days"] == 3
    assert ep14.iloc[1]["duration_days"] == 3


def test_episodes_shorter_cooldown_catches_middle_dip():
    daily = _synth()
    ep5 = detect_episodes(daily, threshold=12, cooldown=5)
    # With N=5, the middle dip after 5 days above should also count → 3 episodes
    assert len(ep5) == 3
