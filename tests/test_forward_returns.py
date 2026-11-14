"""Forward-return math on a toy price series."""

from __future__ import annotations

import pandas as pd
import pytest

from src.metrics import forward_return, mae_in_window, mfe_in_window


def _prices() -> pd.DataFrame:
    # Deterministic path: close = 100, 110, 90, 120, 100
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "open": [100, 105, 100, 95, 115],
            "high": [105, 115, 102, 125, 120],
            "low": [95, 100, 85, 90, 95],
            "close": [100, 110, 90, 120, 100],
            "fng": [50, 50, 50, 50, 50],
        }
    )


def test_forward_return_basic():
    daily = _prices()
    # Enter at idx 0 price 100; H=1 → close 110 → +10%
    assert forward_return(daily, 0, 100.0, 1) == pytest.approx(0.10)
    # H=2 → close 90 → -10%
    assert forward_return(daily, 0, 100.0, 2) == pytest.approx(-0.10)
    # H=4 → close 100 → 0%
    assert forward_return(daily, 0, 100.0, 4) == pytest.approx(0.0)
    # Past end → NaN
    import math

    assert math.isnan(forward_return(daily, 0, 100.0, 10))


def test_mae_mfe():
    daily = _prices()
    # From entry idx 0 price 100, window 3: lows at idx 1,2,3 = 100,85,90 → min 85 → MAE -15%
    mae, lower, ttb = mae_in_window(daily, 0, 100.0, 3)
    assert mae == pytest.approx(-0.15)
    assert lower is True
    assert ttb == 2  # lowest low at idx 2
    mfe = mfe_in_window(daily, 0, 100.0, 3)
    # highs 115,102,125 → 125 → +25%
    assert mfe == pytest.approx(0.25)
