"""Regression tests: backtests must not count new capital as returns."""

import numpy as np
import pandas as pd

from src.backtest import dca_strategy, ladder_strategy


def _flat_daily(n: int = 400, px: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"date": dates, "open": px, "high": px, "low": px, "close": px, "volume": 1.0}
    )


def _episodes(idxs: list[int], daily: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_date": daily["date"].iloc[idxs].values,
            "entry_price": daily["close"].iloc[idxs].values,
            "entry_idx": idxs,
        }
    )


def test_ladder_flat_price_has_no_gain():
    daily = _flat_daily()
    res = ladder_strategy(daily, _episodes([50, 150, 250], daily), [0.1, 0.2], 90, fee=0.0)
    eq = res["equity"]["equity"].values
    assert np.isclose(eq[0], 1.0)
    assert np.allclose(eq, 1.0)  # flat price, no fee -> equity never moves
    assert abs(res["cagr"]) < 1e-9


def test_dca_flat_price_has_zero_irr():
    daily = _flat_daily()
    res = dca_strategy(daily, pd.Series(True, index=daily.index), 1.0, fee=0.0)
    assert abs(res["cagr"]) < 1e-6
    growth = res["equity"]["equity"].dropna().values
    assert np.allclose(growth, 1.0)
