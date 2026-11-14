"""Strategy backtests: buy & hold, lump-sum on signal, DCA during fear, ladder."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("btc_fng_study")


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))


def _cagr(equity: pd.Series, dates: pd.Series) -> float:
    if len(equity) < 2:
        return float("nan")
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start <= 0 or end <= 0:
        return float("nan")
    years = (pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[0])).days / 365.25
    if years <= 0:
        return float("nan")
    return float((end / start) ** (1 / years) - 1.0)


def _money_weighted_irr(flows: np.ndarray, dates: pd.Series, final_value: float) -> float:
    """Annualised money-weighted return (IRR) for daily contributions.

    flows[i] = cash invested on day i (>= 0). final_value is the portfolio value on the last day.
    Solves sum(flow_i * (1+r)^(t_end - t_i)) = final_value by bisection.
    """
    idx = np.nonzero(flows > 0)[0]
    if len(idx) == 0 or final_value <= 0:
        return float("nan")
    t_end = pd.Timestamp(dates.iloc[-1])
    yrs = np.array([(t_end - pd.Timestamp(dates.iloc[i])).days / 365.25 for i in idx])
    cf = flows[idx]

    def fv(r: float) -> float:
        return float(np.sum(cf * (1.0 + r) ** yrs)) - final_value

    lo, hi = -0.99, 10.0
    if fv(lo) > 0 or fv(hi) < 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if fv(mid) > 0:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2)


def buy_and_hold(daily: pd.DataFrame, fee: float = 0.001) -> dict[str, Any]:
    """Buy at first close (after fee), hold to last close."""
    day = daily.reset_index(drop=True)
    entry = float(day.loc[0, "close"]) * (1 + fee)
    units = 1.0 / entry
    equity = day["close"] * units
    # Sell fee at end for consistency
    final = float(equity.iloc[-1]) * (1 - fee)
    eq = equity.copy()
    eq.iloc[-1] = final
    return {
        "name": "buy_and_hold",
        "cagr": _cagr(eq, day["date"]),
        "max_drawdown": _max_drawdown(eq.values),
        "pct_time_invested": 1.0,
        "n_trades": 1,
        "final_equity": final,
        "equity": pd.DataFrame({"date": day["date"], "equity": eq, "strategy": "buy_and_hold"}),
    }


def lump_sum_on_signal(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    hold_days: int,
    fee: float = 0.001,
    initial_cash: float = 1.0,
) -> dict[str, Any]:
    """Deploy full cash on each episode entry when flat; hold H days; cash otherwise.

    Non-overlapping: if a new signal arrives while invested, it is skipped.
    """
    day = daily.reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(day["date"])}
    cash = initial_cash
    units = 0.0
    invested_until = -1
    n_trades = 0
    equity_path = []

    # Sort unique entry dates
    entries = (
        episodes[["entry_date", "entry_price", "entry_idx"]]
        .drop_duplicates(subset=["entry_date"])
        .sort_values("entry_date")
    )
    entry_map = {pd.Timestamp(r.entry_date): r for r in entries.itertuples()}

    for i, row in day.iterrows():
        dt = pd.Timestamp(row["date"])
        px = float(row["close"])

        # Exit if hold period ended
        if units > 0 and i >= invested_until:
            cash = units * px * (1 - fee)
            units = 0.0
            invested_until = -1

        # Enter if signal today and flat
        if units == 0 and dt in entry_map:
            ep = entry_map[dt]
            entry_px = float(ep.entry_price) * (1 + fee)
            units = cash / entry_px
            cash = 0.0
            invested_until = int(ep.entry_idx) + hold_days
            n_trades += 1

        mark = cash + units * px
        equity_path.append(mark)

    eq = pd.Series(equity_path, dtype=float)
    invested_flags = []
    # Recompute invested flags roughly from equity structure
    cash_r = initial_cash
    units_r = 0.0
    inv_until = -1
    for i, row in day.iterrows():
        dt = pd.Timestamp(row["date"])
        if units_r > 0 and i >= inv_until:
            cash_r = units_r * float(row["close"]) * (1 - fee)
            units_r = 0.0
            inv_until = -1
        if units_r == 0 and dt in entry_map:
            ep = entry_map[dt]
            units_r = cash_r / (float(ep.entry_price) * (1 + fee))
            cash_r = 0.0
            inv_until = int(ep.entry_idx) + hold_days
        invested_flags.append(units_r > 0)

    pct_invested = float(np.mean(invested_flags)) if invested_flags else 0.0
    name = f"lump_sum_H{hold_days}"
    return {
        "name": name,
        "cagr": _cagr(eq, day["date"]),
        "max_drawdown": _max_drawdown(eq.values),
        "pct_time_invested": pct_invested,
        "n_trades": n_trades,
        "final_equity": float(eq.iloc[-1]),
        "equity": pd.DataFrame({"date": day["date"], "equity": eq, "strategy": name}),
    }


def dca_strategy(
    daily: pd.DataFrame,
    buy_mask: pd.Series,
    daily_amount: float = 1.0,
    fee: float = 0.001,
    name: str = "dca",
) -> dict[str, Any]:
    """Buy `daily_amount` (USD) on days where buy_mask is True; hold forever."""
    day = daily.reset_index(drop=True)
    mask = buy_mask.reset_index(drop=True).astype(bool)
    units = 0.0
    spent = 0.0
    n_trades = 0
    equity = []
    spent_path = []
    flows = np.zeros(len(day))
    for i, row in day.iterrows():
        if bool(mask.iloc[i]):
            px = float(row["close"]) * (1 + fee)
            units += daily_amount / px
            spent += daily_amount
            flows[i] = daily_amount
            n_trades += 1
        equity.append(units * float(row["close"]))
        spent_path.append(spent)
    eq = pd.Series(equity, dtype=float)
    spent_s = pd.Series(spent_path, dtype=float)
    # Value per $1 invested to date (comparable to the other strategies' growth-of-$1 curves).
    # Contributions are NOT counted as returns.
    growth = (eq / spent_s.where(spent_s > 0)).astype(float)
    avg_cost = spent / units if units > 0 else float("nan")
    final_px = float(day["close"].iloc[-1])
    growth_valid = growth.dropna()
    dd_arr = growth_valid.values if len(growth_valid) else np.array([1.0])
    # Money-weighted annual return (IRR). A plain CAGR on the equity curve would count
    # new contributions as gains, so it is not used for DCA.
    cagr = _money_weighted_irr(flows, day["date"], float(eq.iloc[-1]))
    return {
        "name": name,
        "cagr": cagr,
        "max_drawdown": _max_drawdown(dd_arr),
        "pct_time_invested": float(mask.mean()),
        "n_trades": n_trades,
        "final_equity": float(eq.iloc[-1]),
        "total_invested": spent,
        "avg_cost_basis": avg_cost,
        "final_price": final_px,
        "units": units,
        "return_method": "money_weighted_irr",
        "equity": pd.DataFrame({"date": day["date"], "equity": growth, "strategy": name}),
    }


def ladder_strategy(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    drop_levels: list[float],
    window_days: int,
    fee: float = 0.001,
    capital: float = 1.0,
) -> dict[str, Any]:
    """Split capital into (1 + len(drops)) tranches: entry, then -10%, -20% within window.

    Each episode is treated independently with equal capital allocation across episodes
    (capital / n_episodes), then tranches within the episode.

    All capital is held as cash from day one and only leaves cash when a tranche fills, so the
    equity curve starts at `capital` and later allocations are never counted as gains.
    """
    day = daily.reset_index(drop=True)
    eps = (
        episodes[["entry_date", "entry_price", "entry_idx"]]
        .drop_duplicates(subset=["entry_date"])
        .sort_values("entry_date")
        .reset_index(drop=True)
    )
    if eps.empty:
        return {
            "name": "ladder",
            "cagr": float("nan"),
            "max_drawdown": float("nan"),
            "pct_time_invested": 0.0,
            "n_trades": 0,
            "final_equity": capital,
            "equity": pd.DataFrame(
                {"date": day["date"], "equity": capital, "strategy": "ladder"}
            ),
        }

    n_tranches = 1 + len(drop_levels)
    per_episode = capital / len(eps)
    tranche = per_episode / n_tranches
    n_trades = 0
    # Track per-day mark-to-market of all units bought so far + residual cash
    # Pre-compute fills
    fills: list[tuple[int, float]] = []  # (idx, units)

    for _, ep in eps.iterrows():
        eidx = int(ep["entry_idx"])
        eprice = float(ep["entry_price"])
        # Tranche 1 at entry
        fill_px = eprice * (1 + fee)
        fills.append((eidx, tranche / fill_px))
        n_trades += 1

        for lvl in drop_levels:
            target = eprice * (1 - lvl)
            hit_idx = None
            end = min(eidx + window_days, len(day) - 1)
            for j in range(eidx + 1, end + 1):
                if float(day.loc[j, "low"]) <= target:
                    hit_idx = j
                    break
            if hit_idx is not None:
                # Fill at target (conservative) with fee
                fill_px = target * (1 + fee)
                fills.append((hit_idx, tranche / fill_px))
                n_trades += 1

    # Build equity curve: cash starts at full capital and is drawn down as tranches fill
    units_by_day = np.zeros(len(day))
    cash_by_day = np.full(len(day), float(capital))
    for idx, u in fills:
        units_by_day[idx:] += u
        cash_by_day[idx:] -= tranche
    equity = units_by_day * day["close"].values + cash_by_day
    # Invested when units > 0
    pct_invested = float(np.mean(units_by_day > 0))
    eq = pd.Series(equity, dtype=float)
    return {
        "name": "ladder",
        "cagr": _cagr(eq, day["date"]),
        "max_drawdown": _max_drawdown(eq.values),
        "pct_time_invested": pct_invested,
        "n_trades": n_trades,
        "final_equity": float(eq.iloc[-1]),
        "return_method": "cagr",
        "equity": pd.DataFrame({"date": day["date"], "equity": eq, "strategy": "ladder"}),
    }


def ladder_vs_lump_by_episode(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    drop_levels: list[float],
    window_days: int,
    horizons: list[int],
    fee: float = 0.001,
) -> pd.DataFrame:
    """Compare, per episode, $1 all-in at entry vs $1 split into tranches (entry, -10%, -20%).

    Unfilled tranches stay as cash (0% return). Value is marked at entry_idx + H.
    Returns one summary row per horizon.
    """
    day = daily.reset_index(drop=True)
    eps = episodes.drop_duplicates(subset=["entry_date"]).sort_values("entry_date")
    n_tr = 1 + len(drop_levels)
    rows = []
    for h in horizons:
        lump_r, lad_r = [], []
        for _, ep in eps.iterrows():
            i = int(ep["entry_idx"])
            p = float(ep["entry_price"])
            if i + h >= len(day):
                continue
            exit_px = float(day.loc[i + h, "close"])
            lump_r.append(exit_px / (p * (1 + fee)) - 1.0)
            units = (1.0 / n_tr) / (p * (1 + fee))
            cash = 0.0
            end = min(i + window_days, i + h, len(day) - 1)
            lows = day["low"].values[i + 1 : end + 1]
            for lvl in drop_levels:
                tgt = p * (1 - lvl)
                if len(lows) and (lows <= tgt).any():
                    units += (1.0 / n_tr) / (tgt * (1 + fee))
                else:
                    cash += 1.0 / n_tr
            lad_r.append(units * exit_px + cash - 1.0)
        if not lump_r:
            continue
        lr, dr = np.array(lump_r), np.array(lad_r)
        rows.append(
            {
                "horizon": h,
                "n": len(lr),
                "lump_median": float(np.median(lr)),
                "ladder_median": float(np.median(dr)),
                "lump_mean": float(np.mean(lr)),
                "ladder_mean": float(np.mean(dr)),
                "lump_win_rate": float(np.mean(lr > 0)),
                "ladder_win_rate": float(np.mean(dr > 0)),
                "ladder_beats_lump": int(np.sum(dr > lr)),
                "pct_ladder_beats_lump": float(np.mean(dr > lr)),
                "sample_flag": "low sample" if len(lr) < 10 else "ok",
            }
        )
    return pd.DataFrame(rows)


def run_backtests(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    cfg: dict[str, Any],
    threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all strategy comparisons for one threshold (episodes with default cooldown + base timing)."""
    fee = cfg["fee_per_trade"]
    hold_hs = cfg["backtest_hold_horizons"]
    results = []
    equity_frames = []

    bh = buy_and_hold(daily, fee)
    results.append({k: v for k, v in bh.items() if k != "equity"})
    equity_frames.append(bh["equity"])

    for h in hold_hs:
        r = lump_sum_on_signal(daily, episodes, h, fee)
        results.append({k: v for k, v in r.items() if k != "equity"})
        equity_frames.append(r["equity"])

    # DCA during fear vs always DCA
    fear_mask = daily["fng"] <= threshold
    dca_fear = dca_strategy(
        daily, fear_mask, cfg["dca_daily_amount"], fee, name=f"dca_fear_le_{threshold}"
    )
    dca_all = dca_strategy(
        daily, pd.Series(True, index=daily.index), cfg["dca_daily_amount"], fee, name="dca_always"
    )
    for r in (dca_fear, dca_all):
        row = {k: v for k, v in r.items() if k != "equity"}
        results.append(row)
        equity_frames.append(r["equity"])

    ladder = ladder_strategy(
        daily,
        episodes,
        cfg["ladder_drop_levels"],
        cfg["ladder_window_days"],
        fee,
    )
    results.append({k: v for k, v in ladder.items() if k != "equity"})
    equity_frames.append(ladder["equity"])

    summary = pd.DataFrame(results)
    summary["threshold"] = threshold
    # Cost basis comparison note
    summary["cost_basis_note"] = ""
    if "avg_cost_basis" in dca_fear:
        summary.loc[summary["name"] == dca_fear["name"], "cost_basis_note"] = (
            f"avg_cost={dca_fear['avg_cost_basis']:.2f} vs always={dca_all['avg_cost_basis']:.2f}"
        )
        summary.loc[summary["name"] == dca_all["name"], "cost_basis_note"] = (
            f"avg_cost={dca_all['avg_cost_basis']:.2f}"
        )

    equity = pd.concat(equity_frames, ignore_index=True)
    equity["threshold"] = threshold
    return summary, equity
