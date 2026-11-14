"""Signal-day and episode detection for Fear & Greed thresholds."""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.logging_utils import log_count

logger = logging.getLogger("btc_fng_study")

EntryTiming = Literal["next_open", "signal_close"]


def detect_signal_days(daily: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Every day with F&G <= threshold (Signal A)."""
    mask = daily["fng"] <= threshold
    cols = ["date", "fng", "open", "high", "low", "close"]
    if "fng_class" in daily.columns:
        cols.insert(2, "fng_class")
    out = daily.loc[mask, cols].copy()
    if "fng_class" not in out.columns:
        out["fng_class"] = ""
    out["threshold"] = threshold
    out["signal_type"] = "signal_day"
    log_count(logger, f"Signal days (fng<={threshold})", len(out))
    return out.reset_index(drop=True)


def detect_episodes(
    daily: pd.DataFrame,
    threshold: int,
    cooldown: int,
) -> pd.DataFrame:
    """Episodes (Signal B): first day F&G crosses to <= threshold after >= cooldown days above.

    An episode runs until the first day F&G rises above threshold again.
    Records start, end, duration, and minimum F&G during the episode.
    """
    df = daily[["date", "fng", "open", "high", "low", "close"]].copy().reset_index(drop=True)
    below = df["fng"] <= threshold
    episodes: list[dict[str, Any]] = []

    i = 0
    n = len(df)
    while i < n:
        if not below.iloc[i]:
            i += 1
            continue
        # Check cooldown: need >= cooldown consecutive days above threshold immediately before
        if i == 0:
            # Start of sample — treat as valid episode start (no prior history to require)
            eligible = True
        else:
            lookback = df["fng"].iloc[max(0, i - cooldown) : i]
            eligible = len(lookback) >= cooldown and bool((lookback > threshold).all())

        # Find end of this below-threshold run
        j = i
        while j < n and below.iloc[j]:
            j += 1
        run = df.iloc[i:j]
        end_date = run["date"].iloc[-1]
        # End date = last day still <= threshold; "first day back above" is j if exists
        first_above = df["date"].iloc[j] if j < n else pd.NaT

        if eligible:
            episodes.append(
                {
                    "threshold": threshold,
                    "cooldown": cooldown,
                    "signal_date": run["date"].iloc[0],
                    "end_date": end_date,
                    "first_day_above": first_above,
                    "duration_days": int(len(run)),
                    "min_fng": int(run["fng"].min()),
                    "signal_fng": int(run["fng"].iloc[0]),
                    "signal_open": float(run["open"].iloc[0]),
                    "signal_close": float(run["close"].iloc[0]),
                }
            )
        i = j

    out = pd.DataFrame(episodes)
    log_count(logger, f"Episodes (fng<={threshold}, N={cooldown})", len(out))
    return out


def attach_entry_prices(
    events: pd.DataFrame,
    daily: pd.DataFrame,
    timing: EntryTiming,
    date_col: str = "signal_date",
) -> pd.DataFrame:
    """Attach entry_date and entry_price based on timing rule.

    next_open: entry at open of day t+1 (base case, avoids look-ahead).
    signal_close: entry at close of signal day t (sensitivity).
    Rows without a valid next bar (for next_open) are dropped.
    """
    if events.empty:
        events = events.copy()
        events["entry_date"] = pd.Series(dtype="datetime64[ns]")
        events["entry_price"] = pd.Series(dtype=float)
        events["entry_timing"] = pd.Series(dtype=str)
        events["signal_idx"] = pd.Series(dtype=int)
        events["entry_idx"] = pd.Series(dtype=int)
        return events

    day = daily.reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(day["date"])}
    rows = []
    for _, ev in events.iterrows():
        sig_date = pd.Timestamp(ev[date_col])
        if sig_date not in date_to_idx:
            continue
        sig_idx = date_to_idx[sig_date]
        if timing == "next_open":
            entry_idx = sig_idx + 1
            if entry_idx >= len(day):
                continue  # no next bar
            entry_date = day.loc[entry_idx, "date"]
            entry_price = float(day.loc[entry_idx, "open"])
        elif timing == "signal_close":
            entry_idx = sig_idx
            entry_date = day.loc[entry_idx, "date"]
            entry_price = float(day.loc[entry_idx, "close"])
        else:
            raise ValueError(f"Unknown entry timing: {timing}")

        # Look-ahead safety: for next_open, entry must be strictly after signal
        if timing == "next_open" and entry_idx <= sig_idx:
            raise AssertionError("Look-ahead violation: entry_idx <= signal_idx")

        row = ev.to_dict()
        row["entry_date"] = entry_date
        row["entry_price"] = entry_price
        row["entry_timing"] = timing
        row["signal_idx"] = sig_idx
        row["entry_idx"] = entry_idx
        rows.append(row)

    out = pd.DataFrame(rows)
    log_count(logger, f"Entries with prices ({timing})", len(out))
    return out


def build_all_events(daily: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Build signal-day and episode tables for all thresholds / cooldowns / timings."""
    thresholds = cfg["thresholds"]
    cooldowns = cfg["cooldowns"]
    base_timing: EntryTiming = cfg["entry_timing"]
    sens_timing: EntryTiming = cfg["sensitivity_entry_timing"]

    signal_frames = []
    episode_frames = []

    for thr in thresholds:
        sig = detect_signal_days(daily, thr)
        for timing in (base_timing, sens_timing):
            dated_col = "date"
            # signal days use "date" as signal date
            tmp = sig.rename(columns={"date": "signal_date"})
            attached = attach_entry_prices(tmp, daily, timing, date_col="signal_date")
            if not attached.empty:
                signal_frames.append(attached)

        for n in cooldowns:
            ep = detect_episodes(daily, thr, n)
            for timing in (base_timing, sens_timing):
                attached = attach_entry_prices(ep, daily, timing, date_col="signal_date")
                if not attached.empty:
                    episode_frames.append(attached)

    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    episodes = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()

    # Print headline counts for default cooldown + base timing
    default_n = cfg["default_cooldown"]
    for thr in thresholds:
        n_sig = len(signals[(signals["threshold"] == thr) & (signals["entry_timing"] == base_timing)])
        # signal days are duplicated across timings — count unique signal dates
        n_sig_days = daily["fng"].le(thr).sum()
        n_ep = len(
            episodes[
                (episodes["threshold"] == thr)
                & (episodes["cooldown"] == default_n)
                & (episodes["entry_timing"] == base_timing)
            ]
        )
        logger.info(
            "Threshold <=%s: signal_days=%d, episodes(N=%d, %s)=%d",
            thr,
            int(n_sig_days),
            default_n,
            base_timing,
            n_ep,
        )

    return {"signal_days": signals, "episodes": episodes}


def cycle_label(date: pd.Timestamp, cycles: list[dict[str, str]]) -> str:
    """Map a date to a named market cycle from config."""
    d = pd.Timestamp(date)
    for c in cycles:
        if pd.Timestamp(c["start"]) <= d < pd.Timestamp(c["end"]):
            return c["name"]
    return "other"


def fear_depth_label(fng: int, bins: list[dict[str, Any]]) -> str:
    """Map F&G value to a depth bin name."""
    for b in bins:
        if b["min"] <= fng <= b["max"]:
            return b["name"]
    return "other"


def annotate_regimes(events: pd.DataFrame, daily: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add regime / cycle / fear-depth columns at signal date."""
    if events.empty:
        return events
    day = daily.set_index("date")
    out = events.copy()

    def _lookup(row: pd.Series) -> pd.Series:
        sig = pd.Timestamp(row["signal_date"])
        if sig not in day.index:
            return pd.Series(
                {"above_sma_200": np.nan, "cycle": "other", "fear_depth": "other", "year": sig.year}
            )
        r = day.loc[sig]
        fng_val = int(row.get("signal_fng", row.get("fng", r["fng"])))
        return pd.Series(
            {
                "above_sma_200": bool(r["above_sma_200"]) if pd.notna(r["above_sma_200"]) else np.nan,
                "cycle": cycle_label(sig, cfg["cycles"]),
                "fear_depth": fear_depth_label(fng_val, cfg["fear_depth_bins"]),
                "year": int(sig.year),
            }
        )

    extra = out.apply(_lookup, axis=1)
    return pd.concat([out, extra], axis=1)
