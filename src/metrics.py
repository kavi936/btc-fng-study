"""Forward returns, drawdowns, time-to-bottom, baselines, and bootstrap stats."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("btc_fng_study")


def _safe_pct(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def distribution_stats(x: np.ndarray | pd.Series, prefix: str = "") -> dict[str, Any]:
    """Mean, median, std, percentiles, min, max, skew for a return series."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(arr)
    p = f"{prefix}_" if prefix else ""
    if n == 0:
        keys = ["n", "mean", "median", "std", "p10", "p25", "p75", "p90", "min", "max", "skew"]
        return {f"{p}{k}": (0 if k == "n" else float("nan")) for k in keys}
    return {
        f"{p}n": n,
        f"{p}mean": float(np.mean(arr)),
        f"{p}median": float(np.median(arr)),
        f"{p}std": float(np.std(arr, ddof=1)) if n > 1 else float("nan"),
        f"{p}p10": float(np.percentile(arr, 10)),
        f"{p}p25": float(np.percentile(arr, 25)),
        f"{p}p75": float(np.percentile(arr, 75)),
        f"{p}p90": float(np.percentile(arr, 90)),
        f"{p}min": float(np.min(arr)),
        f"{p}max": float(np.max(arr)),
        f"{p}skew": float(stats.skew(arr)) if n > 2 else float("nan"),
    }


def mark_low_sample(n: int, low_n: int) -> str:
    return "low sample" if n < low_n else "ok"


# ---------------------------------------------------------------------------
# Per-entry metrics
# ---------------------------------------------------------------------------

def _date_index(daily: pd.DataFrame) -> dict[pd.Timestamp, int]:
    """Map normalized dates to row positions."""
    return {pd.Timestamp(d): i for i, d in enumerate(daily["date"])}


def _window_slice(
    daily: pd.DataFrame,
    entry_idx: int,
    window: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Rows strictly after entry with calendar date <= entry_date + window days."""
    day = daily.reset_index(drop=True)
    entry_date = pd.Timestamp(day.loc[entry_idx, "date"])
    end_date = entry_date + pd.Timedelta(days=window)
    sl = day[(day.index > entry_idx) & (day["date"] <= end_date)]
    return sl, end_date


def forward_return(
    daily: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    horizon: int,
    date_to_idx: dict[pd.Timestamp, int] | None = None,
) -> float:
    """close on calendar day entry_date+H (or nearest prior within 3d) / entry - 1."""
    day = daily.reset_index(drop=True)
    entry_date = pd.Timestamp(day.loc[entry_idx, "date"])
    target = entry_date + pd.Timedelta(days=horizon)
    d2i = date_to_idx or _date_index(day)
    j = None
    if target in d2i:
        j = d2i[target]
    else:
        # Allow a small lookback if that exact calendar day is a known gap
        for back in range(1, 4):
            cand = target - pd.Timedelta(days=back)
            if cand in d2i and d2i[cand] > entry_idx:
                j = d2i[cand]
                break
    if j is None or j <= entry_idx:
        return float("nan")
    return float(day.loc[j, "close"] / entry_price - 1.0)


def mae_in_window(
    daily: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    window: int,
) -> tuple[float, bool, int | float]:
    """Max adverse excursion using lows on (entry_date, entry_date+window].

    Returns (mae, made_lower_low, calendar days from entry to lowest low).
    """
    day = daily.reset_index(drop=True)
    sl, _ = _window_slice(day, entry_idx, window)
    if sl.empty:
        return float("nan"), False, float("nan")
    min_low = float(sl["low"].min())
    mae = min_low / entry_price - 1.0
    lower_low = min_low < entry_price
    bottom_row = sl.loc[sl["low"].idxmin()]
    entry_date = pd.Timestamp(day.loc[entry_idx, "date"])
    days_to_bottom = (pd.Timestamp(bottom_row["date"]) - entry_date).days
    return mae, lower_low, days_to_bottom


def mfe_in_window(
    daily: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    window: int,
) -> float:
    """Max favorable excursion using highs on (entry_date, entry_date+window]."""
    day = daily.reset_index(drop=True)
    sl, _ = _window_slice(day, entry_idx, window)
    if sl.empty:
        return float("nan")
    max_high = float(sl["high"].max())
    return max_high / entry_price - 1.0


def days_until_recover(
    daily: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    max_days: int = 365,
) -> float:
    """Calendar days until close back above entry_price; NaN if never within max_days."""
    day = daily.reset_index(drop=True)
    entry_date = pd.Timestamp(day.loc[entry_idx, "date"])
    end_date = entry_date + pd.Timedelta(days=max_days)
    sl = day[(day.index > entry_idx) & (day["date"] <= end_date)]
    for _, row in sl.iterrows():
        if float(row["close"]) >= entry_price:
            return float((pd.Timestamp(row["date"]) - entry_date).days)
    return float("nan")


def days_to_local_bottom(
    daily: pd.DataFrame,
    entry_idx: int,
    recovery: float = 0.20,
    max_scan: int = 730,
) -> float:
    """Calendar days from entry to lowest low before price recovers +recovery off that low."""
    day = daily.reset_index(drop=True)
    entry_date = pd.Timestamp(day.loc[entry_idx, "date"])
    end_date = entry_date + pd.Timedelta(days=max_scan)
    sl = day[(day.index > entry_idx) & (day["date"] <= end_date)]
    if sl.empty:
        return float("nan")
    running_min = float("inf")
    bottom_date = entry_date
    for _, row in sl.iterrows():
        low = float(row["low"])
        if low < running_min:
            running_min = low
            bottom_date = pd.Timestamp(row["date"])
        if running_min < float("inf") and float(row["high"]) >= running_min * (1 + recovery):
            return float((bottom_date - entry_date).days)
    return float("nan")


def days_fng_above(
    daily: pd.DataFrame,
    entry_idx: int,
    level: int,
    max_days: int = 730,
) -> float:
    """Calendar days from entry until F&G is above `level`."""
    day = daily.reset_index(drop=True)
    entry_date = pd.Timestamp(day.loc[entry_idx, "date"])
    end_date = entry_date + pd.Timedelta(days=max_days)
    sl = day[(day.index >= entry_idx) & (day["date"] <= end_date)]
    for _, row in sl.iterrows():
        if int(row["fng"]) > level:
            return float((pd.Timestamp(row["date"]) - entry_date).days)
    return float("nan")


def compute_entry_metrics(
    daily: pd.DataFrame,
    entries: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Compute per-entry forward returns, MAE, MFE, recovery, time-to-bottom."""
    if entries.empty:
        return entries.copy()

    horizons = cfg["horizons"]
    mae_windows = cfg["mae_windows"]
    mfe_windows = cfg["mfe_windows"]
    dd_levels = cfg["further_drawdown_levels"]
    recovery = cfg["local_bottom_recovery"]
    fng_levels = cfg["fng_recovery_levels"]

    day = daily.reset_index(drop=True)
    d2i = _date_index(day)
    records = []

    for _, ev in entries.iterrows():
        entry_idx = int(ev["entry_idx"])
        entry_price = float(ev["entry_price"])
        entry_date = pd.Timestamp(day.loc[entry_idx, "date"])
        rec: dict[str, Any] = ev.to_dict()

        for h in horizons:
            rec[f"ret_{h}d"] = forward_return(day, entry_idx, entry_price, h, d2i)

        for w in mae_windows:
            mae, lower, ttb = mae_in_window(day, entry_idx, entry_price, w)
            rec[f"mae_{w}d"] = mae
            rec[f"lower_low_{w}d"] = lower
            rec[f"ttb_{w}d"] = ttb
            # Flag if bottom falls on the last calendar day of the window
            rec[f"ttb_{w}d_at_window_end"] = bool(pd.notna(ttb) and int(ttb) >= w)
            for lvl in dd_levels:
                pct = int(lvl * 100)
                rec[f"fell_{pct}pct_{w}d"] = bool(pd.notna(mae) and mae <= -lvl)

        for w in mfe_windows:
            rec[f"mfe_{w}d"] = mfe_in_window(day, entry_idx, entry_price, w)

        rec["days_underwater"] = days_until_recover(day, entry_idx, entry_price, 365)
        rec["never_recovered_365"] = bool(pd.isna(rec["days_underwater"]))
        rec["days_to_local_bottom"] = days_to_local_bottom(day, entry_idx, recovery)

        for lvl in fng_levels:
            rec[f"days_fng_above_{lvl}"] = days_fng_above(day, entry_idx, lvl)

        # Cumulative path for average-path chart (0..180 calendar days)
        path = []
        for d in range(0, 181):
            if d == 0:
                path.append(0.0)
                continue
            target = entry_date + pd.Timedelta(days=d)
            j = d2i.get(target)
            if j is None or j <= entry_idx:
                path.append(np.nan)
            else:
                path.append(float(day.loc[j, "close"] / entry_price - 1.0))
        rec["path_0_180"] = path

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregation + baseline
# ---------------------------------------------------------------------------

def aggregate_forward_returns(
    per_entry: pd.DataFrame,
    baseline_returns: dict[int, np.ndarray],
    horizons: list[int],
    low_n: int,
    label: str,
) -> pd.DataFrame:
    """Aggregate return stats vs baseline for each horizon."""
    rows = []
    for h in horizons:
        col = f"ret_{h}d"
        arr = per_entry[col].dropna().values if col in per_entry.columns else np.array([])
        dist = distribution_stats(arr)
        win_rate = float(np.mean(arr > 0)) if len(arr) else float("nan")
        base = baseline_returns.get(h, np.array([]))
        base_dist = distribution_stats(base, prefix="baseline")
        base_wr = float(np.mean(base > 0)) if len(base) else float("nan")
        rows.append(
            {
                "label": label,
                "horizon": h,
                **dist,
                "win_rate": win_rate,
                "baseline_win_rate": base_wr,
                "excess_win_rate": win_rate - base_wr if len(arr) and len(base) else float("nan"),
                "excess_median": dist["median"] - base_dist["baseline_median"]
                if len(arr) and len(base)
                else float("nan"),
                "mean_over_std": (
                    dist["mean"] / dist["std"] if dist["std"] and dist["std"] > 0 else float("nan")
                ),
                "sample_flag": mark_low_sample(dist["n"], low_n),
                **{k: v for k, v in base_dist.items()},
            }
        )
    return pd.DataFrame(rows)


def unconditional_forward_returns(daily: pd.DataFrame, horizons: list[int]) -> dict[int, np.ndarray]:
    """Buy-any-day baseline: forward close-to-close returns over calendar horizons."""
    day = daily.reset_index(drop=True)
    d2i = _date_index(day)
    out: dict[int, np.ndarray] = {}
    for h in horizons:
        rets = []
        for i in range(len(day)):
            r = forward_return(day, i, float(day.loc[i, "close"]), h, d2i)
            if not np.isnan(r):
                rets.append(r)
        out[h] = np.asarray(rets, dtype=float)
    return out


def aggregate_drawdowns(per_entry: pd.DataFrame, mae_windows: list[int], dd_levels: list[float], low_n: int, label: str) -> pd.DataFrame:
    rows = []
    for w in mae_windows:
        col = f"mae_{w}d"
        arr = per_entry[col].dropna().values if col in per_entry.columns else np.array([])
        n = len(arr)
        lower_col = f"lower_low_{w}d"
        pct_lower = float(per_entry[lower_col].mean()) if lower_col in per_entry and n else float("nan")
        row: dict[str, Any] = {
            "label": label,
            "window": w,
            "n": n,
            "sample_flag": mark_low_sample(n, low_n),
            "mae_median": float(np.median(arr)) if n else float("nan"),
            "mae_p75": float(np.percentile(arr, 75)) if n else float("nan"),
            "mae_p90": float(np.percentile(arr, 90)) if n else float("nan"),
            "mae_worst": float(np.min(arr)) if n else float("nan"),
            "pct_lower_low": pct_lower,
        }
        for lvl in dd_levels:
            pct = int(lvl * 100)
            c = f"fell_{pct}pct_{w}d"
            row[f"pct_fell_{pct}"] = float(per_entry[c].mean()) if c in per_entry and n else float("nan")
        # median return / |median MAE| style risk-adjusted (use ret_wd if exists else nearest)
        ret_col = f"ret_{w}d"
        if ret_col in per_entry.columns and n:
            med_ret = float(per_entry[ret_col].median())
            med_mae = row["mae_median"]
            row["median_ret_over_abs_mae"] = (
                med_ret / abs(med_mae) if med_mae and med_mae != 0 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_time_to_bottom(per_entry: pd.DataFrame, windows: list[int], low_n: int, label: str) -> pd.DataFrame:
    rows = []
    for w in windows:
        col = f"ttb_{w}d"
        arr = per_entry[col].dropna().values if col in per_entry.columns else np.array([])
        n = len(arr)
        flag_col = f"ttb_{w}d_at_window_end"
        n_at_end = int(per_entry[flag_col].sum()) if flag_col in per_entry.columns else 0
        rows.append(
            {
                "label": label,
                "window": w,
                "n": n,
                "sample_flag": mark_low_sample(n, low_n),
                "ttb_mean": float(np.mean(arr)) if n else float("nan"),
                "ttb_median": float(np.median(arr)) if n else float("nan"),
                "n_bottom_at_window_end": n_at_end,
                "pct_bottom_at_window_end": _safe_pct(n_at_end, n),
            }
        )
    # Local bottom
    loc = per_entry["days_to_local_bottom"].dropna().values if "days_to_local_bottom" in per_entry else np.array([])
    rows.append(
        {
            "label": label,
            "window": "local_plus20pct",
            "n": len(loc),
            "sample_flag": mark_low_sample(len(loc), low_n),
            "ttb_mean": float(np.mean(loc)) if len(loc) else float("nan"),
            "ttb_median": float(np.median(loc)) if len(loc) else float("nan"),
            "n_bottom_at_window_end": np.nan,
            "pct_bottom_at_window_end": np.nan,
        }
    )
    # Recovery / underwater
    uw = per_entry["days_underwater"].dropna().values if "days_underwater" in per_entry else np.array([])
    never = float(per_entry["never_recovered_365"].mean()) if "never_recovered_365" in per_entry else float("nan")
    rows.append(
        {
            "label": label,
            "window": "underwater_to_entry",
            "n": int(len(per_entry)),
            "sample_flag": mark_low_sample(len(per_entry), low_n),
            "ttb_mean": float(np.mean(uw)) if len(uw) else float("nan"),
            "ttb_median": float(np.median(uw)) if len(uw) else float("nan"),
            "n_bottom_at_window_end": int(per_entry["never_recovered_365"].sum())
            if "never_recovered_365" in per_entry
            else 0,
            "pct_bottom_at_window_end": never,
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def year_matched_sample(
    daily: pd.DataFrame,
    n: int,
    years: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n entry indices, matching year counts when possible."""
    day = daily.reset_index(drop=True)
    day["year"] = day["date"].dt.year
    # Target year counts from the provided years list
    from collections import Counter

    counts = Counter(years)
    idxs: list[int] = []
    for y, c in counts.items():
        pool = day.index[day["year"] == y].tolist()
        if not pool:
            continue
        choose = rng.choice(pool, size=c, replace=len(pool) < c)
        idxs.extend(choose.tolist())
    # Top up if short
    while len(idxs) < n:
        idxs.append(int(rng.integers(0, len(day))))
    return np.asarray(idxs[:n], dtype=int)


def bootstrap_median_pvalue(
    daily: pd.DataFrame,
    signal_medians: dict[int, float],
    n_entries: int,
    years: list[int],
    horizons: list[int],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Empirical p-value: percentile of signal median within random-entry median distribution."""
    rng = np.random.default_rng(seed)
    day = daily.reset_index(drop=True)
    rows = []
    for h in horizons:
        random_medians = []
        d2i = _date_index(day)
        for _ in range(n_boot):
            idxs = year_matched_sample(day, n_entries, years, rng)
            rets = []
            for i in idxs:
                entry_price = float(day.loc[i, "open"])
                r = forward_return(day, i, entry_price, h, d2i)
                if not np.isnan(r):
                    rets.append(r)
            if rets:
                random_medians.append(float(np.median(rets)))
        rm = np.asarray(random_medians)
        sig_med = signal_medians.get(h, float("nan"))
        if len(rm) and pd.notna(sig_med):
            # One-sided: P(random median >= signal median) — higher is better for signal
            p = float(np.mean(rm >= sig_med))
            # Percentile rank of signal within random dist
            pct = float(np.mean(rm < sig_med))
        else:
            p, pct = float("nan"), float("nan")
        rows.append(
            {
                "horizon": h,
                "signal_median": sig_med,
                "random_median_mean": float(np.mean(rm)) if len(rm) else float("nan"),
                "empirical_pvalue_onesided": p,
                "signal_percentile_in_random": pct,
                "n_boot": len(rm),
                "n_entries": n_entries,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    stat: str = "median",
) -> tuple[float, float, float]:
    """Return (point, lo, hi) 95% bootstrap CI for median or win_rate."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats_boot = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        if stat == "median":
            stats_boot.append(float(np.median(sample)))
        elif stat == "win_rate":
            stats_boot.append(float(np.mean(sample > 0)))
        else:
            raise ValueError(stat)
    lo, hi = np.percentile(stats_boot, [2.5, 97.5])
    point = float(np.median(arr)) if stat == "median" else float(np.mean(arr > 0))
    return point, float(lo), float(hi)


def block_bootstrap_ci(
    values: np.ndarray,
    block_length: int,
    n_boot: int,
    seed: int,
    stat: str = "median",
) -> tuple[float, float, float]:
    """Block bootstrap CI for autocorrelated signal-day returns."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    bl = max(1, min(block_length, n))
    rng = np.random.default_rng(seed)
    stats_boot = []
    n_blocks = int(np.ceil(n / bl))
    for _ in range(n_boot):
        blocks = []
        for _b in range(n_blocks):
            start = int(rng.integers(0, n))
            for k in range(bl):
                blocks.append(arr[(start + k) % n])
        sample = np.asarray(blocks[:n])
        if stat == "median":
            stats_boot.append(float(np.median(sample)))
        else:
            stats_boot.append(float(np.mean(sample > 0)))
    lo, hi = np.percentile(stats_boot, [2.5, 97.5])
    point = float(np.median(arr)) if stat == "median" else float(np.mean(arr > 0))
    return point, float(lo), float(hi)


def compute_ci_table(
    per_entry: pd.DataFrame,
    horizons: list[int],
    n_boot: int,
    seed: int,
    label: str,
    use_block: bool = False,
) -> pd.DataFrame:
    rows = []
    for h in horizons:
        col = f"ret_{h}d"
        arr = per_entry[col].dropna().values if col in per_entry.columns else np.array([])
        if use_block:
            med, mlo, mhi = block_bootstrap_ci(arr, h, n_boot, seed, "median")
            wr, wlo, whi = block_bootstrap_ci(arr, h, n_boot, seed + 1, "win_rate")
        else:
            med, mlo, mhi = bootstrap_ci(arr, n_boot, seed, "median")
            wr, wlo, whi = bootstrap_ci(arr, n_boot, seed + 1, "win_rate")
        rows.append(
            {
                "label": label,
                "horizon": h,
                "n": len(arr),
                "median": med,
                "median_ci_lo": mlo,
                "median_ci_hi": mhi,
                "win_rate": wr,
                "win_rate_ci_lo": wlo,
                "win_rate_ci_hi": whi,
                "method": "block" if use_block else "iid",
            }
        )
    return pd.DataFrame(rows)


def average_path(per_entry: pd.DataFrame, baseline_daily: pd.DataFrame, max_days: int = 180) -> pd.DataFrame:
    """Mean/median/IQR cumulative return path from entry, plus unconditional baseline path."""
    if per_entry.empty or "path_0_180" not in per_entry.columns:
        return pd.DataFrame()
    paths = np.vstack(per_entry["path_0_180"].values)
    days = np.arange(0, max_days + 1)
    rows = []
    # Baseline: average path from all days using calendar offsets
    day = baseline_daily.reset_index(drop=True)
    d2i = _date_index(day)
    base_paths = []
    # Subsample for speed on long histories
    step = max(1, len(day) // 800)
    for i in range(0, len(day), step):
        p0 = float(day.loc[i, "close"])
        entry_date = pd.Timestamp(day.loc[i, "date"])
        path = []
        ok = True
        for d in days:
            if d == 0:
                path.append(0.0)
                continue
            j = d2i.get(entry_date + pd.Timedelta(days=int(d)))
            if j is None:
                ok = False
                break
            path.append(float(day.loc[j, "close"] / p0 - 1.0))
        if ok:
            base_paths.append(path)
    base = np.asarray(base_paths) if base_paths else np.empty((0, max_days + 1))

    for d in days:
        col = paths[:, d]
        valid = col[~np.isnan(col)]
        row = {
            "day": int(d),
            "mean": float(np.mean(valid)) if len(valid) else float("nan"),
            "median": float(np.median(valid)) if len(valid) else float("nan"),
            "p25": float(np.percentile(valid, 25)) if len(valid) else float("nan"),
            "p75": float(np.percentile(valid, 75)) if len(valid) else float("nan"),
            "n": len(valid),
        }
        if len(base):
            bcol = base[:, d]
            row["baseline_mean"] = float(np.mean(bcol))
            row["baseline_median"] = float(np.median(bcol))
        else:
            row["baseline_mean"] = float("nan")
            row["baseline_median"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def run_metrics_bundle(
    daily: pd.DataFrame,
    entries: pd.DataFrame,
    cfg: dict[str, Any],
    label: str,
    is_episodes: bool = True,
) -> dict[str, pd.DataFrame]:
    """Full metric suite for one entry set."""
    low_n = cfg["low_sample_n"]
    horizons = cfg["horizons"]
    per = compute_entry_metrics(daily, entries, cfg)
    baseline = unconditional_forward_returns(daily, horizons)
    fwd = aggregate_forward_returns(per, baseline, horizons, low_n, label)
    dd = aggregate_drawdowns(per, cfg["mae_windows"], cfg["further_drawdown_levels"], low_n, label)
    ttb = aggregate_time_to_bottom(per, cfg["time_to_bottom_windows"], low_n, label)

    ci_h = cfg["ci_horizons"]
    ci = compute_ci_table(
        per,
        ci_h,
        cfg["bootstrap_samples"],
        cfg["bootstrap_seed"],
        label,
        use_block=not is_episodes,
    )

    boot = pd.DataFrame()
    if is_episodes and not per.empty:
        years = per["signal_date"].apply(lambda d: pd.Timestamp(d).year).tolist()
        sig_med = {h: float(per[f"ret_{h}d"].median()) for h in ci_h if f"ret_{h}d" in per}
        boot = bootstrap_median_pvalue(
            daily,
            sig_med,
            len(per),
            years,
            ci_h,
            cfg["bootstrap_samples"],
            cfg["bootstrap_seed"],
        )
        boot["label"] = label

    path = average_path(per, daily, 180)
    if not path.empty:
        path["label"] = label

    # Recovery F&G stats
    fng_rows = []
    for lvl in cfg["fng_recovery_levels"]:
        col = f"days_fng_above_{lvl}"
        arr = per[col].dropna().values if col in per.columns else np.array([])
        fng_rows.append(
            {
                "label": label,
                "level": lvl,
                "n": len(arr),
                "mean_days": float(np.mean(arr)) if len(arr) else float("nan"),
                "median_days": float(np.median(arr)) if len(arr) else float("nan"),
                "sample_flag": mark_low_sample(len(arr), low_n),
            }
        )
    fng_rec = pd.DataFrame(fng_rows)

    return {
        "per_entry": per,
        "forward_returns": fwd,
        "drawdowns": dd,
        "time_to_bottom": ttb,
        "ci": ci,
        "bootstrap": boot,
        "avg_path": path,
        "fng_recovery": fng_rec,
    }


def split_and_aggregate(
    daily: pd.DataFrame,
    per_entry: pd.DataFrame,
    cfg: dict[str, Any],
    base_label: str,
) -> pd.DataFrame:
    """Key forward-return stats split by regime / cycle / fear depth."""
    if per_entry.empty:
        return pd.DataFrame()
    horizons = [30, 90, 180, 365]
    low_n = cfg["low_sample_n"]
    baseline = unconditional_forward_returns(daily, horizons)
    frames = []

    split_specs: list[tuple[str, pd.Series]] = [
        ("all", pd.Series(True, index=per_entry.index)),
    ]
    if "above_sma_200" in per_entry.columns:
        split_specs.append(("above_sma200", per_entry["above_sma_200"] == True))  # noqa: E712
        split_specs.append(("below_sma200", per_entry["above_sma_200"] == False))  # noqa: E712
    if "cycle" in per_entry.columns:
        for c in per_entry["cycle"].dropna().unique():
            split_specs.append((f"cycle:{c}", per_entry["cycle"] == c))
    if "fear_depth" in per_entry.columns:
        for c in per_entry["fear_depth"].dropna().unique():
            split_specs.append((f"fear_depth:{c}", per_entry["fear_depth"] == c))
    if "year" in per_entry.columns:
        for y in sorted(per_entry["year"].dropna().unique()):
            split_specs.append((f"year:{y}", per_entry["year"] == y))

    for split_name, mask in split_specs:
        sub = per_entry.loc[mask]
        agg = aggregate_forward_returns(
            sub, baseline, horizons, low_n, f"{base_label}|{split_name}"
        )
        frames.append(agg)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
