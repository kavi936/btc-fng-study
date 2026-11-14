#!/usr/bin/env python3
"""End-to-end runner for the BTC Fear & Greed extreme-fear study.

Phases:
  fetch     — download/cache data, merge, print validation, stop
  events    — signal/episode detection + metrics (+ tests are separate via pytest)
  backtest  — strategy comparisons
  report    — summary.md, CSVs, charts
  all       — events + backtest + report (assumes fetch already done, or runs fetch first)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.logging_utils import setup_logging
from src.fetch import run_fetch, print_fetch_summary
from src.events import build_all_events, annotate_regimes
from src.metrics import run_metrics_bundle, split_and_aggregate
from src.backtest import ladder_vs_lump_by_episode, run_backtests
from src.report import write_reports


def phase_fetch(cfg: dict, refresh: bool) -> None:
    daily, info = run_fetch(cfg, refresh=refresh)
    print_fetch_summary(info)
    _ = daily  # written to processed/


def _load_daily(cfg: dict, refresh: bool):
    processed = cfg["paths"]["processed"] / "daily.parquet"
    if processed.exists() and not refresh:
        import pandas as pd

        daily = pd.read_parquet(processed)
        daily["date"] = pd.to_datetime(daily["date"])
        return daily
    daily, info = run_fetch(cfg, refresh=refresh)
    print_fetch_summary(info)
    return daily


def phase_events(cfg: dict, refresh: bool) -> tuple:
    logger = setup_logging()
    daily = _load_daily(cfg, refresh=False)
    events = build_all_events(daily, cfg)
    events["episodes"] = annotate_regimes(events["episodes"], daily, cfg)
    events["signal_days"] = annotate_regimes(events["signal_days"], daily, cfg)

    default_n = cfg["default_cooldown"]
    base_timing = cfg["entry_timing"]
    sens_timing = cfg["sensitivity_entry_timing"]

    bundles: dict = {}
    # Episodes: all thresholds × cooldowns × both timings
    for thr in cfg["thresholds"]:
        for n in cfg["cooldowns"]:
            for timing in (base_timing, sens_timing):
                mask = (
                    (events["episodes"]["threshold"] == thr)
                    & (events["episodes"]["cooldown"] == n)
                    & (events["episodes"]["entry_timing"] == timing)
                )
                sub = events["episodes"].loc[mask]
                label = f"episodes_thr{thr}_N{n}_{timing}"
                logger.info("Metrics for %s (n=%d)", label, len(sub))
                bundles[label] = run_metrics_bundle(daily, sub, cfg, label, is_episodes=True)

        # Signal days (base + sensitivity) — descriptive; block bootstrap in CI
        for timing in (base_timing, sens_timing):
            mask = (events["signal_days"]["threshold"] == thr) & (
                events["signal_days"]["entry_timing"] == timing
            )
            sub = events["signal_days"].loc[mask]
            label = f"signal_days_thr{thr}_{timing}"
            logger.info("Metrics for %s (n=%d)", label, len(sub))
            bundles[label] = run_metrics_bundle(daily, sub, cfg, label, is_episodes=False)

    # Regime splits on headline episodes (thr 12 & 15, default N, base timing)
    regime_frames = []
    for thr in cfg["thresholds"]:
        label = f"episodes_thr{thr}_N{default_n}_{base_timing}"
        per = bundles[label]["per_entry"]
        regime_frames.append(split_and_aggregate(daily, per, cfg, label))
    import pandas as pd

    regime = pd.concat(regime_frames, ignore_index=True) if regime_frames else pd.DataFrame()

    # Cache intermediate artifacts
    import pickle

    cache_path = cfg["paths"]["processed"] / "analysis_cache.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump({"events": events, "bundles": bundles, "regime": regime, "daily": daily}, f)
    logger.info("Wrote analysis cache to %s", cache_path)
    return daily, events, bundles, regime


def phase_backtest(cfg: dict) -> tuple:
    import pickle

    logger = setup_logging()
    cache_path = cfg["paths"]["processed"] / "analysis_cache.pkl"
    if not cache_path.exists():
        raise FileNotFoundError("Run --phase events first (missing analysis_cache.pkl)")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    daily = cache["daily"]
    events = cache["events"]
    default_n = cfg["default_cooldown"]
    base_timing = cfg["entry_timing"]

    summaries = []
    equities = []
    lvl_frames = []
    for thr in cfg["thresholds"]:
        ep = events["episodes"][
            (events["episodes"]["threshold"] == thr)
            & (events["episodes"]["cooldown"] == default_n)
            & (events["episodes"]["entry_timing"] == base_timing)
        ]
        logger.info("Backtests for threshold <=%s (%d episodes)", thr, len(ep))
        summary, equity = run_backtests(daily, ep, cfg, thr)
        summaries.append(summary)
        equities.append(equity)
        lvl = ladder_vs_lump_by_episode(
            daily,
            ep,
            cfg["ladder_drop_levels"],
            cfg["ladder_window_days"],
            [h for h in cfg["horizons"] if h >= 30],
            cfg["fee_per_trade"],
        )
        lvl.insert(0, "threshold", thr)
        lvl_frames.append(lvl)

    import pandas as pd

    summary = pd.concat(summaries, ignore_index=True)
    equity = pd.concat(equities, ignore_index=True)
    cache["backtest_summary"] = summary
    cache["backtest_equity"] = equity
    cache["ladder_vs_lump"] = pd.concat(lvl_frames, ignore_index=True) if lvl_frames else pd.DataFrame()
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    return daily, events, cache["bundles"], cache["regime"], summary, equity


def phase_report(cfg: dict) -> str:
    import pickle

    cache_path = cfg["paths"]["processed"] / "analysis_cache.pkl"
    if not cache_path.exists():
        raise FileNotFoundError("Run --phase events (and backtest) first")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    if "backtest_summary" not in cache:
        raise FileNotFoundError("Run --phase backtest first")
    md = write_reports(
        cfg,
        cache["daily"],
        cache["events"],
        cache["bundles"],
        cache["backtest_summary"],
        cache["backtest_equity"],
        cache["regime"],
        ladder_vs_lump=cache.get("ladder_vs_lump"),
    )
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC F&G extreme-fear study")
    parser.add_argument(
        "--phase",
        choices=["fetch", "events", "backtest", "report", "all"],
        default="fetch",
        help="Pipeline phase (default: fetch — stops after validation)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of API data (ignore caches)",
    )
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config()
    logger.info("Config loaded from %s", ROOT / "config.yaml")

    if args.phase == "fetch":
        phase_fetch(cfg, args.refresh)
        return

    if args.phase == "events":
        phase_events(cfg, args.refresh)
        return

    if args.phase == "backtest":
        phase_backtest(cfg)
        return

    if args.phase == "report":
        md = phase_report(cfg)
        print("\n" + md)
        return

    if args.phase == "all":
        # Fetch if needed, then full pipeline
        _load_daily(cfg, refresh=args.refresh)
        phase_events(cfg, refresh=False)
        phase_backtest(cfg)
        md = phase_report(cfg)
        print("\n" + md)
        return


if __name__ == "__main__":
    main()
