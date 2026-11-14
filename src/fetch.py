"""Download and cache BTC OHLCV + Fear & Greed; merge into a daily panel."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.logging_utils import log_count

logger = logging.getLogger("btc_fng_study")

REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_PAGES = 0.2


def _cache_path(raw_dir: Path, name: str, ext: str = "parquet") -> Path:
    return raw_dir / f"{name}.{ext}"


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _load_df(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["date"])


def _normalize_dates(s: pd.Series) -> pd.Series:
    """UTC midnight dates (timezone-naive for stable merges)."""
    dt = pd.to_datetime(s, utc=True)
    return dt.dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)


# ---------------------------------------------------------------------------
# BTC OHLCV
# ---------------------------------------------------------------------------

def _parse_binance_klines(raw: list[list[Any]], source: str) -> pd.DataFrame:
    rows = []
    for k in raw:
        rows.append(
            {
                "date": pd.to_datetime(k[0], unit="ms", utc=True),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "price_source": source,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = _normalize_dates(df["date"])
    return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def fetch_binance_klines(
    base_url: str,
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    limit: int = 1000,
) -> pd.DataFrame:
    """Paginate daily klines from a Binance-compatible base URL."""
    url = f"{base_url.rstrip('/')}/api/v3/klines"
    source = base_url.replace("https://", "").split("/")[0]
    all_rows: list[list[Any]] = []
    start_time = 0  # earliest available

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
        }
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)
        last_open = batch[-1][0]
        # Advance past this candle's open
        start_time = last_open + 86_400_000
        if len(batch) < limit:
            break
        time.sleep(SLEEP_BETWEEN_PAGES)

    logger.info("Fetched %d klines from %s", len(all_rows), source)
    return _parse_binance_klines(all_rows, source=source)


def fetch_coinglass_ohlc(api_key: str, base: str, path: str) -> pd.DataFrame:
    """Fallback BTC spot OHLC from CoinGlass (requires API key)."""
    url = f"{base.rstrip('/')}{path}"
    headers = {"CG-API-KEY": api_key}
    # CoinGlass spot price history: symbol BTC, interval 1d
    params = {"symbol": "BTC", "interval": "1d", "limit": 4500}
    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload.get("code")) not in ("0", "200"):
        raise RuntimeError(f"CoinGlass OHLC error: {payload}")
    data = payload.get("data") or []
    rows = []
    for item in data:
        # Flexible field names across CoinGlass versions
        ts = item.get("time") or item.get("t") or item.get("timestamp")
        rows.append(
            {
                "date": pd.to_datetime(ts, unit="ms" if ts and ts > 1e12 else "s", utc=True),
                "open": float(item.get("open", item.get("o"))),
                "high": float(item.get("high", item.get("h"))),
                "low": float(item.get("low", item.get("l"))),
                "close": float(item.get("close", item.get("c"))),
                "volume": float(item.get("volume", item.get("v", 0))),
                "price_source": "coinglass",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("CoinGlass OHLC returned empty data")
    df["date"] = _normalize_dates(df["date"])
    return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def fetch_btc_ohlcv(cfg: dict[str, Any], refresh: bool = False) -> pd.DataFrame:
    """Fetch BTC daily OHLCV with caching and endpoint fallbacks."""
    raw_dir: Path = cfg["paths"]["raw"]
    cache = _cache_path(raw_dir, "btc_ohlcv")
    csv_cache = _cache_path(raw_dir, "btc_ohlcv", "csv")

    if not refresh:
        cached = _load_df(cache)
        if cached is None:
            cached = _load_df(csv_cache)
        if cached is not None and not cached.empty:
            cached["date"] = _normalize_dates(cached["date"])
            log_count(logger, "BTC OHLCV (cached)", len(cached))
            return cached

    binance_cfg = cfg["binance"]
    last_err: Exception | None = None
    for base in binance_cfg["bases"]:
        try:
            df = fetch_binance_klines(
                base,
                symbol=binance_cfg["symbol"],
                interval=binance_cfg["interval"],
                limit=binance_cfg["limit"],
            )
            if df.empty:
                raise RuntimeError(f"Empty klines from {base}")
            _save_df(df, cache)
            _save_df(df, csv_cache)
            log_count(logger, "BTC OHLCV (downloaded)", len(df))
            return df
        except Exception as exc:  # noqa: BLE001 — try next endpoint
            logger.warning("Binance endpoint %s failed: %s", base, exc)
            last_err = exc

    api_key = cfg.get("coinglass_api_key")
    if api_key:
        try:
            df = fetch_coinglass_ohlc(
                api_key, cfg["coinglass"]["base"], cfg["coinglass"]["ohlc_path"]
            )
            _save_df(df, cache)
            _save_df(df, csv_cache)
            log_count(logger, "BTC OHLCV (CoinGlass)", len(df))
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error("CoinGlass OHLC failed: %s", exc)
            last_err = exc

    raise RuntimeError(f"All BTC OHLCV sources failed. Last error: {last_err}")


# ---------------------------------------------------------------------------
# Fear & Greed
# ---------------------------------------------------------------------------

def fetch_alternative_me_fng(url: str) -> pd.DataFrame:
    """Full history from alternative.me (no API key)."""
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("metadata", {}).get("error"):
        raise RuntimeError(f"alternative.me error: {payload['metadata']['error']}")
    data = payload["data"]
    df = pd.DataFrame(data)
    df["date"] = _normalize_dates(pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True))
    df["fng"] = df["value"].astype(int)
    df["fng_class"] = df["value_classification"]
    df = df[["date", "fng", "fng_class"]].drop_duplicates(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_coinglass_fng(api_key: str, base: str, path: str) -> pd.DataFrame:
    """Secondary F&G series from CoinGlass for cross-check."""
    url = f"{base.rstrip('/')}{path}"
    headers = {"CG-API-KEY": api_key}
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload.get("code")) not in ("0", "200"):
        raise RuntimeError(f"CoinGlass F&G error: {payload}")
    data = payload["data"]
    times = data["time_list"]
    values = data["data_list"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(times, unit="ms", utc=True),
            "fng_cg": values,
        }
    )
    df["date"] = _normalize_dates(df["date"])
    return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def validate_fng_calendar(df: pd.DataFrame, value_col: str = "fng") -> dict[str, Any]:
    """Check duplicates and missing calendar days; return a validation summary."""
    dup_mask = df["date"].duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing = full.difference(pd.DatetimeIndex(df["date"]))
    summary = {
        "n_rows": len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "n_duplicates": n_dup,
        "n_missing_days": len(missing),
        "missing_days": [str(d.date()) for d in missing[:50]],  # cap log size
        "value_col": value_col,
    }
    if n_dup:
        logger.warning("F&G duplicates: %d rows", n_dup)
    if len(missing):
        logger.warning("F&G missing calendar days: %d (showing up to 50)", len(missing))
        for d in summary["missing_days"][:10]:
            logger.warning("  missing: %s", d)
    return summary


def fetch_fear_greed(cfg: dict[str, Any], refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch F&G (primary + optional CoinGlass cross-check). Returns (df, validation)."""
    raw_dir: Path = cfg["paths"]["raw"]
    cache = _cache_path(raw_dir, "fng_alternative")
    csv_cache = _cache_path(raw_dir, "fng_alternative", "csv")
    validation: dict[str, Any] = {}

    if not refresh:
        cached = _load_df(cache)
        if cached is None:
            cached = _load_df(csv_cache)
        if cached is not None and not cached.empty:
            cached["date"] = _normalize_dates(cached["date"])
            log_count(logger, "F&G alternative.me (cached)", len(cached))
            validation["alternative"] = validate_fng_calendar(cached)
            return cached, validation

    df = fetch_alternative_me_fng(cfg["alternative_me_fng"])
    _save_df(df, cache)
    _save_df(df, csv_cache)
    # Also cache raw JSON for auditability
    raw_json = raw_dir / "fng_alternative_raw.json"
    resp = requests.get(cfg["alternative_me_fng"], timeout=REQUEST_TIMEOUT)
    raw_json.write_text(resp.text, encoding="utf-8")
    log_count(logger, "F&G alternative.me (downloaded)", len(df))
    validation["alternative"] = validate_fng_calendar(df)

    api_key = cfg.get("coinglass_api_key")
    if api_key:
        try:
            cg = fetch_coinglass_fng(api_key, cfg["coinglass"]["base"], cfg["coinglass"]["fng_path"])
            cg_cache = _cache_path(raw_dir, "fng_coinglass")
            _save_df(cg, cg_cache)
            log_count(logger, "F&G CoinGlass (downloaded)", len(cg))
            validation["coinglass"] = validate_fng_calendar(cg.rename(columns={"fng_cg": "fng"}))
            merged_check = df.merge(cg, on="date", how="inner")
            if not merged_check.empty:
                corr = merged_check["fng"].corr(merged_check["fng_cg"])
                disagree = merged_check[merged_check["fng"] != merged_check["fng_cg"]]
                validation["cross_check"] = {
                    "n_overlap": len(merged_check),
                    "correlation": float(corr) if pd.notna(corr) else None,
                    "n_disagree": len(disagree),
                }
                disagree_path = cfg["paths"]["processed"] / "fng_disagreement.csv"
                disagree.to_csv(disagree_path, index=False)
                logger.info(
                    "F&G cross-check: corr=%.4f, disagree_days=%d",
                    corr if pd.notna(corr) else float("nan"),
                    len(disagree),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CoinGlass F&G cross-check skipped: %s", exc)

    return df, validation


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def find_gaps(dates: pd.Series) -> list[str]:
    """Return list of missing calendar dates between min and max."""
    full = pd.date_range(dates.min(), dates.max(), freq="D")
    missing = full.difference(pd.DatetimeIndex(dates))
    return [str(d.date()) for d in missing]


def merge_daily(
    ohlcv: pd.DataFrame,
    fng: pd.DataFrame,
    processed_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inner-join on UTC date. Log gaps and drop them explicitly (no silent ffill)."""
    price = ohlcv.copy()
    fear = fng.copy()
    price["date"] = _normalize_dates(price["date"])
    fear["date"] = _normalize_dates(fear["date"])

    log_count(logger, "OHLCV before merge", len(price))
    log_count(logger, "F&G before merge", len(fear))

    merged = price.merge(fear, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    log_count(logger, "Merged (inner join)", len(merged))

    gaps = find_gaps(merged["date"])
    info: dict[str, Any] = {
        "n_ohlcv": len(price),
        "n_fng": len(fear),
        "n_merged_raw": len(merged),
        "ohlcv_range": (str(price["date"].min().date()), str(price["date"].max().date())),
        "fng_range": (str(fear["date"].min().date()), str(fear["date"].max().date())),
        "merged_range": (
            str(merged["date"].min().date()) if len(merged) else None,
            str(merged["date"].max().date()) if len(merged) else None,
        ),
        "gaps_found": gaps,
        "n_gaps": len(gaps),
        "price_sources": sorted(merged["price_source"].unique().tolist()) if len(merged) else [],
    }

    if gaps:
        # Gaps are already absent from the inner join (no silent forward-fill).
        # Keep both sides of the history; forward metrics use calendar-date lookups
        # so a missing day does not invent prices and does not truncate the sample.
        logger.warning(
            "Merged series has %d calendar gaps (logged, not forward-filled). "
            "Keeping all %d observed rows; horizons resolve by calendar date.",
            len(gaps),
            len(merged),
        )
        for g in gaps[:20]:
            logger.warning("  gap: %s", g)
        info["gap_rows_dropped"] = 0  # never invented, so nothing to drop
        info["used_longest_contiguous"] = False
        info["gaps_kept_as_holes"] = gaps
    else:
        info["gap_rows_dropped"] = 0
        info["used_longest_contiguous"] = False
        info["gaps_kept_as_holes"] = []

    final_gaps = find_gaps(merged["date"])
    if final_gaps:
        logger.warning(
            "Daily index is not fully contiguous (%d holes). "
            "This is expected when F&G or price is missing those dates.",
            len(final_gaps),
        )

    # Add 200d SMA for regime splits (computed on merged sample)
    merged["sma_200"] = merged["close"].rolling(200, min_periods=200).mean()
    merged["above_sma_200"] = merged["close"] > merged["sma_200"]

    out = processed_dir / "daily.parquet"
    _save_df(merged, out)
    _save_df(merged, processed_dir / "daily.csv")
    log_count(logger, "Merged daily written", len(merged))
    info["n_final"] = len(merged)
    info["final_range"] = (str(merged["date"].min().date()), str(merged["date"].max().date()))

    # Persist validation summary
    summary_path = processed_dir / "fetch_validation.json"
    summary_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return merged, info


def run_fetch(cfg: dict[str, Any], refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """End-to-end fetch + merge. Returns (daily_df, validation_info)."""
    ohlcv = fetch_btc_ohlcv(cfg, refresh=refresh)
    fng, fng_val = fetch_fear_greed(cfg, refresh=refresh)
    daily, merge_info = merge_daily(ohlcv, fng, cfg["paths"]["processed"])
    merge_info["fng_validation"] = fng_val
    return daily, merge_info


def print_fetch_summary(info: dict[str, Any]) -> None:
    """Human-readable validation printout for Phase 1 gate."""
    print("\n" + "=" * 60)
    print("PHASE 1 — DATA FETCH VALIDATION")
    print("=" * 60)
    print(f"OHLCV rows:     {info['n_ohlcv']}  range {info['ohlcv_range'][0]} → {info['ohlcv_range'][1]}")
    print(f"F&G rows:       {info['n_fng']}  range {info['fng_range'][0]} → {info['fng_range'][1]}")
    print(f"Merged (raw):   {info['n_merged_raw']}")
    print(f"Gaps found:     {info['n_gaps']}")
    if info["gaps_found"]:
        shown = info["gaps_found"][:15]
        print(f"  first gaps:   {shown}")
    print(f"Final rows:     {info['n_final']}  range {info['final_range'][0]} → {info['final_range'][1]}")
    print(f"Price sources:  {info['price_sources']}")
    print(f"Gaps (no ffill):{info['n_gaps']} — horizons use calendar-date lookups")
    cc = info.get("fng_validation", {}).get("cross_check")
    if cc:
        print(f"F&G cross-check corr={cc.get('correlation')} disagree={cc.get('n_disagree')}")
    print("=" * 60)
    print("Stop here for confirmation before Phase 2 (events/metrics).")
    print("Re-run with: python run_all.py --phase events  (or --phase all)")
    print("=" * 60 + "\n")
