# BTC Extreme Fear Buy-Signal Study

Reproducible Python research pipeline that asks: **does buying Bitcoin when the Crypto Fear & Greed Index hits extreme levels (≤12 / ≤15) beat buying on a random day?**

It measures win rates, forward returns, further drawdowns, time-to-bottom, and simple strategy backtests (buy & hold, lump-sum, DCA-during-fear, scale-in ladder) — always against an unconditional baseline, because BTC trended up over most of the sample.

**Start here:** [`reports/summary.md`](reports/summary.md) (plain-English findings + tables). Charts are in [`reports/`](reports/).

## Key caveats

- Extreme-fear **episodes are rare** (~16–19 at the headline settings). Treat results as suggestive, not settled.
- Overlapping “every fear day” returns are autocorrelated; significance claims use **episodes**.
- Fear & Greed partly reflects price/volatility itself — compare mentally to a plain “buy the dip” rule.
- Not financial advice.

## Quick start

```bash
python3 -m pip install -r requirements.txt

# Phase 1 — download + validate (stops for confirmation)
python3 run_all.py --phase fetch

# Full pipeline
python3 run_all.py --phase all

# Or step by step
python3 run_all.py --phase events
python3 run_all.py --phase backtest
python3 run_all.py --phase report
```

Use `--refresh` to force re-download of API caches.

Optional: copy `.env.example` to `.env` and set `COINGLASS_API_KEY` for Binance fallback OHLC and an F&G cross-check.

## Data

| Series | Source |
|--------|--------|
| BTC daily OHLCV | Binance public market-data mirror (`data-api.binance.vision`), then Binance.US, then CoinGlass if keyed |
| Fear & Greed | [alternative.me](https://alternative.me/crypto/fear-and-greed-index/) (full history from ~2018-02) |

Cached downloads live under `data/raw/`; the merged daily panel is `data/processed/daily.parquet`.

## Project layout

```
config.yaml      # thresholds, horizons, fees, cooldowns
run_all.py       # --phase fetch|events|backtest|report|all
src/             # fetch, events, metrics, backtest, report
tests/           # look-ahead, forward-return math, episode detection
reports/         # summary.md, CSVs, PNG charts
```

## Tests

```bash
python3 -m pytest -q
```

## License / attribution

Research code for educational use. Fear & Greed data courtesy of alternative.me; price data from Binance public endpoints. Cite those sources if you reuse the charts or numbers.
