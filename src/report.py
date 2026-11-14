"""Generate summary.md, CSVs, and PNG charts into reports/."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("btc_fng_study")


def _footnote(ax: plt.Axes, daily: pd.DataFrame) -> None:
    src = ",".join(sorted(daily["price_source"].dropna().unique().astype(str)))
    d0 = daily["date"].min().date()
    d1 = daily["date"].max().date()
    ax.annotate(
        f"Sources: {src} + alternative.me F&G | {d0} → {d1}",
        xy=(0.0, -0.12),
        xycoords="axes fraction",
        fontsize=8,
        color="#555555",
    )


def chart_price_episodes(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["date"], daily["close"], color="#1a1a1a", lw=1.0, label="BTC close")
    ax.set_yscale("log")
    colors = {12: "#c0392b", 15: "#e67e22"}
    for thr, color in colors.items():
        ep = episodes[(episodes["threshold"] == thr)]
        if ep.empty:
            continue
        ax.scatter(
            ep["signal_date"],
            ep["entry_price"],
            s=36,
            c=color,
            marker="v",
            label=f"Episode F&G≤{thr}",
            zorder=5,
            alpha=0.85,
        )
        for _, r in ep.iterrows():
            ax.axvspan(r["signal_date"], r["end_date"], color=color, alpha=0.08, lw=0)
    ax.set_title("BTC price with Extreme Fear episodes")
    ax.set_ylabel("BTCUSDT (log)")
    ax.legend(loc="upper left")
    _footnote(ax, daily)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def chart_return_boxplots(
    per_signal: pd.DataFrame,
    baseline: dict[int, np.ndarray],
    horizons: list[int],
    out: Path,
    daily: pd.DataFrame,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    data = []
    labels = []
    positions = []
    pos = 0
    for h in horizons:
        col = f"ret_{h}d"
        sig = per_signal[col].dropna().values if col in per_signal else np.array([])
        base = baseline.get(h, np.array([]))
        data.extend([sig, base])
        labels.extend([f"S{h}", f"B{h}"])
        positions.extend([pos, pos + 0.7])
        pos += 2.0
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#3498db" if i % 2 == 0 else "#95a5a6")
        patch.set_alpha(0.7)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks([p + 0.35 for p in range(0, int(pos), 2)])
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Forward return")
    ax.set_title(title)
    ax.legend(
        [bp["boxes"][0], bp["boxes"][1]],
        ["Signal", "Baseline"],
        loc="upper left",
    )
    _footnote(ax, daily)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def chart_win_rates(
    fwd: pd.DataFrame,
    ci: pd.DataFrame,
    out: Path,
    daily: pd.DataFrame,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sub = fwd.sort_values("horizon")
    ax.plot(sub["horizon"], sub["win_rate"], "o-", color="#2980b9", label="Signal")
    ax.plot(sub["horizon"], sub["baseline_win_rate"], "s--", color="#7f8c8d", label="Baseline")
    if not ci.empty:
        c = ci.sort_values("horizon")
        ax.errorbar(
            c["horizon"],
            c["win_rate"],
            yerr=[c["win_rate"] - c["win_rate_ci_lo"], c["win_rate_ci_hi"] - c["win_rate"]],
            fmt="none",
            ecolor="#2980b9",
            capsize=4,
        )
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    _footnote(ax, daily)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def chart_histogram(series: pd.Series, out: Path, title: str, xlabel: str, daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = series.dropna()
    ax.hist(vals, bins=min(20, max(5, len(vals) // 2 or 5)), color="#34495e", edgecolor="white", alpha=0.85)
    if len(vals):
        ax.axvline(vals.median(), color="#e74c3c", ls="--", label=f"median={vals.median():.1f}")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    _footnote(ax, daily)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def chart_avg_path(path: pd.DataFrame, out: Path, daily: pd.DataFrame, title: str) -> None:
    if path.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(path["day"], path["p25"], path["p75"], color="#3498db", alpha=0.2, label="25–75%")
    ax.plot(path["day"], path["mean"], color="#2980b9", lw=1.5, label="Signal mean")
    ax.plot(path["day"], path["median"], color="#1a5276", lw=1.5, ls="--", label="Signal median")
    if "baseline_median" in path.columns:
        ax.plot(path["day"], path["baseline_median"], color="#7f8c8d", lw=1.2, label="Baseline median")
    ax.axhline(0, color="#333", lw=0.7)
    ax.set_xlabel("Days after entry")
    ax.set_ylabel("Cumulative return")
    ax.set_title(title)
    ax.legend()
    _footnote(ax, daily)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def chart_equity(equity: pd.DataFrame, out: Path, daily: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, g in equity.groupby("strategy"):
        ax.plot(g["date"], g["equity"], lw=1.2, label=name)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Equity (log)")
    ax.legend(fontsize=8, loc="upper left")
    _footnote(ax, daily)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{100 * x:.1f}%"


def _fmt_num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:.{digits}f}"


def build_summary_md(
    cfg: dict[str, Any],
    daily: pd.DataFrame,
    episode_counts: dict[str, int],
    headline: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> str:
    """Plain-English findings first, then tables."""
    lines: list[str] = []
    lines.append("# BTC Extreme Fear Buy-Signal Study")
    lines.append("")
    d0, d1 = daily["date"].min().date(), daily["date"].max().date()
    lines.append(f"**Sample:** {d0} → {d1} ({len(daily)} daily bars, inner-join of BTC OHLCV + alternative.me F&G).")
    lines.append("")
    lines.append("## Findings (read this first)")
    lines.append("")
    counts_txt = ", ".join(f"{k}: {v}" for k, v in episode_counts.items())
    lines.append(
        f"- **Small sample warning:** Episode counts are low "
        f"({counts_txt}). Overlapping signal-day returns are *not* independent; "
        "significance claims use episodes only."
    )

    # Pull key stats from headline bundle if present
    fwd = headline.get("forward_returns")
    dd = headline.get("drawdowns")
    ttb = headline.get("time_to_bottom")
    boot = headline.get("bootstrap")
    ci = headline.get("ci")

    if fwd is not None and not fwd.empty:
        for h in (30, 90, 180, 365):
            row = fwd[fwd["horizon"] == h]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(
                f"- **Success rate @ {h}d:** win rate {_fmt_pct(r['win_rate'])} "
                f"(baseline {_fmt_pct(r['baseline_win_rate'])}, excess {_fmt_pct(r['excess_win_rate'])}); "
                f"median return {_fmt_pct(r['median'])} (baseline {_fmt_pct(r.get('baseline_median', float('nan')))}, "
                f"excess {_fmt_pct(r['excess_median'])}); n={int(r['n'])} [{r['sample_flag']}]."
            )
        lines.append(
            "- **Returns:** See forward-return tables for the full distribution "
            "(mean/median/std/percentiles/skew) across all horizons — no single 'best' horizon is cherry-picked."
        )

    if ttb is not None and not ttb.empty:
        row90 = ttb[ttb["window"] == 90]
        if not row90.empty:
            r = row90.iloc[0]
            lines.append(
                f"- **Time to bottom (90d window):** median { _fmt_num(r['ttb_median'], 1) } days, "
                f"mean {_fmt_num(r['ttb_mean'], 1)}; "
                f"{int(r['n_bottom_at_window_end'])} entries had the low on the last day of the window "
                f"(true bottom may be later)."
            )
        loc = ttb[ttb["window"] == "local_plus20pct"]
        if not loc.empty:
            r = loc.iloc[0]
            lines.append(
                f"- **Local bottom (+20% recovery rule):** median {_fmt_num(r['ttb_median'], 1)} days "
                f"(n={int(r['n'])})."
            )

    if dd is not None and not dd.empty:
        row90 = dd[dd["window"] == 90]
        if not row90.empty:
            r = row90.iloc[0]
            lines.append(
                f"- **Further drawdown (90d):** {_fmt_pct(r['pct_lower_low'])} of entries made a lower low; "
                f"median MAE {_fmt_pct(r['mae_median'])}, p90 {_fmt_pct(r['mae_p90'])}, "
                f"worst {_fmt_pct(r['mae_worst'])}; "
                f"fell ≥10%/20%/30%: {_fmt_pct(r.get('pct_fell_10', float('nan')))} / "
                f"{_fmt_pct(r.get('pct_fell_20', float('nan')))} / {_fmt_pct(r.get('pct_fell_30', float('nan')))}."
            )

    if boot is not None and not boot.empty:
        bits = []
        for _, r in boot.iterrows():
            bits.append(
                f"{int(r['horizon'])}d p={_fmt_num(r['empirical_pvalue_onesided'], 3)} "
                f"(pctile {_fmt_num(r['signal_percentile_in_random'], 2)})"
            )
        lines.append(
            "- **Edge vs noise (random-entry bootstrap on episodes):** "
            + "; ".join(bits)
            + ". Low p-values suggest the median forward return is unusual vs year-matched random entries; "
            "high p-values mean it looks like noise given BTC's upward drift."
        )

    if ci is not None and not ci.empty:
        lines.append(
            "- **95% bootstrap CIs (episodes)** for win rate and median return at 30/90/180/365d "
            "are in the CI table below — interpret cautiously given small n."
        )

    lines.append(
        "- **Entry timing:** Base case enters at the **open of day t+1** after F&G prints; "
        "sensitivity uses close of day t. Both are reported."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Thresholds: {cfg['thresholds']}")
    lines.append(f"- Horizons: {cfg['horizons']}")
    lines.append(f"- Cooldowns: {cfg['cooldowns']} (headline default N={cfg['default_cooldown']})")
    lines.append(f"- Fee: {cfg['fee_per_trade']}")
    lines.append(f"- Entry timing (base): `{cfg['entry_timing']}`")
    lines.append("")

    def _table_section(title: str, df: pd.DataFrame | None) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if df is None or df.empty:
            lines.append("_No data._")
            lines.append("")
            return
        # Keep markdown tables readable
        show = df.copy()
        if len(show.columns) > 14:
            prefer = [
                c
                for c in [
                    "label",
                    "horizon",
                    "window",
                    "n",
                    "sample_flag",
                    "win_rate",
                    "baseline_win_rate",
                    "excess_win_rate",
                    "median",
                    "baseline_median",
                    "excess_median",
                    "mean",
                    "std",
                    "mean_over_std",
                    "mae_median",
                    "pct_lower_low",
                    "ttb_median",
                    "name",
                    "cagr",
                    "max_drawdown",
                    "pct_time_invested",
                    "n_trades",
                    "empirical_pvalue_onesided",
                    "median_ci_lo",
                    "median_ci_hi",
                    "win_rate_ci_lo",
                    "win_rate_ci_hi",
                ]
                if c in show.columns
            ]
            show = show[prefer]
        lines.append(show.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    _table_section("Forward returns by horizon", tables.get("forward_returns_by_horizon"))
    _table_section("Drawdown / MAE stats", tables.get("drawdown_stats"))
    _table_section("Time to bottom / recovery", tables.get("time_to_bottom"))
    _table_section("Bootstrap / empirical p-values", tables.get("bootstrap"))
    _table_section("95% CIs (episodes)", tables.get("ci"))
    _table_section("Backtests", tables.get("backtests"))
    lines.append(
        "_Return column: `cagr` is time-weighted for buy & hold, lump-sum and ladder (all capital "
        "counted from day one). For DCA rows it is the money-weighted IRR, and DCA equity curves "
        "show value per $1 invested, so contributions are never counted as gains._"
    )
    lines.append("")
    _table_section(
        "Ladder vs all-in, per episode (unfilled tranches held as cash)", tables.get("ladder_vs_lump")
    )
    _table_section("Regime splits", tables.get("regime_splits"))
    _table_section("F&G recovery after signal", tables.get("fng_recovery"))

    lines.append("## Charts")
    lines.append("")
    lines.append("PNG charts are saved alongside this file in `reports/`.")
    lines.append("")
    return "\n".join(lines)


def write_reports(
    cfg: dict[str, Any],
    daily: pd.DataFrame,
    events: dict[str, pd.DataFrame],
    metric_bundles: dict[str, dict[str, pd.DataFrame]],
    backtest_summary: pd.DataFrame,
    backtest_equity: pd.DataFrame,
    regime_splits: pd.DataFrame,
    ladder_vs_lump: pd.DataFrame | None = None,
) -> str:
    """Write all CSVs, charts, summary.md. Returns summary markdown text."""
    reports: Path = cfg["paths"]["reports"]
    reports.mkdir(parents=True, exist_ok=True)

    # --- CSVs ---
    ep = events["episodes"]
    sig = events["signal_days"]
    # Headline episodes: default cooldown + base timing, both thresholds
    default_n = cfg["default_cooldown"]
    base_timing = cfg["entry_timing"]
    ep_head = ep[(ep["cooldown"] == default_n) & (ep["entry_timing"] == base_timing)].copy()
    sig_head = sig[sig["entry_timing"] == base_timing].copy()

    ep_head.to_csv(reports / "episodes.csv", index=False)
    sig_head.to_csv(reports / "signal_days.csv", index=False)
    # Full grids also useful
    ep.to_csv(reports / "episodes_full.csv", index=False)
    sig.to_csv(reports / "signal_days_full.csv", index=False)

    fwd_frames = []
    dd_frames = []
    ttb_frames = []
    boot_frames = []
    ci_frames = []
    fng_frames = []
    path_frames = []

    for label, bundle in metric_bundles.items():
        if not bundle["forward_returns"].empty:
            fwd_frames.append(bundle["forward_returns"])
        if not bundle["drawdowns"].empty:
            dd_frames.append(bundle["drawdowns"])
        if not bundle["time_to_bottom"].empty:
            ttb_frames.append(bundle["time_to_bottom"])
        if bundle.get("bootstrap") is not None and not bundle["bootstrap"].empty:
            boot_frames.append(bundle["bootstrap"])
        if not bundle["ci"].empty:
            ci_frames.append(bundle["ci"])
        if not bundle["fng_recovery"].empty:
            fng_frames.append(bundle["fng_recovery"])
        if bundle.get("avg_path") is not None and not bundle["avg_path"].empty:
            path_frames.append(bundle["avg_path"])

    fwd_all = pd.concat(fwd_frames, ignore_index=True) if fwd_frames else pd.DataFrame()
    dd_all = pd.concat(dd_frames, ignore_index=True) if dd_frames else pd.DataFrame()
    ttb_all = pd.concat(ttb_frames, ignore_index=True) if ttb_frames else pd.DataFrame()
    boot_all = pd.concat(boot_frames, ignore_index=True) if boot_frames else pd.DataFrame()
    ci_all = pd.concat(ci_frames, ignore_index=True) if ci_frames else pd.DataFrame()
    fng_all = pd.concat(fng_frames, ignore_index=True) if fng_frames else pd.DataFrame()
    path_all = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()

    fwd_all.to_csv(reports / "forward_returns_by_horizon.csv", index=False)
    dd_all.to_csv(reports / "drawdown_stats.csv", index=False)
    ttb_all.to_csv(reports / "time_to_bottom.csv", index=False)
    backtest_summary.to_csv(reports / "backtests.csv", index=False)
    regime_splits.to_csv(reports / "regime_splits.csv", index=False)
    if not boot_all.empty:
        boot_all.to_csv(reports / "bootstrap_pvalues.csv", index=False)
    if not ci_all.empty:
        ci_all.to_csv(reports / "confidence_intervals.csv", index=False)
    if not path_all.empty:
        path_all.to_csv(reports / "average_paths.csv", index=False)
    if not fng_all.empty:
        fng_all.to_csv(reports / "fng_recovery.csv", index=False)
    if not backtest_equity.empty:
        backtest_equity.to_csv(reports / "backtest_equity.csv", index=False)
    if ladder_vs_lump is not None and not ladder_vs_lump.empty:
        ladder_vs_lump.to_csv(reports / "ladder_vs_lump.csv", index=False)

    # Headline key for summary: primary threshold 12, episodes, default N, base timing
    primary_label = f"episodes_thr12_N{default_n}_{base_timing}"
    # Fallbacks
    if primary_label not in metric_bundles:
        keys = [k for k in metric_bundles if k.startswith("episodes_thr12")]
        primary_label = keys[0] if keys else next(iter(metric_bundles), "")
    headline = metric_bundles.get(primary_label, {})

    episode_counts = {
        f"≤{thr} N={default_n}": int(
            ((ep_head["threshold"] == thr)).sum()
        )
        for thr in cfg["thresholds"]
    }

    tables = {
        "forward_returns_by_horizon": fwd_all,
        "drawdown_stats": dd_all,
        "time_to_bottom": ttb_all,
        "bootstrap": boot_all,
        "ci": ci_all[ci_all["label"].astype(str).str.startswith("episodes")] if not ci_all.empty else ci_all,
        "backtests": backtest_summary,
        "ladder_vs_lump": ladder_vs_lump,
        "regime_splits": regime_splits,
        "fng_recovery": fng_all,
    }
    md = build_summary_md(cfg, daily, episode_counts, headline, tables)
    (reports / "summary.md").write_text(md, encoding="utf-8")

    # --- Charts ---
    from src.metrics import unconditional_forward_returns

    baseline = unconditional_forward_returns(daily, cfg["horizons"])

    chart_price_episodes(
        daily,
        ep_head,
        reports / "price_with_episodes.png",
    )

    # Use primary headline per-entry for charts
    per = headline.get("per_entry", pd.DataFrame())
    if not per.empty:
        chart_return_boxplots(
            per,
            baseline,
            cfg["horizons"],
            reports / "returns_boxplots.png",
            daily,
            f"Forward returns: episodes F&G≤12 (N={default_n}) vs baseline",
        )
        hf = headline.get("forward_returns", pd.DataFrame())
        hci = headline.get("ci", pd.DataFrame())
        chart_win_rates(
            hf,
            hci,
            reports / "win_rate_by_horizon.png",
            daily,
            f"Win rate by horizon: episodes F&G≤12 (N={default_n})",
        )
        if "ttb_90d" in per.columns:
            chart_histogram(
                per["ttb_90d"],
                reports / "days_to_bottom_90d.png",
                "Days to bottom (90d window)",
                "Days after entry",
                daily,
            )
        if "mae_90d" in per.columns:
            chart_histogram(
                per["mae_90d"],
                reports / "mae_90d.png",
                "MAE distribution (90d window)",
                "Max adverse excursion",
                daily,
            )
        path = headline.get("avg_path", pd.DataFrame())
        chart_avg_path(
            path,
            reports / "average_path_180d.png",
            daily,
            "Average path 0–180d after entry (episodes F&G≤12)",
        )

    if not backtest_equity.empty:
        # Chart for threshold 12 if present
        eq12 = backtest_equity[backtest_equity["threshold"] == 12]
        if eq12.empty:
            eq12 = backtest_equity
        chart_equity(
            eq12,
            reports / "equity_curves.png",
            daily,
            "Strategy equity curves vs buy & hold",
        )

    logger.info("Wrote reports to %s", reports)
    return md
