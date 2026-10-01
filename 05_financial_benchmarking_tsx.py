"""
SCRIPT 5 — Financial benchmarking and event analysis
=====================================================

Teck Resources (TECK-B.TO) vs the S&P/TSX Composite (^GSPTSE), 2001–2024.

Data
----
Both series are **real** adjusted prices downloaded from Yahoo Finance and
cached in ``data/raw/`` on first run (see ``src/finance.load_prices``). If the
download fails and no cache exists, the script stops — it never substitutes
simulated data.

Analyses
--------
1. Annual risk–return metrics for Teck and the index (mean, geometric mean,
   volatility, Sharpe, Sortino, VaR/CVaR 95%, maximum drawdown, beta, Treynor).
2. Difference in mean annual returns (paired and Welch t-tests).
3. Event analysis around the Mount Polley tailings breach (4 Aug 2014, an
   Imperial Metals operation — not a Teck asset):
     a. daily market-model event study (estimation window −260…−11 trading
        days, event window 0…+2) — the standard design (MacKinlay, 1997);
     b. annual mean-adjusted abnormal return, kept as an exploratory check.
   Years and dates are matched by label, never by array position.
4. Figures: cumulative growth of 1 CAD, risk–return scatter, drawdowns.

Author: Cesar Yoshio Machado Pedroza
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import config
from finance import (
    load_prices, annual_returns, risk_metrics, mean_difference_test,
    event_study_market_model_daily, event_study_annual_mean_adjusted,
)

TABLES = config.OUTPUTS_TABLES
FIGURES = config.OUTPUTS_FIGURES


def main(refresh: bool = False) -> None:
    print("=" * 78)
    print(f"FINANCIAL BENCHMARKING — {config.TICKER} vs {config.BENCHMARK_NAME}")
    print("=" * 78)

    teck_px = load_prices(config.TICKER, config.PRICE_START, config.PRICE_END,
                          config.DATA_RAW, refresh=refresh)
    idx_px = load_prices(config.BENCHMARK_TICKER, config.PRICE_START, config.PRICE_END,
                         config.DATA_RAW, refresh=refresh)

    years = range(config.START_YEAR, config.END_YEAR + 1)
    r_teck = annual_returns(teck_px).reindex(years).dropna()
    r_idx = annual_returns(idx_px).reindex(years).dropna()
    r_teck, r_idx = r_teck.align(r_idx, join="inner")
    print(f"Annual returns: {r_teck.index.min()}–{r_teck.index.max()} (n = {len(r_teck)})")

    pd.DataFrame({"Teck": r_teck, config.BENCHMARK_NAME: r_idx}).to_csv(
        TABLES / "annual_returns_teck_vs_index.csv", index_label="year")

    # 1. Metrics ---------------------------------------------------------------
    rf = config.RISK_FREE_RATE
    m_teck = risk_metrics(r_teck, rf, market=r_idx)
    m_idx = risk_metrics(r_idx, rf, market=r_idx)
    metrics = pd.DataFrame({"Teck Resources": m_teck, config.BENCHMARK_NAME: m_idx})
    metrics.to_csv(TABLES / "financial_metrics.csv", index_label="metric")
    print(f"\nRisk-free rate = {rf:.2%}\n")
    print(metrics.round(4).to_string())

    # 2. Mean-difference tests ---------------------------------------------------
    tests = mean_difference_test(r_teck, r_idx)
    pd.DataFrame([tests]).to_csv(TABLES / "teck_vs_index_mean_tests.csv", index=False)
    print(f"\nPaired t = {tests['paired_t']:.3f} (p = {tests['paired_p']:.4f}); "
          f"Welch t = {tests['welch_t']:.3f} (p = {tests['welch_p']:.4f})")

    # 3. Event analysis ------------------------------------------------------------
    daily, daily_table = event_study_market_model_daily(
        teck_px, idx_px, config.EVENT_DATE, event_name=config.EVENT_NAME)
    annual = event_study_annual_mean_adjusted(
        r_teck, config.EVENT_YEAR, event_name=config.EVENT_NAME)
    ev = pd.DataFrame([daily.as_dict(), annual.as_dict()])
    ev.to_csv(TABLES / "event_study_mount_polley.csv", index=False)
    daily_table.to_csv(TABLES / "event_study_mount_polley_daily_ar.csv", index_label="date")
    print("\nEvent analysis:")
    print(ev[["method", "event_window", "abnormal_return_t0", "car",
              "t_stat_car", "p_value_car"]].round(4).to_string(index=False))

    # 4. Figures ---------------------------------------------------------------------
    growth = pd.DataFrame({
        "Teck Resources": (1 + r_teck).cumprod(),
        config.BENCHMARK_NAME: (1 + r_idx).cumprod(),
    })
    start = pd.DataFrame({c: [1.0] for c in growth.columns}, index=[growth.index.min() - 1])
    growth = pd.concat([start, growth])  # value 1.0 at the end of the base year

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(growth.index, growth["Teck Resources"], lw=2, label="Teck Resources")
    ax.plot(growth.index, growth[config.BENCHMARK_NAME], lw=2, ls="--",
            label=config.BENCHMARK_NAME)
    # x-axis = year-end points; place the event at its date within the year
    ev_date = pd.Timestamp(config.EVENT_DATE)
    ev_x = ev_date.year - 1 + ev_date.dayofyear / 365.25
    ax.axvline(ev_x, color="red", ls=":", lw=1.5,
               label="Mount Polley (2014) — Imperial Metals")
    ax.set_yscale("log")
    ax.set_xlabel("Year-end")
    ax.set_ylabel("Growth of 1 CAD (log scale)")
    ax.set_title(f"Cumulative returns: Teck Resources vs {config.BENCHMARK_NAME} "
                 f"({r_teck.index.min()}–{r_teck.index.max()})")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "cumulative_returns_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, m in [("Teck Resources", m_teck), (config.BENCHMARK_NAME, m_idx)]:
        ax.scatter(m["volatility"] * 100, m["mean_return"] * 100, s=150, edgecolor="black",
                   label=name, zorder=3)
    ax.set_xlabel("Volatility (annual, %)")
    ax.set_ylabel("Mean annual return (%)")
    ax.set_title("Risk vs return")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "risk_return_scatter.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name in growth.columns:
        dd = growth[name] / growth[name].cummax() - 1
        ax.plot(dd.index, dd * 100, label=name)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdowns from previous year-end peak")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "drawdowns.png", dpi=300)
    plt.close(fig)

    print("\nSaved tables to outputs/tables and figures to outputs/figures.")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
