"""
SCRIPT 5 — Financial benchmarking and event analysis
=====================================================

Teck Resources (TECK-B.TO) vs the S&P/TSX Composite (^GSPTSE), 2001–2024.

Data
----
All series are **real** prices downloaded from Yahoo Finance and cached in
``data/raw/`` on first run (see ``src/finance.load_prices``). If a download
fails and no cache exists, the script stops — it never substitutes simulated
data.

Comparisons
-----------
^GSPTSE is a price index: it excludes dividends. Comparing it with a dividend-
adjusted Teck series would mix two bases, so the script reports:

* **Main (price vs price), 2001–2024** — Teck closes adjusted for splits only
  vs ^GSPTSE. Written to ``financial_metrics.csv`` and the other main tables.
* **Check (total return vs total return), 2002–2024** — dividend-adjusted Teck
  vs XIC.TO (an ETF tracking the S&P/TSX Capped Composite, distributions
  reinvested). Written to the ``*_total_return.csv`` tables.

Analyses
--------
1. Annual risk–return metrics (mean, geometric mean, volatility, Sharpe,
   Sortino, VaR/CVaR 95%, maximum drawdown, annual beta, Treynor) plus the
   beta estimated on daily returns over the same years.
2. Difference in mean annual returns (paired and Welch t-tests).
3. Event analysis around the Mount Polley tailings breach (4 Aug 2014, an
   Imperial Metals operation — not a Teck asset):
     a. daily market-model event study (estimation window −260…−11 trading
        days, event window 0…+2) — the standard design (MacKinlay, 1997),
        with ^GSPTSE as the market and, as a robustness check, XIC.TO;
     b. annual mean-adjusted abnormal return, kept as an exploratory check.
   Years and dates are matched by label, never by array position.
4. Figures (main comparison): cumulative growth of 1 CAD, risk–return
   scatter, drawdowns.

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
    load_prices, annual_returns, daily_returns, beta, risk_metrics,
    mean_difference_test, event_study_market_model_daily,
    event_study_annual_mean_adjusted,
)

TABLES = config.OUTPUTS_TABLES
FIGURES = config.OUTPUTS_FIGURES


def compare(stock_px: pd.Series, market_px: pd.Series, years: range, rf: float):
    """Annual metrics, daily beta and mean tests for one stock/benchmark pair."""
    r_s = annual_returns(stock_px).reindex(years).dropna()
    r_m = annual_returns(market_px).reindex(years).dropna()
    r_s, r_m = r_s.align(r_m, join="inner")
    m_s = risk_metrics(r_s, rf, market=r_m)
    m_m = risk_metrics(r_m, rf, market=r_m)
    d = pd.concat({"s": daily_returns(stock_px), "m": daily_returns(market_px)},
                  axis=1, join="inner").dropna()
    d = d[(d.index.year >= r_s.index.min()) & (d.index.year <= r_s.index.max())]
    m_s["beta_daily"] = beta(d["s"], d["m"])
    m_m["beta_daily"] = 1.0
    return r_s, r_m, m_s, m_m, mean_difference_test(r_s, r_m)


def main(refresh: bool = False) -> None:
    print("=" * 78)
    print(f"FINANCIAL BENCHMARKING — {config.TICKER} vs {config.BENCHMARK_NAME}")
    print("=" * 78)

    load = lambda t, adj: load_prices(t, config.PRICE_START, config.PRICE_END,
                                      config.DATA_RAW, refresh=refresh, adjusted=adj)
    teck_price = load(config.TICKER, False)        # splits only (price return)
    teck_total = load(config.TICKER, True)         # dividends reinvested
    idx_px = load(config.BENCHMARK_TICKER, True)   # price index (no dividends to adjust)
    tr_px = load(config.TOTAL_RETURN_BENCHMARK_TICKER, True)

    rf = config.RISK_FREE_RATE
    print(f"Risk-free rate = {rf:.2%}")

    # 1–2. Main comparison: price vs price ---------------------------------------
    years = range(config.START_YEAR, config.END_YEAR + 1)
    r_teck, r_idx, m_teck, m_idx, tests = compare(teck_price, idx_px, years, rf)
    print(f"\n[Main] price vs price: {r_teck.index.min()}–{r_teck.index.max()} "
          f"(n = {len(r_teck)})")
    pd.DataFrame({"Teck (price)": r_teck, config.BENCHMARK_NAME: r_idx}).to_csv(
        TABLES / "annual_returns_teck_vs_index.csv", index_label="year")
    metrics = pd.DataFrame({"Teck Resources (price)": m_teck, config.BENCHMARK_NAME: m_idx})
    metrics.to_csv(TABLES / "financial_metrics.csv", index_label="metric")
    pd.DataFrame([tests]).to_csv(TABLES / "teck_vs_index_mean_tests.csv", index=False)
    print(metrics.round(4).to_string())
    print(f"Paired t = {tests['paired_t']:.3f} (p = {tests['paired_p']:.4f}); "
          f"Welch t = {tests['welch_t']:.3f} (p = {tests['welch_p']:.4f})")

    # 1–2b. Check: total return vs total return ----------------------------------
    years_tr = range(config.TOTAL_RETURN_START_YEAR, config.END_YEAR + 1)
    r_tt, r_tr, m_tt, m_tr, tests_tr = compare(teck_total, tr_px, years_tr, rf)
    print(f"\n[Check] total return vs total return: {r_tt.index.min()}–{r_tt.index.max()} "
          f"(n = {len(r_tt)})")
    pd.DataFrame({"Teck (total return)": r_tt, config.TOTAL_RETURN_BENCHMARK_NAME: r_tr}).to_csv(
        TABLES / "annual_returns_total_return.csv", index_label="year")
    metrics_tr = pd.DataFrame({"Teck Resources (total return)": m_tt,
                               config.TOTAL_RETURN_BENCHMARK_NAME: m_tr})
    metrics_tr.to_csv(TABLES / "financial_metrics_total_return.csv", index_label="metric")
    pd.DataFrame([tests_tr]).to_csv(TABLES / "mean_tests_total_return.csv", index=False)
    print(metrics_tr.round(4).to_string())
    print(f"Paired t = {tests_tr['paired_t']:.3f} (p = {tests_tr['paired_p']:.4f}); "
          f"Welch t = {tests_tr['welch_t']:.3f} (p = {tests_tr['welch_p']:.4f})")

    # 3. Event analysis ------------------------------------------------------------
    # Over a three-day window dividends are immaterial; the total-return Teck
    # series is used so that an ex-dividend day could not show up as a loss.
    daily, daily_table = event_study_market_model_daily(
        teck_total, idx_px, config.EVENT_DATE, event_name=config.EVENT_NAME)
    daily_xic, _ = event_study_market_model_daily(
        teck_total, tr_px, config.EVENT_DATE,
        event_name=f"{config.EVENT_NAME} — market: {config.TOTAL_RETURN_BENCHMARK_TICKER}")
    annual = event_study_annual_mean_adjusted(
        annual_returns(teck_total).reindex(years).dropna(), config.EVENT_YEAR,
        event_name=config.EVENT_NAME)
    ev = pd.DataFrame([daily.as_dict(), daily_xic.as_dict(), annual.as_dict()])
    ev.to_csv(TABLES / "event_study_mount_polley.csv", index=False)
    daily_table.to_csv(TABLES / "event_study_mount_polley_daily_ar.csv", index_label="date")
    print("\nEvent analysis:")
    print(f"Day 0 = {daily_table.index[0].date()} (first trading day on or after "
          f"{config.EVENT_DATE})")
    print(ev[["method", "event", "event_window", "beta", "abnormal_return_t0", "car",
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
    ax.set_title(f"Cumulative price returns: Teck Resources vs {config.BENCHMARK_NAME} "
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
    ax.set_title("Risk vs return (price returns)")
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
