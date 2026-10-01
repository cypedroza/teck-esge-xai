"""
Financial metrics and event-study utilities.

All functions are pure (no I/O) except ``load_prices``, which downloads
adjusted prices from Yahoo Finance and caches them under ``data/raw``.

Design rules
------------
* Returns are always handled as ``pd.Series`` **indexed by year** (annual) or
  by date (daily). Positional indexing is never used to locate an event, which
  removes the off-by-one risk of mixing a full table index with a series that
  had missing values dropped.
* Nothing is ever simulated. If market data cannot be obtained the code raises
  an error instead of silently substituting random numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ════════════════════════════════════════════════════════════════════════════
# DATA ACQUISITION
# ════════════════════════════════════════════════════════════════════════════

def load_prices(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path,
    refresh: bool = False,
) -> pd.Series:
    """Daily dividend/split-adjusted close prices for ``ticker``.

    The first successful download is cached as CSV so that later runs (and
    other researchers) use exactly the same series. Delete the cache file or
    pass ``refresh=True`` to download again.

    Raises
    ------
    RuntimeError
        If no cached file exists and the download fails or returns no data.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("^", "IDX_").replace(".", "_")
    cache_file = cache_dir / f"prices_{safe}_{start}_{end}.csv"

    if cache_file.exists() and not refresh:
        s = pd.read_csv(cache_file, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = ticker
        return s

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("yfinance is required: pip install yfinance") from exc

    hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if hist is None or hist.empty:
        raise RuntimeError(
            f"No price data returned for {ticker}. Check the ticker and your "
            "internet connection. The pipeline does not fall back to simulated data."
        )
    s = hist["Close"].copy()
    # Remove timezone so the cache round-trips cleanly
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s.name = ticker
    s.to_frame().to_csv(cache_file)
    return s


def annual_returns(prices: pd.Series) -> pd.Series:
    """Calendar-year simple returns (decimal) from daily prices.

    Uses the last available close of each year; the first year is dropped
    because it has no previous year-end close.
    """
    year_end = prices.groupby(prices.index.year).last()
    r = year_end.pct_change().dropna()
    r.index = r.index.astype(int)
    r.index.name = "year"
    return r


def daily_returns(prices: pd.Series) -> pd.Series:
    """Daily simple returns (decimal)."""
    return prices.pct_change().dropna()


# ════════════════════════════════════════════════════════════════════════════
# RISK–RETURN METRICS
# ════════════════════════════════════════════════════════════════════════════

def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the cumulative wealth path (decimal, ≤ 0).

    The wealth path starts at 1.0 *before* the first return, so a loss in the
    very first period is counted.
    """
    wealth = np.concatenate([[1.0], (1.0 + returns.to_numpy()).cumprod()])
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1.0).min())


def downside_deviation(returns: pd.Series, target: float = 0.0) -> float:
    """Root-mean-square of shortfalls below ``target`` (Sortino & Price, 1994)."""
    shortfall = np.minimum(returns.to_numpy() - target, 0.0)
    return float(np.sqrt(np.mean(shortfall ** 2)))


def beta(asset: pd.Series, market: pd.Series) -> float:
    """OLS beta of ``asset`` on ``market`` over their common index."""
    a, m = asset.align(market, join="inner")
    if len(a) < 3:
        raise ValueError("Need at least 3 overlapping observations to estimate beta.")
    return float(np.cov(a, m, ddof=1)[0, 1] / np.var(m, ddof=1))


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def risk_metrics(
    returns: pd.Series,
    risk_free: float,
    market: Optional[pd.Series] = None,
    alpha_level: float = 0.05,
) -> dict:
    """Standard risk–return summary for a return series (decimal units).

    Volatility uses the sample standard deviation (ddof = 1). VaR/CVaR are
    historical (empirical quantile), reported as returns (negative = loss).
    """
    r = returns.dropna()
    mean = float(r.mean())
    vol = float(r.std(ddof=1))
    var_q = float(np.quantile(r, alpha_level))
    tail = r[r <= var_q]
    out = {
        "n_periods": int(len(r)),
        "mean_return": mean,
        "geometric_mean_return": float((1 + r).prod() ** (1 / len(r)) - 1),
        "volatility": vol,
        "sharpe": (mean - risk_free) / vol if vol > 0 else np.nan,
        "sortino": _safe_div(mean - risk_free, downside_deviation(r, risk_free)),
        f"VaR_{int((1 - alpha_level) * 100)}": var_q,
        f"CVaR_{int((1 - alpha_level) * 100)}": float(tail.mean()),
        "max_drawdown": max_drawdown(r),
    }
    if market is not None:
        b = beta(r, market)
        out["beta"] = b
        out["treynor"] = (mean - risk_free) / b if b != 0 else np.nan
    return out


def mean_difference_test(a: pd.Series, b: pd.Series) -> dict:
    """Welch t-test and paired t-test for a difference in mean returns.

    Annual returns of a stock and its benchmark are measured over the same
    years, so the paired test is the more appropriate one; Welch is reported
    for comparison with an independent-samples reading.
    """
    a_, b_ = a.align(b, join="inner")
    welch = stats.ttest_ind(a_, b_, equal_var=False)
    paired = stats.ttest_rel(a_, b_)
    return {
        "n": int(len(a_)),
        "welch_t": float(welch.statistic),
        "welch_p": float(welch.pvalue),
        "paired_t": float(paired.statistic),
        "paired_p": float(paired.pvalue),
    }


# ════════════════════════════════════════════════════════════════════════════
# EVENT STUDIES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class EventStudyResult:
    method: str
    event: str
    estimation_window: str
    event_window: str
    n_estimation: int
    alpha: float
    beta: float
    abnormal_return_t0: float
    car: float
    t_stat_car: float
    p_value_car: float

    def as_dict(self) -> dict:
        return asdict(self)


def event_study_market_model_daily(
    stock_prices: pd.Series,
    market_prices: pd.Series,
    event_date: str,
    estimation_window: Tuple[int, int] = (-260, -11),
    event_window: Tuple[int, int] = (0, 2),
    event_name: str = "event",
) -> Tuple[EventStudyResult, pd.DataFrame]:
    """Classic market-model event study on daily returns (MacKinlay, 1997).

    * Day 0 is the first trading day on or after ``event_date``.
    * Windows are in trading days relative to day 0 (inclusive bounds).
      The default estimation window (−260, −11) gives 250 trading days and a
      10-day gap that keeps the event out of the estimation sample.
    * Expected return:  E[R_i,t] = α + β · R_m,t, estimated by OLS.
    * CAR test:  t = CAR / (σ_ε · √L), with L the event-window length and
      σ_ε the residual standard deviation of the estimation regression
      (df = N − 2).

    Returns the summary and a day-by-day table of AR and CAR.
    """
    rs = daily_returns(stock_prices)
    rm = daily_returns(market_prices)
    df = pd.concat({"stock": rs, "market": rm}, axis=1, join="inner").dropna()
    if df.empty:
        raise ValueError("Stock and market series have no overlapping dates.")

    pos0 = int(df.index.searchsorted(pd.Timestamp(event_date)))
    if pos0 >= len(df):
        raise ValueError(f"Event date {event_date} is after the last available date.")

    e0, e1 = estimation_window
    w0, w1 = event_window
    if pos0 + e0 < 0 or pos0 + w1 >= len(df):
        raise ValueError("Not enough data around the event for the requested windows.")

    est = df.iloc[pos0 + e0: pos0 + e1 + 1]
    evt = df.iloc[pos0 + w0: pos0 + w1 + 1]

    slope, intercept, *_ = stats.linregress(est["market"], est["stock"])
    resid = est["stock"] - (intercept + slope * est["market"])
    sigma = float(np.sqrt((resid ** 2).sum() / (len(est) - 2)))

    ar = evt["stock"] - (intercept + slope * evt["market"])
    car = float(ar.sum())
    L = len(evt)
    t_car = car / (sigma * np.sqrt(L))
    p_car = float(2 * (1 - stats.t.cdf(abs(t_car), df=len(est) - 2)))

    table = pd.DataFrame({
        "relative_day": range(w0, w0 + L),
        "stock_return": evt["stock"].to_numpy(),
        "market_return": evt["market"].to_numpy(),
        "abnormal_return": ar.to_numpy(),
    }, index=evt.index)
    table["CAR"] = table["abnormal_return"].cumsum()

    result = EventStudyResult(
        method="market model, daily (OLS)",
        event=f"{event_name} ({event_date}; day 0 = {evt.index[0].date()})",
        estimation_window=f"[{e0}, {e1}] trading days",
        event_window=f"[{w0}, {w1}] trading days",
        n_estimation=int(len(est)),
        alpha=float(intercept),
        beta=float(slope),
        abnormal_return_t0=float(ar.iloc[-w0]) if w0 <= 0 else float("nan"),
        car=car,
        t_stat_car=float(t_car),
        p_value_car=p_car,
    )
    return result, table


def event_study_annual_mean_adjusted(
    returns: pd.Series,
    event_year: int,
    estimation_years: int = 5,
    post_years: int = 2,
    event_name: str = "event",
) -> EventStudyResult:
    """Annual mean-adjusted abnormal return (exploratory; low power).

    Expected return = mean of the ``estimation_years`` years immediately
    before ``event_year``. The test uses the prediction-error variance
    σ²·(1 + 1/n) for a single new observation, with n − 1 degrees of freedom.
    Years are looked up by **label**, never by position.
    """
    r = returns.dropna()
    if event_year not in r.index:
        raise KeyError(f"Year {event_year} not found in the return series.")
    est_years = [y for y in range(event_year - estimation_years, event_year)]
    missing = [y for y in est_years if y not in r.index]
    if missing:
        raise KeyError(f"Estimation years missing from returns: {missing}")

    est = r.loc[est_years]
    mu = float(est.mean())
    sd = float(est.std(ddof=1))
    n = len(est)

    ar0 = float(r.loc[event_year] - mu)
    window = [y for y in range(event_year, event_year + post_years + 1) if y in r.index]
    car = float((r.loc[window] - mu).sum())
    L = len(window)
    se_car = sd * np.sqrt(L * (1 + 1 / n))
    t_car = car / se_car
    p_car = float(2 * (1 - stats.t.cdf(abs(t_car), df=n - 1)))

    return EventStudyResult(
        method="mean-adjusted, annual",
        event=f"{event_name} ({event_year})",
        estimation_window=f"{est_years[0]}–{est_years[-1]}",
        event_window=f"{window[0]}–{window[-1]}",
        n_estimation=n,
        alpha=mu,
        beta=float("nan"),
        abnormal_return_t0=ar0,
        car=car,
        t_stat_car=float(t_car),
        p_value_car=p_car,
    )
