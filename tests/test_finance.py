import numpy as np
import pandas as pd
import pytest

from finance import (annual_returns, beta, event_study_annual_mean_adjusted,
                     event_study_market_model_daily, max_drawdown, risk_metrics)


def test_annual_returns_use_year_end_closes():
    idx = pd.to_datetime(["2001-06-01", "2001-12-31", "2002-12-31", "2003-12-31"])
    px = pd.Series([5.0, 10.0, 15.0, 12.0], index=idx)
    r = annual_returns(px)
    assert list(r.index) == [2002, 2003]
    assert r.loc[2002] == pytest.approx(0.5)
    assert r.loc[2003] == pytest.approx(-0.2)


def test_max_drawdown_counts_first_period_loss():
    r = pd.Series([-0.5, 1.0, -0.25])
    # wealth: 1 → 0.5 → 1.0 → 0.75 ; worst = -50%
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_beta_recovers_known_slope():
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0, 0.1, 500))
    a = 0.01 + 1.7 * m + pd.Series(rng.normal(0, 0.001, 500))
    assert beta(a, m) == pytest.approx(1.7, abs=0.01)


def test_risk_metrics_sharpe():
    r = pd.Series([0.10, 0.20, 0.30])
    out = risk_metrics(r, risk_free=0.0)
    assert out["sharpe"] == pytest.approx(0.20 / r.std(ddof=1))


def test_annual_event_study_matches_by_label_not_position():
    # 2001 has no return (first year) — the classic source of an off-by-one bug.
    years = list(range(2002, 2018))
    r = pd.Series(0.0, index=years, dtype=float)
    r.loc[2009:2013] = [0.05, 0.15, 0.10, 0.08, 0.12]  # mean 0.10
    r.loc[2014] = -0.40   # event year
    r.loc[2015] = 0.90    # next year (must NOT be read as the event year)
    res = event_study_annual_mean_adjusted(r, 2014, estimation_years=5, post_years=0)
    assert res.abnormal_return_t0 == pytest.approx(-0.50)
    assert res.estimation_window == "2009–2013"


def test_annual_event_study_missing_year_raises():
    r = pd.Series([0.1, 0.2], index=[2012, 2013])
    with pytest.raises(KeyError):
        event_study_annual_mean_adjusted(r, 2014)


def test_daily_market_model_recovers_parameters_and_event():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2013-01-01", "2014-12-31")
    rm = rng.normal(0.0003, 0.01, len(dates))
    rs = 0.0002 + 1.2 * rm + rng.normal(0, 0.002, len(dates))
    day0 = dates.searchsorted(pd.Timestamp("2014-08-04"))
    rs[day0] -= 0.05                         # injected abnormal return on day 0
    mkt = pd.Series(100 * np.cumprod(1 + rm), index=dates)
    stk = pd.Series(50 * np.cumprod(1 + rs), index=dates)
    res, table = event_study_market_model_daily(stk, mkt, "2014-08-04")
    assert res.beta == pytest.approx(1.2, abs=0.05)
    assert res.abnormal_return_t0 == pytest.approx(-0.05, abs=0.01)
    assert res.p_value_car < 0.05
    assert len(table) == 3
