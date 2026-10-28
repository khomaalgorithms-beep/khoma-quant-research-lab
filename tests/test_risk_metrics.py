
import math

import numpy as np
import pandas as pd
import pytest

from src import risk_metrics as rm
from tests.helpers import returns_series


# --- total return and CAGR ---------------------------------------------------

def test_total_return_compounds_daily_returns():
    # +10% then -10% is NOT break-even: 1.1 * 0.9 - 1 = -1%
    assert rm.total_return(returns_series([0.10, -0.10])) == pytest.approx(-0.01)


def test_total_return_of_empty_series_is_nan():
    assert math.isnan(rm.total_return(pd.Series(dtype=float)))


def test_cagr_one_year_equals_total_return():
    daily = (1.10) ** (1 / 252) - 1  # 252 identical days that compound to +10%
    r = returns_series([daily] * 252)
    assert rm.cagr(r) == pytest.approx(0.10)


def test_cagr_two_years():
    # +21% over two years is 10% per year, because 1.1^2 = 1.21
    daily = (1.21) ** (1 / 504) - 1
    assert rm.cagr(returns_series([daily] * 504)) == pytest.approx(0.10)


def test_cagr_is_nan_for_short_samples():
    # 3 months of data should not be annualized into a headline CAGR
    assert math.isnan(rm.cagr(returns_series([0.01] * 60)))


def test_cagr_when_portfolio_is_wiped_out():
    r = returns_series([0.0] * 251 + [-1.0])
    assert rm.cagr(r) == -1.0


# --- volatility, Sharpe, Sortino ------------------------------------------------

def test_annualized_volatility_known_value():
    n = 252
    r = returns_series([0.01, -0.01] * (n // 2))
    # mean is 0, every deviation is 0.01 -> sample std = 0.01 * sqrt(n / (n - 1))
    expected = 0.01 * math.sqrt(n / (n - 1)) * math.sqrt(252)
    assert rm.annualized_volatility(r) == pytest.approx(expected)


def test_volatility_needs_two_observations():
    assert math.isnan(rm.annualized_volatility(returns_series([0.01])))


def test_sharpe_ratio_known_value():
    n = 252
    r = returns_series([0.02, 0.0] * (n // 2))  # mean 1%, deviation 1%
    std = 0.01 * math.sqrt(n / (n - 1))
    assert rm.sharpe_ratio(r) == pytest.approx(0.01 / std * math.sqrt(252))


def test_sharpe_ratio_subtracts_risk_free_rate():
    n = 252
    r = returns_series([0.02, 0.0] * (n // 2))
    rf_daily = (1.05) ** (1 / 252) - 1
    std = 0.01 * math.sqrt(n / (n - 1))
    expected = (0.01 - rf_daily) / std * math.sqrt(252)
    assert rm.sharpe_ratio(r, risk_free_rate=0.05) == pytest.approx(expected)


def test_sharpe_is_nan_when_volatility_is_zero():
    # Constant returns -> zero volatility -> dividing by zero would give infinity
    assert math.isnan(rm.sharpe_ratio(returns_series([0.001] * 300)))


def test_sortino_ratio_known_value():
    r = returns_series([0.02, -0.01, 0.02, -0.01])
    excess = np.array([0.02, -0.01, 0.02, -0.01])
    downside_dev = math.sqrt(np.mean(np.minimum(excess, 0) ** 2)) * math.sqrt(252)
    expected = excess.mean() * 252 / downside_dev
    assert rm.sortino_ratio(r) == pytest.approx(expected)


def test_sortino_is_nan_without_down_days():
    assert math.isnan(rm.sortino_ratio(returns_series([0.01, 0.02, 0.005])))


# --- drawdowns ------------------------------------------------------------------

def test_max_drawdown_known_path():
    # Values: 100 -> 120 -> 90 -> 95 -> 130. Worst fall: 120 -> 90 = -25%
    values = np.array([100, 120, 90, 95, 130], dtype=float)
    r = returns_series(list(values[1:] / values[:-1] - 1))
    assert rm.max_drawdown(r) == pytest.approx(-0.25)


def test_drawdown_counts_loss_on_first_day():
    assert rm.max_drawdown(returns_series([-0.10, 0.05])) == pytest.approx(-0.10)


def test_drawdown_series_is_never_positive():
    rng = np.random.default_rng(1)
    dd = rm.drawdown_series(returns_series(list(rng.normal(0, 0.01, 500))))
    assert (dd <= 1e-12).all()


def test_calmar_ratio_known_value():
    daily = (1.10) ** (1 / 252) - 1
    returns = [daily] * 252
    returns[10] = -0.20  # one bad day
    r = returns_series(returns)
    assert rm.calmar_ratio(r) == pytest.approx(rm.cagr(r) / abs(rm.max_drawdown(r)))


def test_calmar_is_nan_without_drawdown():
    assert math.isnan(rm.calmar_ratio(returns_series([0.001] * 300)))


def test_drawdown_recovery_dates():
    values = np.array([100, 120, 90, 95, 125, 130], dtype=float)
    r = returns_series(list(values[1:] / values[:-1] - 1))
    info = rm.drawdown_recovery(r)
    assert info["peak"] == r.index[0]       # day the value was 120
    assert info["trough"] == r.index[1]     # day the value was 90
    assert info["recovery"] == r.index[3]   # first day back above 120 (125)
    assert info["days_to_recover"] == 2


# --- simple statistics ----------------------------------------------------------

def test_percent_positive_best_and_worst_day():
    r = returns_series([0.01, -0.02, 0.0, 0.03])
    assert rm.percent_positive_days(r) == pytest.approx(0.5)  # flat day is not positive
    assert rm.best_day(r) == pytest.approx(0.03)
    assert rm.worst_day(r) == pytest.approx(-0.02)


def test_annual_returns_by_calendar_year():
    index = pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"])
    r = pd.Series([0.10, 0.10, -0.05, 0.0], index=index)
    annual = rm.annual_returns(r)
    assert annual[2020] == pytest.approx(0.21)
    assert annual[2021] == pytest.approx(-0.05)


def test_performance_summary_on_backtest_frame():
    n = 300
    frame = pd.DataFrame({
        "strategy_return": [np.nan] + [0.001] * n,
        "weight": [0.0] + [0.5] * n,
        "turnover": [0.0, 0.5] + [0.0] * (n - 1),
    }, index=pd.bdate_range("2020-01-01", periods=n + 1))
    s = rm.performance_summary(frame)
    assert s["avg_exposure"] == pytest.approx(0.5)
    assert s["annual_turnover"] == pytest.approx(0.5 / (n / 252))
    assert s["total_return"] == pytest.approx(1.001**n - 1)
    assert math.isnan(s["sharpe"])  # constant returns: zero volatility


def test_format_metric_handles_nan():
    assert rm.format_metric("sharpe", float("nan")) == "n/a"
    assert rm.format_metric("cagr", 0.1234) == "12.3%"
    assert rm.format_metric("sharpe", 1.234) == "1.23"
