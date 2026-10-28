
import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    diversification_ratio,
    equal_weights,
    normalize_weights,
    portfolio_volatility,
    rebalance_dates,
    risk_contributions,
    simulate_portfolio,
    validate_weights,
)


@pytest.fixture
def two_assets() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2021-01-01", periods=300)
    returns = rng.normal(0.0003, [0.012, 0.005], size=(300, 2))
    returns[0] = 0.0
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=index, columns=["A", "B"])


# --- weights ------------------------------------------------------------------------

def test_equal_weights_sum_to_one():
    w = equal_weights(["A", "B", "C", "D"])
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["A"] == pytest.approx(0.25)


@pytest.mark.parametrize("weights", [{"A": 0.7, "B": 0.7}, {"A": 1.2, "B": -0.2}, {}])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        validate_weights(weights)


def test_normalize_weights():
    assert normalize_weights({"A": 3, "B": 1}) == {"A": 0.75, "B": 0.25}


# --- rebalancing -----------------------------------------------------------------------

def test_monthly_rebalance_dates_are_first_trading_day_of_month():
    index = pd.bdate_range("2021-01-01", "2021-04-30")
    flags = rebalance_dates(index, "M")
    flagged = list(index[flags])
    assert flagged == [pd.Timestamp("2021-02-01"), pd.Timestamp("2021-03-01"), pd.Timestamp("2021-04-01")]


def test_never_rebalanced_portfolio_equals_buy_and_hold_of_each_asset(two_assets):
    sim = simulate_portfolio(two_assets, {"A": 0.6, "B": 0.4}, rebalance=None, cost_bps=0)
    expected = 10_000 * (0.6 * two_assets["A"] / two_assets["A"].iloc[0]
                         + 0.4 * two_assets["B"] / two_assets["B"].iloc[0])
    pd.testing.assert_series_equal(sim["result"]["equity"], expected, check_names=False)


def test_weights_drift_between_rebalances(two_assets):
    sim = simulate_portfolio(two_assets, {"A": 0.6, "B": 0.4}, rebalance=None, cost_bps=0)
    final_a = sim["weights"]["A"].iloc[-1]
    value_a = 0.6 * two_assets["A"].iloc[-1] / two_assets["A"].iloc[0]
    value_b = 0.4 * two_assets["B"].iloc[-1] / two_assets["B"].iloc[0]
    assert final_a == pytest.approx(value_a / (value_a + value_b))


def test_daily_rebalancing_return_is_weighted_average(two_assets):
    sim = simulate_portfolio(two_assets, {"A": 0.5, "B": 0.5}, rebalance="D", cost_bps=0)
    asset_returns = two_assets.pct_change().iloc[1:]
    expected = 0.5 * asset_returns["A"] + 0.5 * asset_returns["B"]
    pd.testing.assert_series_equal(sim["result"]["portfolio_return"].iloc[1:], expected, check_names=False)


def test_weights_return_to_target_on_rebalance_days(two_assets):
    sim = simulate_portfolio(two_assets, {"A": 0.6, "B": 0.4}, rebalance="M", cost_bps=0)
    rebalanced = sim["result"]["rebalanced"]
    assert rebalanced.sum() > 0
    assert np.allclose(sim["weights"].loc[rebalanced, "A"], 0.6)


def test_rebalancing_costs_match_turnover(two_assets):
    free = simulate_portfolio(two_assets, {"A": 0.6, "B": 0.4}, rebalance="M", cost_bps=0)
    paid = simulate_portfolio(two_assets, {"A": 0.6, "B": 0.4}, rebalance="M", cost_bps=10)
    # Turnover is identical (costs are taken proportionally and don't change weights)...
    pd.testing.assert_series_equal(free["result"]["turnover"], paid["result"]["turnover"])
    # ...and the cost on each day is turnover * 10 bps
    pd.testing.assert_series_equal(paid["result"]["cost"], paid["result"]["turnover"] * 0.001,
                                   check_names=False)
    assert paid["result"]["equity"].iloc[-1] < free["result"]["equity"].iloc[-1]


def test_contributions_add_up_to_portfolio_return(two_assets):
    sim = simulate_portfolio(two_assets, {"A": 0.3, "B": 0.7}, rebalance="Q", cost_bps=0)
    total = sim["contributions"].sum(axis=1).iloc[1:]
    pd.testing.assert_series_equal(total, sim["result"]["portfolio_return"].iloc[1:], check_names=False)


def test_missing_prices_are_rejected(two_assets):
    broken = two_assets.copy()
    broken.iloc[5, 0] = np.nan
    with pytest.raises(ValueError):
        simulate_portfolio(broken, {"A": 0.5, "B": 0.5})


# --- risk analytics ----------------------------------------------------------------------

def test_identical_assets_give_no_diversification(two_assets):
    prices = pd.DataFrame({"A": two_assets["A"], "A2": two_assets["A"]})
    w = {"A": 0.5, "A2": 0.5}
    single_vol = prices["A"].pct_change().std() * np.sqrt(252)
    assert portfolio_volatility(w, prices) == pytest.approx(single_vol)
    assert diversification_ratio(w, prices) == pytest.approx(1.0)


def test_risk_contributions_sum_to_one(two_assets):
    table = risk_contributions({"A": 0.5, "B": 0.5}, two_assets)
    assert table["risk_contribution"].sum() == pytest.approx(1.0)
    # A is the more volatile asset, so it should carry more than half the risk
    assert table.loc["A", "risk_contribution"] > 0.5


def test_mixing_uncorrelated_assets_lowers_volatility(two_assets):
    w = {"A": 0.5, "B": 0.5}
    weighted_avg = 0.5 * two_assets["A"].pct_change().std() + 0.5 * two_assets["B"].pct_change().std()
    assert portfolio_volatility(w, two_assets) < weighted_avg * np.sqrt(252)
