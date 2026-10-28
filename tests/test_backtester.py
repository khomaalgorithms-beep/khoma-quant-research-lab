
import numpy as np
import pandas as pd
import pytest

from src.backtester import run_backtest, run_strategy_backtest
from tests.helpers import prices_from_returns


def no_costs(**kwargs):
    return {"commission_bps": 0.0, "slippage_bps": 0.0, **kwargs}


# --- basic accounting -----------------------------------------------------------------

def test_first_row_is_the_starting_point(random_prices):
    bt = run_backtest(random_prices, pd.Series(1.0, index=random_prices.index), initial_capital=5_000)
    assert bt["equity"].iloc[0] == 5_000
    assert np.isnan(bt["strategy_return"].iloc[0])


def test_buy_and_hold_matches_price_growth_without_costs(random_prices):
    weights = pd.Series(1.0, index=random_prices.index)
    bt = run_backtest(random_prices, weights, initial_capital=10_000, **no_costs(execution_delay=0))
    # Bought at the close of day 0 -> equity grows exactly like the price from then on
    expected = 10_000 * random_prices / random_prices.iloc[0]
    pd.testing.assert_series_equal(bt["equity"], expected, check_names=False)


def test_default_delay_buys_one_day_later(random_prices):
    weights = pd.Series(1.0, index=random_prices.index)
    bt = run_backtest(random_prices, weights, initial_capital=10_000, **no_costs())
    # Signal at close of day 0, trade at close of day 1 -> growth measured from P_1
    assert bt["equity"].iloc[1] == pytest.approx(10_000)
    assert bt["equity"].iloc[-1] == pytest.approx(10_000 * random_prices.iloc[-1] / random_prices.iloc[1])


def test_cash_earns_cash_rate():
    prices = prices_from_returns([0.01] * 252)  # 253 prices -> 252 returns
    bt = run_backtest(prices, pd.Series(0.0, index=prices.index), initial_capital=100.0,
                      cash_rate=0.05, **no_costs())
    assert bt["equity"].iloc[-1] == pytest.approx(105.0)


# --- signal timing and look-ahead bias ----------------------------------------------------

def test_signal_is_executed_with_a_lag():
    prices = prices_from_returns([0.01] * 10)
    target = pd.Series(0.0, index=prices.index)
    target.iloc[3] = 1.0  # decide on day 3's close

    bt0 = run_backtest(prices, target, **no_costs(execution_delay=0))
    assert bt0["weight"].iloc[4] == 1.0 and bt0["weight"].sum() == 1.0  # earns day 4's return

    bt1 = run_backtest(prices, target, **no_costs(execution_delay=1))
    assert bt1["weight"].iloc[5] == 1.0 and bt1["weight"].sum() == 1.0  # earns day 5's return


def test_engine_cannot_profit_from_same_day_information():
    """
    LOOK-AHEAD TEST.
    Returns alternate +5%, -5%. The "strategy" buys whenever TODAY's return was
    positive. If the engine (wrongly) applied today's signal to today's return,
    it would capture every +5% day and get rich. A correct engine applies the
    signal to a LATER day - which is always a -5% day here - so it must lose money.
    """
    prices = prices_from_returns([0.05, -0.05] * 50)
    peeking_signal = (prices.pct_change() > 0).astype(float)

    bt = run_backtest(prices, peeking_signal, initial_capital=10_000, **no_costs(execution_delay=0))
    assert bt["equity"].iloc[-1] < 10_000
    # Every day it was invested, the realized return was negative
    invested = bt[bt["weight"] > 0]
    assert (invested["asset_return"] < 0).all()


def test_future_prices_do_not_change_past_results(random_prices):
    """Shocking prices after day 400 must not change anything up to day 400."""
    params = {"short_window": 20, "long_window": 60}
    original = run_strategy_backtest(random_prices, "Volatility-Targeted Trend", params)
    shocked_prices = random_prices.copy()
    shocked_prices.iloc[400:] *= 1.5
    shocked = run_strategy_backtest(shocked_prices, "Volatility-Targeted Trend", params)
    pd.testing.assert_frame_equal(original.iloc[:400], shocked.iloc[:400])


# --- transaction costs and turnover ----------------------------------------------------------

def test_one_purchase_costs_exactly_the_cost_rate(random_prices):
    weights = pd.Series(1.0, index=random_prices.index)
    free = run_backtest(random_prices, weights, **no_costs())
    paid = run_backtest(random_prices, weights, commission_bps=5, slippage_bps=5)
    # Buy-and-hold trades once (100% turnover) at 10 bps = 0.1%
    assert paid["equity"].iloc[-1] == pytest.approx(free["equity"].iloc[-1] * (1 - 0.001))
    assert paid["turnover"].sum() == pytest.approx(1.0)


def test_turnover_counts_entries_and_exits():
    prices = prices_from_returns([0.0] * 12)
    target = pd.Series(0.0, index=prices.index)
    target.iloc[2:6] = 1.0  # enter, hold, exit
    bt = run_backtest(prices, target, **no_costs())
    assert bt["turnover"].sum() == pytest.approx(2.0)  # 100% in + 100% out


def test_turnover_includes_drift_rebalancing():
    prices = prices_from_returns([0.10, 0.0, 0.0])
    target = pd.Series(0.5, index=prices.index)
    bt = run_backtest(prices, target, **no_costs(execution_delay=0))
    # Day 1: hold 50%, asset +10% -> drifted weight = 0.55 / 1.05.
    # Day 2: trading back to 50% requires |0.5 - 0.55/1.05| turnover.
    assert bt["turnover"].iloc[2] == pytest.approx(abs(0.5 - 0.55 / 1.05))


def test_higher_costs_never_improve_results(random_prices):
    params = {"short_window": 20, "long_window": 60}
    finals = [run_strategy_backtest(random_prices, "Moving Average Trend (50/200)", params,
                                    slippage_bps=bps)["equity"].iloc[-1] for bps in (0, 10, 50)]
    assert finals[0] >= finals[1] >= finals[2]


# --- leverage and input validation ------------------------------------------------------------

def test_leverage_is_rejected_by_default(random_prices):
    with pytest.raises(ValueError, match="leverage"):
        run_backtest(random_prices, pd.Series(1.5, index=random_prices.index))


def test_leverage_allowed_when_enabled():
    prices = prices_from_returns([0.01] * 5)
    bt = run_backtest(prices, pd.Series(2.0, index=prices.index), allow_leverage=True,
                      **no_costs(execution_delay=0))
    assert bt["strategy_return"].iloc[2] == pytest.approx(0.02)


def test_short_selling_is_rejected(random_prices):
    with pytest.raises(ValueError, match="short"):
        run_backtest(random_prices, pd.Series(-1.0, index=random_prices.index))


@pytest.mark.parametrize("bad", ["nan", "unsorted", "duplicate", "too_short"])
def test_bad_price_data_is_rejected(random_prices, bad):
    prices = random_prices.iloc[:50].copy()
    if bad == "nan":
        prices.iloc[10] = np.nan
    elif bad == "unsorted":
        prices = prices.iloc[::-1]
    elif bad == "duplicate":
        prices = pd.concat([prices, prices.iloc[[-1]]])
    elif bad == "too_short":
        prices = prices.iloc[:1]
    with pytest.raises(ValueError):
        run_backtest(prices, pd.Series(1.0, index=prices.index))


def test_results_are_reproducible(random_prices):
    a = run_strategy_backtest(random_prices, "Volatility-Targeted Trend")
    b = run_strategy_backtest(random_prices, "Volatility-Targeted Trend")
    pd.testing.assert_frame_equal(a, b)
