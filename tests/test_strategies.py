
import math

import numpy as np
import pandas as pd
import pytest

from src.indicators import rolling_volatility, simple_moving_average, simple_returns
from src.strategies import (
    STRATEGIES,
    buy_and_hold,
    get_target_weights,
    moving_average_trend,
    volatility_targeted_trend,
)
from tests.helpers import make_dates, prices_from_returns


# --- indicators -----------------------------------------------------------------

def test_simple_moving_average_known_values():
    prices = pd.Series([1, 2, 3, 4, 5], index=make_dates(5), dtype=float)
    sma = simple_moving_average(prices, 3)
    assert sma.isna().sum() == 2              # not enough history for the first 2 days
    assert sma.iloc[2] == pytest.approx(2.0)  # (1+2+3)/3
    assert sma.iloc[4] == pytest.approx(4.0)  # (3+4+5)/3


def test_simple_returns_first_value_is_nan():
    prices = pd.Series([100.0, 110.0, 99.0], index=make_dates(3))
    r = simple_returns(prices)
    assert math.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.10)
    assert r.iloc[2] == pytest.approx(-0.10)


def test_rolling_volatility_known_value():
    prices = prices_from_returns([0.01, -0.01] * 20)
    vol = rolling_volatility(prices, window=10)
    expected = 0.01 * math.sqrt(10 / 9) * math.sqrt(252)
    assert vol.iloc[-1] == pytest.approx(expected)


# --- strategy signals --------------------------------------------------------------

def test_buy_and_hold_is_always_fully_invested(random_prices):
    weights = buy_and_hold(random_prices)
    assert (weights == 1.0).all()


def test_moving_average_in_cash_during_warm_up():
    prices = prices_from_returns([0.001] * 300)
    weights = moving_average_trend(prices, 50, 200)
    assert (weights.iloc[:199] == 0.0).all()  # 200-day average doesn't exist yet
    assert weights.iloc[199] == 1.0            # rising prices: short MA above long MA


def test_moving_average_exits_in_downtrend():
    prices = prices_from_returns([0.001] * 300 + [-0.004] * 200)
    weights = moving_average_trend(prices, 50, 200)
    assert weights.iloc[299] == 1.0
    assert weights.iloc[-1] == 0.0


def test_moving_average_signal_matches_manual_rule(random_prices):
    weights = moving_average_trend(random_prices, 20, 60)
    short = random_prices.rolling(20).mean()
    long = random_prices.rolling(60).mean()
    expected = (short > long).astype(float)
    pd.testing.assert_series_equal(weights, expected, check_names=False)


def test_moving_average_rejects_bad_windows(random_prices):
    with pytest.raises(ValueError):
        moving_average_trend(random_prices, 200, 50)


def test_volatility_target_sizes_position_correctly():
    # Returns alternate +2% / -1%: prices drift upward (trend = 1) and the
    # 60-day volatility is known exactly: deviations are +/-1.5% around a 0.5% mean.
    prices = prices_from_returns([0.02, -0.01] * 200)
    weights = volatility_targeted_trend(prices, 50, 200, target_volatility=0.15, volatility_window=60)
    sigma = 0.015 * math.sqrt(60 / 59) * math.sqrt(252)
    assert weights.iloc[-1] == pytest.approx(0.15 / sigma)


def test_volatility_target_never_exceeds_max_weight():
    prices = prices_from_returns([0.02, -0.01] * 200)
    weights = volatility_targeted_trend(prices, target_volatility=5.0, max_weight=1.0)
    assert weights.max() == pytest.approx(1.0)


def test_volatility_target_handles_zero_volatility():
    # Constant +0.1% every day: an uptrend with zero volatility.
    # target / 0 would be infinite - the strategy must cap it, not crash or return inf/NaN.
    prices = prices_from_returns([0.001] * 300)
    weights = volatility_targeted_trend(prices)
    assert np.isfinite(weights).all()
    assert weights.iloc[-1] == pytest.approx(1.0)


# --- look-ahead bias in signals ---------------------------------------------------------

@pytest.mark.parametrize("name", list(STRATEGIES))
def test_signals_do_not_use_future_prices(random_prices, name):
    """
    A signal on day t may only depend on prices up to day t.
    So computing the strategy on data CUT OFF at day t must give exactly the
    same signals (up to day t) as computing it on the full history.
    """
    params = {"short_window": 20, "long_window": 60} if name != "Buy and Hold" else {}
    full = get_target_weights(name, random_prices, **params)
    for cutoff in [100, 250, 400]:
        partial = get_target_weights(name, random_prices.iloc[:cutoff], **params)
        pd.testing.assert_series_equal(partial, full.iloc[:cutoff])


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_changing_future_prices_does_not_change_past_signals(random_prices, name):
    params = {"short_window": 20, "long_window": 60} if name != "Buy and Hold" else {}
    original = get_target_weights(name, random_prices, **params)
    shocked_prices = random_prices.copy()
    shocked_prices.iloc[300:] *= 0.5  # a fake crash in the "future"
    shocked = get_target_weights(name, shocked_prices, **params)
    pd.testing.assert_series_equal(original.iloc[:300], shocked.iloc[:300])
