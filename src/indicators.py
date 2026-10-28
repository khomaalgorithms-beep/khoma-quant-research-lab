

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Daily simple returns: r_t = P_t / P_(t-1) - 1.

    The first row has no previous price, so it is NaN.
    """
    return prices.pct_change(fill_method=None)


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Daily log returns: ln(P_t / P_(t-1)). Useful because they add up over time."""
    return np.log(prices / prices.shift(1))


def simple_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Average of the last `window` prices, including today.

    The first (window - 1) values are NaN because there is not enough
    history yet. We do NOT fill them in - the strategy treats NaN as
    "no signal".
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    return prices.rolling(window=window, min_periods=window).mean()


def rolling_volatility(
    prices: pd.Series,
    window: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """
    Rolling standard deviation of daily returns over the last `window` days.

    Annualized volatility = daily standard deviation * sqrt(252).
    The sqrt comes from assuming daily returns are roughly independent, so
    variances add: Var(year) = 252 * Var(day).
    """
    if window < 2:
        raise ValueError("window must be at least 2 to compute a standard deviation")
    daily_returns = simple_returns(prices)
    vol = daily_returns.rolling(window=window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol
