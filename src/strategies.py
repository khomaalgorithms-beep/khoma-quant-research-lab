

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import rolling_volatility, simple_moving_average


def buy_and_hold(prices: pd.Series) -> pd.Series:
    """
    Strategy 1 - Buy and Hold (the benchmark).

    Rule: always hold 100% of the asset.
    """
    return pd.Series(1.0, index=prices.index, name="target_weight")


def moving_average_trend(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.Series:
    """
    Strategy 2 - Moving Average Trend Following ("golden cross / death cross").

    Rules, evaluated after the close of each day t:
        * If SMA(short) > SMA(long): target weight = 1.0 (invested)
        * Otherwise:                  target weight = 0.0 (cash)
        * Before both averages exist (first `long_window - 1` days): 0.0

    Idea: when the recent average price is above the longer-term average,
    the asset is in an uptrend. Trend followers stay invested in uptrends
    and step aside in downtrends, accepting some whipsaw losses in return
    for (hopefully) avoiding the worst of long bear markets.
    """
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")

    short_ma = simple_moving_average(prices, short_window)
    long_ma = simple_moving_average(prices, long_window)

    # Comparisons with NaN are False, so the warm-up period becomes 0 (cash).
    in_uptrend = short_ma > long_ma
    return in_uptrend.astype(float).rename("target_weight")


def volatility_targeted_trend(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
    target_volatility: float = 0.15,
    volatility_window: int = 60,
    max_weight: float = 1.0,
) -> pd.Series:
    """
    Strategy 3 - Volatility-Targeted Trend Following.

    Step 1 (trend filter): use the same rule as moving_average_trend.
    Step 2 (position size): when in an uptrend, size the position so that
            the EXPECTED portfolio volatility is roughly `target_volatility`:

                weight_t = min( target_volatility / sigma_t , max_weight )

            where sigma_t is the annualized volatility of the last
            `volatility_window` daily returns (known at the close of day t).

    Why it works mathematically: if the asset's volatility is sigma, holding
    a fraction w of it gives a portfolio volatility of about w * sigma
    (cash has ~zero volatility). Setting w = target / sigma makes
    w * sigma = target.

    Assumptions:
        * recent volatility is a reasonable forecast of near-future volatility
          (volatility tends to cluster, but it can jump suddenly - e.g. a crash
          that starts on a calm day will not be caught in advance)
        * max_weight <= 1.0 means NO leverage: in calm markets the strategy
          simply stays at 100%, it never borrows to reach the target.

    Defaults are conservative and were NOT optimized: 15% is close to the
    long-run volatility of broad US equities, and 60 days is a common
    "about one quarter" lookback that reacts without being too jumpy.
    """
    if target_volatility <= 0:
        raise ValueError("target_volatility must be positive")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")

    trend = moving_average_trend(prices, short_window, long_window)
    sigma = rolling_volatility(prices, window=volatility_window, annualize=True)

    raw_weight = target_volatility / sigma
    # sigma can be 0 (flat prices) -> inf, or NaN during warm-up. Handle both.
    raw_weight = raw_weight.replace([np.inf, -np.inf], max_weight)
    sized = raw_weight.clip(upper=max_weight).fillna(0.0)

    return (trend * sized).rename("target_weight")


# ---------------------------------------------------------------------------
# Registry used by the dashboard, notebook and reports.
# "params" lists the default value for every tunable input.
# ---------------------------------------------------------------------------

STRATEGIES = {
    "Buy and Hold": {
        "function": buy_and_hold,
        "params": {},
        "description": "Always 100% invested. The baseline every other strategy must beat.",
    },
    "Moving Average Trend (50/200)": {
        "function": moving_average_trend,
        "params": {"short_window": 50, "long_window": 200},
        "description": "100% invested when the 50-day average is above the 200-day average, otherwise cash.",
    },
    "Volatility-Targeted Trend": {
        "function": volatility_targeted_trend,
        "params": {
            "short_window": 50,
            "long_window": 200,
            "target_volatility": 0.15,
            "volatility_window": 60,
            "max_weight": 1.0,
        },
        "description": "Trend filter plus position sizing: weight = min(15% / recent volatility, 100%).",
    },
}


def get_target_weights(strategy_name: str, prices: pd.Series, **params) -> pd.Series:
    """Look up a strategy by name and run it, using defaults for missing params."""
    if strategy_name not in STRATEGIES:
        raise KeyError(f"Unknown strategy '{strategy_name}'. Options: {list(STRATEGIES)}")
    spec = STRATEGIES[strategy_name]
    full_params = {**spec["params"], **params}
    return spec["function"](prices, **full_params)
