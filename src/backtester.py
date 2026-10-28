
from __future__ import annotations

import pandas as pd

from src.strategies import get_target_weights

TRADING_DAYS_PER_YEAR = 252


def run_backtest(
    prices: pd.Series,
    target_weights: pd.Series,
    initial_capital: float = 10_000.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 5.0,
    execution_delay: int = 1,
    cash_rate: float = 0.0,
    allow_leverage: bool = False,
) -> pd.DataFrame:
    """
    Simulate a strategy on one asset.

    Parameters
    ----------
    prices          : adjusted close prices, indexed by date.
    target_weights  : desired weight (0..1) decided at each day's close.
    initial_capital : starting portfolio value in dollars.
    commission_bps  : commission per trade, in basis points of traded value (1 bp = 0.01%).
    slippage_bps    : estimated slippage / spread cost, in basis points of traded value.
    execution_delay : extra trading days between signal and trade (see module docstring).
    cash_rate       : annual interest earned on uninvested cash (0.0 by default).
    allow_leverage  : if False (default), weights above 1.0 raise an error.

    Returns
    -------
    DataFrame indexed by date with columns:
        price, asset_return, target_weight, weight, turnover, cost,
        gross_return, strategy_return, equity, cost_dollars
    The first row is the starting point: equity = initial_capital and the
    returns are NaN (there is no previous day to measure a return from).
    """
    _validate_inputs(prices, target_weights, initial_capital, execution_delay, allow_leverage)

    target = target_weights.reindex(prices.index).fillna(0.0)
    asset_return = prices.pct_change(fill_method=None)

    # The weight actually held during each day (see TIMING MODEL above).
    weight = target.shift(1 + execution_delay).fillna(0.0)

    daily_cash_rate = (1 + cash_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    r = asset_return.fillna(0.0)
    gross_return = weight * r + (1 - weight) * daily_cash_rate
    # Day 0 is the starting point: no time has passed, so nothing is earned
    # (not even cash interest). Without this, cash would earn one extra day.
    gross_return.iloc[0] = 0.0

    # Weight at the end of the day, after prices moved but before trading.
    drifted_weight = weight * (1 + r) / (1 + gross_return)
    weight_before_trade = drifted_weight.shift(1).fillna(0.0)  # we start 100% in cash
    turnover = (weight - weight_before_trade).abs()

    cost_rate = (commission_bps + slippage_bps) / 10_000
    cost = turnover * cost_rate

    # The trade (and its cost) happens at the start of the holding period,
    # then the remaining capital earns the day's gross return.
    strategy_return = (1 - cost) * (1 + gross_return) - 1
    equity = initial_capital * (1 + strategy_return).cumprod()

    result = pd.DataFrame({
        "price": prices,
        "asset_return": asset_return,
        "target_weight": target,
        "weight": weight,
        "turnover": turnover,
        "cost": cost,
        "gross_return": gross_return,
        "strategy_return": strategy_return,
        "equity": equity,
    })
    result["cost_dollars"] = result["cost"] * result["equity"].shift(1).fillna(initial_capital)

    # Row 0 is the starting point, not a real trading day for the strategy.
    first = result.index[0]
    result.loc[first, ["asset_return", "gross_return", "strategy_return"]] = float("nan")
    result.loc[first, "equity"] = initial_capital
    return result


def _validate_inputs(
    prices: pd.Series,
    target_weights: pd.Series,
    initial_capital: float,
    execution_delay: int,
    allow_leverage: bool,
) -> None:
    if len(prices) < 2:
        raise ValueError("Need at least 2 prices to run a backtest.")
    if prices.isna().any():
        raise ValueError("Prices contain NaN. Clean the data first (data_loader.clean_price_data).")
    if (prices <= 0).any():
        raise ValueError("Prices must be positive.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Prices must be sorted from oldest to newest.")
    if prices.index.has_duplicates:
        raise ValueError("Prices contain duplicate dates.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if execution_delay < 0 or int(execution_delay) != execution_delay:
        raise ValueError("execution_delay must be a non-negative whole number of days.")
    if (target_weights < 0).any():
        raise ValueError("Negative weights (short selling) are not supported.")
    if not allow_leverage and (target_weights > 1.0 + 1e-9).any():
        raise ValueError(
            "A target weight above 1.0 means borrowing money (leverage). "
            "Set allow_leverage=True only if that is intended."
        )


def run_strategy_backtest(
    prices: pd.Series,
    strategy_name: str,
    strategy_params: dict | None = None,
    **backtest_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper: compute a strategy's target weights, then backtest them."""
    weights = get_target_weights(strategy_name, prices, **(strategy_params or {}))
    return run_backtest(prices, weights, **backtest_kwargs)
