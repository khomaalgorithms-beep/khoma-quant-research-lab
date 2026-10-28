
from __future__ import annotations

import pandas as pd

from src.backtester import run_strategy_backtest
from src.risk_metrics import performance_summary


def chronological_split_date(index: pd.DatetimeIndex, train_fraction: float = 0.6) -> pd.Timestamp:
    """
    Return the first date of the TEST period.

    The first `train_fraction` of the dates (in time order) are the training
    period; everything after is the test period. Never shuffle time series:
    shuffling would let the model "train" on 2022 and "test" on 2021.
    """
    if not 0.1 <= train_fraction <= 0.9:
        raise ValueError("train_fraction should be between 0.1 and 0.9")
    if len(index) < 20:
        raise ValueError("Not enough data to split into train and test periods.")
    split_position = int(len(index) * train_fraction)
    return index[split_position]


def split_backtest(backtest: pd.DataFrame, split_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut a backtest result into (train, test) at split_date (test starts ON split_date)."""
    train = backtest.loc[backtest.index < split_date]
    test = backtest.loc[backtest.index >= split_date]
    return train, test


def train_test_report(
    prices: pd.Series,
    strategy_name: str,
    strategy_params: dict | None,
    split_date: pd.Timestamp,
    risk_free_rate: float = 0.0,
    **backtest_kwargs,
) -> pd.DataFrame:
    """Metrics for the same strategy in the training period vs. the test period."""
    backtest = run_strategy_backtest(prices, strategy_name, strategy_params, **backtest_kwargs)
    train, test = split_backtest(backtest, split_date)
    return pd.DataFrame({
        "In-sample (train)": performance_summary(train, risk_free_rate),
        "Out-of-sample (test)": performance_summary(test, risk_free_rate),
    })


def moving_average_parameter_search(
    prices: pd.Series,
    split_date: pd.Timestamp,
    short_windows: list[int] = (20, 50, 100),
    long_windows: list[int] = (100, 150, 200, 250),
    risk_free_rate: float = 0.0,
    **backtest_kwargs,
) -> pd.DataFrame:
    """
    Try every (short, long) moving-average pair, rank them by TRAIN-period Sharpe,
    and show the TEST-period Sharpe next to it.

    The point of this table is to demonstrate overfitting: the pair that looks
    best in-sample is usually not the best out-of-sample, and its test result
    is typically worse than its train result. Only the train column may be used
    to CHOOSE parameters.
    """
    rows = []
    for short in short_windows:
        for long in long_windows:
            if short >= long:
                continue
            bt = run_strategy_backtest(
                prices, "Moving Average Trend (50/200)",
                {"short_window": short, "long_window": long}, **backtest_kwargs,
            )
            train, test = split_backtest(bt, split_date)
            train_stats = performance_summary(train, risk_free_rate)
            test_stats = performance_summary(test, risk_free_rate)
            rows.append({
                "short_window": short,
                "long_window": long,
                "train_sharpe": train_stats["sharpe"],
                "test_sharpe": test_stats["sharpe"],
                "train_cagr": train_stats["cagr"],
                "test_cagr": test_stats["cagr"],
            })
    table = pd.DataFrame(rows).sort_values("train_sharpe", ascending=False, na_position="last")
    table["rank_by_train"] = range(1, len(table) + 1)
    return table.reset_index(drop=True)


def cost_sensitivity(
    prices: pd.Series,
    strategy_name: str,
    strategy_params: dict | None = None,
    cost_levels_bps: list[float] = (0, 5, 10, 25, 50),
    risk_free_rate: float = 0.0,
    **backtest_kwargs,
) -> pd.DataFrame:
    """
    Re-run the backtest with different total trading costs (in basis points
    per unit of turnover) to see how fragile the result is.
    """
    backtest_kwargs = {k: v for k, v in backtest_kwargs.items()
                       if k not in ("commission_bps", "slippage_bps")}
    rows = []
    for bps in cost_levels_bps:
        bt = run_strategy_backtest(prices, strategy_name, strategy_params,
                                   commission_bps=0.0, slippage_bps=bps, **backtest_kwargs)
        stats = performance_summary(bt, risk_free_rate)
        rows.append({
            "cost_bps": bps,
            "cagr": stats["cagr"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
            "total_cost_dollars": bt["cost_dollars"].sum(),
        })
    return pd.DataFrame(rows)
