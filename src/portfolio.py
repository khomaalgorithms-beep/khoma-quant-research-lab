
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

REBALANCE_OPTIONS = {
    "Monthly": "M",
    "Quarterly": "Q",
    "Annually": "Y",
    "Never (let weights drift)": None,
}


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def equal_weights(tickers: list[str]) -> dict[str, float]:
    """1/N in every asset."""
    if not tickers:
        raise ValueError("Need at least one ticker.")
    return {t: 1.0 / len(tickers) for t in tickers}


def validate_weights(weights: dict[str, float], tolerance: float = 1e-6) -> dict[str, float]:
    """
    Check that weights are long-only and add up to 100%.
    Raises ValueError with a helpful message otherwise.
    """
    if not weights:
        raise ValueError("No weights given.")
    if any(w < 0 for w in weights.values()):
        raise ValueError("Weights cannot be negative (no short selling).")
    total = sum(weights.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"Weights must sum to 1.0 (100%), but they sum to {total:.4f}.")
    return weights


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Scale positive weights so they sum to 1, e.g. {A: 3, B: 1} -> {A: 0.75, B: 0.25}."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weights must add up to a positive number.")
    return {t: w / total for t, w in weights.items()}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def rebalance_dates(index: pd.DatetimeIndex, frequency: str | None) -> pd.Series:
    """
    True on the first trading day of each new period ("M", "Q", "Y"), or every
    day for "D". The very first day is False (that is the initial purchase).
    """
    if frequency is None:
        return pd.Series(False, index=index)
    if frequency == "D":
        flags = pd.Series(True, index=index)
    else:
        periods = pd.Series(index.to_period(frequency), index=index)
        flags = periods != periods.shift(1)
    flags.iloc[0] = False
    return flags


def simulate_portfolio(
    prices: pd.DataFrame,
    target_weights: dict[str, float],
    rebalance: str | None = "M",
    cost_bps: float = 5.0,
    initial_capital: float = 10_000.0,
) -> dict[str, pd.DataFrame]:
    """
    Simulate a fixed-target-weight portfolio day by day.

    Returns a dictionary with:
        "result"        : DataFrame with portfolio_return, equity, turnover, cost
        "weights"       : actual weights after each day's close (after any rebalance)
        "contributions" : each asset's contribution to the day's return (w_i * r_i)
    """
    validate_weights(target_weights)
    missing = [t for t in target_weights if t not in prices.columns]
    if missing:
        raise KeyError(f"No price data for: {missing}")
    if len(prices) < 2:
        raise ValueError("Need at least 2 days of prices.")
    if prices.isna().any().any():
        raise ValueError("Prices contain NaN - align/clean them first.")

    tickers = list(target_weights)
    prices = prices[tickers]
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0).to_numpy()
    target = np.array([target_weights[t] for t in tickers])
    is_rebalance_day = rebalance_dates(prices.index, rebalance).to_numpy()
    cost_rate = cost_bps / 10_000

    n_days, n_assets = asset_returns.shape
    weights_history = np.zeros((n_days, n_assets))
    contributions = np.zeros((n_days, n_assets))
    portfolio_return = np.zeros(n_days)
    turnover = np.zeros(n_days)
    equity = np.zeros(n_days)

    # Day 0: buy the target weights at the close (start from 100% cash).
    turnover[0] = np.abs(target - 0.0).sum()
    portfolio_return[0] = -turnover[0] * cost_rate
    equity[0] = initial_capital * (1 + portfolio_return[0])
    weights = target.copy()
    weights_history[0] = weights

    for t in range(1, n_days):
        r = asset_returns[t]
        contributions[t] = weights * r
        gross = contributions[t].sum()

        # Let the weights drift with the day's price moves.
        weights = weights * (1 + r) / (1 + gross)

        cost = 0.0
        if is_rebalance_day[t]:
            turnover[t] = np.abs(target - weights).sum()
            cost = turnover[t] * cost_rate
            weights = target.copy()

        portfolio_return[t] = (1 + gross) * (1 - cost) - 1
        equity[t] = equity[t - 1] * (1 + portfolio_return[t])
        weights_history[t] = weights

    index = prices.index
    result = pd.DataFrame({
        "portfolio_return": portfolio_return,
        "equity": equity,
        "turnover": turnover,
        "cost": turnover * cost_rate,
        "rebalanced": is_rebalance_day,
    }, index=index)
    return {
        "result": result,
        "weights": pd.DataFrame(weights_history, index=index, columns=tickers),
        "contributions": pd.DataFrame(contributions, index=index, columns=tickers),
    }


def portfolio_as_backtest(simulation: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Reshape a portfolio simulation into the same column names a single-asset
    backtest uses, so risk_metrics.performance_summary() works on both.
    (A fully invested portfolio has exposure 1.0 every day.)
    """
    result = simulation["result"]
    return pd.DataFrame({
        "strategy_return": result["portfolio_return"],
        "equity": result["equity"],
        "weight": simulation["weights"].sum(axis=1),
        "turnover": result["turnover"],
    })


# ---------------------------------------------------------------------------
# Descriptive risk analytics (full-sample, backward-looking)
# ---------------------------------------------------------------------------

def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Correlation of daily returns between every pair of assets."""
    return prices.pct_change(fill_method=None).dropna().corr()


def annualized_covariance(prices: pd.DataFrame) -> pd.DataFrame:
    """Covariance matrix of daily returns, scaled to annual units (x 252)."""
    return prices.pct_change(fill_method=None).dropna().cov() * TRADING_DAYS_PER_YEAR


def portfolio_volatility(weights: dict[str, float], prices: pd.DataFrame) -> float:
    """
    Fixed-weight portfolio volatility:  sigma_p = sqrt( w^T * Sigma * w )

    This is why diversification works: when assets are not perfectly
    correlated, the off-diagonal terms of Sigma are small (or negative),
    so sigma_p is less than the weighted average of individual volatilities.
    """
    tickers = list(weights)
    cov = annualized_covariance(prices[tickers]).to_numpy()
    w = np.array([weights[t] for t in tickers])
    return float(np.sqrt(w @ cov @ w))


def risk_contributions(weights: dict[str, float], prices: pd.DataFrame) -> pd.DataFrame:
    """
    How much of the portfolio's volatility comes from each asset.

        risk contribution share_i = w_i * (Sigma w)_i / (w^T Sigma w)

    The shares add up to 100%. A 50/50 stock/bond portfolio usually gets far
    MORE than 50% of its risk from stocks, which is an important insight
    for anyone who thinks "equal dollars" means "equal risk".
    """
    tickers = list(weights)
    cov = annualized_covariance(prices[tickers]).to_numpy()
    w = np.array([weights[t] for t in tickers])
    port_var = w @ cov @ w
    marginal = cov @ w
    standalone_vol = np.sqrt(np.diag(cov))
    return pd.DataFrame({
        "weight": w,
        "standalone_volatility": standalone_vol,
        "risk_contribution": w * marginal / port_var if port_var > 0 else np.nan,
    }, index=tickers)


def diversification_ratio(weights: dict[str, float], prices: pd.DataFrame) -> float:
    """
    Weighted average of individual volatilities divided by portfolio volatility.
    1.0 = no diversification benefit (all assets perfectly correlated);
    higher = more risk was cancelled out by combining assets.
    """
    table = risk_contributions(weights, prices)
    weighted_avg_vol = float((table["weight"] * table["standalone_volatility"]).sum())
    port_vol = portfolio_volatility(weights, prices)
    return weighted_avg_vol / port_vol if port_vol > 0 else float("nan")


def return_contribution_summary(simulation: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Sum of each asset's daily contributions (w_i * r_i) over the whole period.

    This is an ARITHMETIC attribution: it does not include compounding, so the
    numbers add up to the sum of daily portfolio returns, not exactly to the
    compounded total return. It is still a fair picture of who did the work.
    """
    return simulation["contributions"].sum()
