
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# CAGR on less than a year of data "annualizes" a lucky month into a huge
# yearly number. Below this many daily returns we report NaN instead.
MIN_DAYS_FOR_ANNUALIZED_RETURN = 252


def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns, dtype=float).dropna()


def daily_risk_free_rate(annual_rate: float) -> float:
    """Convert an annual risk-free rate into the equivalent daily rate (compounded)."""
    return (1 + annual_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1


def wealth_index(returns: pd.Series, starting_value: float = 1.0) -> pd.Series:
    """Growth of `starting_value` invested: V_t = V_0 * (1+r_1)(1+r_2)...(1+r_t)."""
    return starting_value * (1 + _clean(returns)).cumprod()


# ---------------------------------------------------------------------------
# Return metrics
# ---------------------------------------------------------------------------

def total_return(returns: pd.Series) -> float:
    """Total return = (1+r_1)(1+r_2)...(1+r_n) - 1."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float((1 + r).prod() - 1)


def number_of_years(returns: pd.Series) -> float:
    """Length of the sample in years, counting 252 trading days per year."""
    return len(_clean(returns)) / TRADING_DAYS_PER_YEAR


def cagr(returns: pd.Series, min_days: int = MIN_DAYS_FOR_ANNUALIZED_RETURN) -> float:
    """
    Compound Annual Growth Rate - the constant yearly return that would turn
    the starting value into the ending value over the same number of years:

        CAGR = (1 + total return) ^ (1 / years) - 1

    Returns NaN if there are fewer than `min_days` daily returns.
    """
    r = _clean(returns)
    if len(r) < max(min_days, 1):
        return float("nan")
    growth = 1 + total_return(r)
    if growth <= 0:
        return -1.0  # the portfolio was wiped out
    return float(growth ** (1 / number_of_years(r)) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    """
    Annualized volatility = std(daily returns) * sqrt(252).

    Uses the sample standard deviation (ddof=1). Needs at least 2 returns.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


# ---------------------------------------------------------------------------
# Risk-adjusted return metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe ratio = average excess return / volatility of returns, annualized.

        excess_t = r_t - rf_daily
        Sharpe   = mean(excess) / std(excess) * sqrt(252)

    "How much return above cash did I earn per unit of total risk?"
    Returns NaN when volatility is (numerically) zero.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - daily_risk_free_rate(risk_free_rate)
    std = excess.std(ddof=1)
    if not np.isfinite(std) or std < 1e-12:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Sortino ratio - like Sharpe, but only DOWNSIDE volatility counts as risk.

        downside_deviation = sqrt( mean( min(excess_t, 0)^2 ) ) * sqrt(252)
        Sortino = mean(excess) * 252 / downside_deviation

    Days with positive excess return contribute 0 to the downside deviation.
    Returns NaN if there were no days below the risk-free rate.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - daily_risk_free_rate(risk_free_rate)
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_dev < 1e-12:
        return float("nan")
    return float(excess.mean() * TRADING_DAYS_PER_YEAR / downside_dev)


# ---------------------------------------------------------------------------
# Drawdowns
# ---------------------------------------------------------------------------

def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Percentage below the highest value reached so far:

        drawdown_t = V_t / max(V_0..V_t) - 1      (always <= 0)

    The running peak starts at the initial value (1.0), so a loss on day 1
    already counts as a drawdown.
    """
    wealth = wealth_index(returns)
    running_peak = wealth.cummax().clip(lower=1.0)
    return wealth / running_peak - 1


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative number (-0.35 = -35%)."""
    dd = drawdown_series(returns)
    if dd.empty:
        return float("nan")
    return float(dd.min())


def calmar_ratio(returns: pd.Series) -> float:
    """Calmar ratio = CAGR / |max drawdown|. NaN if there was no drawdown or CAGR is NaN."""
    mdd = max_drawdown(returns)
    growth = cagr(returns)
    if not np.isfinite(mdd) or mdd == 0 or not np.isfinite(growth):
        return float("nan")
    return float(growth / abs(mdd))


def drawdown_recovery(returns: pd.Series) -> dict:
    """
    Describe the worst drawdown: when it started (peak), bottomed (trough),
    and recovered (first date back at the old peak, if that happened).
    """
    dd = drawdown_series(returns)
    if dd.empty or dd.min() == 0:
        return {"peak": None, "trough": None, "recovery": None,
                "days_peak_to_trough": 0, "days_to_recover": None}

    trough = dd.idxmin()
    before_trough = dd.loc[:trough]
    at_peak = before_trough[before_trough == 0]
    peak = at_peak.index[-1] if not at_peak.empty else dd.index[0]
    after_trough = dd.loc[trough:]
    recovered = after_trough[after_trough >= 0]
    recovery = recovered.index[0] if not recovered.empty else None

    return {
        "peak": peak,
        "trough": trough,
        "recovery": recovery,
        "days_peak_to_trough": int(dd.index.get_loc(trough) - dd.index.get_loc(peak)),
        "days_to_recover": (int(dd.index.get_loc(recovery) - dd.index.get_loc(trough))
                            if recovery is not None else None),
    }


# ---------------------------------------------------------------------------
# Simple descriptive statistics
# ---------------------------------------------------------------------------

def percent_positive_days(returns: pd.Series) -> float:
    """Share of days with a return strictly above zero. Flat (0%) days count as not positive."""
    r = _clean(returns)
    return float((r > 0).mean()) if not r.empty else float("nan")


def best_day(returns: pd.Series) -> float:
    r = _clean(returns)
    return float(r.max()) if not r.empty else float("nan")


def worst_day(returns: pd.Series) -> float:
    r = _clean(returns)
    return float(r.min()) if not r.empty else float("nan")


def annual_returns(returns: pd.Series) -> pd.Series:
    """Calendar-year returns (the first/last years may be partial years)."""
    r = _clean(returns)
    return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)


# ---------------------------------------------------------------------------
# Summary of a backtest
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "total_return": "Total return",
    "cagr": "CAGR",
    "annual_volatility": "Annualized volatility",
    "sharpe": "Sharpe ratio",
    "sortino": "Sortino ratio",
    "max_drawdown": "Maximum drawdown",
    "calmar": "Calmar ratio",
    "pct_positive_days": "% positive days",
    "worst_day": "Worst day",
    "best_day": "Best day",
    "avg_exposure": "Average exposure",
    "annual_turnover": "Annual turnover",
}

PERCENT_METRICS = {"total_return", "cagr", "annual_volatility", "max_drawdown",
                   "pct_positive_days", "worst_day", "best_day", "avg_exposure",
                   "annual_turnover"}


def performance_summary(backtest: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """
    Compute all headline metrics for a backtest result (from backtester.run_backtest)
    or any slice of one.

    avg_exposure    = average weight held in the asset (1.0 = always fully invested)
    annual_turnover = total one-way turnover per year (1.0 = traded the whole
                      portfolio's value once per year)
    """
    returns = backtest["strategy_return"].dropna()
    valid_rows = backtest.loc[returns.index]
    years = number_of_years(returns)

    return {
        "start": returns.index.min(),
        "end": returns.index.max(),
        "years": years,
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "annual_volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free_rate),
        "sortino": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
        "pct_positive_days": percent_positive_days(returns),
        "worst_day": worst_day(returns),
        "best_day": best_day(returns),
        "avg_exposure": float(valid_rows["weight"].mean()) if "weight" in valid_rows else float("nan"),
        "annual_turnover": (float(valid_rows["turnover"].sum() / years)
                            if "turnover" in valid_rows and years > 0 else float("nan")),
    }


def summary_table(summaries: dict[str, dict]) -> pd.DataFrame:
    """Turn {strategy_name: performance_summary(...)} into a metrics-by-strategy table."""
    rows = {name: {METRIC_LABELS[k]: s[k] for k in METRIC_LABELS} for name, s in summaries.items()}
    return pd.DataFrame(rows)


def format_metric(key: str, value: float) -> str:
    """Format a metric for display: percentages with 1-2 decimals, ratios with 2."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if key in PERCENT_METRICS:
        return f"{value:.2%}" if key in {"worst_day", "best_day"} else f"{value:.1%}"
    return f"{value:.2f}"


def formatted_summary_table(summaries: dict[str, dict]) -> pd.DataFrame:
    """Same as summary_table but with every value formatted as text for display."""
    rows = {name: {METRIC_LABELS[k]: format_metric(k, s[k]) for k in METRIC_LABELS}
            for name, s in summaries.items()}
    return pd.DataFrame(rows)
