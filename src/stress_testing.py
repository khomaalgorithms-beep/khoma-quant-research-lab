
from __future__ import annotations

import pandas as pd

from src.risk_metrics import annualized_volatility, max_drawdown, total_return

CRISIS_PERIODS = {
    "2008 Global Financial Crisis": ("2007-10-09", "2009-03-09"),
    "2020 COVID-19 Crash": ("2020-02-19", "2020-03-23"),
    "2022 Equity Bear Market": ("2022-01-03", "2022-10-12"),
}


def window_is_covered(index: pd.DatetimeIndex, start: str, end: str) -> bool:
    """True if the data starts on/before `start` and ends on/after `end`."""
    if len(index) == 0:
        return False
    return index.min() <= pd.Timestamp(start) and index.max() >= pd.Timestamp(end)


def analyze_window(returns: pd.Series, equity: pd.Series, start: str, end: str) -> dict:
    """
    Stats for one stress window.

    Recovery: the number of trading days after the window ends until equity
    climbs back to the level it had at the window's start. If that never
    happens in the data, recovery is reported as not observed.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    window_returns = returns.loc[start_ts:end_ts].dropna()

    # Equity just BEFORE the window starts is the level we want to recover.
    before = equity.loc[equity.index < start_ts]
    starting_level = before.iloc[-1] if not before.empty else equity.iloc[0]
    after = equity.loc[end_ts:]
    recovered = after[after >= starting_level]
    days_to_recover = (int(after.index.get_loc(recovered.index[0])) if not recovered.empty else None)

    return {
        "start": start_ts.date(),
        "end": end_ts.date(),
        "trading_days": len(window_returns),
        "period_return": total_return(window_returns),
        "max_drawdown": max_drawdown(window_returns),
        "annualized_volatility": annualized_volatility(window_returns),
        "days_to_recover_after_window": days_to_recover,
        "recovered_by": recovered.index[0].date() if not recovered.empty else None,
    }


def stress_test(
    returns: pd.Series,
    equity: pd.Series,
    periods: dict[str, tuple[str, str]] = CRISIS_PERIODS,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Run analyze_window for every period the data fully covers.

    Returns (results_table, skipped) where skipped maps a period name to the
    reason it was not analyzed.
    """
    rows, skipped = {}, {}
    for name, (start, end) in periods.items():
        if not window_is_covered(returns.index, start, end):
            skipped[name] = (
                f"Data covers {returns.index.min().date()} to {returns.index.max().date()}, "
                f"but this period needs {start} to {end}. Not analyzed."
            )
            continue
        rows[name] = analyze_window(returns, equity, start, end)
    return pd.DataFrame(rows).T, skipped
