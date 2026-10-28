
import numpy as np
import pandas as pd


def make_dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    """n consecutive business days."""
    return pd.bdate_range(start, periods=n, name="Date")


def prices_from_returns(returns: list[float] | np.ndarray, start_price: float = 100.0,
                        start: str = "2020-01-01") -> pd.Series:
    """
    Build a price series whose daily returns are exactly `returns`.
    The series has len(returns) + 1 prices (the first is start_price).
    """
    values = start_price * np.cumprod(np.concatenate([[1.0], 1 + np.asarray(returns, dtype=float)]))
    return pd.Series(values, index=make_dates(len(values), start), name="price")


def returns_series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    return pd.Series(values, index=make_dates(len(values), start), dtype=float)


