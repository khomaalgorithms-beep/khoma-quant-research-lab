
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

# Gaps longer than this (in calendar days) get flagged in the quality report.
# Normal weekends/holidays are 3-4 days, so anything above 5 is suspicious.
MAX_NORMAL_GAP_DAYS = 5


class DataDownloadError(Exception):
    """Raised when market data cannot be downloaded or is unusable."""


# ---------------------------------------------------------------------------
# Downloading and caching
# ---------------------------------------------------------------------------

def _to_date(value: str | date | datetime) -> date:
    """Convert a 'YYYY-MM-DD' string, date, or datetime into a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _cache_path(ticker: str, start: date, end: date, cache_dir: Path) -> Path:
    return cache_dir / f"{ticker}_{start.isoformat()}_{end.isoformat()}.csv"


def read_price_csv(path: str | Path) -> pd.DataFrame:
    """Read a price CSV saved by this project (Date index + OHLCV columns)."""
    df = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    return df


def _flatten_yfinance_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Newer yfinance versions return MultiIndex columns like ('Close', 'SPY')
    even for a single ticker. Keep only the first level ('Close').
    """
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(0)
    return raw


def download_price_history(
    ticker: str,
    start: str | date,
    end: str | date,
    use_cache: bool = True,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for one ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : symbol such as "SPY".
    start, end : inclusive date range, "YYYY-MM-DD".
    use_cache : read from / save to a local CSV cache.

    Returns
    -------
    DataFrame indexed by date with columns Open, High, Low, Close, Adj Close, Volume.
    The data is NOT cleaned yet - pass it to clean_price_data().

    Raises
    ------
    DataDownloadError if the download fails or returns nothing.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")
    start_date, end_date = _to_date(start), _to_date(end)
    if start_date >= end_date:
        raise ValueError(f"Start date {start_date} must be before end date {end_date}.")

    path = _cache_path(ticker, start_date, end_date, cache_dir)
    if use_cache and path.exists():
        return read_price_csv(path)

    try:
        import yfinance as yf

        # yfinance treats `end` as exclusive, so add one day to include the end date.
        # auto_adjust=False keeps BOTH "Close" and "Adj Close".
        raw = yf.download(
            ticker,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # network errors, API changes, etc.
        raise DataDownloadError(f"Download failed for {ticker}: {exc}") from exc

    # yfinance often reports failures by returning an EMPTY frame instead of raising.
    if raw is None or raw.empty:
        raise DataDownloadError(
            f"No data returned for {ticker} between {start_date} and {end_date}. "
            "Check the symbol, the date range, and your internet connection."
        )

    raw = _flatten_yfinance_columns(raw)
    missing = [col for col in PRICE_COLUMNS if col not in raw.columns]
    if missing:
        raise DataDownloadError(f"Downloaded data for {ticker} is missing columns: {missing}")

    df = raw[PRICE_COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    # Only cache completed history. If the range includes today, the latest bar
    # may still change, so caching it could freeze an incomplete number.
    if use_cache and end_date < date.today():
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    return df


# ---------------------------------------------------------------------------
# Validation and cleaning
# ---------------------------------------------------------------------------

def check_price_data(df: pd.DataFrame, price_column: str = "Adj Close") -> dict:
    """
    Inspect a price DataFrame and report problems WITHOUT changing it.

    Returns a dictionary describing the data quality, for example:
        {"rows": 5000, "duplicate_dates": 0, "is_sorted": True, ...}
    """
    if price_column not in df.columns:
        raise KeyError(f"Column '{price_column}' not found. Columns: {list(df.columns)}")

    prices = df[price_column]
    dates = pd.Series(df.index)
    sorted_dates = dates.sort_values().drop_duplicates()
    gaps = sorted_dates.diff().dt.days
    large_gaps = sorted_dates[gaps > MAX_NORMAL_GAP_DAYS]

    return {
        "rows": len(df),
        "first_date": df.index.min(),
        "last_date": df.index.max(),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "is_sorted": bool(df.index.is_monotonic_increasing),
        "missing_prices": int(prices.isna().sum()),
        "non_positive_prices": int((prices <= 0).sum()),
        "missing_by_column": {col: int(df[col].isna().sum()) for col in df.columns},
        "large_gaps": [d.date().isoformat() for d in large_gaps],
    }


def clean_price_data(df: pd.DataFrame, price_column: str = "Adj Close") -> tuple[pd.DataFrame, dict]:
    """
    Clean a price DataFrame and return (clean_df, report).

    What it does:
        * sorts dates in chronological order
        * removes duplicate dates (keeps the first occurrence)
        * removes rows where the research price is missing or <= 0

    What it deliberately does NOT do:
        * forward-fill or interpolate missing prices. A filled price would
          create a fake 0% return day. When a day is dropped, the next return
          simply covers two days of real price movement.
    """
    report = check_price_data(df, price_column)

    clean = df.sort_index()
    clean = clean[~clean.index.duplicated(keep="first")]
    bad_price = clean[price_column].isna() | (clean[price_column] <= 0)
    clean = clean[~bad_price].copy()

    report["rows_removed"] = len(df) - len(clean)
    report["rows_after_cleaning"] = len(clean)
    if len(clean) < 2:
        raise ValueError("Fewer than 2 valid price rows remain after cleaning.")
    return clean, report


def align_prices(price_series: dict[str, pd.Series]) -> tuple[pd.DataFrame, int]:
    """
    Combine several assets' price series into one table with one column per ticker.

    Only dates where EVERY asset has a price are kept (an "inner join").
    This avoids inventing prices for an asset on a day it did not trade.

    Returns (aligned_prices, number_of_dates_dropped).
    """
    combined = pd.concat(price_series, axis=1).sort_index()
    aligned = combined.dropna(how="any")
    return aligned, len(combined) - len(aligned)


# ---------------------------------------------------------------------------
# Synthetic data (clearly labeled, reproducible, no internet required)
# ---------------------------------------------------------------------------

# name: (annual drift, annual volatility, start price) in the calm regime
SYNTHETIC_ASSETS = {
    "SYN_STOCKS": (0.09, 0.15, 100.0),
    "SYN_TECH": (0.12, 0.21, 100.0),
    "SYN_SMALLCAP": (0.08, 0.19, 100.0),
    "SYN_BONDS": (0.035, 0.07, 100.0),
    "SYN_GOLD": (0.05, 0.15, 100.0),
}

# Correlation between the assets' daily shocks (same order as SYNTHETIC_ASSETS)
SYNTHETIC_CORRELATION = np.array([
    [1.00, 0.85, 0.80, -0.20, 0.05],
    [0.85, 1.00, 0.70, -0.15, 0.05],
    [0.80, 0.70, 1.00, -0.15, 0.05],
    [-0.20, -0.15, -0.15, 1.00, 0.25],
    [0.05, 0.05, 0.05, 0.25, 1.00],
])

# In the "stress" regime, equity-like assets fall and become more volatile.
STRESS_DRIFT = np.array([-0.25, -0.30, -0.30, 0.06, 0.08])
STRESS_VOL_MULTIPLIER = np.array([2.0, 2.0, 2.0, 1.2, 1.3])


def generate_synthetic_market(
    start: str = "2005-01-03",
    end: str = "2024-12-31",
    seed: int = 42,
    prob_enter_stress: float = 0.004,
    prob_exit_stress: float = 0.04,
) -> dict[str, pd.DataFrame]:
    """
    Simulate a small, fake market of 5 assets for testing and offline demos.

    Model: each day the market is either "calm" or "stress" (a two-state
    Markov chain). Daily log returns are correlated normal random draws whose
    mean and volatility depend on the regime. This produces trends, crashes
    and recoveries without copying any real asset's history.

    THE OUTPUT IS SYNTHETIC. It must never be presented as real market data.
    Synthetic assets have no dividends or splits, so Adj Close == Close.
    """
    rng = np.random.default_rng(seed)
    # Weekdays minus US federal holidays: about 251 days per year, close to
    # the ~252 trading days of the NYSE that annualization assumes.
    holidays = USFederalHolidayCalendar().holidays(start, end)
    dates = pd.bdate_range(start, end, freq="C", holidays=holidays, name="Date")
    n_days, n_assets = len(dates), len(SYNTHETIC_ASSETS)

    calm_drift = np.array([v[0] for v in SYNTHETIC_ASSETS.values()])
    calm_vol = np.array([v[1] for v in SYNTHETIC_ASSETS.values()])
    start_prices = np.array([v[2] for v in SYNTHETIC_ASSETS.values()])
    cholesky = np.linalg.cholesky(SYNTHETIC_CORRELATION)

    # 1) simulate the regime path
    in_stress = np.zeros(n_days, dtype=bool)
    for t in range(1, n_days):
        if in_stress[t - 1]:
            in_stress[t] = rng.random() > prob_exit_stress
        else:
            in_stress[t] = rng.random() < prob_enter_stress

    # 2) simulate correlated daily log returns
    shocks = rng.standard_normal((n_days, n_assets)) @ cholesky.T
    drift = np.where(in_stress[:, None], STRESS_DRIFT, calm_drift)
    vol = np.where(in_stress[:, None], calm_vol * STRESS_VOL_MULTIPLIER, calm_vol)
    daily_vol = vol / np.sqrt(252)
    log_returns = (drift / 252 - 0.5 * daily_vol**2) + daily_vol * shocks
    log_returns[0] = 0.0
    closes = start_prices * np.exp(np.cumsum(log_returns, axis=0))

    # 3) build plausible OHLCV bars around each close
    market = {}
    for i, name in enumerate(SYNTHETIC_ASSETS):
        close = closes[:, i]
        prev_close = np.concatenate([[close[0]], close[:-1]])
        open_ = prev_close * (1 + rng.normal(0, daily_vol[:, i] * 0.25))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, daily_vol[:, i] * 0.5)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, daily_vol[:, i] * 0.5)))
        volume = rng.integers(1_000_000, 5_000_000, n_days)
        market[name] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close,
             "Adj Close": close, "Volume": volume},
            index=dates,
        ).round({"Open": 4, "High": 4, "Low": 4, "Close": 4, "Adj Close": 4})
    return market


def load_sample_prices(ticker: str, sample_dir: Path = SAMPLE_DIR) -> pd.DataFrame:
    """Load one of the bundled synthetic sample CSVs (e.g. 'SYN_STOCKS')."""
    path = sample_dir / f"{ticker.upper()}.csv"
    if not path.exists():
        available = sorted(p.stem for p in sample_dir.glob("SYN_*.csv"))
        raise FileNotFoundError(
            f"No sample file for {ticker}. Available: {available}. "
            "Run `python scripts/make_sample_data.py` to regenerate them."
        )
    return read_price_csv(path)


# ---------------------------------------------------------------------------
# One function the dashboard and notebook use for everything
# ---------------------------------------------------------------------------

def load_prices(
    tickers: list[str],
    start: str | date,
    end: str | date,
    source: str = "yahoo",
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Load adjusted close prices for several tickers, cleaned and aligned.

    Parameters
    ----------
    source : "yahoo" (real data via yfinance) or "synthetic" (bundled sample data).

    Returns
    -------
    prices : DataFrame, one column per ticker, only dates where all have prices.
    info   : {"reports": {...per ticker quality report...},
              "errors": {ticker: message}, "dates_dropped_in_alignment": int,
              "source": source}
    Tickers that fail are reported in info["errors"] instead of crashing.
    """
    start_ts, end_ts = pd.Timestamp(_to_date(start)), pd.Timestamp(_to_date(end))
    series, reports, errors = {}, {}, {}

    for ticker in tickers:
        try:
            if source == "yahoo":
                raw = download_price_history(ticker, start, end, use_cache=use_cache)
            elif source == "synthetic":
                raw = load_sample_prices(ticker).loc[start_ts:end_ts]
            else:
                raise ValueError(f"Unknown data source: {source}")
            clean, report = clean_price_data(raw)
            series[ticker.upper()] = clean["Adj Close"]
            reports[ticker.upper()] = report
        except (DataDownloadError, FileNotFoundError, ValueError, KeyError) as exc:
            errors[ticker.upper()] = str(exc)

    if not series:
        return pd.DataFrame(), {"reports": reports, "errors": errors,
                                "dates_dropped_in_alignment": 0, "source": source}

    prices, dropped = align_prices(series)
    return prices, {"reports": reports, "errors": errors,
                    "dates_dropped_in_alignment": dropped, "source": source}
