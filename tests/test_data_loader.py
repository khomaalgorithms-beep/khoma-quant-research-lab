
import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl


def make_ohlcv(n: int = 10, start: str = "2021-01-04") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=n, name="Date")
    close = np.linspace(100, 110, n)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
                         "Adj Close": close * 0.98, "Volume": 1_000_000}, index=index)


# --- validation and cleaning ----------------------------------------------------------------

def test_check_detects_duplicates_unsorted_missing_and_gaps():
    df = make_ohlcv(10)
    df.iloc[3, df.columns.get_loc("Adj Close")] = np.nan
    df = pd.concat([df, df.iloc[[2]]])                     # duplicate date, out of order
    late = make_ohlcv(1, start="2021-03-01")               # a gap of several weeks
    df = pd.concat([df, late])
    report = dl.check_price_data(df)
    assert report["duplicate_dates"] == 1
    assert report["is_sorted"] is False
    assert report["missing_prices"] == 1
    assert report["large_gaps"] == ["2021-03-01"]


def test_clean_sorts_dedupes_and_drops_missing_without_filling():
    df = make_ohlcv(10)
    df.iloc[4, df.columns.get_loc("Adj Close")] = np.nan
    messy = pd.concat([df.iloc[::-1], df.iloc[[0]]])
    clean, report = dl.clean_price_data(messy)

    assert clean.index.is_monotonic_increasing
    assert not clean.index.has_duplicates
    assert len(clean) == 9                                  # 10 dates minus the missing one
    assert df.index[4] not in clean.index                   # dropped, NOT forward-filled
    assert report["rows_removed"] == 2
    # Every surviving price is an original value - nothing was invented
    pd.testing.assert_series_equal(clean["Adj Close"], df["Adj Close"].drop(df.index[4]))


def test_clean_rejects_non_positive_prices():
    df = make_ohlcv(5)
    df.iloc[2, df.columns.get_loc("Adj Close")] = 0.0
    clean, _ = dl.clean_price_data(df)
    assert len(clean) == 4


def test_clean_fails_if_too_little_data_remains():
    df = make_ohlcv(2)
    df["Adj Close"] = np.nan
    with pytest.raises(ValueError):
        dl.clean_price_data(df)


def test_align_prices_keeps_only_common_dates():
    a = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]))
    b = pd.Series([5.0, 6.0], index=pd.to_datetime(["2021-01-05", "2021-01-06"]))
    aligned, dropped = dl.align_prices({"A": a, "B": b})
    assert list(aligned.index) == list(b.index)
    assert dropped == 1


# --- synthetic data ----------------------------------------------------------------------

def test_synthetic_data_is_reproducible_and_labeled():
    first = dl.generate_synthetic_market(start="2020-01-01", end="2020-12-31", seed=1)
    second = dl.generate_synthetic_market(start="2020-01-01", end="2020-12-31", seed=1)
    other = dl.generate_synthetic_market(start="2020-01-01", end="2020-12-31", seed=2)
    assert all(name.startswith("SYN_") for name in first)
    pd.testing.assert_frame_equal(first["SYN_STOCKS"], second["SYN_STOCKS"])
    assert not first["SYN_STOCKS"].equals(other["SYN_STOCKS"])
    bars = first["SYN_STOCKS"]
    assert (bars["High"] >= bars[["Open", "Close"]].max(axis=1)).all()
    assert (bars["Low"] <= bars[["Open", "Close"]].min(axis=1)).all()


def test_bundled_sample_files_load():
    df = dl.load_sample_prices("SYN_STOCKS")
    assert set(dl.PRICE_COLUMNS) <= set(df.columns)
    assert len(df) > 1000


# --- downloading (yfinance is replaced by fakes - no internet needed) -------------------------

def test_empty_download_raises_clear_error(monkeypatch, tmp_path):
    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(dl.DataDownloadError, match="No data returned"):
        dl.download_price_history("FAKE", "2020-01-01", "2020-12-31", cache_dir=tmp_path)


def test_network_error_raises_clear_error(monkeypatch, tmp_path):
    import yfinance

    def broken(*args, **kwargs):
        raise ConnectionError("no internet")
    monkeypatch.setattr(yfinance, "download", broken)
    with pytest.raises(dl.DataDownloadError, match="Download failed"):
        dl.download_price_history("SPY", "2020-01-01", "2020-12-31", cache_dir=tmp_path)


def test_download_flattens_columns_and_uses_cache(monkeypatch, tmp_path):
    import yfinance
    calls = {"count": 0}
    raw = make_ohlcv(5)
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["SPY"]])  # newer yfinance format

    def fake_download(*args, **kwargs):
        calls["count"] += 1
        assert kwargs["auto_adjust"] is False  # we need both Close and Adj Close
        return raw
    monkeypatch.setattr(yfinance, "download", fake_download)

    first = dl.download_price_history("spy", "2021-01-01", "2021-01-31", cache_dir=tmp_path)
    second = dl.download_price_history("SPY", "2021-01-01", "2021-01-31", cache_dir=tmp_path)
    assert list(first.columns) == dl.PRICE_COLUMNS
    assert calls["count"] == 1                     # second call came from the cache
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_invalid_dates_and_tickers_raise():
    with pytest.raises(ValueError):
        dl.download_price_history("SPY", "2021-01-01", "2020-01-01")
    with pytest.raises(ValueError):
        dl.download_price_history("  ", "2020-01-01", "2021-01-01")


def test_load_prices_reports_failures_instead_of_crashing():
    prices, info = dl.load_prices(["SYN_STOCKS", "NOT_A_SAMPLE"], "2010-01-01", "2012-12-31",
                                  source="synthetic")
    assert list(prices.columns) == ["SYN_STOCKS"]
    assert "NOT_A_SAMPLE" in info["errors"]
