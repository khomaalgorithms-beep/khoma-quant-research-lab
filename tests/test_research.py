
import numpy as np
import pandas as pd
import pytest

from src.backtester import run_strategy_backtest
from src.reporting import LIMITATIONS, generate_html_report, generate_markdown_report
from src.research import chronological_split_date, cost_sensitivity, split_backtest
from src.risk_metrics import format_metric, performance_summary
from src.stress_testing import stress_test
from tests.helpers import prices_from_returns


# --- train / test split ------------------------------------------------------------------

def test_split_is_chronological_and_complete(random_prices):
    bt = run_strategy_backtest(random_prices, "Buy and Hold")
    split = chronological_split_date(bt.index, train_fraction=0.6)
    train, test = split_backtest(bt, split)
    assert train.index.max() < test.index.min()        # no overlap, train strictly first
    assert len(train) + len(test) == len(bt)            # nothing lost
    assert len(train) == int(len(bt) * 0.6)


def test_split_rejects_extreme_fractions(random_prices):
    with pytest.raises(ValueError):
        chronological_split_date(random_prices.index, train_fraction=0.99)


def test_cost_sensitivity_is_monotonic(random_prices):
    table = cost_sensitivity(random_prices, "Moving Average Trend (50/200)",
                             {"short_window": 20, "long_window": 60}, cost_levels_bps=[0, 10, 50])
    assert table["total_cost_dollars"].iloc[0] == 0
    assert table["cagr"].is_monotonic_decreasing


# --- stress testing ------------------------------------------------------------------------

def test_stress_test_skips_periods_without_data():
    prices = prices_from_returns([0.001] * 300, start="2015-01-01")
    bt = run_strategy_backtest(prices, "Buy and Hold")
    table, skipped = stress_test(bt["strategy_return"], bt["equity"])
    assert table.empty
    assert len(skipped) == 3
    assert all("Not analyzed" in msg for msg in skipped.values())


def test_stress_test_known_window():
    index = pd.bdate_range("2020-01-01", "2020-06-30")
    r = pd.Series(0.0, index=index)
    crash = (index >= "2020-02-19") & (index <= "2020-03-23")
    r[crash] = -0.01
    after = index > "2020-03-23"
    r[after] = 0.02
    equity = 100 * (1 + r).cumprod()
    table, skipped = stress_test(r, equity, {"Test crash": ("2020-02-19", "2020-03-23")})
    n = int(crash.sum())
    row = table.loc["Test crash"]
    assert row["period_return"] == pytest.approx(0.99**n - 1)
    assert row["max_drawdown"] == pytest.approx(0.99**n - 1)
    # Recovery: need (1.02)^k >= 1 / 0.99^n
    k = int(np.ceil(np.log(1 / 0.99**n) / np.log(1.02)))
    assert row["days_to_recover_after_window"] == k
    assert skipped == {}


# --- reports -------------------------------------------------------------------------------

def test_report_numbers_come_from_the_backtest(random_prices):
    backtests = {"Buy and Hold": run_strategy_backtest(random_prices, "Buy and Hold")}
    md = generate_markdown_report({"Asset": "TEST"}, backtests, "Buy and Hold", data_note="synthetic")
    summary = performance_summary(backtests["Buy and Hold"])
    assert format_metric("total_return", summary["total_return"]) in md
    assert format_metric("max_drawdown", summary["max_drawdown"]) in md
    assert LIMITATIONS[0] in md
    assert "synthetic" in md


def test_html_report_is_self_contained(random_prices):
    backtests = {"Buy and Hold": run_strategy_backtest(random_prices, "Buy and Hold")}
    page = generate_html_report({"Asset": "TEST"}, backtests, "Buy and Hold")
    assert page.startswith("<!doctype html>")
    assert "data:image/png;base64," in page
