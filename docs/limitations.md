# Limitations and Research Integrity

## Historical backtests do not predict the future

A backtest answers one narrow question: *"What would have happened if this exact rule had been followed over this exact period, under these assumptions?"* It does not tell you what will happen next. Markets change: interest-rate regimes, market structure, and the behavior of other investors all shift over time. A rule that worked in the past may have worked for reasons that no longer exist, or simply by luck.

## Overfitting

Overfitting means tuning a strategy until it fits the **noise** of past data rather than a real, repeatable effect. If you test 100 parameter combinations, some will look excellent purely by chance.

What this project does about it:
- It uses common, pre-chosen defaults (50/200 moving averages, 15% target volatility, 60-day lookback) instead of optimized values.
- It includes a chronological train/test split and a parameter-search table that shows how the best training-period choice does on unseen test data.

What it does **not** do: it can't prevent you from looking at the test results, changing the parameters, and looking again. Once you do that, the test period stops being truly out-of-sample.

## Survivorship bias

The assets you can download today are the ones that **survived**. Funds that closed and companies that went bankrupt or were delisted are missing. Testing only survivors makes historical results look better than what an investor could actually have achieved at the time.

## Selection bias

SPY, QQQ, IWM, TLT and GLD were chosen with hindsight. They are well known *because* they existed and grew over the period. Choosing the assets, the dates, or the strategy after seeing how things turned out builds hindsight into the result.

## Data quality

- Yahoo Finance is free and unofficial. Data can contain errors, gaps, or later revisions, and `yfinance` can break when Yahoo changes its website.
- Adjusted prices are recalculated by Yahoo when new dividends occur, so old adjusted values can change slightly between downloads.
- The data is daily only, so intraday moves (including the true worst point of a crash day) are invisible.
- Yahoo's terms don't allow redistributing its data. This repository therefore ships **no Yahoo data**. Downloads are cached locally and ignored by Git.
- The bundled sample data is **synthetic**. It is useful for testing the code and says nothing about real markets.

## Execution and costs

- Every trade is assumed to fill at the daily close, plus a fixed cost in basis points. Real fills depend on spreads, liquidity, order type, and market impact.
- Costs are a single percentage of traded value. Fixed per-trade fees, taxes, and short-term capital gains are not modeled.
- The volatility-targeted strategy rebalances to its exact target every day, which creates small trades. A real implementation would probably use a tolerance band.
- Transaction-cost sensitivity is shown explicitly, because a strategy that only works at zero cost doesn't really work.

## Statistics

- Sharpe and Sortino ratios are noisy estimates. Over a few years, the uncertainty around a Sharpe ratio is large.
- Annualization assumes returns are roughly independent from day to day, which they are not exactly (volatility clusters).
- Stress-test windows use rounded, approximate peak and trough dates of the S&P 500, which won't match every asset's own peak and trough.
- Portfolio correlations and risk contributions use the full sample. Correlations change over time and often rise during crises, exactly when diversification is needed most.

## Bottom line

Use the results to **understand risk and trade-offs**: how deep drawdowns were, how costs and timing change outcomes, and how diversification behaves. Do not use them as evidence that any strategy will make money in the future. Nothing in this project is investment advice.
