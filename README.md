# KHOMA Quant Research Lab

I started this project because I kept seeing trading strategies online with great-looking backtests and no explanation
of how they were tested. I wanted to build my own backtester and understand every step: where the data comes from, when
a trade actually happens, what it costs, and how to tell if a result is real or just luck.

It tests a few simple strategies on historical ETF data (SPY, QQQ, IWM, TLT, GLD by default), compares them to buy and
hold, and looks at portfolio risk and diversification. There's a Streamlit dashboard for exploring results and a test
suite that checks the math.

This is a learning project, not a trading system. Nothing here is investment advice.

## What it does

You pick some tickers and a date range, and the app downloads daily prices from Yahoo Finance. From there you can:

- look at prices, moving averages and rolling volatility for each asset
- backtest a strategy against buy and hold, with trading costs included
- compare all three strategies side by side (returns, Sharpe, drawdowns, yearly returns)
- split the data into a training period and a test period to check for overfitting
- see how results change as trading costs go up
- build a portfolio with your own weights and rebalancing schedule, and see where its risk comes from
- check how each strategy did during 2008, the March 2020 crash, and 2022
- export a report as Markdown or HTML

If you don't have internet, there's a set of synthetic (simulated) price files so the app still runs. The app labels
that data as synthetic everywhere it shows up.

## The strategies

**Buy and hold.** Put everything in the asset on day one and never sell. This is the benchmark.

**50/200 moving average.** Be fully invested when the 50-day average price is above the 200-day average, otherwise sit
in cash. It's the classic "golden cross" rule.

**Volatility-targeted trend.** Same trend rule, but the position size shrinks when the market gets volatile. The weight
is `min(15% / recent volatility, 100%)`, using the last 60 days of returns. It never borrows, so in calm markets it just
stays at 100%.

I used standard default settings on purpose instead of tuning them. Tuning parameters until the backtest looks good is
the easiest way to fool yourself.

## How the backtest handles timing

This was the part I spent the most time on. A lot of simple backtests calculate a signal from today's closing price and
then pretend they bought at that same close, which isn't possible.

Here, a signal is calculated after the close on day t, the trade happens at the close on day t+1, and the position earns
its first return on day t+2. You can switch to the optimistic same-close version in the sidebar to see how much it
changes the results.

Costs are charged in basis points every time the portfolio trades, including the small trades needed when a position
drifts away from its target weight.

The full list of formulas and assumptions is in [docs/methodology.md](docs/methodology.md).

## Running it

You need Python 3.11 or newer.

```bash
git clone https://github.com/<your-username>/khoma-quant-research-lab.git
cd khoma-quant-research-lab
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

It opens at http://localhost:8501.

To run the whole analysis without the dashboard and save the results to a folder:

```bash
python scripts/generate_sample_outputs.py --source yahoo --asset SPY --out reports/spy
```

There's also a step-by-step notebook in `notebooks/research_walkthrough.ipynb`.

## Tests

```bash
pytest
```

The tests use small made-up price series where I can work out the right answer by hand. For example, prices going 100,
120, 90, 95, 130 should give a max drawdown of exactly -25%.

A few tests exist only to catch look-ahead bias. My favorite one feeds the engine a strategy that cheats: in a market
that alternates +5% and -5% days, it buys after every up day. If the engine accidentally let it trade on the same day it
saw the signal, it would make money every time. A correct engine makes it lose money, and the test checks for that.

## Results

<!-- Fill this in after running with real data. Example of what to write:
     "On SPY from 2005 to 2024, the 50/200 strategy cut the max drawdown from about X% to Y%,
     but finished with lower total return because it missed the start of most recoveries." -->

_Coming soon. I'm running the backtests on real data and will add the numbers and screenshots here._

## Things to keep in mind

- These are historical simulations. A strategy working in the past doesn't mean it will work in the future.
- Every trade fills at the daily close plus a flat cost. Real trading is messier.
- Yahoo Finance data is free and sometimes has errors.
- I picked ETFs that exist today, which means the asset list is biased toward things that survived and did well.
- One train/test split is a pretty weak test. Walk-forward testing would be better.

More detail in [docs/limitations.md](docs/limitations.md).

## Project layout

```
app.py                  Streamlit dashboard
src/data_loader.py      downloading, caching and cleaning prices
src/indicators.py       returns, moving averages, volatility
src/strategies.py       the three strategies
src/backtester.py       trade timing, costs and equity
src/risk_metrics.py     CAGR, Sharpe, Sortino, drawdowns, etc.
src/research.py         train/test split, parameter search, cost sensitivity
src/portfolio.py        multi-asset portfolios and rebalancing
src/stress_testing.py   2008 / 2020 / 2022 analysis
src/visualization.py    charts
src/reporting.py        Markdown and HTML reports
tests/                  pytest tests
notebooks/              research walkthrough notebook
data/sample/            synthetic sample data (not real prices)
docs/                   methodology and limitations
```

## What I want to add next

- only rebalance the volatility strategy when the weight moves by more than a few percent
- walk-forward testing instead of a single split
- rolling correlations, especially stocks vs bonds in 2022
- use actual T-bill yields as the cash return

## About

My name is Dmytrii Khoma. 
I'm a Finance student at the University of Tampa (class of 2030), learning Python and quantitative finance. I also run
KhomaAlgorithms, where I work on systematic trading tools. None of that code is in this repo.


## License

MIT. See [LICENSE](LICENSE).
