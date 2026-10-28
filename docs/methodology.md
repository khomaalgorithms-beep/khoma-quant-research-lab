# Methodology

This page explains exactly how every number in the project is produced. If a result can't be traced back to a formula here, it shouldn't be trusted.

## 1. Data

- **Source:** Yahoo Finance daily bars via the free `yfinance` library, or the bundled **synthetic** sample data (a seeded random simulation for offline use and tests).
- **Price used for research:** `Adj Close`, which is adjusted for splits and dividends. Its daily percentage change approximates the **total return** an investor received. `Close` (unadjusted) is kept for reference only.
- **Cleaning:** dates are sorted, duplicate dates are removed (first kept), and rows with a missing or non-positive price are **dropped**. Missing prices are never forward-filled or interpolated, because a filled price creates a fake 0% day.
- **Multiple assets:** only dates on which *every* selected asset has a price are kept (inner join).

## 2. Returns

Daily simple return:

$$r_t = \frac{P_t}{P_{t-1}} - 1$$

Compounding: a value $V_0$ grows to $V_n = V_0 \prod_{t=1}^{n}(1 + r_t)$.

## 3. Strategies

Every strategy outputs a **target weight** $s_t \in [0, 1]$ after seeing the close of day $t$. Strategies never see future prices — the test suite checks this by cutting the data at different dates and confirming that past signals never change.

| Strategy | Rule after the close of day $t$ |
|---|---|
| Buy and Hold | $s_t = 1$ |
| Moving Average Trend | $s_t = 1$ if $\text{SMA}_{50}(t) > \text{SMA}_{200}(t)$, else $0$; $0$ during the 200-day warm-up |
| Volatility-Targeted Trend | $s_t = \text{trend}_t \times \min\left(\dfrac{\sigma^*}{\hat\sigma_t},\ w_{\max}\right)$ |

where $\text{SMA}_n(t)$ is the average of the last $n$ closes including day $t$, $\sigma^*$ is the target volatility (default 15%), $\hat\sigma_t$ is the annualized standard deviation of the last 60 daily returns, and $w_{\max}$ is the maximum weight (default 1.0, meaning **no leverage**).

**Why volatility targeting works (approximately):** holding a fraction $w$ of an asset with volatility $\sigma$ and keeping the rest in cash gives portfolio volatility $\approx w\sigma$. Choosing $w = \sigma^*/\hat\sigma$ makes that $\approx \sigma^*$. This relies on recent volatility being a reasonable forecast of near-term volatility. Volatility does tend to cluster, but sudden shocks that begin on a calm day are not anticipated.

## 4. Backtest timing (look-ahead prevention)

1. The signal $s_t$ uses prices up to the close of day $t$.
2. The trade is assumed to happen at the close of day $t+1$ (default `execution_delay = 1`), because you can't know a closing price and trade at that same close.
3. The new position earns the return of day $t+2$.

So the weight held during day $t$ is:

$$w_t = s_{t-1-d}, \qquad d = \text{execution delay}$$

Setting $d = 0$ gives the common textbook version (trade at the same close as the signal). It is available, but labeled as optimistic.

## 5. Portfolio accounting and costs

Gross daily return, with any uninvested cash earning a daily cash rate $c$ (default 0):

$$g_t = w_t\, r_t + (1 - w_t)\, c$$

Before trading, the weight has **drifted** with the day's prices:

$$\tilde w_{t} = \frac{w_t (1 + r_t)}{1 + g_t}$$

Turnover is the size of the trade needed to reach the new weight, and the cost is charged on it:

$$\text{turnover}_t = |w_t - \tilde w_{t-1}|, \qquad \text{cost}_t = \text{turnover}_t \times \frac{\text{commission}_{bps} + \text{slippage}_{bps}}{10{,}000}$$

Net return and equity:

$$R_t = (1 - \text{cost}_t)(1 + g_t) - 1, \qquad E_t = E_0 \prod_{i=1}^{t}(1 + R_i)$$

Weights above 1 (leverage) and below 0 (shorting) are rejected unless leverage is explicitly enabled.

## 6. Performance metrics

All metrics use daily strategy returns $R_t$, $n$ = number of daily returns, and 252 trading days per year.

| Metric | Formula | Notes |
|---|---|---|
| Total return | $\prod (1+R_t) - 1$ | |
| CAGR | $(1 + \text{TR})^{252/n} - 1$ | Reported as n/a for less than 252 days, to avoid annualizing a lucky month |
| Annualized volatility | $\text{std}(R_t)\sqrt{252}$ | Sample std (ddof = 1) |
| Sharpe ratio | $\dfrac{\overline{R - r_f}}{\text{std}(R - r_f)}\sqrt{252}$ | $r_f$ = daily equivalent of the annual risk-free rate; n/a if volatility is zero |
| Sortino ratio | $\dfrac{252 \cdot \overline{R - r_f}}{\sqrt{252}\sqrt{\overline{\min(R - r_f, 0)^2}}}$ | Only below-target days count as risk |
| Maximum drawdown | $\min_t \left(\dfrac{V_t}{\max_{s \le t} V_s} - 1\right)$ | Peak includes the starting value |
| Calmar ratio | $\text{CAGR} / \lvert\text{MDD}\rvert$ | n/a if there was no drawdown |
| % positive days | share of days with $R_t > 0$ | Days in cash (0%) are not positive |
| Best / worst day | $\max R_t$, $\min R_t$ | |
| Average exposure | mean of $w_t$ | 1.0 = always fully invested |
| Annual turnover | $\sum \text{turnover}_t / (n/252)$ | 1.0 = traded the full portfolio value once per year |

The daily risk-free rate is $r_f = (1 + \text{annual rate})^{1/252} - 1$.

## 7. Multi-asset portfolios

- The user picks **fixed target weights** (equal or custom). No returns are used to choose them.
- Day 0: buy the targets at the close (cost charged on 100% turnover).
- Between rebalances, each weight drifts: $w_{i,t} = \dfrac{w_{i,t-1}(1 + r_{i,t})}{1 + \sum_j w_{j,t-1} r_{j,t}}$.
- Rebalance dates are the **first trading day of each new month, quarter, or year**. That is known from the calendar alone, so no future information is used. On those days, turnover $= \sum_i |w_i^{\text{target}} - w_{i,t}|$ and the cost is charged.
- **Portfolio volatility (fixed weights):** $\sigma_p = \sqrt{w^\top \Sigma w}$, with $\Sigma$ the annualized covariance of daily returns.
- **Risk contribution share:** $\dfrac{w_i (\Sigma w)_i}{w^\top \Sigma w}$, which sums to 100%.
- **Diversification ratio:** $\dfrac{\sum_i w_i \sigma_i}{\sigma_p}$, where 1.0 means no diversification benefit.
- **Return contribution:** $\sum_t w_{i,t-1}\, r_{i,t}$ (arithmetic, so compounding is not included).

These portfolio statistics use the **full sample** and are descriptive. They show what happened, and none of them claims to identify an "optimal" portfolio.

## 8. Stress testing

Fixed windows are measured from the S&P 500's approximate peak to its approximate trough:

| Period | Window |
|---|---|
| 2008 Global Financial Crisis | 2007-10-09 to 2009-03-09 |
| 2020 COVID-19 Crash | 2020-02-19 to 2020-03-23 |
| 2022 Equity Bear Market | 2022-01-03 to 2022-10-12 |

For each window the dashboard reports the period return, the maximum drawdown inside the window, annualized volatility, and the number of trading days after the window until equity regained its pre-window level (if that happened within the data). A window is analyzed only if the data covers all of it, and with synthetic data the named crises are not analyzed at all.

## 9. In-sample vs. out-of-sample

The history is split **chronologically**: the first X% of dates is the training period and the rest is the test period. Shuffling is never used. The backtest runs on the full history and is then sliced, so 200-day indicators are already "warm" at the start of the test period. This is not leakage, because every indicator only looks backward. A grid of moving-average windows is ranked by training-period Sharpe so you can see how the best in-sample choice performs out-of-sample.
