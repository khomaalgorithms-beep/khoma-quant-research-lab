
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # draw without needing a screen (works on servers and in tests)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

# Fixed categorical order: series 1 is always blue, 2 orange, 3 aqua, ...
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT_PRIMARY = "#1B2430"
TEXT_MUTED = "#6B7280"
GRID_COLOR = "#E5E7EB"
SURFACE = "#FFFFFF"
# Diverging scale for correlations: red (-1) -> gray (0) -> blue (+1)
DIVERGING_SCALE = [[0.0, "#c43c3b"], [0.25, "#e89a92"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1.0, "#1c5cab"]]


def color_for(i: int) -> str:
    return SERIES_COLORS[i % len(SERIES_COLORS)]


def _style(fig: go.Figure, title: str, y_title: str = "", y_format: str | None = None,
           height: int = 420) -> go.Figure:
    """Apply one consistent, quiet look to every Plotly chart."""
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": TEXT_PRIMARY}, "x": 0, "xanchor": "left"},
        height=height,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "Inter, Segoe UI, Helvetica, Arial, sans-serif", "color": TEXT_PRIMARY, "size": 12},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID_COLOR, tickfont={"color": TEXT_MUTED})
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, title=y_title, tickfont={"color": TEXT_MUTED},
                     tickformat=y_format)
    return fig


# ---------------------------------------------------------------------------
# Plotly charts (dashboard)
# ---------------------------------------------------------------------------

def price_chart(prices: pd.Series, moving_averages: dict[str, pd.Series] | None = None,
                title: str = "Price") -> go.Figure:
    """Price line, optionally with moving averages on top."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Adjusted close",
                             line={"color": color_for(0), "width": 2}))
    for i, (label, series) in enumerate((moving_averages or {}).items(), start=1):
        fig.add_trace(go.Scatter(x=series.index, y=series, name=label,
                                 line={"color": color_for(i), "width": 1.5}))
    return _style(fig, title, "Price")


def normalized_price_chart(prices: pd.DataFrame, title: str = "Growth of $1") -> go.Figure:
    """Every asset rebased to 1.0 on the first date, so different price levels compare fairly."""
    rebased = prices / prices.iloc[0]
    fig = go.Figure()
    for i, col in enumerate(rebased.columns):
        fig.add_trace(go.Scatter(x=rebased.index, y=rebased[col], name=col,
                                 line={"color": color_for(i), "width": 2}))
    return _style(fig, title, "Value of $1", y_format="$.2f")


def equity_chart(equity_curves: dict[str, pd.Series], title: str = "Equity curve",
                 log_scale: bool = False) -> go.Figure:
    """Portfolio value over time for one or more strategies."""
    fig = go.Figure()
    for i, (name, equity) in enumerate(equity_curves.items()):
        fig.add_trace(go.Scatter(x=equity.index, y=equity, name=name,
                                 line={"color": color_for(i), "width": 2},
                                 hovertemplate="%{y:$,.0f}"))
    fig = _style(fig, title, "Portfolio value", y_format="$,.0f")
    if log_scale:
        fig.update_yaxes(type="log")
    return fig


def drawdown_chart(drawdowns: dict[str, pd.Series], title: str = "Drawdown from previous peak") -> go.Figure:
    """Drawdown (negative %) over time - shows how deep and how long the losses were."""
    fig = go.Figure()
    for i, (name, dd) in enumerate(drawdowns.items()):
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=name, line={"color": color_for(i), "width": 1.5},
                                 hovertemplate="%{y:.1%}"))
    return _style(fig, title, "Drawdown", y_format=".0%", height=320)


def annual_returns_chart(annual: pd.DataFrame, title: str = "Calendar-year returns") -> go.Figure:
    """Grouped bars: one group per year, one bar per strategy."""
    fig = go.Figure()
    for i, col in enumerate(annual.columns):
        fig.add_trace(go.Bar(x=annual.index.astype(str), y=annual[col], name=col,
                             marker={"color": color_for(i)}, hovertemplate="%{y:.1%}"))
    fig = _style(fig, title, "Return", y_format=".0%", height=360)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.add_hline(y=0, line={"color": TEXT_MUTED, "width": 1})
    return fig


def rolling_volatility_chart(vols: dict[str, pd.Series], title: str = "Rolling annualized volatility") -> go.Figure:
    fig = go.Figure()
    for i, (name, vol) in enumerate(vols.items()):
        fig.add_trace(go.Scatter(x=vol.index, y=vol, name=name, line={"color": color_for(i), "width": 1.5},
                                 hovertemplate="%{y:.1%}"))
    return _style(fig, title, "Volatility", y_format=".0%", height=320)


def exposure_chart(weights: pd.Series, title: str = "Position size (weight in the asset)") -> go.Figure:
    fig = go.Figure(go.Scatter(x=weights.index, y=weights, name="Weight", fill="tozeroy",
                               line={"color": color_for(0), "width": 1}, hovertemplate="%{y:.0%}"))
    fig = _style(fig, title, "Weight", y_format=".0%", height=260)
    fig.update_yaxes(range=[0, max(1.05, float(weights.max()) * 1.05)])
    return fig


def correlation_heatmap(corr: pd.DataFrame, title: str = "Correlation of daily returns") -> go.Figure:
    """Correlation matrix on a fixed -1..+1 diverging scale with the value printed in each cell."""
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1, colorscale=DIVERGING_SCALE,
        text=np.round(corr.values, 2), texttemplate="%{text:.2f}",
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        xgap=2, ygap=2, colorbar={"title": "corr"},
    ))
    fig = _style(fig, title, height=420)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", gridcolor=SURFACE)
    return fig


def bar_chart(values: pd.Series, title: str, y_format: str = ".0%") -> go.Figure:
    """Simple single-series bar chart (e.g. risk contribution by asset)."""
    fig = go.Figure(go.Bar(x=values.index.astype(str), y=values.values, marker={"color": color_for(0)},
                           hovertemplate="%{x}: %{y:" + y_format + "}<extra></extra>"))
    fig = _style(fig, title, y_format=y_format, height=320)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Matplotlib charts (static images for reports and the notebook)
# ---------------------------------------------------------------------------

def static_equity_and_drawdown(equity_curves: dict[str, pd.Series], title: str = "") -> plt.Figure:
    """Two stacked panels: equity curves on top, drawdowns underneath."""
    fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})
    for i, (name, equity) in enumerate(equity_curves.items()):
        ax_eq.plot(equity.index, equity, color=color_for(i), linewidth=1.6, label=name)
        drawdown = equity / equity.cummax() - 1
        ax_dd.plot(drawdown.index, drawdown, color=color_for(i), linewidth=1.2)

    ax_eq.set_ylabel("Portfolio value")
    ax_eq.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax_eq.legend(frameon=False, loc="upper left")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for ax in (ax_eq, ax_dd):
        ax.grid(color=GRID_COLOR, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax_eq.set_title(title, loc="left", fontsize=12)
    fig.tight_layout()
    return fig
