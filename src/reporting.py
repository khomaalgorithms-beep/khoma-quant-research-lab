
from __future__ import annotations

import base64
import html
import io
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from src.risk_metrics import (
    METRIC_LABELS,
    drawdown_recovery,
    format_metric,
    performance_summary,
)
from src.visualization import static_equity_and_drawdown

LIMITATIONS = [
    "Results are historical simulations. They are not predictions and do not guarantee future returns.",
    "Execution is simplified: trades fill at the daily close plus a fixed cost in basis points. "
    "Real fills, spreads and market impact can be worse.",
    "Taxes, borrowing costs, and fund expense ratios beyond what is reflected in adjusted prices are not modeled.",
    "Yahoo Finance data is free and unofficial; it can contain errors, revisions and gaps.",
    "The tested assets were chosen with hindsight (selection and survivorship bias). "
    "Assets that failed or were delisted are not in the test.",
    "Strategy parameters are common defaults, but any parameter seen to work on past data "
    "may be overfitted. Check the out-of-sample (test) period before trusting a result.",
]


def _summaries(backtests: dict[str, pd.DataFrame], risk_free_rate: float) -> dict[str, dict]:
    return {name: performance_summary(bt, risk_free_rate) for name, bt in backtests.items()}


def _metrics_table_rows(summaries: dict[str, dict]) -> list[list[str]]:
    rows = []
    for key, label in METRIC_LABELS.items():
        rows.append([label] + [format_metric(key, s[key]) for s in summaries.values()])
    return rows


def _drawdown_rows(backtests: dict[str, pd.DataFrame]) -> list[list[str]]:
    rows = []
    for name, bt in backtests.items():
        info = drawdown_recovery(bt["strategy_return"].dropna())
        fmt = lambda d: d.date().isoformat() if d is not None else "not recovered in data"  # noqa: E731
        rows.append([
            name,
            fmt(info["peak"]) if info["peak"] is not None else "n/a",
            fmt(info["trough"]) if info["trough"] is not None else "n/a",
            fmt(info["recovery"]) if info["peak"] is not None else "n/a",
            str(info["days_to_recover"]) if info["days_to_recover"] is not None else "-",
        ])
    return rows


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _settings_rows(settings: dict) -> list[list[str]]:
    return [[str(k), str(v)] for k, v in settings.items()]


def _benchmark_comparison(summaries: dict[str, dict], benchmark_name: str) -> list[str]:
    """Plain-English lines comparing each strategy with the benchmark."""
    if benchmark_name not in summaries:
        return []
    bench = summaries[benchmark_name]
    lines = []
    for name, s in summaries.items():
        if name == benchmark_name:
            continue
        parts = []
        for key, label in [("cagr", "CAGR"), ("annual_volatility", "volatility"),
                           ("max_drawdown", "max drawdown"), ("sharpe", "Sharpe")]:
            parts.append(f"{label} {format_metric(key, s[key])} vs {format_metric(key, bench[key])}")
        lines.append(f"{name} vs {benchmark_name}: " + "; ".join(parts) + ".")
    return lines


def generate_markdown_report(
    settings: dict,
    backtests: dict[str, pd.DataFrame],
    benchmark_name: str,
    risk_free_rate: float = 0.0,
    data_note: str = "",
) -> str:
    """
    Build a Markdown research summary.

    settings  : the assumptions to print (strategy, assets, dates, capital, parameters, costs, ...)
    backtests : {strategy_name: backtest DataFrame}
    data_note : e.g. a warning that the data is synthetic
    """
    summaries = _summaries(backtests, risk_free_rate)
    names = list(summaries)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts = [
        "# KHOMA Quant Research Lab - Research Summary",
        f"_Generated {generated} from calculated backtest results._",
    ]
    if data_note:
        parts.append(f"> **Data note:** {data_note}")
    parts += [
        "## Settings and assumptions",
        _markdown_table(["Setting", "Value"], _settings_rows(settings)),
        "## Performance statistics",
        _markdown_table(["Metric"] + names, _metrics_table_rows(summaries)),
        "## Benchmark comparison",
        "\n".join(f"- {line}" for line in _benchmark_comparison(summaries, benchmark_name)) or "- (no benchmark)",
        "## Worst drawdown",
        _markdown_table(["Strategy", "Peak", "Trough", "Recovered", "Days trough to recovery"],
                        _drawdown_rows(backtests)),
        "## Research limitations",
        "\n".join(f"- {item}" for item in LIMITATIONS),
        "---\n_Historical research only. Not investment advice._",
    ]
    return "\n\n".join(parts) + "\n"


def _figure_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _html_table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def generate_html_report(
    settings: dict,
    backtests: dict[str, pd.DataFrame],
    benchmark_name: str,
    risk_free_rate: float = 0.0,
    data_note: str = "",
) -> str:
    """Same content as the Markdown report, as one self-contained HTML file with an embedded chart."""
    summaries = _summaries(backtests, risk_free_rate)
    names = list(summaries)
    chart = _figure_to_base64(
        static_equity_and_drawdown({n: bt["equity"] for n, bt in backtests.items()})
    )
    note = f'<p class="note"><strong>Data note:</strong> {html.escape(data_note)}</p>' if data_note else ""
    comparison = "".join(f"<li>{html.escape(line)}</li>" for line in _benchmark_comparison(summaries, benchmark_name))
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in LIMITATIONS)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KHOMA Research Summary</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1B2430;
         max-width: 960px; margin: 32px auto; padding: 0 16px; line-height: 1.5; background: #fff; }}
  h1 {{ font-size: 26px; margin-bottom: 4px; }} h2 {{ font-size: 18px; margin-top: 32px;
         border-bottom: 1px solid #E5E7EB; padding-bottom: 4px; }}
  .meta {{ color: #6B7280; font-size: 13px; }}
  .note {{ background: #FFF7E6; border-left: 4px solid #eda100; padding: 8px 12px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ border-bottom: 1px solid #E5E7EB; padding: 6px 10px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #F3F5F8; }}
  img {{ max-width: 100%; }}
</style></head><body>
<h1>KHOMA Quant Research Lab - Research Summary</h1>
<p class="meta">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} from calculated backtest results.</p>
{note}
<h2>Settings and assumptions</h2>{_html_table(["Setting", "Value"], _settings_rows(settings))}
<h2>Equity and drawdown</h2><img alt="Equity curve and drawdown chart" src="data:image/png;base64,{chart}">
<h2>Performance statistics</h2>{_html_table(["Metric"] + names, _metrics_table_rows(summaries))}
<h2>Benchmark comparison</h2><ul>{comparison}</ul>
<h2>Worst drawdown</h2>{_html_table(["Strategy", "Peak", "Trough", "Recovered", "Days trough to recovery"],
                                     _drawdown_rows(backtests))}
<h2>Research limitations</h2><ul>{limitations}</ul>
<p class="meta">Historical research only. Not investment advice.</p>
</body></html>
"""
