"""Standalone HTML report generator.

Produces a self-contained HTML strategy analysis report with embedded
charts (via plotly CDN) and auto-discovered metrics grouped by category.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np

from stratstat.report._charts import (
    _cumulative_returns,
    _drawdown_series,
    _ensure_plotly,
    _monthly_heatmap_data,
    _to_array,
)
from stratstat.report._common import discover_and_format


def generate_report(
    returns: Any,
    output_path: str | Path,
    *,
    benchmark: Any | None = None,
    periods_per_year: int | None = None,
    title: str | None = None,
    include_benchmark_metrics: bool = True,
) -> None:
    """Generate a self-contained HTML strategy analysis report.

    Produces a single ``.html`` file containing:
    * Equity curve chart
    * Drawdown chart
    * Monthly returns heatmap
    * Benchmark comparison chart (if *benchmark* provided)
    * Grouped statistics tables (auto-discovered from registry)
    * Methodology references

    Requires ``plotly`` (``pip install stratstat[report]``).

    Parameters
    ----------
    returns: ``ReturnsInput`` or array-like strategy returns.
    output_path: File path for the output HTML file.  Parent
        directories are created automatically.
    benchmark: Optional benchmark returns (1-D array-like).
    periods_per_year: Annualization factor.
    title: Report title (default: "Strategy Analysis Report").
    include_benchmark_metrics: If True (default) and *benchmark* is
        provided, include benchmark-tier statistics.
    """
    _ensure_plotly()
    import plotly.graph_objects as go
    from plotly.io import to_html as fig_to_html_div

    from stratstat.inputs import BenchmarkInput, ReturnsInput

    # -- Normalise inputs -------------------------------------------------
    r = _to_array(returns)
    strat = r[:, 0] if r.ndim > 1 and r.shape[1] >= 1 else r.ravel()
    p = Path(output_path) if isinstance(output_path, str) else output_path
    title = title or "Strategy Analysis Report"

    # -- Generate charts --------------------------------------------------
    charts_html: list[str] = []

    # Equity curve
    cum = _cumulative_returns(strat.reshape(-1, 1))[:, 0]
    x = np.arange(len(cum))
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=x, y=cum, mode="lines", name="Strategy",
        line={"color": "steelblue"},
        hovertemplate="%{y:.3f}<extra></extra>",
    ))
    if benchmark is not None:
        bench_arr = np.asarray(benchmark, dtype=np.float64).ravel()
        n_b = min(len(bench_arr), len(strat))
        bench_cum = np.cumprod(1.0 + np.where(
            np.isfinite(bench_arr[:n_b]), bench_arr[:n_b], 0.0))
        fig_eq.add_trace(go.Scatter(
            x=np.arange(len(bench_cum)), y=bench_cum,
            mode="lines", name="Benchmark",
            line={"dash": "dash", "color": "gray"},
            hovertemplate="%{y:.3f}<extra></extra>",
        ))
    fig_eq.update_layout(
        title="Equity Curve",
        yaxis_title="Cumulative Return",
        xaxis_title="Period",
        hovermode="x unified",
    )
    charts_html.append(fig_to_html_div(fig_eq, full_html=False, include_plotlyjs=False))
    charts_html.append("")

    # Drawdown
    dd = _drawdown_series(strat.reshape(-1, 1))[:, 0]
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=x, y=dd, mode="lines", name="Drawdown",
        fill="tozeroy",
        fillcolor="rgba(220, 50, 50, 0.3)",
        line={"color": "crimson"},
        hovertemplate="%{y:.1%}<extra></extra>",
    ))
    fig_dd.update_layout(
        title="Drawdown",
        yaxis_title="Drawdown",
        xaxis_title="Period",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )
    charts_html.append(fig_to_html_div(fig_dd, full_html=False, include_plotlyjs=False))
    charts_html.append("")

    # Monthly heatmap
    grid, years, months = _monthly_heatmap_data(strat)
    if grid.size > 0:
        fig_hm = go.Figure(data=go.Heatmap(
            z=grid,
            x=months[:grid.shape[1]] if grid.shape[1] <= 12 else months,
            y=[str(y) for y in years],
            colorscale="RdYlGn",
            zmid=0,
            texttemplate="%{z:.1%}",
            textfont={"size": 10},
            hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
        ))
        fig_hm.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            yaxis={"autorange": "reversed"},
        )
        charts_html.append(
            fig_to_html_div(fig_hm, full_html=False, include_plotlyjs=False)
        )
        charts_html.append("")

    # -- Statistics tables ------------------------------------------------
    inp = ReturnsInput(strat, periods_per_year=periods_per_year)
    sections = discover_and_format(
        inp, ["descriptive", "risk", "risk_adjusted", "inference"]
    )

    # Benchmark metrics
    if benchmark is not None and include_benchmark_metrics:
        bench_arr = np.asarray(benchmark, dtype=np.float64).ravel()
        bm_inp = BenchmarkInput(
            returns=strat, benchmark=bench_arr, periods_per_year=periods_per_year
        )
        bm_sections = discover_and_format(bm_inp, ["benchmark"])
        sections.extend(bm_sections)

    stats_html = _build_stats_html(sections)

    # -- Methodology references -------------------------------------------
    refs_html = _build_methodology_html(sections)

    # -- Assemble full HTML -----------------------------------------------
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    n_metrics = sum(len(s["metrics"]) for s in sections)

    html = _HTML_TEMPLATE.format(
        title=title,
        date=date_str,
        n_metrics=n_metrics,
        n_periods=len(strat),
        ppy=periods_per_year or "N/A",
        charts="\n".join(charts_html),
        statistics=stats_html,
        methodology=refs_html,
    )

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_stats_html(sections: list[dict[str, Any]]) -> str:
    """Build the statistics section HTML with grouped tables."""
    parts: list[str] = ['<div class="statistics">']
    parts.append('<h2>Statistics</h2>')

    for section in sections:
        label = section["section"]
        metrics = section["metrics"]
        if not metrics:
            continue
        parts.append(f"<h3>{label}</h3>")
        parts.append("<table>")
        parts.append(
            "<thead><tr>"
            "<th>Metric</th><th>Value</th><th>Citation</th>"
            "</tr></thead>"
        )
        parts.append("<tbody>")
        for i, m in enumerate(metrics):
            bg = ' style="background:#f8fafc"' if i % 2 == 0 else ""
            val = m["value"]
            if isinstance(val, (float, np.floating)):
                val_str = f"{float(val):.6g}" if np.isfinite(val) else "N/A"
            elif val is None:
                val_str = "N/A"
            elif isinstance(val, np.ndarray):
                if val.size <= 5:
                    inner = ", ".join(
                        f"{float(v):.6g}" for v in val.ravel()
                    )
                    val_str = f"[{inner}]"
                else:
                    val_str = f"array{val.shape}"
            else:
                val_str = str(val)

            ref = m.get("ref", "")
            ref_attr = ref.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
            ref_short = (ref[:120] + "…") if len(ref) > 120 else ref

            parts.append(
                f"<tr{bg}>"
                f'<td class="metric-name">{m["name"]}</td>'
                f'<td class="metric-val">{val_str}</td>'
                f'<td class="metric-meta" title="{ref_attr}">{ref_short}</td>'
                f"</tr>"
            )
        parts.append("</tbody></table>")

    parts.append("</div>")
    return "\n".join(parts)


def _build_methodology_html(sections: list[dict[str, Any]]) -> str:
    """Build methodology footnotes from metric references."""
    seen: set[str] = set()
    refs: list[str] = []
    for section in sections:
        for m in section["metrics"]:
            ref = m.get("ref", "")
            if ref and ref not in seen:
                seen.add(ref)
                refs.append(ref)

    if not refs:
        return ""

    parts: list[str] = ['<div class="methodology">']
    parts.append("<h2>Methodology</h2>")
    parts.append("<ol>")
    for ref in refs:
        ref_escaped = ref.replace("&", "&amp;").replace("<", "&lt;")
        parts.append(f"<li>{ref_escaped}</li>")
    parts.append("</ol>")
    parts.append("</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body {{
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: #f8fafc;
    color: #1a202c;
}}
h1 {{
    color: #1a365d;
    border-bottom: 3px solid #3182ce;
    padding-bottom: 10px;
    font-size: 28px;
}}
h2 {{
    margin-top: 40px;
    padding-bottom: 6px;
    font-size: 20px;
    color: #3182ce;
    border-bottom: 2px solid #3182ce;
}}
h3 {{
    color: #555;
    margin-top: 20px;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-bottom: 10px;
}}
th {{
    background: #edf2f7;
    text-align: left;
    padding: 7px 10px;
    border-bottom: 2px solid #cbd5e0;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
td {{
    padding: 5px 10px;
    border-bottom: 1px solid #e2e8f0;
}}
tr:hover {{
    background: #f7fafc;
}}
.metric-name {{
    font-weight: 600;
    font-family: ui-monospace, monospace;
    font-size: 12px;
}}
.metric-val {{
    font-family: ui-monospace, monospace;
    color: #2d3748;
}}
.metric-meta {{
    color: #a0aec0;
    font-size: 10px;
    max-width: 400px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
.summary {{
    color: #666;
    font-size: 14px;
    margin-bottom: 30px;
}}
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 5px;
    color: white;
}}
.charts {{
    margin: 20px 0;
}}
.charts .plotly-graph-div {{
    margin-bottom: 30px;
}}
.methodology {{
    margin-top: 40px;
    font-size: 12px;
    color: #666;
}}
.methodology ol {{
    padding-left: 20px;
}}
.methodology li {{
    margin-bottom: 4px;
}}
</style>
</head>
<body>

<h1>📊 {title}</h1>
<p class="summary">
    {n_periods} periods &nbsp;|&nbsp;
    Periods/year: {ppy} &nbsp;|&nbsp;
    <strong>{n_metrics} metrics</strong> across sections
    &nbsp;|&nbsp; Generated: {date}
</p>

<div class="charts">
{charts}
</div>

{statistics}

{methodology}

</body>
</html>"""
