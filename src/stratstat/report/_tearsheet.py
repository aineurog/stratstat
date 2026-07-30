"""Single-strategy tear sheet.

Composes an equity curve, drawdown chart, monthly heatmap, and
summary statistics table into a single multi-panel figure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from stratstat.report._charts import (
    _cumulative_returns,
    _drawdown_series,
    _ensure_plotly,
    _monthly_heatmap_data,
    _to_array,
)


def _stats_table_data(
    r: NDArray[np.floating],
    periods_per_year: int | None,
) -> dict[str, Any]:
    """Compute a summary statistics dictionary for a single strategy.

    Delegates to registered metrics where possible.
    """
    from stratstat.inputs import ReturnsInput
    from stratstat.registry import _compute_one

    inp = ReturnsInput(r.ravel(), periods_per_year=periods_per_year)
    stats: dict[str, Any] = {}

    metric_names = [
        "cagr", "annualized_volatility", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio", "skewness", "excess_kurtosis",
        "var", "cvar",
    ]
    for name in metric_names:
        try:
            result = _compute_one(inp, name)
            val = result.value
            if isinstance(val, np.ndarray) and val.shape != ():
                val = float(val.flat[0])
            stats[name] = float(val) if val is not None else np.nan
        except (ValueError, KeyError):
            stats[name] = np.nan

    return stats


def tear_sheet(
    returns: Any,
    benchmark: Any | None = None,
    periods_per_year: int | None = None,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Single-strategy tear sheet.

    Produces a four-panel figure:
    1. Equity curve (cumulative return)
    2. Drawdown chart
    3. Monthly returns heatmap
    4. Summary statistics table

    Parameters
    ----------
    returns: ReturnsInput or 1-D array of strategy returns.
    benchmark: Optional benchmark returns (1-D array).
    periods_per_year: Annualization factor.
    title: Optional overall title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    r = _to_array(returns)
    strat = r[:, 0] if r.ndim > 1 and r.shape[1] >= 1 else r.ravel()

    cum = _cumulative_returns(strat.reshape(-1, 1))[:, 0]
    dd = _drawdown_series(strat.reshape(-1, 1))[:, 0]
    grid, years, months = _monthly_heatmap_data(strat)

    # Stats
    stats = _stats_table_data(strat, periods_per_year)

    # Build figure with 2×2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Equity Curve",
            "Drawdown",
            "Monthly Returns Heatmap",
            "Summary Statistics",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "table"}],
        ],
        row_heights=[0.5, 0.5],
        column_widths=[0.55, 0.45],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    # -- Panel 1: Equity Curve --------------------------------------------
    x = np.arange(len(cum))
    fig.add_trace(go.Scatter(
        x=x, y=cum, mode="lines", name="Strategy",
        line={"color": "steelblue"},
        hovertemplate="%{y:.3f}<extra></extra>",
    ), row=1, col=1)

    if benchmark is not None:
        bench_arr = np.asarray(benchmark, dtype=np.float64).ravel()
        n_b = min(len(bench_arr), len(strat))
        bench_cum = np.cumprod(1.0 + np.where(np.isfinite(bench_arr[:n_b]),
                                              bench_arr[:n_b], 0.0))
        fig.add_trace(go.Scatter(
            x=np.arange(len(bench_cum)), y=bench_cum,
            mode="lines", name="Benchmark",
            line={"dash": "dash", "color": "gray"},
            hovertemplate="%{y:.3f}<extra></extra>",
        ), row=1, col=1)

    # -- Panel 2: Drawdown ------------------------------------------------
    fig.add_trace(go.Scatter(
        x=x, y=dd, mode="lines", name="Drawdown",
        fill="tozeroy",
        fillcolor="rgba(220, 50, 50, 0.3)",
        line={"color": "crimson"},
        hovertemplate="%{y:.1%}<extra></extra>",
    ), row=1, col=2)

    # -- Panel 3: Monthly Heatmap -----------------------------------------
    if grid.size > 0:
        fig.add_trace(go.Heatmap(
            z=grid,
            x=months[:grid.shape[1]] if grid.shape[1] <= 12 else months,
            y=[str(y) for y in years],
            colorscale="RdYlGn",
            zmid=0,
            texttemplate="%{z:.1%}",
            textfont={"size": 9},
            hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
            showscale=False,
        ), row=2, col=1)

    # -- Panel 4: Stats Table ---------------------------------------------
    header_vals = list(stats.keys())
    cell_vals = [[f"{stats[k]:.4f}" if not np.isnan(stats[k]) else "N/A"
                   for k in header_vals]]
    header_labels = [k.replace("_", " ").title() for k in header_vals]

    fig.add_trace(go.Table(
        header={"values": header_labels,
                    "font": {"size": 10},
                    "align": "center"},
        cells={"values": cell_vals,
                   "font": {"size": 10},
                   "align": "center",
                   "height": 25},
    ), row=2, col=2)

    fig.update_layout(
        title=title or "Strategy Tear Sheet",
        hovermode="x unified",
        showlegend=False,
        height=900,
    )
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_xaxes(title_text="Period", row=1, col=2)

    return fig
