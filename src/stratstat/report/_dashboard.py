"""Multi-strategy comparison dashboard.

Composes equity curves overlay, rolling-metric charts, a correlation
heatmap, and a ranking table into a single multi-panel figure.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from stratstat.report._charts import (
    _compute_rolling_metric,
    _cumulative_returns,
    _ensure_plotly,
    _to_array,
)

# Key metrics shown in the ranking table.  These are the most commonly
# used performance indicators.  Each must be a registered returns-tier
# metric that accepts (ReturnsInput) and returns a scalar float value.
_RANKING_METRICS = [
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
]


def _correlation_matrix(r: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute Pearson correlation matrix with NaN handling."""
    n_strat = r.shape[1]
    corr: NDArray[np.floating] = np.full((n_strat, n_strat), np.nan)
    for i in range(n_strat):
        for j in range(i, n_strat):
            mask = np.isfinite(r[:, i]) & np.isfinite(r[:, j])
            if mask.sum() >= 3:
                c = np.corrcoef(r[mask, i], r[mask, j])[0, 1]
                corr[i, j] = c
                corr[j, i] = c
    return corr


def dashboard(
    returns: Any,
    periods_per_year: int | None = None,
    rolling_window: int = 60,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Multi-strategy comparison dashboard.

    Produces a four-panel figure:
    1. Equity curves overlay (all strategies)
    2. Rolling Sharpe ratio overlay
    3. Correlation heatmap
    4. Performance ranking table (auto-discovered metrics)

    Parameters
    ----------
    returns: CompareInput or 2-D array of shape ``(n_periods, n_strategies)``.
    periods_per_year: Annualization factor.
    rolling_window: Window size for rolling metrics.
    title: Optional overall title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    r = _to_array(returns)
    n_strat = r.shape[1]
    labels = [f"S{i + 1}" for i in range(n_strat)]

    cum = _cumulative_returns(r)
    x = np.arange(cum.shape[0])

    # Compute rolling Sharpe for each strategy
    rolling_sharpes = [
        _compute_rolling_metric(r[:, i : i + 1], "sharpe_ratio", rolling_window, periods_per_year)
        for i in range(n_strat)
    ]

    # Correlation
    corr = _correlation_matrix(r)

    # Compute performance stats for ranking — dynamic discovery
    rankings = _compute_rankings(r, periods_per_year)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Equity Curves",
            f"Rolling Sharpe Ratio ({rolling_window}-period)",
            "Correlation Matrix",
            "Performance Rankings",
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

    # -- Panel 1: Equity Curves -------------------------------------------
    colors = ["steelblue", "crimson", "green", "orange", "purple", "brown"]
    for i in range(n_strat):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=cum[:, i],
                mode="lines",
                name=labels[i],
                line={"color": colors[i % len(colors)]},
                hovertemplate="%{y:.3f}<extra>" + labels[i] + "</extra>",
            ),
            row=1,
            col=1,
        )

    # -- Panel 2: Rolling Sharpe ------------------------------------------
    for i in range(n_strat):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rolling_sharpes[i],
                mode="lines",
                name=labels[i],
                line={"color": colors[i % len(colors)]},
                hovertemplate="%{y:.3f}<extra>" + labels[i] + "</extra>",
            ),
            row=1,
            col=2,
        )

    # -- Panel 3: Correlation Heatmap -------------------------------------
    fig.add_trace(
        go.Heatmap(
            z=corr,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            texttemplate="%{z:.2f}",
            textfont={"size": 11},
            hovertemplate="Corr(%{x}, %{y}) = %{z:.3f}<extra></extra>",
            showscale=False,
        ),
        row=2,
        col=1,
    )

    # -- Panel 4: Rankings Table ------------------------------------------
    # Build columns: Strategy | metric_1 | metric_2 | ...
    metric_names_ordered = sorted(
        rankings, key=lambda n: _RANKING_METRICS.index(n) if n in _RANKING_METRICS else 99
    )
    header_vals = ["Strategy"] + [n.replace("_", " ").title() for n in metric_names_ordered]
    cell_data: list[list[str]] = [
        labels,
    ]
    for mname in metric_names_ordered:
        arr = rankings[mname]
        cell_data.append(
            [f"{arr[i]:.4f}" if not np.isnan(arr[i]) else "N/A" for i in range(n_strat)]
        )

    fig.add_trace(
        go.Table(
            header={"values": header_vals, "font": {"size": 10}, "align": "center"},
            cells={"values": cell_data, "font": {"size": 10}, "align": "center", "height": 25},
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=title or "Multi-Strategy Dashboard",
        hovermode="x unified",
        height=900,
    )
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
    return fig


def _compute_rankings(
    r: NDArray[np.floating],
    periods_per_year: int | None,
) -> dict[str, NDArray[np.floating]]:
    """Compute per-strategy performance metrics for the ranking table.

    Uses the registry to discover and compute all metrics listed in
    ``_RANKING_METRICS``.  Returns a dict mapping metric name to an
    ndarray of per-strategy values.
    """
    from stratstat.inputs import ReturnsInput
    from stratstat.registry import _compute_one

    n_strat = r.shape[1]
    results: dict[str, NDArray[np.floating]] = {}

    for mname in _RANKING_METRICS:
        arr = np.full(n_strat, np.nan, dtype=np.float64)
        for i in range(n_strat):
            inp = ReturnsInput(r[:, i], periods_per_year=periods_per_year)
            try:
                result = _compute_one(inp, mname)
                val = result.value
                if isinstance(val, np.ndarray):
                    if val.size == 1:
                        arr[i] = float(val.flat[0])
                    else:
                        arr[i] = np.nan
                else:
                    arr[i] = cast(float, val) if val is not None else np.nan
            except (ValueError, KeyError):
                arr[i] = np.nan
        results[mname] = arr

    return results
