"""Individual chart functions for the report module.

Each function accepts either raw arrays or StratStat input containers,
calls ``_ensure_plotly()`` lazily, and returns a ``plotly.graph_objects.Figure``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _ensure_plotly() -> None:
    """Check that plotly is installed; raise a helpful error if not."""
    try:
        import plotly  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "plotly is required for the report module. "
            "Install it with: pip install stratstat[report]"
        ) from err


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_array(data: Any) -> NDArray[np.floating]:
    """Normalise any accepted input to a 2-D numpy array ``(n_periods, n)``."""
    from stratstat.inputs import ReturnsInput

    if isinstance(data, ReturnsInput):
        return data.values
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _cumulative_returns(r: NDArray[np.floating]) -> NDArray[np.floating]:
    """Cumulative return index from period returns ``(n_periods, n_cols)``.

    Fills NaN with 0 so the index doesn't break.

    .. note::

        This is a **chart helper**, not a metric.  Core metrics such as
        ``cumulative_return`` (in ``descriptive.py``) compute the same
        formula but return a ``MetricResult``.  The chart module needs
        the raw array series for plotting, so it keeps a separate
        implementation rather than extracting values from a
        ``MetricResult``.  If the formula logic becomes complex enough
        to warrant sharing, it should be extracted into
        ``core._utils`` as a single source of truth.
    """
    r_filled = np.where(np.isfinite(r), r, 0.0)
    return np.cumprod(1.0 + r_filled, axis=0)


def _drawdown_series(r: NDArray[np.floating]) -> NDArray[np.floating]:
    """Drawdown (underwater) series for each column ``(n_periods, n_cols)``.

    Drawdown at *t* = (peak - current) / peak, where peak is the running
    maximum of the cumulative return index.  Values are returned as
    negative percentages (e.g. -0.15 = 15 % drawdown).

    .. note::

        This is a **chart helper**, not a metric.  Core metrics such as
        ``max_drawdown`` (in ``risk.py``) compute the same drawdown
        formula but return a ``MetricResult`` with the single worst
        value.  The chart module needs the full drawdown *series* for
        plotting, so it keeps a separate implementation.  See also
        ``_cumulative_returns`` above.
    """
    cum = _cumulative_returns(r)
    running_max = np.maximum.accumulate(cum, axis=0)
    dd = np.where(running_max > 0, cum / running_max - 1.0, 0.0)
    return dd


def _recovery_mask(dd: NDArray[np.floating]) -> NDArray[np.bool_]:
    """Boolean mask for periods in recovery (drawdown < 0 but not at trough)."""
    trough = np.minimum.accumulate(dd, axis=0)
    # In recovery when current dd is above the running trough (but still negative)
    # and the trough is negative (there was a drawdown)
    recovering = (dd > trough) & (dd < 0.0)
    # Also mark the final recovery-to-zero as recovery
    # (when dd goes from negative back to 0)
    for col in range(dd.shape[1]):
        neg_mask = dd[:, col] < 0.0
        if np.any(neg_mask):
            last_neg = np.max(np.where(neg_mask)[0])
            # After last negative period, mark next period as recovery
            if last_neg + 1 < dd.shape[0] and dd[last_neg + 1, col] == 0.0:
                recovering[last_neg + 1, col] = True
    return recovering


def _monthly_heatmap_data(
    r: NDArray[np.floating],
) -> tuple[NDArray[np.floating], list[int], list[str]]:
    """Reshape returns into a years × months grid.

    Returns ``(grid, years, month_labels)`` where *grid* has shape
    ``(n_years, 12)`` with NaN for missing months.

    Requires ``pandas``, which is a core dependency of StratStat and
    therefore always available when the report module is used.
    """
    import pandas as pd

    series = r[:, 0] if r.ndim == 2 and r.shape[1] >= 1 else r.ravel()

    n = len(series)
    # Generate a monthly date range — assume month-end frequency
    dr = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="ME")
    monthly_returns: pd.Series = pd.Series(series, index=dr)

    # Pivot to years × months
    df = monthly_returns.groupby(
        [monthly_returns.index.year,  # type: ignore[attr-defined]
         monthly_returns.index.month]  # type: ignore[attr-defined]
    ).apply(
        lambda x: (1.0 + x).prod() - 1.0  # type: ignore[operator]  # compound monthly return
    )
    if df.empty:
        return np.empty((0, 12)), [], []

    grid_df = df.unstack()
    years = list(grid_df.index)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    grid = grid_df.to_numpy(dtype=np.float64)
    return grid, years, months


def _compute_rolling_metric(
    r: NDArray[np.floating],
    metric_name: str,
    window: int,
    periods_per_year: int | None,
) -> NDArray[np.floating]:
    """Compute a rolling metric over a returns series."""
    from stratstat.inputs import ReturnsInput
    from stratstat.registry import _compute_one

    n = r.shape[0]
    values = np.full(n, np.nan)
    for t in range(window - 1, n):
        win = r[t - window + 1 : t + 1]
        inp = ReturnsInput(win, periods_per_year=periods_per_year)
        try:
            result = _compute_one(inp, metric_name)
            val = result.value
            if not isinstance(val, np.ndarray) or val.shape == ():
                values[t] = float(val)
            else:
                values[t] = float(val.flat[0])
        except (ValueError, KeyError):
            values[t] = np.nan
    return values


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------


def equity_curve(
    returns: Any,
    title: str | None = None,
    label: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Equity curve (cumulative return index).

    Parameters
    ----------
    returns: ReturnsInput or array-like of shape ``(n_periods,)`` or
        ``(n_periods, n_strategies)``.
    title: Optional chart title.
    label: Optional legend label for the equity curve.
    """
    _ensure_plotly()
    import plotly.graph_objects as go

    r = _to_array(returns)
    cum = _cumulative_returns(r)
    fig = go.Figure()
    for col in range(cum.shape[1]):
        lbl = label or (f"Strategy {col + 1}" if cum.shape[1] > 1 else "Strategy")
        if cum.shape[1] > 1 and label is None:
            lbl = f"Strategy {col + 1}"
        fig.add_trace(go.Scatter(
            y=cum[:, col],
            mode="lines",
            name=lbl,
            hovertemplate="%{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title=title or "Equity Curve",
        yaxis_title="Cumulative Return",
        xaxis_title="Period",
        hovermode="x unified",
    )
    return fig


def drawdown_chart(
    returns: Any,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Underwater equity (drawdown) chart with recovery-period shading.

    Parameters
    ----------
    returns: ReturnsInput or array-like of shape ``(n_periods,)`` or
        ``(n_periods, n_strategies)``.
    title: Optional chart title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go

    r = _to_array(returns)
    dd = _drawdown_series(r)
    recovering = _recovery_mask(dd)

    fig = go.Figure()
    x = np.arange(dd.shape[0])
    for col in range(dd.shape[1]):
        lbl = f"Strategy {col + 1}" if dd.shape[1] > 1 else "Drawdown"
        # Main drawdown trace
        fig.add_trace(go.Scatter(
            x=x,
            y=dd[:, col],
            mode="lines",
            name=lbl,
            fill="tozeroy",
            fillcolor="rgba(220, 50, 50, 0.3)",
            line={"color": "crimson"},
            hovertemplate="%{y:.1%}<extra></extra>",
        ))
        # Recovery shading
        rec_mask = recovering[:, col]
        if np.any(rec_mask):
            # Find contiguous recovery segments
            fig.add_trace(go.Scatter(
                x=x[rec_mask],
                y=dd[rec_mask, col],
                mode="lines",
                name=f"{lbl} (recovery)",
                fill="tozeroy",
                fillcolor="rgba(50, 180, 80, 0.25)",
                line={"color": "green", "width": 1},
                showlegend=False,
                hovertemplate="%{y:.1%}<extra></extra>",
            ))
    fig.update_layout(
        title=title or "Drawdown",
        yaxis_title="Drawdown",
        xaxis_title="Period",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )
    return fig


def monthly_heatmap(
    returns: Any,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Monthly returns heatmap (years × months).

    Parameters
    ----------
    returns: ReturnsInput or 1-D array of monthly returns.
    title: Optional chart title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go

    r = _to_array(returns)
    grid, years, months = _monthly_heatmap_data(r)

    if grid.size == 0:
        fig = go.Figure()
        fig.update_layout(title=title or "Monthly Returns Heatmap")
        return fig

    # Build text annotations
    annots = []
    for yi, year in enumerate(years):
        for mi in range(12):
            val = grid[yi, mi] if mi < grid.shape[1] else np.nan
            if not np.isnan(val):
                annots.append({
                    "x": months[mi] if mi < len(months) else str(mi + 1),
                    "y": str(year),
                    "text": f"{val:.1%}",
                    "showarrow": False,
                    "font": {"color": "white" if abs(val) > 0.03 else "black", "size": 10},
                })

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=months[:grid.shape[1]] if grid.shape[1] <= 12 else months,
        y=[str(y) for y in years],
        colorscale="RdYlGn",
        zmid=0,
        texttemplate="%{z:.1%}",
        textfont={"size": 10},
        hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title=title or "Monthly Returns Heatmap",
        xaxis_title="Month",
        yaxis_title="Year",
        yaxis={"autorange": "reversed"},
    )
    return fig


def rolling_metric_chart(
    returns: Any,
    metric_name: str,
    window: int = 60,
    periods_per_year: int | None = None,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Rolling-metric time series.

    Computes *metric_name* over a rolling window and plots the result.

    Parameters
    ----------
    returns: ReturnsInput or array-like.
    metric_name: Name of a registered metric (e.g. "sharpe_ratio").
    window: Rolling window size in periods.
    periods_per_year: Annualization factor passed to the metric.
    title: Optional chart title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go

    r = _to_array(returns)
    rolling_vals = _compute_rolling_metric(r, metric_name, window, periods_per_year)

    fig = go.Figure()
    x = np.arange(len(rolling_vals))
    fig.add_trace(go.Scatter(
        x=x,
        y=rolling_vals,
        mode="lines",
        name=metric_name.replace("_", " ").title(),
        hovertemplate="%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title or f"Rolling {metric_name.replace('_', ' ').title()} ({window}-period window)",
        yaxis_title=metric_name.replace("_", " ").title(),
        xaxis_title="Period",
        hovermode="x unified",
    )
    return fig


def cumulative_return_chart(
    returns: Any,
    benchmark: Any | None = None,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Cumulative return index, optionally with a benchmark overlay.

    Parameters
    ----------
    returns: Strategy returns (ReturnsInput or array-like).
    benchmark: Optional benchmark returns (array-like, 1-D).
    title: Optional chart title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go

    r = _to_array(returns)
    cum = _cumulative_returns(r)
    x = np.arange(cum.shape[0])

    fig = go.Figure()
    for col in range(cum.shape[1]):
        lbl = f"Strategy {col + 1}" if cum.shape[1] > 1 else "Strategy"
        fig.add_trace(go.Scatter(
            x=x, y=cum[:, col],
            mode="lines", name=lbl,
            hovertemplate="%{y:.3f}<extra></extra>",
        ))

    if benchmark is not None:
        bench_arr = np.asarray(benchmark, dtype=np.float64).ravel()
        bench_cum = np.cumprod(1.0 + np.where(np.isfinite(bench_arr), bench_arr, 0.0))
        fig.add_trace(go.Scatter(
            x=x, y=bench_cum,
            mode="lines", name="Benchmark",
            line={"dash": "dash", "color": "gray"},
            hovertemplate="%{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        title=title or "Cumulative Return",
        yaxis_title="Cumulative Return",
        xaxis_title="Period",
        hovermode="x unified",
    )
    return fig


def benchmark_overlay_chart(
    returns: Any,
    benchmark: Any,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Two-panel chart: cumulative return (top) and active return (bottom).

    Parameters
    ----------
    returns: Strategy returns (ReturnsInput or 1-D array).
    benchmark: Benchmark returns (1-D array).
    title: Optional chart title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    r = _to_array(returns)
    bench_raw = np.asarray(benchmark, dtype=np.float64).ravel()

    strat = r[:, 0] if r.ndim > 1 and r.shape[1] >= 1 else r.ravel()

    # Align lengths
    n = min(len(strat), len(bench_raw))
    strat = np.asarray(strat[:n], dtype=np.float64)
    bench_arr: NDArray[np.floating] = np.asarray(bench_raw[:n], dtype=np.float64)

    cum_strat = np.cumprod(1.0 + np.where(np.isfinite(strat), strat, 0.0))
    cum_bench = np.cumprod(1.0 + np.where(np.isfinite(bench_arr), bench_arr, 0.0))
    active = strat - bench_arr
    cum_active = np.cumsum(np.where(np.isfinite(active), active, 0.0))

    x = np.arange(n)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4],
                        subplot_titles=("Cumulative Return", "Active Return"))

    fig.add_trace(go.Scatter(
        x=x, y=cum_strat, mode="lines", name="Strategy",
        hovertemplate="%{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=cum_bench, mode="lines", name="Benchmark",
        line={"dash": "dash", "color": "gray"},
        hovertemplate="%{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=cum_active, mode="lines", name="Cumulative Active Return",
        fill="tozeroy", fillcolor="rgba(100, 100, 200, 0.2)",
        hovertemplate="%{y:.3f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        title=title or "Benchmark Comparison",
        hovermode="x unified",
        showlegend=True,
    )
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Active Return", row=2, col=1)
    fig.update_xaxes(title_text="Period", row=2, col=1)
    return fig


def trade_markers_chart(
    trades: Any,
    returns: Any | None = None,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Equity curve with trade entry/exit markers and P&L highlights.

    Per-trade P&L bars are colored green (win), crimson (loss), or
    gray (tie) based solely on the sign of ``pnl``.  The optional
    ``side`` field is accepted but not currently used for coloring;
    long/short breakdown is available via the trade-tier metrics
    in ``stratstat.core.trades``.

    Parameters
    ----------
    trades: TradeInput or dict with a ``pnl`` key and optional
        ``side`` key.
    returns: Optional portfolio-level returns for the equity curve
        background (not yet implemented).
    title: Optional chart title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from stratstat.inputs import TradeInput

    trd = trades if isinstance(trades, TradeInput) else TradeInput(trades=trades)

    pnl_arr = trd.pnl
    n_trades = len(pnl_arr)
    x_trades = np.arange(1, n_trades + 1)

    # Cumulative P&L from trades
    cum_pnl = np.cumsum(pnl_arr)

    # Win/loss markers
    wins = pnl_arr > 0
    losses = pnl_arr < 0

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=("Cumulative P&L", "Per-Trade P&L"),
    )

    # Cumulative P&L
    fig.add_trace(go.Scatter(
        x=x_trades, y=cum_pnl, mode="lines", name="Cumulative P&L",
        line={"color": "steelblue"},
        hovertemplate="Trade %{x}<br>Cum P&L: %{y:.3%}<extra></extra>",
    ), row=1, col=1)

    # Per-trade P&L bars
    colors = np.where(wins, "green", np.where(losses, "crimson", "gray"))
    fig.add_trace(go.Bar(
        x=x_trades, y=pnl_arr, name="P&L per Trade",
        marker_color=colors,
        hovertemplate="Trade %{x}: %{y:.2%}<extra></extra>",
    ), row=2, col=1)

    # Add a zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black",
                  opacity=0.3, row=2, col=1)

    win_count = int(np.sum(wins))
    loss_count = int(np.sum(losses))

    fig.update_layout(
        title=title or f"Trade Markers ({win_count}W / {loss_count}L)",
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_yaxes(title_text="Cumulative P&L", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="P&L", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Trade Number", row=2, col=1)
    return fig
