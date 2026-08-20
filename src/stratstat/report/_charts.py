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
    dates: Any | None = None,
) -> tuple[NDArray[np.floating], list[int], list[str]]:
    """Reshape returns into a years × months grid.

    Returns ``(grid, years, month_labels)`` where *grid* has shape
    ``(n_years, 12)`` with NaN for missing months.

    Requires ``pandas``, which is a core dependency of StratStat and
    therefore always available when the report module is used.

    Parameters
    ----------
    r: Returns array.
    dates: Optional date index, one per return period.  When omitted,
        a month-end range ending today is assumed.
    """
    import pandas as pd

    series = r[:, 0] if r.ndim == 2 and r.shape[1] >= 1 else r.ravel()

    n = len(series)
    if dates is not None:
        index = pd.to_datetime(dates)
        if len(index) != n:
            raise ValueError(
                f"dates length ({len(index)}) must match returns length ({n})."
            )
    else:
        # Generate a monthly date range — assume month-end frequency.
        # "ME" requires pandas >= 2.2; fall back to "M" on older versions.
        try:
            index = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="ME")
        except ValueError:
            index = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="M")
    monthly_returns: pd.Series = pd.Series(series, index=index)

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
    """Compute a rolling metric over a returns series.

    Delegates to :func:`stratstat.core.returns.wrappers.rolling` and
    extracts the raw value array for charting.
    """
    from stratstat.core.returns.wrappers import rolling
    from stratstat.inputs import ReturnsInput

    n = r.shape[0]

    # If window exceeds data length, return all NaN (chart can still render).
    if window > n:
        out: NDArray[np.floating] = np.full(n, np.nan, dtype=np.float64)
        return out

    # Pass periods_per_year via a temporary ReturnsInput so that
    # rolling() sets it on each window slice.  Do NOT pass it as a
    # metric_kwarg — the underlying metric reads it from the input.
    tmp = ReturnsInput(r.ravel(), periods_per_year=periods_per_year)
    result = rolling(tmp, metric_name, window)
    raw: NDArray[np.floating] = np.asarray(result.value, dtype=np.float64)
    return raw


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
    dates: Any | None = None,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Monthly returns heatmap (years × months).

    Parameters
    ----------
    returns: ReturnsInput or 1-D array of monthly returns.
    dates: Optional date index, one per return period.  When omitted,
        a month-end range ending today is assumed.
    title: Optional chart title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go

    r = _to_array(returns)
    grid, years, months = _monthly_heatmap_data(r, dates=dates)

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


def _trade_excursions(
    trd: Any,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute per-trade dollar MFE and MAE excursions.

    Mirrors the ``mfe`` and ``mae`` metrics in
    :mod:`stratstat.core.trades`: for each trade, MFE is the largest
    favorable dollar move and MAE the largest adverse dollar move,
    both relative to entry price.  Returns two arrays of length
    ``n_trades`` (NaN where the path is missing or invalid).
    """
    itp_list = trd.intratrade_prices
    is_long = trd.is_long
    assert itp_list is not None
    assert is_long is not None

    n = min(len(itp_list), len(is_long))
    mfe_vals: NDArray[np.floating] = np.full(n, np.nan)
    mae_vals: NDArray[np.floating] = np.full(n, np.nan)
    for j in range(n):
        path = itp_list[j]
        if len(path) < 2 or not np.isfinite(path[0]):
            continue
        entry = path[0]
        if is_long[j]:
            mfe_vals[j] = np.nanmax(path) - entry
            mae_vals[j] = entry - np.nanmin(path)
        else:
            mfe_vals[j] = entry - np.nanmin(path)
            mae_vals[j] = np.nanmax(path) - entry
    return mfe_vals, mae_vals


def trade_markers_chart(
    trades: Any,
    returns: Any | None = None,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Equity curve with trade P&L highlights and MFE/MAE overlays.

    The top panel shows the portfolio equity curve built from
    ``returns`` when they are provided; otherwise it shows the
    cumulative P&L across trades.  The bottom panel shows per-trade
    P&L bars colored green (win), crimson (loss), or gray (tie).  When
    the trade log carries ``intratrade_prices`` and ``side``, each
    trade's MFE and MAE dollar excursions are overlaid as markers.

    Parameters
    ----------
    trades: TradeInput or dict with a ``pnl`` key and optional
        ``side`` and ``intratrade_prices`` keys.
    returns: Optional portfolio-level returns for the equity curve
        background.
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

    r_arr = None if returns is None else np.asarray(returns, dtype=np.float64).ravel()
    has_equity = r_arr is not None and r_arr.size > 0

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            "Equity Curve" if has_equity else "Cumulative P&L",
            "Per-Trade P&L",
        ),
    )

    # Top panel: equity curve or cumulative P&L
    if has_equity:
        assert r_arr is not None
        equity = np.cumprod(1.0 + r_arr)
        x_equity = np.arange(equity.size)
        fig.add_trace(go.Scatter(
            x=x_equity, y=equity, mode="lines", name="Equity Curve",
            line={"color": "steelblue"},
            hovertemplate="Period %{x}<br>Equity: %{y:.3f}<extra></extra>",
        ), row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Equity", tickformat=".3f", row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=x_trades, y=cum_pnl, mode="lines", name="Cumulative P&L",
            line={"color": "steelblue"},
            hovertemplate="Trade %{x}<br>Cum P&L: %{y:.3%}<extra></extra>",
        ), row=1, col=1)
        fig.update_xaxes(title_text="Trade Number", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative P&L", tickformat=".0%", row=1, col=1)

    # Per-trade P&L bars
    colors = np.where(wins, "green", np.where(losses, "crimson", "gray"))
    fig.add_trace(go.Bar(
        x=x_trades, y=pnl_arr, name="P&L per Trade",
        marker_color=colors,
        hovertemplate="Trade %{x}: %{y:.2%}<extra></extra>",
    ), row=2, col=1)

    # MFE/MAE excursion markers
    if trd.has_intratrade and trd.has_side:
        mfe_vals, mae_vals = _trade_excursions(trd)
        x_exc = np.arange(1, mfe_vals.size + 1)
        fig.add_trace(go.Scatter(
            x=x_exc, y=mfe_vals, mode="markers", name="MFE",
            marker={"symbol": "triangle-up", "color": "green", "size": 9},
            hovertemplate="Trade %{x}<br>MFE: %{y:.2f}<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=x_exc, y=-mae_vals, mode="markers", name="MAE",
            marker={"symbol": "triangle-down", "color": "crimson", "size": 9},
            hovertemplate="Trade %{x}<br>MAE: %{y:.2f}<extra></extra>",
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
    fig.update_yaxes(title_text="P&L", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Trade Number", row=2, col=1)
    return fig


def trade_duration_histogram(
    trades: Any,
    bins: int | None = None,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Histogram of trade holding-period durations.

    Requires duration data in the trade log (``duration`` or
    ``entry_time`` / ``exit_time`` fields).

    Parameters
    ----------
    trades: TradeInput or dict with ``duration`` (or ``entry_time`` /
        ``exit_time``) and optionally ``pnl`` for win/loss coloring.
    bins: Number of histogram bins (auto if None).
    title: Optional chart title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go

    from stratstat.inputs import TradeInput

    trd = trades if isinstance(trades, TradeInput) else TradeInput(trades=trades)

    if not trd.has_duration:
        fig = go.Figure()
        fig.update_layout(
            title=title or "Trade Duration Distribution (no duration data)",
        )
        return fig

    dur = trd.duration
    assert dur is not None
    valid = dur[np.isfinite(dur)]

    if len(valid) == 0:
        fig = go.Figure()
        fig.update_layout(title=title or "Trade Duration Distribution")
        return fig

    n = len(valid)
    if bins is None:
        bins = max(8, min(50, int(np.sqrt(n))))

    # Split by win/loss for stacked histogram
    pnl = trd.pnl
    if len(valid) == len(pnl):
        win_dur = valid[pnl > 0.0]
        loss_dur = valid[pnl < 0.0]
    else:
        win_dur = np.array([], dtype=np.float64)
        loss_dur = np.array([], dtype=np.float64)

    # Use consistent bin edges across both traces
    all_min = float(np.min(valid))
    all_max = float(np.max(valid))
    if all_max <= all_min:
        all_max = all_min + 1.0

    fig = go.Figure()
    if len(win_dur) > 0:
        fig.add_trace(go.Histogram(
            x=win_dur,
            xbins={"start": all_min, "end": all_max, "size": (all_max - all_min) / bins},
            name=f"Wins ({len(win_dur)})",
            marker_color="green",
            marker_opacity=0.7,
            hovertemplate="%{x:.1f} periods<br>%{y} trades<extra></extra>",
        ))
    if len(loss_dur) > 0:
        fig.add_trace(go.Histogram(
            x=loss_dur,
            xbins={"start": all_min, "end": all_max, "size": (all_max - all_min) / bins},
            name=f"Losses ({len(loss_dur)})",
            marker_color="crimson",
            marker_opacity=0.7,
            hovertemplate="%{x:.1f} periods<br>%{y} trades<extra></extra>",
        ))

    fig.update_layout(
        title=title or "Trade Duration Distribution",
        xaxis_title="Holding Period Duration",
        yaxis_title="Number of Trades",
        barmode="overlay",
        bargap=0.05,
        hovermode="x unified",
    )
    return fig


def returns_distribution(
    returns: Any,
    bins: int = 50,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Histogram of period returns with a fitted normal curve overlay.

    Parameters
    ----------
    returns: ReturnsInput or 1-D array of period returns.
    bins: Number of histogram bins (default 50).
    title: Optional chart title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go

    r = _to_array(returns)
    strat = r[:, 0] if r.ndim > 1 and r.shape[1] >= 1 else r.ravel()
    strat = strat[np.isfinite(strat)]

    if len(strat) == 0:
        fig = go.Figure()
        fig.update_layout(title=title or "Returns Distribution")
        return fig

    mu = float(np.mean(strat))
    sigma = float(np.std(strat, ddof=1))

    # Normal curve overlay (pure numpy — no scipy dependency)
    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    pdf = np.exp(-0.5 * ((x_range - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=strat,
        nbinsx=bins,
        name="Returns",
        histnorm="probability density",
        marker_color="steelblue",
        marker_opacity=0.7,
        hovertemplate="%{x:.2%}<br>Density: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_range,
        y=pdf,
        mode="lines",
        name=f"Normal (μ={mu:.4f}, σ={sigma:.4f})",
        line={"color": "crimson", "width": 2},
        hovertemplate="%{x:.2%}<br>%{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title or "Returns Distribution",
        xaxis_title="Return",
        yaxis_title="Density",
        xaxis_tickformat=".1%",
        bargap=0.05,
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Exposure charts
# ---------------------------------------------------------------------------


def exposure_over_time(
    positions: Any,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Gross and net exposure over time from position weights.

    Parameters
    ----------
    positions: 2-D array of shape ``(n_periods, n_assets)`` or ExposureInput.
    title: Optional chart title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go

    from stratstat.inputs import ExposureInput

    if isinstance(positions, ExposureInput):
        w = positions.positions
    else:
        w = np.asarray(positions, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"Expected 2-D positions array, got shape {w.shape}")

    gross = np.sum(np.abs(w), axis=1)
    net = np.sum(w, axis=1)
    x = np.arange(len(gross))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=gross, mode="lines", name="Gross Exposure",
        line={"color": "steelblue"},
        hovertemplate="%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=net, mode="lines", name="Net Exposure",
        line={"color": "darkorange"},
        hovertemplate="%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)
    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.3,
                  annotation_text="100%")
    fig.update_layout(
        title=title or "Exposure Over Time",
        yaxis_title="Exposure",
        xaxis_title="Period",
        hovermode="x unified",
    )
    return fig


def effective_n_chart(
    positions: Any,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Effective number of positions (1 / sum(w_i^2)) over time.

    Parameters
    ----------
    positions: 2-D array of shape ``(n_periods, n_assets)`` or ExposureInput.
    title: Optional chart title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go

    from stratstat.inputs import ExposureInput

    if isinstance(positions, ExposureInput):
        w = positions.positions
    else:
        w = np.asarray(positions, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"Expected 2-D positions array, got shape {w.shape}")

    w2 = np.sum(w ** 2, axis=1)
    eff_n = np.where(w2 > 0, 1.0 / w2, 0.0)
    x = np.arange(len(eff_n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=eff_n, mode="lines", name="Effective N",
        fill="tozeroy", fillcolor="rgba(100, 150, 220, 0.15)",
        line={"color": "steelblue"},
        hovertemplate="%{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title=title or "Effective N (Concentration)",
        yaxis_title="Effective N",
        xaxis_title="Period",
        hovermode="x unified",
    )
    return fig


def exposure_heatmap(
    positions: Any,
    title: str | None = None,
) -> plotly.graph_objects.Figure:  # type: ignore[name-defined]  # noqa: F821
    """Asset-level exposure heatmap (assets × time).

    Only suitable for a modest number of assets (≤30).  For larger
    universes, consider aggregating.

    Parameters
    ----------
    positions: 2-D array of shape ``(n_periods, n_assets)`` or ExposureInput.
    title: Optional chart title.
    """
    _ensure_plotly()
    import plotly.graph_objects as go

    from stratstat.inputs import ExposureInput

    if isinstance(positions, ExposureInput):
        w = positions.positions
    else:
        w = np.asarray(positions, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"Expected 2-D positions array, got shape {w.shape}")

    n_assets = w.shape[1]
    if n_assets > 30:
        fig = go.Figure()
        fig.update_layout(
            title=title or "Exposure Heatmap (skipped — too many assets)",
        )
        return fig

    fig = go.Figure(data=go.Heatmap(
        z=w.T,
        x=[str(i) for i in range(w.shape[0])],
        y=[f"A{i + 1}" for i in range(n_assets)],
        colorscale="RdBu",
        zmid=0,
        hovertemplate="Period %{x}<br>%{y}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title or "Exposure Heatmap",
        xaxis_title="Period",
        yaxis_title="Asset",
        yaxis={"autorange": "reversed"} if n_assets > 1 else {},
    )
    return fig
