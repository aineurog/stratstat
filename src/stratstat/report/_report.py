"""Standalone HTML report generator.

Produces a self-contained HTML strategy analysis report with embedded
charts and auto-discovered metrics, organised into top-level tabs:
Overview, Performance, Exposure, Trades, Benchmark.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from stratstat.report._charts import (
    _cumulative_returns,
    _drawdown_series,
    _ensure_plotly,
    _monthly_heatmap_data,
    _to_array,
)
from stratstat.report._common import discover_and_format


def _ensure_weasyprint() -> None:
    """Check that weasyprint is installed; raise a helpful error if not."""
    try:
        import weasyprint  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "weasyprint is required for PDF output. "
            "Install it with: pip install stratstat[pdf]"
        ) from err


# ---------------------------------------------------------------------------
# Summary / Overview metric names
# ---------------------------------------------------------------------------

_SUMMARY_METRICS: list[str] = [
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "calmar_ratio",
    "sortino_ratio",
    "var",
    "positive_period_ratio",
]

_SUMMARY_LABELS: dict[str, str] = {
    "cagr": "CAGR",
    "annualized_volatility": "Volatility",
    "sharpe_ratio": "Sharpe",
    "max_drawdown": "Max Drawdown",
    "calmar_ratio": "Calmar",
    "sortino_ratio": "Sortino",
    "var": "VaR (95%)",
    "positive_period_ratio": "Win Rate",
}

_OVERVIEW_METRICS: list[str] = [
    "cagr",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "downside_deviation",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "max_drawdown_duration",
    "var",
    "cvar",
    "skewness",
    "kurtosis",
    "positive_period_ratio",
]

# ---------------------------------------------------------------------------
# Helpers: MetricSet → sections conversion
# ---------------------------------------------------------------------------


def _metric_set_to_sections(
    metric_set: Any,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert a MetricSet to the sections format used by the report.

    Returns the same shape as :func:`discover_and_format` so the rest
    of the report pipeline (stats tables, ref index, methodology) works
    unchanged whether metrics came from a pre-computed MetricSet or
    were discovered on-the-fly.
    """
    from stratstat.results import _CATEGORY_LABELS, _CATEGORY_ORDER

    # Group metrics by primary category
    groups: dict[str, list[Any]] = {}
    for mr in metric_set:
        primary = mr.category[0] if mr.category else "other"
        if categories is not None and primary not in categories:
            continue
        groups.setdefault(primary, []).append(mr)

    # Build sections in display order
    ordered = sorted(
        groups.items(), key=lambda kv: _CATEGORY_ORDER.get(kv[0], 99)
    )
    sections: list[dict[str, Any]] = []
    for cat_name, metrics in ordered:
        label = _CATEGORY_LABELS.get(cat_name, cat_name.title())
        formatted: list[dict[str, Any]] = []
        for mr in metrics:
            val = mr.value
            if isinstance(val, np.ndarray) and val.size == 1:
                val = float(val.flat[0])
            formatted.append({
                "name": mr.name,
                "value": val,
                "ref": mr.meta.get("ref", ""),
            })
        sections.append({"section": label, "metrics": formatted})

    return sections


def _lookup_metrics(
    metric_set: Any,
    names: list[str],
) -> list[dict[str, Any]]:
    """Extract named metrics from a MetricSet as {name, value, ref} dicts.

    Missing metrics get ``value=np.nan`` and empty ref.
    """
    by_name: dict[str, Any] = {}
    for mr in metric_set:
        by_name[mr.name] = mr

    result: list[dict[str, Any]] = []
    for name in names:
        mr = by_name.get(name)
        if mr is not None:
            val = mr.value
            if isinstance(val, np.ndarray) and val.size == 1:
                val = float(val.flat[0])
            result.append({
                "name": name,
                "value": val,
                "ref": mr.meta.get("ref", ""),
            })
        else:
            result.append({"name": name, "value": np.nan, "ref": ""})

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    returns: Any,
    output_path: str | Path,
    *,
    benchmark: Any | None = None,
    positions: Any | None = None,
    asset_returns: Any | None = None,
    trades: Any | None = None,
    periods_per_year: int | None = None,
    title: str | None = None,
    include_benchmark_metrics: bool = True,
    metrics: Any | None = None,
) -> None:
    """Generate a self-contained HTML strategy analysis report.

    Produces a single ``.html`` (or ``.pdf``) file with top-level tabs:

    * **Overview** — hero cards, equity curve, drawdown, key statistics
    * **Performance** — distribution, heatmap, rolling metrics, full stats
    * **Exposure** — exposure charts and stats (when *positions* provided)
    * **Trades** — trade P&L and duration charts (when *trades* provided)
    * **Benchmark** — benchmark overlay and stats (when *benchmark* provided)

    Requires ``plotly`` (``pip install stratstat[report]``).
    PDF output also requires ``weasyprint`` (``pip install stratstat[pdf]``).

    Parameters
    ----------
    returns: ``ReturnsInput`` or array-like strategy returns.
    output_path: File path (``.html`` or ``.pdf``).  Parent directories
        are created automatically.
    benchmark: Optional benchmark returns (1-D array-like).
    positions: Optional 2-D position weights ``(n_periods, n_assets)``.
    asset_returns: Optional asset-level returns for book-level exposure
        metrics.
    trades: Optional trade log (dict or DataFrame with ``pnl`` column,
        and optionally ``side``, ``duration``, etc.).
    periods_per_year: Annualization factor.
    title: Report title (default: "Strategy Analysis Report").
    include_benchmark_metrics: If True (default) and *benchmark* is
        provided, include benchmark-tier statistics.
    metrics: Optional pre-computed :class:`MetricSet`.  When provided,
        statistics tables are populated from these values instead of
        recomputing on-the-fly.  Charts are always built from raw data
        regardless.  Useful in notebook workflows where metrics have
        already been computed with :func:`compute_all`.
    """
    _ensure_plotly()
    import plotly.graph_objects as go
    from plotly.io import to_html as _fig_to_html_div

    def _html_div(fig: Any, include_plotlyjs: bool = False) -> str:
        """Wrap a plotly figure as an HTML div string (responsive)."""
        return cast(str, _fig_to_html_div(
            fig, full_html=False, include_plotlyjs=include_plotlyjs,
            config={"responsive": True},
        ))

    # Trigger metric registration so the registry is populated.
    import stratstat.core.benchmark  # noqa: F401
    import stratstat.core.exposure  # noqa: F401
    import stratstat.core.returns.descriptive  # noqa: F401
    import stratstat.core.returns.inference  # noqa: F401
    import stratstat.core.returns.risk  # noqa: F401
    import stratstat.core.returns.risk_adjusted  # noqa: F401
    import stratstat.core.trades  # noqa: F401
    from stratstat.inputs import BenchmarkInput, ReturnsInput

    # -- Normalise inputs -------------------------------------------------
    r = _to_array(returns)
    # Honour an annualization factor carried on a passed ReturnsInput.
    if periods_per_year is None and isinstance(returns, ReturnsInput):
        periods_per_year = returns.periods_per_year
    strat = r[:, 0] if r.ndim > 1 and r.shape[1] >= 1 else r.ravel()
    p = Path(output_path) if isinstance(output_path, str) else output_path
    title = title or "Strategy Analysis Report"
    inp = ReturnsInput(strat, periods_per_year=periods_per_year)

    # -- Build input containers for optional layers -----------------------
    bench_arr: np.ndarray[Any, Any] | None = None
    bm_inp = None
    if benchmark is not None and include_benchmark_metrics:
        bench_arr = np.asarray(benchmark, dtype=np.float64).ravel()
        bm_inp = BenchmarkInput(
            returns=strat,
            benchmark=bench_arr[:len(strat)],
            periods_per_year=periods_per_year,
        )

    exp_inp = None
    if positions is not None:
        from stratstat.inputs import ExposureInput
        exp_inp = ExposureInput(
            positions=positions,
            returns=asset_returns,
            periods_per_year=periods_per_year,
        )

    trd_inp = None
    if trades is not None:
        from stratstat.inputs import TradeInput
        trd_inp = TradeInput(trades=trades, periods_per_year=periods_per_year)

    # -- Build all chart figures (first one embeds plotly.js) -------------
    first_chart = True
    all_charts: dict[str, str] = {}

    # --- Overview charts ---
    cum = _cumulative_returns(strat.reshape(-1, 1))[:, 0]
    x_idx = np.arange(len(cum))

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=x_idx, y=cum, mode="lines", name="Strategy",
        line={"color": "steelblue"},
        hovertemplate="%{y:.3f}<extra></extra>",
    ))
    if bench_arr is not None:
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
        title="Equity Curve", yaxis_title="Cumulative Return",
        xaxis_title="Period", hovermode="x unified",
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
    )

    dd = _drawdown_series(strat.reshape(-1, 1))[:, 0]
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=x_idx, y=dd, mode="lines", name="Drawdown",
        fill="tozeroy", fillcolor="rgba(220, 50, 50, 0.3)",
        line={"color": "crimson"},
        hovertemplate="%{y:.1%}<extra></extra>",
    ))
    fig_dd.update_layout(
        title="Drawdown", yaxis_title="Drawdown",
        xaxis_title="Period", yaxis_tickformat=".0%",
        hovermode="x unified",
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
    )

    # Equity curve + drawdown as a 2-col row
    eq_fig_div = _html_div(fig_eq, include_plotlyjs=first_chart)
    first_chart = False
    dd_fig_div = _html_div(fig_dd)

    all_charts["equity_curve"] = (
        '<div class="chart-row">'
        f'<div class="chart-col">{eq_fig_div}</div>'
        f'<div class="chart-col">{dd_fig_div}</div>'
        '</div>'
    )

    # --- Performance charts ---
    from stratstat.report._charts import returns_distribution
    fig_dist = returns_distribution(strat, title="Returns Distribution")
    fig_dist.update_layout(margin={"l": 50, "r": 20, "t": 40, "b": 40})

    grid, years, months = _monthly_heatmap_data(strat)
    fig_hm = go.Figure()
    if grid.size > 0:
        fig_hm.add_trace(go.Heatmap(
            z=grid,
            x=months[:grid.shape[1]] if grid.shape[1] <= 12 else months,
            y=[str(y) for y in years],
            colorscale="RdYlGn", zmid=0,
            texttemplate="%{z:.1%}", textfont={"size": 10},
            hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
        ))
    fig_hm.update_layout(
        title="Monthly Returns Heatmap",
        xaxis_title="Month", yaxis_title="Year",
        yaxis={"autorange": "reversed"},
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
    )

    all_charts["distribution_heatmap"] = (
        '<div class="chart-row">'
        f'<div class="chart-col">'
        f'{_html_div(fig_dist)}'
        f'</div>'
        f'<div class="chart-col">'
        f'{_html_div(fig_hm)}'
        f'</div>'
        '</div>'
    )

    # --- Rolling metrics ---
    from stratstat.report._charts import rolling_metric_chart
    fig_rs = rolling_metric_chart(
        inp, "sharpe_ratio", window=60, periods_per_year=periods_per_year,
        title="Rolling Sharpe Ratio (60-period window)",
    )
    fig_rs.update_layout(margin={"l": 50, "r": 20, "t": 40, "b": 40})
    fig_rv = rolling_metric_chart(
        inp, "annualized_volatility", window=60,
        periods_per_year=periods_per_year,
        title="Rolling Volatility (60-period window)",
    )
    fig_rv.update_layout(margin={"l": 50, "r": 20, "t": 40, "b": 40})

    all_charts["rolling"] = (
        '<div class="rolling-section">'
        f'{_html_div(fig_rs)}'
        f'{_html_div(fig_rv)}'
        '</div>'
    )

    # --- Exposure charts ---
    if exp_inp is not None:
        from stratstat.report._charts import (
            effective_n_chart,
            exposure_heatmap,
            exposure_over_time,
        )

        fig_exp = exposure_over_time(exp_inp, title="Exposure Over Time")
        fig_exp.update_layout(margin={"l": 50, "r": 20, "t": 40, "b": 40})
        fig_eff = effective_n_chart(exp_inp, title="Effective N")
        fig_eff.update_layout(margin={"l": 50, "r": 20, "t": 40, "b": 40})
        fig_ehm = exposure_heatmap(exp_inp, title="Exposure Heatmap")
        fig_ehm.update_layout(margin={"l": 50, "r": 20, "t": 40, "b": 40})

        all_charts["exposure"] = (
            '<div class="chart-row">'
            f'<div class="chart-col">'
            f'{_html_div(fig_exp)}'
            f'</div>'
            f'<div class="chart-col">'
            f'{_html_div(fig_eff)}'
            f'</div>'
            '</div>'
            '<div class="full-width-chart">'
            f'{_html_div(fig_ehm)}'
            '</div>'
        )

    # --- Trade charts ---
    if trd_inp is not None:
        from stratstat.report._charts import (
            trade_duration_histogram,
            trade_markers_chart,
        )

        fig_trades = trade_markers_chart(trd_inp, title="Trade P&L Analysis")
        fig_trades.update_layout(margin={"l": 50, "r": 20, "t": 40, "b": 40})

        trade_parts = [
            '<div class="full-width-chart">'
            f'{_html_div(fig_trades)}'
            '</div>',
        ]

        if trd_inp.has_duration:
            fig_dur = trade_duration_histogram(trd_inp, title="Trade Duration Distribution")
            fig_dur.update_layout(margin={"l": 50, "r": 20, "t": 40, "b": 40})
            trade_parts.append(
                '<div class="full-width-chart">'
                f'{_html_div(fig_dur)}'
                '</div>'
            )

        all_charts["trades"] = "\n".join(trade_parts)

    # --- Benchmark charts ---
    if bm_inp is not None:
        from stratstat.report._charts import benchmark_overlay_chart
        fig_bm = benchmark_overlay_chart(strat, bench_arr,
                                         title="Benchmark Comparison")
        fig_bm.update_layout(margin={"l": 50, "r": 20, "t": 40, "b": 40})
        all_charts["benchmark"] = (
            '<div class="full-width-chart">'
            f'{_html_div(fig_bm)}'
            '</div>'
        )

    # -- Collect per-tab statistics ---------------------------------------
    if metrics is not None:
        # Use pre-computed MetricSet — convert to sections format.
        perf_sections = _metric_set_to_sections(
            metrics, ["descriptive", "risk", "risk_adjusted", "inference"]
        )
        exp_sections = (
            _metric_set_to_sections(metrics, ["exposure"])
            if exp_inp is not None else []
        )
        trd_sections = (
            _metric_set_to_sections(metrics, ["trades"])
            if trd_inp is not None else []
        )
        bm_sections = (
            _metric_set_to_sections(metrics, ["benchmark"])
            if bm_inp is not None else []
        )
        # Overview: extract curated metrics from the MetricSet.
        overview_metrics = _lookup_metrics(metrics, _OVERVIEW_METRICS)
        summary_metrics = _lookup_metrics(metrics, _SUMMARY_METRICS)
    else:
        # Discover and compute metrics on-the-fly from input containers.
        perf_sections = discover_and_format(
            inp, ["descriptive", "risk", "risk_adjusted", "inference"]
        )
        exp_sections = (
            discover_and_format(exp_inp, ["exposure"])
            if exp_inp is not None else []
        )
        trd_sections = (
            discover_and_format(trd_inp, ["trades"])
            if trd_inp is not None else []
        )
        bm_sections = (
            discover_and_format(bm_inp, ["benchmark"])
            if bm_inp is not None else []
        )
        overview_metrics = None
        summary_metrics = None

    # -- Build tabs -------------------------------------------------------
    tabs: list[dict[str, Any]] = []

    # Overview tab
    overview_content = _build_overview_content(
        summary_html=_build_summary_html(
            inp, metric_set=summary_metrics
        ),
        data_summary_html=_build_data_summary(
            n_periods=len(strat),
            periods_per_year=periods_per_year,
            n_trades=trd_inp.n_trades if trd_inp else None,
            n_assets=exp_inp.positions.shape[1] if exp_inp else None,
        ),
        charts_html=all_charts["equity_curve"],
        inp=inp,
        key_metrics=overview_metrics,
    )
    tabs.append({"id": "overview", "label": "Overview", "content": overview_content})

    # Performance tab
    perf_ref_index = _build_ref_index(perf_sections)
    perf_content = _build_domain_tab_content(
        charts_html=all_charts["distribution_heatmap"]
                   + all_charts["rolling"],
        sections=perf_sections,
        ref_index=perf_ref_index,
    )
    tabs.append({"id": "performance", "label": "Performance", "content": perf_content})

    # Exposure tab
    if exp_inp is not None:
        exp_ref_index = _build_ref_index(exp_sections)
        exp_content = _build_domain_tab_content(
            charts_html=all_charts.get("exposure", ""),
            sections=exp_sections,
            ref_index=exp_ref_index,
        )
        tabs.append({"id": "exposure", "label": "Exposure", "content": exp_content})

    # Trades tab
    if trd_inp is not None:
        trd_ref_index = _build_ref_index(trd_sections)
        trd_content = _build_domain_tab_content(
            charts_html=all_charts.get("trades", ""),
            sections=trd_sections,
            ref_index=trd_ref_index,
        )
        tabs.append({"id": "trades", "label": "Trades", "content": trd_content})

    # Benchmark tab
    if bm_inp is not None:
        bm_ref_index = _build_ref_index(bm_sections)
        bm_content = _build_domain_tab_content(
            charts_html=all_charts.get("benchmark", ""),
            sections=bm_sections,
            ref_index=bm_ref_index,
        )
        tabs.append({"id": "benchmark", "label": "Benchmark", "content": bm_content})

    # -- Assemble ----------------------------------------------------------
    all_sections = perf_sections + exp_sections + trd_sections + bm_sections
    n_metrics_total = sum(len(s["metrics"]) for s in all_sections)
    date_str = datetime.date.today().strftime("%Y-%m-%d")

    nav_html = _build_tab_nav(tabs)
    panels_html = _build_tab_panels(tabs)

    html = _HTML_TEMPLATE.format(
        title=title,
        date=date_str,
        n_metrics=n_metrics_total,
        n_periods=len(strat),
        ppy=periods_per_year or "N/A",
        freq="Daily" if periods_per_year and periods_per_year > 200 else "Monthly",
        tab_nav=nav_html,
        tab_panels=panels_html,
    )

    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".pdf":
        _ensure_weasyprint()
        from weasyprint import HTML
        HTML(string=html).write_pdf(p)
    else:
        p.write_text(html)


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------


def _build_overview_content(
    summary_html: str,
    data_summary_html: str,
    charts_html: str,
    inp: Any,
    key_metrics: list[dict[str, Any]] | None = None,
) -> str:
    """Build the Overview tab content: cards, data bar, charts, key stats.

    When *key_metrics* is provided (pre-computed from a MetricSet), the
    individual ``_compute_one`` calls are skipped and those values are
    used directly.
    """
    from stratstat.registry import _compute_one

    # Key statistics table — curated set
    if key_metrics is not None:
        metrics = key_metrics
    else:
        metrics = []
        for name in _OVERVIEW_METRICS:
            ref = ""
            try:
                result = _compute_one(inp, name)
                val = result.value
                if isinstance(val, np.ndarray):
                    val = float(val.flat[0]) if val.size == 1 else val
                ref = result.meta.get("ref", "")
            except Exception:
                val = np.nan
            metrics.append({"name": name, "value": val, "ref": ref})

    # Build ref index and stats table
    ref_index: dict[str, int] = {}
    ref_counter = 1
    for m in metrics:
        ref = m.get("ref", "")
        if ref and ref not in ref_index:
            ref_index[ref] = ref_counter
            ref_counter += 1

    section = {"section": "Key Statistics", "metrics": metrics}
    stats_html = _build_stats_html([section], ref_index)
    refs_html = _build_methodology_html(ref_index)

    return f"""{summary_html}
{data_summary_html}
{charts_html}
<div class="tab-stats">
{stats_html}
{refs_html}
</div>"""


def _build_domain_tab_content(
    charts_html: str,
    sections: list[dict[str, Any]],
    ref_index: dict[str, int],
) -> str:
    """Build content for a domain tab: charts + stats tables + methodology."""
    stats_html = _build_stats_html(sections, ref_index)
    refs_html = _build_methodology_html(ref_index)

    return f"""{charts_html}
<div class="tab-stats">
{stats_html}
{refs_html}
</div>"""


def _build_tab_nav(tabs: list[dict[str, Any]]) -> str:
    """Build the top-level tab navigation bar."""
    parts: list[str] = ['<div class="tab-nav">']
    for i, tab in enumerate(tabs):
        active = " active" if i == 0 else ""
        parts.append(
            f'<button class="tab-nav-btn{active}" '
            f'onclick="switchTab(event, \'tab-{tab["id"]}\')">'
            f'{tab["label"]}</button>'
        )
    parts.append('</div>')
    return "\n".join(parts)


def _build_tab_panels(tabs: list[dict[str, Any]]) -> str:
    """Build all tab content panels."""
    parts: list[str] = ['<div class="tab-panels">']
    for i, tab in enumerate(tabs):
        active = " active" if i == 0 else ""
        parts.append(
            f'<div id="tab-{tab["id"]}" class="tab-panel{active}">'
            f'{tab["content"]}'
            f'</div>'
        )
    parts.append('</div>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Data summary bar
# ---------------------------------------------------------------------------


def _build_data_summary(
    n_periods: int,
    periods_per_year: int | None = None,
    n_trades: int | None = None,
    n_assets: int | None = None,
) -> str:
    """Build the data summary bar below hero cards."""
    freq = "Daily" if periods_per_year and periods_per_year > 200 else "Monthly"
    ppy = periods_per_year or "N/A"
    parts = [f"{n_periods} periods | {freq} | {ppy} ppy"]
    if n_trades is not None:
        parts.append(f"{n_trades} trades")
    if n_assets is not None:
        parts.append(f"{n_assets} assets")
    return f'<div class="data-summary">{" &nbsp;|&nbsp; ".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Summary cards
# ---------------------------------------------------------------------------


def _build_summary_html(
    inp: Any, metric_set: list[dict[str, Any]] | None = None
) -> str:
    """Build the hero summary cards from key metrics.

    When *metric_set* is provided (pre-computed from a MetricSet via
    :func:`_lookup_metrics`), the individual ``_compute_one`` calls are
    skipped and those values are used directly.
    """
    from stratstat.registry import _compute_one

    by_name = (
        {m["name"]: m for m in metric_set} if metric_set is not None else {}
    )

    cards: list[str] = []
    for name in _SUMMARY_METRICS:
        label = _SUMMARY_LABELS.get(name, name)
        if name in by_name:
            val = by_name[name]["value"]
            val_str = _fmt_scalar(val)
        else:
            try:
                result = _compute_one(inp, name)
                val = result.value
                if isinstance(val, np.ndarray):
                    val = float(val.flat[0]) if val.size == 1 else val
                val_str = _fmt_scalar(val)
            except Exception:
                val_str = "N/A"
        cards.append(f"""<div class="card">
            <div class="card-label">{label}</div>
            <div class="card-value">{val_str}</div>
        </div>""")
    joined = "\n".join(cards)
    return f'<div class="cards">\n{joined}\n</div>'


# ---------------------------------------------------------------------------
# Statistics tables
# ---------------------------------------------------------------------------


def _build_stats_html(
    sections: list[dict[str, Any]],
    ref_index: dict[str, int],
) -> str:
    """Build statistics section: sequential tables with h3 headers."""
    if not sections:
        return ""

    if len(sections) == 1:
        # Single section — one table, no sub-headers needed
        inner = _build_one_table(sections[0], ref_index)
        return f'<div class="stats-block">{inner}</div>'

    # Multiple sections — each with an h3 heading
    parts: list[str] = ['<div class="stats-block">']
    for sec in sections:
        parts.append(f'<h3>{sec["section"]}</h3>')
        parts.append(_build_one_table(sec, ref_index))
    parts.append('</div>')
    return "\n".join(parts)


def _build_one_table(
    section: dict[str, Any],
    ref_index: dict[str, int],
) -> str:
    """Build a single category table with citation number column."""
    metrics = section["metrics"]
    if not metrics:
        return ""

    parts: list[str] = ["<table>"]
    parts.append(
        "<thead><tr>"
        "<th>Metric</th><th>Value</th><th>Ref</th>"
        "</tr></thead>"
    )
    parts.append("<tbody>")
    for i, m in enumerate(metrics):
        bg = ' style="background:#f8fafc"' if i % 2 == 0 else ""
        val = m["value"]
        val_str = _fmt_scalar(val)

        ref = m.get("ref", "")
        ref_num = ref_index.get(ref, "")
        if ref_num:
            ref_cell = f'<a href="#ref-{ref_num}">[{ref_num}]</a>'
        else:
            ref_cell = '<span class="no-ref">—</span>'

        parts.append(
            f"<tr{bg}>"
            f'<td class="metric-name">{m["name"]}</td>'
            f'<td class="metric-val">{val_str}</td>'
            f'<td class="metric-ref">{ref_cell}</td>'
            f"</tr>"
        )
    parts.append("</tbody></table>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------


def _build_ref_index(sections: list[dict[str, Any]]) -> dict[str, int]:
    """Build a citation ref -> number index from a list of sections."""
    ref_index: dict[str, int] = {}
    ref_counter = 1
    for sec in sections:
        for m in sec["metrics"]:
            ref = m.get("ref", "")
            if ref and ref not in ref_index:
                ref_index[ref] = ref_counter
                ref_counter += 1
    return ref_index


def _build_methodology_html(ref_index: dict[str, int]) -> str:
    """Build numbered methodology section with anchor targets."""
    if not ref_index:
        return ""

    refs_sorted = sorted(ref_index.items(), key=lambda kv: kv[1])

    parts: list[str] = ['<div class="methodology">']
    parts.append("<h3>Methodology</h3>")
    parts.append("<ol>")
    for ref_text, num in refs_sorted:
        ref_escaped = ref_text.replace("&", "&amp;").replace("<", "&lt;")
        parts.append(f'<li id="ref-{num}">{ref_escaped}</li>')
    parts.append("</ol>")
    parts.append("</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def _fmt_scalar(val: Any) -> str:
    """Format a metric value for display."""
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return "N/A"
        if np.isposinf(val):
            return "∞"
        if np.isneginf(val):
            return "-∞"
        return f"{float(val):.6g}"
    if isinstance(val, np.ndarray):
        if val.size <= 5:
            inner = ", ".join(_fmt_scalar(float(v)) for v in val.ravel())
            return f"[{inner}]"
        return f"array{val.shape}"
    if val is None:
        return "N/A"
    return str(val)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
/* -- reset & base ------------------------------------------------------ */
body {{
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px 24px;
    background: #f8fafc;
    color: #1a202c;
}}

/* -- header ------------------------------------------------------------ */
h1 {{
    color: #1a365d;
    border-bottom: 3px solid #3182ce;
    padding-bottom: 10px;
    font-size: 28px;
    margin-bottom: 6px;
}}
.summary-line {{
    color: #666;
    font-size: 13px;
    margin-bottom: 20px;
}}

/* -- hero summary cards ------------------------------------------------ */
.cards {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}}
.card {{
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 18px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
}}
.card-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #718096;
    margin-bottom: 6px;
}}
.card-value {{
    font-size: 22px;
    font-weight: 700;
    font-family: ui-monospace, monospace;
    color: #1a365d;
}}

/* -- data summary bar -------------------------------------------------- */
.data-summary {{
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    color: #4a5568;
    margin-bottom: 24px;
    text-align: center;
}}

/* -- top-level tab navigation ------------------------------------------ */
.tab-nav {{
    display: flex;
    gap: 2px;
    border-bottom: 3px solid #3182ce;
    margin-bottom: 24px;
}}
.tab-nav-btn {{
    padding: 10px 24px;
    border: none;
    background: #edf2f7;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    border-radius: 8px 8px 0 0;
    color: #4a5568;
    transition: background .15s;
}}
.tab-nav-btn:hover {{
    background: #e2e8f0;
}}
.tab-nav-btn.active {{
    background: #3182ce;
    color: white;
}}

/* -- tab panels -------------------------------------------------------- */
.tab-panels {{
    /* container for all panels */
}}
.tab-panel {{
    display: none;
}}
.tab-panel.active {{
    display: block;
}}

/* -- section headers --------------------------------------------------- */
h2 {{
    margin-top: 0;
    padding-bottom: 6px;
    font-size: 20px;
    color: #3182ce;
    border-bottom: 2px solid #3182ce;
}}
h3 {{
    font-size: 15px;
    color: #2d3748;
    margin-top: 24px;
    margin-bottom: 10px;
    padding-bottom: 4px;
    border-bottom: 1px solid #e2e8f0;
}}

/* -- chart rows -------------------------------------------------------- */
.chart-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
}}
.chart-col {{
    min-width: 0;
}}
.chart-col .plotly-graph-div {{
    width: 100%;
}}
.rolling-section {{
    margin-bottom: 24px;
}}
.rolling-section .plotly-graph-div {{
    margin-bottom: 16px;
    width: 100%;
}}

/* -- full-width charts ------------------------------------------------- */
.full-width-chart {{
    margin-bottom: 24px;
}}
.full-width-chart .plotly-graph-div {{
    width: 100%;
}}

/* -- tables ------------------------------------------------------------ */
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-bottom: 16px;
}}
thead th {{
    background: #edf2f7;
    text-align: left;
    padding: 7px 10px;
    border-bottom: 2px solid #cbd5e0;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
tbody td {{
    padding: 5px 10px;
    border-bottom: 1px solid #e2e8f0;
    vertical-align: top;
}}
tbody tr:hover {{
    background: #f7fafc;
}}
.metric-name {{
    font-weight: 600;
    font-family: ui-monospace, monospace;
    font-size: 12px;
    white-space: nowrap;
    width: 35%;
}}
.metric-val {{
    font-family: ui-monospace, monospace;
    color: #2d3748;
    width: 50%;
}}
.metric-ref {{
    font-size: 11px;
    text-align: center;
    width: 15%;
    font-family: ui-monospace, monospace;
}}
.metric-ref a {{
    color: #3182ce;
    text-decoration: none;
}}
.metric-ref a:hover {{
    text-decoration: underline;
}}
.no-ref {{
    color: #cbd5e0;
}}

/* -- stats block ------------------------------------------------------- */
.tab-stats {{
    margin-top: 24px;
}}

/* -- methodology ------------------------------------------------------- */
.methodology {{
    margin-top: 32px;
    border-top: 1px solid #e2e8f0;
    padding-top: 16px;
    font-size: 12px;
    color: #555;
}}
.methodology ol {{
    padding-left: 22px;
    columns: 2;
    column-gap: 40px;
}}
.methodology li {{
    margin-bottom: 6px;
    break-inside: avoid;
}}

/* -- print / PDF ------------------------------------------------------- */
@page {{
    size: A4;
    margin: 2cm;
}}
@media print {{
    body {{
        max-width: none;
        padding: 0;
    }}
    h1 {{ font-size: 22px; }}
    h2 {{ font-size: 16px; }}
    .cards {{
        grid-template-columns: repeat(4, 1fr);
    }}
    .tab-nav {{ display: none; }}
    .tab-panel {{
        display: block !important;
        page-break-before: always;
    }}
    .tab-panel:first-of-type {{
        page-break-before: avoid;
    }}
    table {{ page-break-inside: avoid; }}
    .chart-row .plotly-graph-div {{ page-break-inside: avoid; }}
    .methodology ol {{ columns: 1; }}
}}
</style>
</head>
<body>

<h1>{title}</h1>
<p class="summary-line">
    {n_periods} periods &nbsp;|&nbsp;
    {freq} &nbsp;|&nbsp;
    Periods/year: {ppy} &nbsp;|&nbsp;
    {n_metrics} metrics &nbsp;|&nbsp;
    {date}
</p>

{tab_nav}

{tab_panels}

<script>
function switchTab(evt, tabId) {{
    var panels = document.querySelectorAll('.tab-panel');
    for (var i = 0; i < panels.length; i++) {{
        panels[i].classList.remove('active');
    }}
    var btns = document.querySelectorAll('.tab-nav-btn');
    for (var i = 0; i < btns.length; i++) {{
        btns[i].classList.remove('active');
    }}
    var panel = document.getElementById(tabId);
    panel.classList.add('active');
    evt.currentTarget.classList.add('active');
    // Resize plotly charts that were hidden
    var gds = panel.querySelectorAll('.plotly-graph-div');
    for (var j = 0; j < gds.length; j++) {{
        if (gds[j]._fullLayout) {{
            Plotly.relayout(gds[j], {{}});
        }}
    }}
    // Fallback: dispatch resize to handle any responsive config
    window.dispatchEvent(new Event('resize'));
}}
</script>

</body>
</html>"""
