"""Reporting module — Plotly-based visualizations.

This module depends on core but never the reverse. It is an optional extra
(``pip install stratstat[report]``). Importing stratstat must not require plotly.

All imports of plotly are lazy (inside function bodies) so that the report
subpackage can be imported without plotly installed — only calling a
visualization function triggers the import check.
"""

from __future__ import annotations

from typing import Any

from stratstat.report._charts import (
    _ensure_plotly,
    benchmark_overlay_chart,
    cumulative_return_chart,
    drawdown_chart,
    equity_curve,
    monthly_heatmap,
    rolling_metric_chart,
    trade_markers_chart,
)
from stratstat.report._common import category_order, collect_metrics, discover_and_format
from stratstat.report._dashboard import dashboard
from stratstat.report._export import to_html, to_image, to_json, to_latex, to_markdown
from stratstat.report._report import generate_report
from stratstat.report._tearsheet import tear_sheet

__all__ = [
    # Charts
    "equity_curve",
    "drawdown_chart",
    "monthly_heatmap",
    "rolling_metric_chart",
    "cumulative_return_chart",
    "benchmark_overlay_chart",
    "trade_markers_chart",
    # Compositions
    "tear_sheet",
    "dashboard",
    "generate_report",
    # Helpers
    "collect_metrics",
    "discover_and_format",
    "category_order",
    # Export
    "to_html",
    "to_image",
    "to_markdown",
    "to_latex",
    "to_json",
]
