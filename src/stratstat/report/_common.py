"""Shared helpers for report compositions.

Functions in this module query the metric registry to dynamically
discover applicable metrics, avoiding hardcoded metric name lists.
"""

from __future__ import annotations

from typing import Any

from stratstat.results import _CATEGORY_LABELS, _CATEGORY_ORDER, MetricSet


def collect_metrics(
    input_data: Any,
    categories: list[str],
    **kwargs: Any,
) -> dict[str, MetricSet]:
    """Compute all registered metrics for each specified category.

    For each primary category in *categories*, routes the input container to
    the matching ``compute_all`` tier and calls it with ``category=category``.

    Returns a dict mapping ``category_name -> MetricSet``, ordered
    by ``_CATEGORY_ORDER``.  Categories with no registered metrics
    are omitted.

    Parameters
    ----------
    input_data: A StratStat input object (ReturnsInput, BenchmarkInput, etc.).
    categories: Primary category tags to discover (e.g. ``["descriptive", "risk"]``).
    **kwargs: Passed through to each metric's computation.
    """
    from stratstat import compute_all
    from stratstat.inputs import (
        BenchmarkInput,
        CompareInput,
        ExposureInput,
        ReturnsInput,
        TradeInput,
    )

    result: dict[str, MetricSet] = {}
    for cat in categories:
        if isinstance(input_data, BenchmarkInput):
            ms = compute_all(
                returns=input_data.returns,
                benchmark=input_data.benchmark,
                periods_per_year=input_data.periods_per_year,
                rf=input_data.rf,
                category=cat,
                **kwargs,
            )
        elif isinstance(input_data, ReturnsInput):
            ms = compute_all(returns=input_data, category=cat, **kwargs)
        elif isinstance(input_data, ExposureInput):
            ms = compute_all(exposure=input_data, category=cat, **kwargs)
        elif isinstance(input_data, TradeInput):
            ms = compute_all(trades=input_data, category=cat, **kwargs)
        elif isinstance(input_data, CompareInput):
            ms = compute_all(compare=input_data, category=cat, **kwargs)
        else:
            ms = compute_all(returns=input_data, category=cat, **kwargs)
        if len(ms) > 0:
            result[cat] = ms
    return result


def discover_and_format(
    input_data: Any,
    categories: list[str],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Compute and format metrics grouped by category for table display.

    Returns a list of ``{"section": str, "metrics": list[dict]}`` dicts,
    ordered by category display priority.  Each metrics dict has keys
    ``name``, ``value``, and ``ref`` suitable for table rendering.

    Parameters
    ----------
    input_data: A StratStat input object.
    categories: Primary category tags to include.
    **kwargs: Passed through to each metric's computation.
    """
    grouped = collect_metrics(input_data, categories, **kwargs)
    sections: list[dict[str, Any]] = []

    # Re-order by _CATEGORY_ORDER
    ordered = sorted(grouped.items(), key=lambda kv: _CATEGORY_ORDER.get(kv[0], 99))

    for cat_name, ms in ordered:
        label = _CATEGORY_LABELS.get(cat_name, cat_name.title())
        metrics: list[dict[str, Any]] = []
        for mr in ms:
            import numpy as np

            val = mr.value
            if isinstance(val, np.ndarray) and val.size == 1:
                val = float(val.flat[0])
            metrics.append(
                {
                    "name": mr.name,
                    "value": val,
                    "ref": mr.meta.get("ref", ""),
                }
            )
        sections.append({"section": label, "metrics": metrics})

    return sections


def category_order() -> list[str]:
    """Return primary categories in standard display order."""
    return sorted(_CATEGORY_ORDER, key=lambda k: _CATEGORY_ORDER[k])
