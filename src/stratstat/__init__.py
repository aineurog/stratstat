"""
StratStat — Quantitative strategy evaluation statistics.
"""

from typing import Any

from stratstat.conventions import get_default, set_default
from stratstat.inputs import BenchmarkInput, CompareInput, ExposureInput, ReturnsInput, TradeInput
from stratstat.registry import get_metric, list_metrics, register_metric
from stratstat.results import MetricResult, MetricSet

__all__ = [
    "register_metric",
    "list_metrics",
    "get_metric",
    "MetricResult",
    "MetricSet",
    "ReturnsInput",
    "ExposureInput",
    "TradeInput",
    "BenchmarkInput",
    "CompareInput",
    "get_default",
    "set_default",
]

# compute() and compute_all() are defined in registry.py and re-exported here
# for convenient top-level access. They will be wired once the registry has
# registered metrics to dispatch to.
def compute(input_data: Any, metric_name: str, **kwargs: Any) -> MetricResult:
    """Compute a single metric on the given input."""
    from stratstat.registry import _compute_one

    return _compute_one(input_data, metric_name, **kwargs)


def compute_all(input_data: Any, category: str | None = None, **kwargs: Any) -> MetricSet:
    """Compute all matching metrics on the given input."""
    from stratstat.registry import _compute_all

    return _compute_all(input_data, category=category, **kwargs)
