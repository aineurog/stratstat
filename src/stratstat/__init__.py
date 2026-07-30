"""
StratStat — Quantitative strategy evaluation statistics.
"""

from stratstat.registry import register_metric, list_metrics, get_metric
from stratstat.results import MetricResult, MetricSet
from stratstat.inputs import ReturnsInput, ExposureInput, TradeInput
from stratstat.conventions import get_default, set_default

__all__ = [
    "register_metric",
    "list_metrics",
    "get_metric",
    "MetricResult",
    "MetricSet",
    "ReturnsInput",
    "ExposureInput",
    "TradeInput",
    "get_default",
    "set_default",
]

# compute() and compute_all() are defined in registry.py and re-exported here
# for convenient top-level access. They will be wired once the registry has
# registered metrics to dispatch to.
def compute(input_data, metric_name, **kwargs) -> MetricResult:
    """Compute a single metric on the given input."""
    from stratstat.registry import _compute_one

    return _compute_one(input_data, metric_name, **kwargs)


def compute_all(input_data, category=None, **kwargs) -> MetricSet:
    """Compute all matching metrics on the given input."""
    from stratstat.registry import _compute_all

    return _compute_all(input_data, category=category, **kwargs)
