"""
StratStat — Quantitative strategy evaluation statistics.
"""

from typing import Any

# Import the metric implementation modules so their @register_metric
# decorators populate the registry at import time.  Without these side-effect
# imports ``list_metrics()`` is empty and ``compute()`` raises
# ``UnknownMetricError``.
import stratstat.core.benchmark  # noqa: F401
import stratstat.core.compare  # noqa: F401
import stratstat.core.exposure  # noqa: F401
import stratstat.core.returns.descriptive  # noqa: F401
import stratstat.core.returns.inference  # noqa: F401
import stratstat.core.returns.risk  # noqa: F401
import stratstat.core.returns.risk_adjusted  # noqa: F401
import stratstat.core.trades  # noqa: F401
from stratstat.conventions import get_default, set_default
from stratstat.core.returns.wrappers import by_regime, rolling
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
    "rolling",
    "by_regime",
]

# compute() and compute_all() are defined here for convenient top-level access.
# They accept raw data (numpy, pandas, polars) or pre-built Input objects.
def compute(
    input_data: Any,
    metric_name: str,
    *,
    periods_per_year: int | None = None,
    rf: float = 0.0,
    **kwargs: Any,
) -> MetricResult:
    """Compute a single metric on the given data.

    Parameters
    ----------
    input_data:
        Raw returns (numpy array, pandas Series/DataFrame, polars
        Series/DataFrame) or a pre-built :class:`ReturnsInput` /
        :class:`BenchmarkInput` / etc.
    metric_name:
        Registered metric name (e.g. ``"sharpe_ratio"``).
    periods_per_year:
        Annualisation factor (252 for daily, 12 for monthly, etc.).
        Only needed when passing raw data; ignored if *input_data* is
        already an Input object.
    rf:
        Risk-free rate per period (default 0.0).  Only used by
        benchmark-tier and compare-tier metrics.
    **kwargs:
        Forwarded to the metric function (e.g. ``return_type="log"``).

    Returns
    -------
    MetricResult
        The computed value with metadata (name, category, ref, etc.).
    """
    from stratstat.registry import _compute_one

    return _compute_one(
        input_data, metric_name,
        periods_per_year=periods_per_year, rf=rf, **kwargs,
    )


def compute_all(
    input_data: Any,
    category: str | None = None,
    *,
    periods_per_year: int | None = None,
    rf: float = 0.0,
    **kwargs: Any,
) -> MetricSet:
    """Compute all registered metrics matching *category* on the given data.

    Parameters
    ----------
    input_data:
        Raw returns or a pre-built Input object.
    category:
        Primary category tag to filter by (e.g. ``"risk"``,
        ``"descriptive"``).  If ``None``, computes all registered
        metrics.
    periods_per_year:
        Annualisation factor.  Only needed when passing raw data.
    rf:
        Risk-free rate per period (default 0.0).
    **kwargs:
        Forwarded to each metric function.

    Returns
    -------
    MetricSet
        Ordered collection of :class:`MetricResult` objects.
    """
    from stratstat.registry import _compute_all

    return _compute_all(
        input_data, category=category,
        periods_per_year=periods_per_year, rf=rf, **kwargs,
    )
