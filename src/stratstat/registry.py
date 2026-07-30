"""Metric registry — decorator-based registration and discovery.

Every metric is registered via @register_metric(...). The registry powers
compute(), compute_all(), list_metrics(), and the generic rolling() wrapper.
Adding a new metric must never require editing a central dispatch block.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from stratstat.results import MetricResult, MetricSet

_registry: dict[str, dict[str, Any]] = {}


def register_metric(
    name: str,
    requires: str,
    category: tuple[str, ...] = (),
    backend: str = "vectorized",
    ref: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a metric function in the global registry.

    Args:
        name: Unique metric name (e.g. "sharpe_ratio").
        requires: Input tier — "returns", "exposure", "trades", "benchmark", or "compare".
        category: Axis-2 classification tags (e.g. ("risk_adjusted", "returns")).
        backend: Computation profile — "vectorized", "sequential", or "resampling".
        ref: Citation string for the formula.

    Returns:
        The decorated function, unchanged.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _registry[name] = {
            "func": func,
            "requires": requires,
            "category": category,
            "backend": backend,
            "ref": ref,
        }
        return func

    return decorator


def list_metrics(
    requires: str | None = None,
    category: str | None = None,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """List registered metrics, optionally filtered.

    Args:
        requires: Filter by input tier ("returns", "exposure", etc.).
        category: Filter by statistical category tag.
        backend: Filter by computation profile.

    Returns:
        List of metric metadata dicts (name, requires, category, backend, ref).
    """
    results = []
    for name, meta in _registry.items():
        if requires is not None and meta["requires"] != requires:
            continue
        if category is not None and category not in meta["category"]:
            continue
        if backend is not None and meta["backend"] != backend:
            continue
        results.append(
            {
                "name": name,
                "requires": meta["requires"],
                "category": meta["category"],
                "backend": meta["backend"],
                "ref": meta["ref"],
            }
        )
    return results


def get_metric(name: str) -> dict[str, Any]:
    """Look up a registered metric by name.

    Raises:
        UnknownMetricError: If the metric is not registered.
    """
    from stratstat.exceptions import UnknownMetricError

    if name not in _registry:
        raise UnknownMetricError(f"Unknown metric: {name!r}")
    return _registry[name]


def _compute_one(input_data: Any, metric_name: str, **kwargs: Any) -> MetricResult:
    """Compute a single metric. Wired to the public compute() in __init__.py."""
    from stratstat.exceptions import UnknownMetricError
    from stratstat.inputs import ReturnsInput
    from stratstat.results import MetricResult

    if metric_name not in _registry:
        raise UnknownMetricError(f"Unknown metric: {metric_name!r}")

    entry = _registry[metric_name]
    func = entry["func"]
    requires = entry["requires"]

    if requires == "returns":
        ret_inp = (
            input_data
            if isinstance(input_data, ReturnsInput)
            else ReturnsInput(input_data)
        )
        return cast(MetricResult, func(ret_inp, **kwargs))

    if requires == "exposure":
        from stratstat.inputs import ExposureInput

        exp_inp = (
            input_data
            if isinstance(input_data, ExposureInput)
            else ExposureInput(input_data)
        )
        return cast(MetricResult, func(exp_inp, **kwargs))

    if requires == "trades":
        from stratstat.inputs import TradeInput

        trd_inp = (
            input_data
            if isinstance(input_data, TradeInput)
            else TradeInput(trades=input_data)
        )
        return cast(MetricResult, func(trd_inp, **kwargs))

    if requires == "benchmark":
        from stratstat.inputs import BenchmarkInput

        bench_inp = (
            input_data
            if isinstance(input_data, BenchmarkInput)
            else BenchmarkInput(input_data)
        )
        return cast(MetricResult, func(bench_inp, **kwargs))

    # Other input tiers wired in later phases.
    raise NotImplementedError(
        f"Input tier {requires!r} not yet implemented for metric {metric_name!r}"
    )


def _compute_all(input_data: Any, category: str | None = None, **kwargs: Any) -> MetricSet:
    """Compute all matching metrics. Wired to the public compute_all() in __init__.py."""
    from stratstat.results import MetricSet

    matches = list_metrics(category=category)
    results = []
    for m in matches:
        results.append(_compute_one(input_data, m["name"], **kwargs))
    return MetricSet(results=results)
