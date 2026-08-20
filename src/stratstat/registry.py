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
        category: Filter by primary statistical category tag (the first
            element of the ``category`` tuple, e.g. ``"risk"``).
        backend: Filter by computation profile.

    Returns:
        List of metric metadata dicts (name, requires, category, backend, ref).
    """
    results = []
    for name, meta in _registry.items():
        if requires is not None and meta["requires"] != requires:
            continue
        if category is not None and (
            not meta["category"] or meta["category"][0] != category
        ):
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


def _build_input(
    input_data: Any, requires: str, **kwargs: Any
) -> tuple[Any, dict[str, Any]]:
    """Build the :class:`Input` container a metric tier needs.

    Extracts ``periods_per_year`` and ``rf`` from *kwargs* and forwards them
    to the Input constructor, returning ``(input_object, cleaned_kwargs)``
    with those constructor-only params removed.

    Raises ``TypeError`` or ``ValueError`` when *input_data* cannot be coerced
    to the requested tier — the signal ``compute_all`` uses to skip a metric
    whose tier does not match the data actually provided.
    """
    ppy = kwargs.pop("periods_per_year", None)
    rf = kwargs.pop("rf", 0.0)

    inp: Any
    if requires == "returns":
        from stratstat.inputs import ReturnsInput

        inp = (
            input_data
            if isinstance(input_data, ReturnsInput)
            else ReturnsInput(input_data, periods_per_year=ppy)
        )
        return inp, kwargs

    if requires == "exposure":
        from stratstat.inputs import ExposureInput

        inp = (
            input_data
            if isinstance(input_data, ExposureInput)
            else ExposureInput(input_data, periods_per_year=ppy)
        )
        return inp, kwargs

    if requires == "trades":
        from stratstat.inputs import TradeInput

        inp = (
            input_data
            if isinstance(input_data, TradeInput)
            else TradeInput(trades=input_data, periods_per_year=ppy)
        )
        return inp, kwargs

    if requires == "benchmark":
        from stratstat.inputs import BenchmarkInput

        inp = (
            input_data
            if isinstance(input_data, BenchmarkInput)
            else BenchmarkInput(input_data, periods_per_year=ppy, rf=rf)
        )
        return inp, kwargs

    if requires == "compare":
        from stratstat.inputs import CompareInput

        inp = (
            input_data
            if isinstance(input_data, CompareInput)
            else CompareInput(input_data, periods_per_year=ppy, rf=rf)
        )
        return inp, kwargs

    raise NotImplementedError(f"Input tier {requires!r} not yet implemented")


def _compute_one(input_data: Any, metric_name: str, **kwargs: Any) -> MetricResult:
    """Compute a single metric. Wired to the public compute() in __init__.py.

    Accepts raw data (numpy, pandas, polars) or a pre-built Input object.
    When raw data is passed, *periods_per_year* and *rf* are extracted from
    **kwargs and forwarded to the appropriate Input constructor so that
    annualisation metadata flows through correctly.
    """
    from stratstat.exceptions import UnknownMetricError
    from stratstat.results import MetricResult

    if metric_name not in _registry:
        raise UnknownMetricError(f"Unknown metric: {metric_name!r}")

    entry = _registry[metric_name]
    func = entry["func"]
    inp, clean_kwargs = _build_input(input_data, entry["requires"], **kwargs)
    return cast(MetricResult, func(inp, **clean_kwargs))


def _compute_all(input_data: Any, category: str | None = None, **kwargs: Any) -> MetricSet:
    """Compute all matching metrics. Wired to the public compute_all() in __init__.py.

    Metrics that are legitimately inapplicable to the given input are skipped
    (not silently swallowed): their names are recorded in
    ``MetricSet.meta["skipped"]``.  A metric is skipped when

    * its input tier cannot be satisfied by the data provided (a raw dict
      cannot feed the returns tier, a raw array cannot feed the trades
      tier, and so on), or
    * it declares a required keyword parameter that was not supplied in
      *kwargs*, or
    * it raises :class:`~stratstat.exceptions.MetricNotApplicableError`.

    Any other exception propagates, so genuine bugs are not masked.
    """
    from inspect import Parameter, signature

    from stratstat.exceptions import MetricNotApplicableError
    from stratstat.results import MetricSet

    matches = list_metrics(category=category)
    results: list[MetricResult] = []
    skipped: list[str] = []
    for m in matches:
        name = m["name"]
        func = _registry[name]["func"]

        # A metric whose required keyword parameters aren't supplied cannot
        # be computed in this batch (e.g. block_bootstrap_ci needs
        # ``target_metric``).
        missing: list[str] = []
        try:
            sig_params = list(signature(func).parameters.values())
        except (TypeError, ValueError):
            sig_params = []
        for p in sig_params[1:]:  # drop the input-container parameter
            if p.name in kwargs:
                continue
            if p.default is not Parameter.empty:
                continue
            if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
                missing.append(p.name)
        if missing:
            skipped.append(name)
            continue

        # Coerce the input to this metric's tier. A coercion failure means
        # the data cannot satisfy this metric's tier, so it is skipped.
        try:
            inp, clean_kwargs = _build_input(input_data, m["requires"], **kwargs)
        except (TypeError, ValueError):
            skipped.append(name)
            continue

        try:
            results.append(func(inp, **clean_kwargs))
        except MetricNotApplicableError:
            skipped.append(name)

    meta = {"skipped": skipped} if skipped else {}
    return MetricSet(results=results, meta=meta)
