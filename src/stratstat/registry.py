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
    alias_of: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a metric function in the global registry.

    Args:
        name: Unique metric name (e.g. "sharpe_ratio").
        requires: Input tier — "returns", "exposure", "trades", "benchmark", or "compare".
        category: Axis-2 classification tags (e.g. ("risk_adjusted", "returns")).
        backend: Computation profile — "vectorized", "sequential", or "resampling".
        ref: Citation string for the formula.
        alias_of: Canonical metric this one duplicates (e.g. a period-level
            twin of a trade-level metric).  ``compute_all`` drops aliases so
            the canonical metric is reported once.

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
            "alias_of": alias_of,
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
        List of metric metadata dicts (name, requires, category, backend, ref,
        alias_of).
    """
    results = []
    for name, meta in _registry.items():
        if requires is not None and meta["requires"] != requires:
            continue
        if category is not None and (not meta["category"] or meta["category"][0] != category):
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
                "alias_of": meta["alias_of"],
            }
        )
    return results


def requires_of(name: str) -> str | None:
    """Return the input tier a metric requires, or None if unregistered."""
    entry = _registry.get(name)
    return entry["requires"] if entry is not None else None


def get_metric(name: str) -> dict[str, Any]:
    """Look up a registered metric by name.

    Raises:
        UnknownMetricError: If the metric is not registered.
    """
    from stratstat.exceptions import UnknownMetricError

    if name not in _registry:
        raise UnknownMetricError(f"Unknown metric: {name!r}")
    return _registry[name]


def _container_params(cls: Any, *, exclude: set[str]) -> set[str]:
    """Keyword parameters *cls* accepts, minus the ones the caller supplies
    positionally or by another route.

    Derived from the signature rather than hardcoded, so a parameter added to
    an Input container becomes reachable through ``compute()`` without editing
    the dispatch code.
    """
    from inspect import signature

    try:
        params = set(signature(cls.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return set()
    return params - {"self"} - exclude


def _take(kwargs: dict[str, Any], names: set[str]) -> dict[str, Any]:
    """Remove and return the entries of *kwargs* whose keys are in *names*."""
    return {k: kwargs.pop(k) for k in list(kwargs) if k in names}


def _build_input(input_data: Any, requires: str, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Build the :class:`Input` container a metric tier needs.

    Extracts ``periods_per_year`` from *kwargs* and forwards it to the Input
    constructor, returning ``(input_object, cleaned_kwargs)`` with that
    constructor-only param removed.

    Secondary constructor data is forwarded too, so ``benchmark=``,
    ``equity=``, ``weights=`` and the rest reach the container they belong to
    instead of falling through to the metric function and raising an error
    that blamed the metric for not accepting them.  Only parameters the target
    container actually declares are taken; anything else is left in *kwargs*
    and still raises, so a typo stays loud.

    This is also how ``rf`` is routed.  ``BenchmarkInput`` and ``CompareInput``
    declare it, so it is consumed here for those tiers.  ``ReturnsInput`` does
    not, so it stays in *kwargs* and reaches the 12 returns tier metrics that
    declare it themselves; popping it unconditionally silently discarded the
    caller's rate and left ``meta["rf"]`` falsely reporting 0.0.  The trades
    and exposure tiers have no consumer, and ``_compute_one`` drops it there.

    Forwarding happens only when this function constructs the container.  A
    pre-built container already carries its own data, so kwargs are left alone
    rather than being silently swallowed against an object that will ignore
    them.

    Raises ``TypeError`` or ``ValueError`` when *input_data* cannot be coerced
    to the requested tier — the signal ``compute_all`` uses to skip a metric
    whose tier does not match the data actually provided.
    """
    ppy = kwargs.pop("periods_per_year", None)

    if requires == "returns":
        from stratstat.inputs import ReturnsInput

        if isinstance(input_data, ReturnsInput):
            return input_data, kwargs
        extra = _take(kwargs, _container_params(ReturnsInput, exclude={"data", "periods_per_year"}))
        return ReturnsInput(input_data, periods_per_year=ppy, **extra), kwargs

    if requires == "exposure":
        from stratstat.inputs import ExposureInput

        if isinstance(input_data, ExposureInput):
            return input_data, kwargs
        extra = _take(
            kwargs, _container_params(ExposureInput, exclude={"positions", "periods_per_year"})
        )
        return ExposureInput(input_data, periods_per_year=ppy, **extra), kwargs

    if requires == "trades":
        from stratstat.inputs import TradeInput

        if isinstance(input_data, TradeInput):
            return input_data, kwargs
        extra = _take(kwargs, _container_params(TradeInput, exclude={"trades", "periods_per_year"}))
        return TradeInput(trades=input_data, periods_per_year=ppy, **extra), kwargs

    if requires == "benchmark":
        from stratstat.inputs import BenchmarkInput

        if isinstance(input_data, BenchmarkInput):
            return input_data, kwargs
        extra = _take(
            kwargs, _container_params(BenchmarkInput, exclude={"returns", "periods_per_year"})
        )
        return BenchmarkInput(input_data, periods_per_year=ppy, **extra), kwargs

    if requires == "compare":
        from stratstat.inputs import CompareInput

        if isinstance(input_data, CompareInput):
            return input_data, kwargs
        extra = _take(
            kwargs, _container_params(CompareInput, exclude={"returns", "periods_per_year"})
        )
        return CompareInput(input_data, periods_per_year=ppy, **extra), kwargs

    raise NotImplementedError(f"Input tier {requires!r} not yet implemented")


def _compute_one(input_data: Any, metric_name: str, **kwargs: Any) -> MetricResult:
    """Compute a single metric. Wired to the public compute() in __init__.py.

    Accepts raw data (numpy, pandas, polars) or a pre-built Input object.
    When raw data is passed, *periods_per_year* and *rf* are extracted from
    **kwargs and forwarded to the appropriate Input constructor so that
    annualisation metadata flows through correctly.

    ``rf`` is accepted on every entry point for signature uniformity, but only
    some metrics declare it.  It is dropped for those that do not rather than
    raising, since callers routinely pass it alongside *periods_per_year*.
    Every other unrecognised keyword still raises ``TypeError``, so typos are
    not swallowed.
    """
    from stratstat.exceptions import UnknownMetricError
    from stratstat.results import MetricResult

    if metric_name not in _registry:
        raise UnknownMetricError(f"Unknown metric: {metric_name!r}")

    entry = _registry[metric_name]
    func = entry["func"]
    inp, clean_kwargs = _build_input(input_data, entry["requires"], **kwargs)
    if "rf" in clean_kwargs and "rf" not in _param_names(func):
        del clean_kwargs["rf"]
    return cast(MetricResult, func(inp, **clean_kwargs))


def _compute_all(
    returns: Any = None,
    trades: Any = None,
    benchmark: Any = None,
    exposure: Any = None,
    compare: Any = None,
    *,
    periods_per_year: int | None = None,
    rf: float = 0.0,
    schema: Any = None,
    columns: Any = None,
    include_returns: bool = True,
    include_trades: bool = True,
    include_benchmark: bool = True,
    include_exposure: bool = True,
    include_compare: bool = True,
    deduplicate: bool = True,
    category: str | None = None,
    tiers: list[str] | None = None,
    **kwargs: Any,
) -> MetricSet:
    """Compute every metric across all input tiers.  Wired to the public
    ``compute_all()`` in ``__init__.py``.

    Each of the five input tiers runs only when its data is present **and**
    its ``include_*`` flag is ``True`` (and, when *tiers* is given, when the
    tier is listed).  The mapping is:

    * ``returns`` tier ← *returns*
    * ``trades`` tier ← *trades*
    * ``benchmark`` tier ← *returns* (strategy) + *benchmark* (benchmark)
    * ``exposure`` tier ← *exposure*
    * ``compare`` tier ← *compare* (runs only when explicitly provided)

    Metrics are never silently dropped; every omission is recorded in the
    returned ``MetricSet.meta`` under:

    * ``"skipped"`` — metrics whose required keyword parameters were not
      supplied, or which raised
      :class:`~stratstat.exceptions.MetricNotApplicableError`.
    * ``"excluded_resampling"`` — resampling-backend metrics (Monte Carlo,
      bootstrap CIs, PBO, White's Reality Check), which are always excluded
      from ``compute_all`` because they are expensive and need their own
      parameters.
    * ``"excluded_tiers"`` — tiers that did not run (data absent or
      ``include_*`` flag false).
    * ``"deduplicated"`` — alias metrics dropped because their canonical
      ``alias_of`` metric also ran (see ``deduplicate``).

    Any other exception propagates, so genuine bugs are not masked.
    """
    from stratstat.exceptions import MetricNotApplicableError
    from stratstat.inputs import (
        BenchmarkInput,
        CompareInput,
        ExposureInput,
        ReturnsInput,
        TradeInput,
    )
    from stratstat.results import MetricSet

    # -- reject keywords nothing could ever consume ---------------------
    # The per-metric filter below is necessary, since every metric takes
    # different parameters, but on its own it means a typo is dropped without
    # a word: ``compute_all(returns=r, rff=0.04)`` returned a result identical
    # to the run without the typo.  Validate up front instead, against the
    # union over every registered metric and every Input container.
    #
    # The union has to span *all* registered metrics, not just the ones that
    # end up running.  Fifteen parameters belong only to resampling metrics,
    # which this function always excludes, so checking against the candidates
    # alone would reject ``target_metric`` and friends as typos.
    _reject_unknown_kwargs(kwargs)

    # -- one schema, shared by every tier ------------------------------
    # Resolved once here rather than per container, so a single call cannot
    # end up with tiers mapped inconsistently.  ``columns=`` at this level
    # mirrors the Schema fields; the single-tier entry points translate their
    # flatter shorthand before calling in.
    from stratstat.schema import _coerce

    schema = _coerce(schema, columns, tier=None)

    # -- which tiers are requested -------------------------------------
    tier_names = ["returns", "trades", "benchmark", "exposure", "compare"]
    if tiers is not None:
        tier_names = [t for t in tier_names if t in set(tiers)]

    flags = {
        "returns": include_returns,
        "trades": include_trades,
        "benchmark": include_benchmark,
        "exposure": include_exposure,
        "compare": include_compare,
    }

    # -- build one Input object per active tier -------------------------
    inputs: dict[str, Any] = {}
    excluded_tiers: list[str] = []

    for tier in tier_names:
        if not flags.get(tier, True):
            excluded_tiers.append(tier)
            continue

        if tier == "returns":
            if returns is None:
                excluded_tiers.append(tier)
                continue
            inputs[tier] = (
                returns
                if isinstance(returns, ReturnsInput)
                else ReturnsInput(returns, periods_per_year=periods_per_year, schema=schema)
            )

        elif tier == "trades":
            if trades is None:
                excluded_tiers.append(tier)
                continue
            inputs[tier] = (
                trades
                if isinstance(trades, TradeInput)
                else TradeInput(trades=trades, periods_per_year=periods_per_year, schema=schema)
            )

        elif tier == "benchmark":
            if returns is None or benchmark is None:
                excluded_tiers.append(tier)
                continue
            strategy = _raw_values(returns)
            inputs[tier] = BenchmarkInput(
                strategy,
                benchmark=benchmark,
                periods_per_year=periods_per_year,
                rf=rf,
                schema=schema,
            )

        elif tier == "exposure":
            if exposure is None:
                excluded_tiers.append(tier)
                continue
            inputs[tier] = (
                exposure
                if isinstance(exposure, ExposureInput)
                else ExposureInput(exposure, periods_per_year=periods_per_year, schema=schema)
            )

        elif tier == "compare":
            if compare is None:
                excluded_tiers.append(tier)
                continue
            if isinstance(compare, CompareInput):
                inputs[tier] = compare
            else:
                strategy = _raw_values(compare)
                inputs[tier] = CompareInput(
                    strategy,
                    benchmark=benchmark,
                    periods_per_year=periods_per_year,
                    rf=rf,
                    schema=schema,
                )

    # -- first pass: collect (name, func, tier) that could run ----------
    # Resampling metrics are always excluded; alias metrics are held aside
    # for dedup; required-kwarg gaps are skipped up front.
    candidates: list[tuple[str, Any, str]] = []
    skipped: list[str] = []
    excluded_resampling: list[str] = []
    aliases: list[str] = []

    for tier in tier_names:
        if tier not in inputs:
            continue
        inp = inputs[tier]
        for m in list_metrics(requires=tier):
            name = m["name"]
            func = _registry[name]["func"]

            if category is not None and (not m["category"] or m["category"][0] != category):
                continue

            if m["backend"] == "resampling":
                excluded_resampling.append(name)
                continue

            if m["alias_of"]:
                aliases.append(name)

            if _missing_required(func, kwargs):
                skipped.append(name)
                continue

            candidates.append((name, func, tier))

    # -- dedup: drop an alias only when its canonical metric also runs --
    run_names = {name for name, _, _ in candidates}
    deduplicated: list[str] = []
    if deduplicate:
        kept: list[tuple[str, Any, str]] = []
        for name, func, tier in candidates:
            canonical = _registry[name]["alias_of"]
            if canonical and canonical in run_names:
                deduplicated.append(name)
                continue
            kept.append((name, func, tier))
        candidates = kept

    # -- second pass: execute -------------------------------------------
    results: list[MetricResult] = []
    consumed: set[str] = set()
    for name, func, tier in candidates:
        inp = inputs[tier]
        params = _param_names(func)
        clean_kwargs = {k: v for k, v in kwargs.items() if k in params}
        consumed |= set(clean_kwargs)
        # ``rf`` is a named parameter of this function, so it never appears in
        # **kwargs.  Returns-tier metrics declare it on the function rather
        # than on the container, so it has to be injected explicitly or the
        # caller's rate is silently ignored.  Benchmark and compare metrics
        # read it off their Input and do not declare it.
        if "rf" in params and "rf" not in clean_kwargs:
            clean_kwargs["rf"] = rf
        try:
            results.append(func(inp, **clean_kwargs))
        except MetricNotApplicableError:
            skipped.append(name)

    meta: dict[str, Any] = {}
    if skipped:
        meta["skipped"] = skipped
    if excluded_resampling:
        meta["excluded_resampling"] = excluded_resampling
    if excluded_tiers:
        meta["excluded_tiers"] = excluded_tiers
    if deduplicated:
        meta["deduplicated"] = deduplicated
    # Recognised, but nothing that actually ran took it: a parameter for a
    # resampling metric, or for a tier that did not run.  Not an error, but it
    # had no effect, and saying so beats letting the caller assume it did.
    unused = sorted(set(kwargs) - consumed)
    if unused:
        meta["unused_kwargs"] = unused
    return MetricSet(results=results, meta=meta)


def _known_kwargs() -> set[str]:
    """Every keyword any metric or any Input container could consume."""
    from stratstat.inputs import (
        BenchmarkInput,
        CompareInput,
        ExposureInput,
        ReturnsInput,
        TradeInput,
    )

    known: set[str] = set()
    for entry in _registry.values():
        known |= _param_names(entry["func"])
    for cls in (ReturnsInput, TradeInput, BenchmarkInput, ExposureInput, CompareInput):
        known |= _container_params(cls, exclude=set())
    return known


def _reject_unknown_kwargs(kwargs: dict[str, Any]) -> None:
    """Raise ``TypeError`` for keywords no metric and no container declares.

    Names a close match when there is one, since the whole point is that the
    caller meant something and it was being thrown away.
    """
    from difflib import get_close_matches

    known = _known_kwargs()
    unknown = sorted(set(kwargs) - known)
    if not unknown:
        return

    parts = []
    for name in unknown:
        close = get_close_matches(name, sorted(known), n=1)
        parts.append(f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    raise TypeError(
        "compute_all() got unexpected keyword argument(s): "
        + ", ".join(parts)
        + ". No registered metric or input container accepts them."
    )


def _raw_values(data: Any) -> Any:
    """Return the raw return matrix for benchmark/compare tiers.

    Accepts pre-built return-bearing Input objects and unwraps them to their
    numpy values; anything else is returned unchanged.
    """
    from stratstat.inputs import BenchmarkInput, CompareInput, ReturnsInput

    if isinstance(data, ReturnsInput):
        return data.values
    if isinstance(data, BenchmarkInput):
        return data.returns
    if isinstance(data, CompareInput):
        return data.returns
    return data


def _param_names(func: Callable[..., Any]) -> set[str]:
    """Return the set of keyword parameter names a metric accepts."""
    from inspect import signature

    try:
        return set(signature(func).parameters)
    except (TypeError, ValueError):
        return set()


def _missing_required(func: Callable[..., Any], kwargs: dict[str, Any]) -> bool:
    """True if *func* has a required keyword parameter absent from *kwargs*."""
    from inspect import Parameter, signature

    try:
        sig_params = list(signature(func).parameters.values())
    except (TypeError, ValueError):
        return False
    for p in sig_params[1:]:  # drop the input-container parameter
        if p.name in kwargs:
            continue
        if p.default is not Parameter.empty:
            continue
        if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
            return True
    return False
