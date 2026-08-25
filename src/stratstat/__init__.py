"""
StratStat — Quantitative strategy evaluation statistics.
"""

__version__ = "1.0.0"

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
from stratstat.schema import Schema, clear_schema, get_schema, set_schema

__all__ = [
    "__version__",
    "register_metric",
    "list_metrics",
    "get_metric",
    "compute",
    "compute_all",
    "compute_returns",
    "compute_trades",
    "compute_benchmark",
    "compute_exposure",
    "compute_compare",
    "MetricResult",
    "MetricSet",
    "ReturnsInput",
    "ExposureInput",
    "TradeInput",
    "BenchmarkInput",
    "CompareInput",
    "get_default",
    "set_default",
    "Schema",
    "set_schema",
    "get_schema",
    "clear_schema",
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
        input_data,
        metric_name,
        periods_per_year=periods_per_year,
        rf=rf,
        **kwargs,
    )


def compute_all(
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
    """Compute every metric across all five input tiers in one call.

    Provide the data you have; each tier runs only when its data is present
    (and its ``include_*`` flag is ``True``).  The mapping is:

    * ``returns`` → returns-tier metrics (strategy returns).
    * ``trades`` → trade-tier metrics (a trade log with a ``pnl`` column).
    * ``benchmark`` → benchmark-tier metrics (strategy returns +
      benchmark returns).
    * ``exposure`` → exposure-tier metrics (positions / weights).
    * ``compare`` → compare-tier metrics (multi-strategy returns); this tier
      runs only when ``compare`` is explicitly provided.

    Parameters
    ----------
    returns:
        Strategy returns (numpy, pandas, polars, or a :class:`ReturnsInput`).
    trades:
        Trade log (dict, pandas/polars DataFrame, or a :class:`TradeInput`).
    benchmark:
        Benchmark returns (numpy, pandas, polars).
    exposure:
        Positions / weights (numpy, pandas, polars, or an
        :class:`ExposureInput`).
    compare:
        Multi-strategy returns for compare-tier metrics.  The compare tier
        runs only when this is provided.
    periods_per_year:
        Annualisation factor (252 for daily, 12 for monthly, etc.).
    rf:
        Risk-free rate per period (default 0.0).  Used by benchmark-tier and
        compare-tier metrics.
    include_returns, include_trades, include_benchmark, include_exposure,
    include_compare:
        Per-tier switches (all default ``True``).
    deduplicate:
        Drop period-level "twin" metrics when their trade-level canonical
        (e.g. ``avg_up_period`` vs ``avg_win``) also runs (default ``True``).
    category:
        Primary statistical tag to filter by (e.g. ``"risk"``,
        ``"descriptive"``).  ``None`` computes all matching metrics.
    tiers:
        Optional explicit list of tiers to run (e.g. ``["returns"]``).
    **kwargs:
        Forwarded to metric functions (e.g. ``confidence=0.95``).

    Returns
    -------
    MetricSet
        Ordered collection of :class:`MetricResult` objects.  Its ``meta``
        records ``"skipped"``, ``"excluded_resampling"``,
        ``"excluded_tiers"``, and ``"deduplicated"`` names so nothing is
        dropped silently.
    """
    from stratstat.registry import _compute_all
    from stratstat.schema import Schema

    if schema is not None and columns is not None:
        raise TypeError("Pass either schema= or columns=, not both.")
    if columns is not None:
        schema = Schema(**dict(columns))

    return _compute_all(
        returns=returns,
        trades=trades,
        benchmark=benchmark,
        exposure=exposure,
        compare=compare,
        periods_per_year=periods_per_year,
        rf=rf,
        schema=schema,
        include_returns=include_returns,
        include_trades=include_trades,
        include_benchmark=include_benchmark,
        include_exposure=include_exposure,
        include_compare=include_compare,
        deduplicate=deduplicate,
        category=category,
        tiers=tiers,
        **kwargs,
    )


def compute_returns(
    data: Any,
    *,
    periods_per_year: int | None = None,
    rf: float = 0.0,
    category: str | None = None,
    schema: Any = None,
    columns: Any = None,
    **kwargs: Any,
) -> MetricSet:
    """Compute every returns-tier metric on strategy returns.

    Parameters
    ----------
    data:
        Strategy returns (numpy, pandas, polars, or a :class:`ReturnsInput`).
    periods_per_year:
        Annualisation factor (252 for daily, etc.).
    rf:
        Risk-free rate per period (unused by returns-tier metrics; accepted
        for signature uniformity).
    category:
        Primary statistical tag to filter by (e.g. ``"risk"``).
    schema:
        A :class:`Schema` mapping canonical names to your columns.
    columns:
        Inline shorthand for ``schema``.  On this single-tier entry point it
        names the returns column directly, e.g. ``columns={"returns": "pct"}``.
    **kwargs:
        Forwarded to metric functions.

    Returns
    -------
    MetricSet
        All returns-tier metrics (resampling metrics excluded).
    """
    from stratstat.registry import _compute_all
    from stratstat.schema import Schema

    if schema is not None and columns is not None:
        raise TypeError("Pass either schema= or columns=, not both.")
    if columns is not None:
        schema = Schema(**dict(columns))

    return _compute_all(
        returns=data,
        periods_per_year=periods_per_year,
        rf=rf,
        category=category,
        tiers=["returns"],
        schema=schema,
        **kwargs,
    )


def compute_trades(
    trades: Any,
    *,
    periods_per_year: int | None = None,
    category: str | None = None,
    columns: Any = None,
    **kwargs: Any,
) -> MetricSet:
    """Compute every trade-tier metric on a trade log.

    Parameters
    ----------
    trades:
        Trade log (dict with a ``pnl`` column, pandas/polars DataFrame, or a
        :class:`TradeInput`).
    periods_per_year:
        Annualisation factor (required by holding-period metrics).
    category:
        Primary statistical tag to filter by (trade-tier metrics use
        ``"trades"``).
    columns:
        Inline column mapping, canonical name to the name in your trade log,
        e.g. ``{"side": "direction"}``.  This entry point covers one tier, so
        the mapping is written flat rather than nested under ``trades``.
        Shorthand for ``schema=Schema(trades=...)``; pass one or the other.
    **kwargs:
        Forwarded to metric functions.

    Returns
    -------
    MetricSet
        All trade-tier metrics.
    """
    from stratstat.registry import _compute_all

    schema_arg = kwargs.pop("schema", None)
    if columns is not None:
        # The tier is unambiguous here, so a flat mapping is what a caller
        # would naturally write. Normalise it to the nested form before
        # handing off, so downstream sees exactly one shape.
        from stratstat.schema import Schema

        if schema_arg is not None:
            raise TypeError(
                "Pass either schema= or columns=, not both. "
                "columns= is shorthand that builds a Schema."
            )
        schema_arg = Schema(trades=dict(columns))

    return _compute_all(
        trades=trades,
        periods_per_year=periods_per_year,
        category=category,
        tiers=["trades"],
        schema=schema_arg,
        **kwargs,
    )


def compute_benchmark(
    returns: Any,
    benchmark: Any,
    *,
    periods_per_year: int | None = None,
    rf: float = 0.0,
    category: str | None = None,
    schema: Any = None,
    columns: Any = None,
    **kwargs: Any,
) -> MetricSet:
    """Compute every benchmark-tier metric on strategy + benchmark returns.

    Parameters
    ----------
    returns:
        Strategy returns (numpy, pandas, polars, or a :class:`ReturnsInput`).
    benchmark:
        Benchmark returns (numpy, pandas, polars).
    periods_per_year:
        Annualisation factor.
    rf:
        Risk-free rate per period (default 0.0).
    category:
        Primary statistical tag to filter by (benchmark-tier metrics use
        ``"benchmark"``).
    **kwargs:
        Forwarded to metric functions.

    Returns
    -------
    MetricSet
        All benchmark-tier metrics.
    """
    from stratstat.registry import _compute_all
    from stratstat.schema import Schema

    if schema is not None and columns is not None:
        raise TypeError("Pass either schema= or columns=, not both.")
    if columns is not None:
        schema = Schema(**dict(columns))

    return _compute_all(
        returns=returns,
        benchmark=benchmark,
        periods_per_year=periods_per_year,
        rf=rf,
        category=category,
        tiers=["benchmark"],
        schema=schema,
        **kwargs,
    )


def compute_exposure(
    positions: Any,
    *,
    returns: Any = None,
    benchmark: Any = None,
    benchmark_weights: Any = None,
    equity: Any = None,
    periods_per_year: int | None = None,
    category: str | None = None,
    schema: Any = None,
    columns: Any = None,
    **kwargs: Any,
) -> MetricSet:
    """Compute every exposure-tier metric on a positions/weights matrix.

    Parameters
    ----------
    positions:
        Position weights of shape ``(n_periods, n_assets)``.
    returns:
        Asset-level returns of shape ``(n_periods, n_assets)`` (optional).
    benchmark:
        Benchmark returns (optional, for long/short beta metrics).
    benchmark_weights:
        Benchmark constituent weights (optional, for active share).
    equity:
        Portfolio equity curve (optional; derived from positions + returns if
        omitted).
    periods_per_year:
        Annualisation factor (required by turnover).
    category:
        Primary statistical tag to filter by (exposure-tier metrics use
        ``"exposure"``).
    **kwargs:
        Forwarded to metric functions.

    Returns
    -------
    MetricSet
        All exposure-tier metrics.
    """
    from stratstat.inputs import ExposureInput
    from stratstat.registry import _compute_all
    from stratstat.schema import Schema

    if schema is not None and columns is not None:
        raise TypeError("Pass either schema= or columns=, not both.")
    if columns is not None:
        schema = Schema(**dict(columns))

    if isinstance(positions, ExposureInput):
        inp = positions
    else:
        inp = ExposureInput(
            positions,
            returns=returns,
            benchmark=benchmark,
            benchmark_weights=benchmark_weights,
            equity=equity,
            periods_per_year=periods_per_year,
            schema=schema,
        )
    return _compute_all(
        exposure=inp,
        category=category,
        tiers=["exposure"],
        **kwargs,
    )


def compute_compare(
    returns: Any,
    *,
    weights: Any = None,
    benchmark: Any = None,
    periods_per_year: int | None = None,
    rf: float = 0.0,
    category: str | None = None,
    schema: Any = None,
    columns: Any = None,
    **kwargs: Any,
) -> MetricSet:
    """Compute every compare-tier metric on multi-strategy returns.

    Parameters
    ----------
    returns:
        Strategy returns of shape ``(n_periods, n_strategies)`` (at least two
        strategies for most metrics).
    weights:
        Strategy weights of shape ``(n_strategies,)`` (defaults to equal
        weight where needed).
    benchmark:
        Benchmark returns (required only by White's Reality Check).
    periods_per_year:
        Annualisation factor (required by JK test and PBO).
    rf:
        Risk-free rate per period (default 0.0).
    category:
        Primary statistical tag to filter by (compare-tier metrics use
        ``"relative"``).
    **kwargs:
        Forwarded to metric functions.

    Returns
    -------
    MetricSet
        All compare-tier metrics (resampling metrics excluded).
    """
    from stratstat.inputs import CompareInput
    from stratstat.registry import _compute_all
    from stratstat.schema import Schema

    if schema is not None and columns is not None:
        raise TypeError("Pass either schema= or columns=, not both.")
    if columns is not None:
        schema = Schema(**dict(columns))

    if isinstance(returns, CompareInput):
        inp = returns
    else:
        inp = CompareInput(
            returns,
            weights=weights,
            benchmark=benchmark,
            periods_per_year=periods_per_year,
            rf=rf,
            schema=schema,
        )
    return _compute_all(
        compare=inp,
        category=category,
        tiers=["compare"],
        **kwargs,
    )
