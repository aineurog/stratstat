"""Generic wrappers for returns-tier metrics.

Applies any registered ``returns``-tier metric over rolling windows or
grouped by regime labels.  These are **wrapper** functions, not metrics
themselves — they use the registry but are not registered in it.

Reference
---------
* Section 5.1 ``rolling`` — Zivot & Wang (2006, *Modeling Financial
  Time Series with S-PLUS*, Ch. 3).
* Section 5.2 ``by_regime`` — Ang & Bekaert (2004), "How Regimes
  Affect Asset Allocation," *Financial Analysts Journal*, 60(2).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from stratstat.exceptions import MetricNotApplicableError
from stratstat.inputs import ReturnsInput
from stratstat.registry import _compute_one, get_metric
from stratstat.results import MetricResult

_ROLLING_REF = (
    "Zivot & Wang (2006, Modeling Financial Time Series with S-PLUS, Ch. 3)"
)

_REGIME_REF = (
    "Ang & Bekaert (2004), 'How Regimes Affect Asset Allocation,' "
    "Financial Analysts Journal, 60(2)"
)


def rolling(
    input_data: Any,
    metric_name: str,
    window: int,
    periods_per_year: int | None = None,
    **metric_kwargs: Any,
) -> MetricResult:
    """Apply a registered metric over a rolling window.

    Slides a window of ``window`` periods across the returns and
    recomputes the named metric on each window.  The first ``window - 1``
    entries of the output are NaN.

    Args:
        input_data: ``ReturnsInput`` or 1-D array-like.
        metric_name: Name of a registered ``returns``-tier metric.
        window: Number of periods per rolling window (must be ≥ 2).
        periods_per_year: Annualization factor.  Used only if
            *input_data* is not already a ``ReturnsInput``; otherwise
            the value from the input takes precedence.
        **metric_kwargs: Passed through to the metric function.

    Returns:
        MetricResult with ``ndarray([n_periods])`` as value.
        ``meta["metric"]`` records the inner metric name,
        ``meta["window"]`` records the window size.
    """
    get_metric(metric_name)  # raise UnknownMetricError early for unknown metrics

    ret = _to_returns_input(input_data, periods_per_year=periods_per_year)
    r = ret.values[:, 0]
    n = len(r)

    if window < 2:
        raise ValueError("window must be at least 2.")
    if window > n:
        raise ValueError(
            f"window ({window}) exceeds number of periods ({n})."
        )

    values = np.full(n, np.nan, dtype=np.float64)
    ppy = ret.periods_per_year

    for t in range(window - 1, n):
        win = r[t - window + 1 : t + 1]
        inp = ReturnsInput(win, periods_per_year=ppy)
        try:
            result = _compute_one(inp, metric_name, **metric_kwargs)
            val = result.value
            if isinstance(val, np.ndarray) and val.shape != ():
                val = val.flat[0]
            values[t] = cast(float, val)
        except MetricNotApplicableError:
            values[t] = np.nan

    name = f"rolling_{window}_{metric_name}"

    return MetricResult(
        name=name,
        value=values,
        category=("rolling", "returns"),
        periods_per_year=ppy,
        meta={
            "ref": _ROLLING_REF,
            "metric": metric_name,
            "window": window,
        },
    )


def by_regime(
    input_data: Any,
    metric_name: str,
    regime_labels: Any,
    periods_per_year: int | None = None,
    **metric_kwargs: Any,
) -> MetricResult:
    """Group returns by regime label and compute a metric per regime.

    ``regime_labels`` is an array-like of the **same length** as the
    returns.  Each unique label value defines one regime, and the named
    metric is computed separately on the subset of returns belonging to
    that regime.

    Args:
        input_data: ``ReturnsInput`` or 1-D array-like.
        metric_name: Name of a registered ``returns``-tier metric.
        regime_labels: Array-like of same length as returns (int, str,
            or bool labels).
        periods_per_year: Annualization factor.  Used only if
            *input_data* is not already a ``ReturnsInput``; otherwise
            the value from the input takes precedence.
        **metric_kwargs: Passed through to the metric function.

    Returns:
        MetricResult with ``ndarray([n_regimes])`` as value.
        ``meta["regime_labels"]`` records the unique sorted labels.
    """
    get_metric(metric_name)  # raise UnknownMetricError early for unknown metrics

    ret = _to_returns_input(input_data, periods_per_year=periods_per_year)
    r = ret.values[:, 0]
    n = len(r)

    labels = np.asarray(regime_labels).ravel()

    if labels.shape[0] != n:
        raise ValueError(
            f"regime_labels length ({labels.shape[0]}) must match "
            f"number of periods ({n})."
        )

    unique_labels = np.unique(labels)
    n_regimes = len(unique_labels)
    values = np.full(n_regimes, np.nan, dtype=np.float64)
    ppy = ret.periods_per_year

    for i, label in enumerate(unique_labels):
        mask = labels == label
        regime_r = r[mask]
        if len(regime_r) < 2:
            # Not enough data — leave as NaN.
            continue
        inp = ReturnsInput(regime_r, periods_per_year=ppy)
        try:
            result = _compute_one(inp, metric_name, **metric_kwargs)
            val = result.value
            if isinstance(val, np.ndarray) and val.shape != ():
                val = val.flat[0]
            values[i] = cast(float, val)
        except MetricNotApplicableError:
            values[i] = np.nan

    name = f"{metric_name}_by_regime"

    return MetricResult(
        name=name,
        value=values,
        category=("regime", "returns"),
        periods_per_year=ppy,
        meta={
            "ref": _REGIME_REF,
            "metric": metric_name,
            "regime_labels": unique_labels.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_returns_input(
    data: Any, periods_per_year: int | None = None
) -> ReturnsInput:
    """Normalise *data* to a single-strategy ``ReturnsInput``.

    Raises:
        ValueError: If multi-strategy input is provided.
    """
    if isinstance(data, ReturnsInput):
        inp = data
    else:
        inp = ReturnsInput(data, periods_per_year=periods_per_year)

    if not inp.is_single:
        raise ValueError(
            "rolling() and by_regime() require single-strategy input."
        )
    return inp
