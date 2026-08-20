"""Internal utility functions shared across core modules.

Includes annualization factor computation, NaN handling, vectorization helpers,
and numba dispatch logic.

IMPORTANT: If this file grows past ~300 lines, it should be split into
core/_utils/ subpackage with separate modules for annualization, NaN handling,
and numba dispatch.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def annualization_factor(periods_per_year: int | None) -> float:
    """Return the annualization factor for the given frequency.

    Args:
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly, etc.).
            If None, returns 1.0 (no annualization).

    Returns:
        Annualization factor as a float.
    """
    if periods_per_year is None:
        return 1.0
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    return float(periods_per_year)


def is_numba_available() -> bool:
    """Check if numba is installed and usable.

    Returns:
        True if numba can be imported.
    """
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


# Threshold (in inner-loop element operations) above which the numba path is
# preferred over the pure-numpy fallback.  Below this, the one-time numba JIT
# compile cost (hundreds of milliseconds with ``cache=False``) dominates the
# savings, so small inputs stay on numpy.
_NUMBA_MIN_WORK = 1_000_000


def numba_worthwhile(work: int) -> bool:
    """Return True when the estimated work justifies the numba compile cost.

    The numba kernels compile on first use, which costs a few hundred
    milliseconds.  For inputs that represent only a few thousand inner-loop
    iterations the numpy fallback finishes faster than numba can compile, so
    the resampling dispatchers call this helper to keep small workloads on the
    numpy path.

    Args:
        work: Approximate number of inner-loop element operations.

    Returns:
        True if the numba path should be preferred over the numpy fallback.
    """
    return work >= _NUMBA_MIN_WORK


def nanmean(a: NDArray[np.floating], axis: int | None = None) -> NDArray[np.floating] | float:
    """Compute the mean ignoring NaN values. Drop-in for when numba is unavailable."""
    return np.nanmean(a, axis=axis)  # type: ignore[no-any-return]


def nanstd(
    a: NDArray[np.floating], axis: int | None = None, ddof: int = 1
) -> NDArray[np.floating] | float:
    """Compute the standard deviation ignoring NaN values."""
    return np.nanstd(a, axis=axis, ddof=ddof)  # type: ignore[no-any-return]


def sample_skewness(a: NDArray[np.floating]) -> float:
    """Bias-corrected (adjusted) sample skewness for a 1-D array.

    Uses the same formula as ``_sample_skewness`` in ``inference.py``
    and the ``skewness`` metric in ``descriptive.py``:

    .. math::
        \\gamma_1 = \\frac{n_{\\text{eff}}}
        {(n_{\\text{eff}}-1)(n_{\\text{eff}}-2)}
        \\sum_{i} z_i^3

    where :math:`z_i = (a_i - \\bar{a}) / \\sigma` and
    :math:`\\sigma` is the sample standard deviation (ddof=1).

    Returns ``NaN`` for fewer than 3 valid observations or zero
    variance.
    """
    valid_mask = np.isfinite(a)
    n_eff = np.sum(valid_mask)
    if n_eff < 3:
        return np.nan

    mean = np.nanmean(a)
    std = np.nanstd(a, ddof=1)
    if std < 1e-15:
        return np.nan

    z = (a - mean) / std
    z = np.where(np.isnan(z), 0.0, z)
    m3 = np.nansum(z**3)

    factor = n_eff / ((n_eff - 1.0) * (n_eff - 2.0))
    return float(factor * m3)


def ols_beta(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Ordinary least-squares beta: Cov(x, y) / Var(y).

    Drops periods where either series is NaN.  Returns NaN if fewer
    than 3 valid overlapping observations.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xc = x[mask]
    yc = y[mask]
    cov = np.cov(xc, yc, ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def compute_cagr(r: NDArray[np.floating], periods_per_year: float) -> NDArray[np.floating]:
    """Compute CAGR for each strategy column.

    Formula:
        CAGR = exp(P * nanmean(log(1 + r))) - 1

    where P is ``periods_per_year``.

    Args:
        r: Returns array of shape (n_periods, n_strategies).
        periods_per_year: Annualization factor.

    Returns:
        CAGR array of shape (n_strategies,).
    """
    log_returns: NDArray[np.floating] = np.log(1.0 + r)
    mean_log: NDArray[np.floating] = np.nanmean(log_returns, axis=0)
    arr: NDArray[np.floating] = np.exp(mean_log * periods_per_year) - 1.0
    return arr
