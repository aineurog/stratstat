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


def nanmean(a: NDArray[np.floating], axis: int | None = None) -> NDArray[np.floating] | float:
    """Compute the mean ignoring NaN values. Drop-in for when numba is unavailable."""
    return np.nanmean(a, axis=axis)  # type: ignore[no-any-return]


def nanstd(
    a: NDArray[np.floating], axis: int | None = None, ddof: int = 1
) -> NDArray[np.floating] | float:
    """Compute the standard deviation ignoring NaN values."""
    return np.nanstd(a, axis=axis, ddof=ddof)  # type: ignore[no-any-return]
