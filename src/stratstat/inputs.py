"""Input containers — validate and normalize user-provided data.

Each container accepts pandas.Series/DataFrame, polars.Series/DataFrame,
or numpy.ndarray, normalizes to numpy internally, and exposes what tiers
of metrics are computable given what was actually provided.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(data: Any) -> np.ndarray:
    """Normalize any accepted input type to a numpy array.

    Accepts: numpy.ndarray, pandas.Series, pandas.DataFrame,
             polars.Series, polars.DataFrame.
    """
    if isinstance(data, np.ndarray):
        return data

    # pandas
    try:
        import pandas as pd
    except ImportError:
        pass
    else:
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data.to_numpy(dtype=float)

    # polars
    try:
        import polars as pl
    except ImportError:
        pass
    else:
        if isinstance(data, pl.Series):
            return data.to_numpy()
        if isinstance(data, pl.DataFrame):
            return data.to_numpy()

    raise TypeError(
        f"Unsupported input type: {type(data).__name__}. "
        f"Expected numpy.ndarray, pandas.Series/DataFrame, or polars.Series/DataFrame."
    )


class ReturnsInput:
    """Wraps a returns series or matrix.

    Accepts a single strategy (1-D) or multiple strategies (2-D columns).
    All inputs are normalized to numpy arrays.
    """

    def __init__(self, data: Any, periods_per_year: int | None = None):
        self._raw = data
        arr = _to_numpy(data)

        # Ensure at least 1-D
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)  # single-column for uniform batch handling
        elif arr.ndim > 2:
            raise ValueError(
                f"Returns data must be 1-D or 2-D, got {arr.ndim}-D array."
            )

        self.values: np.ndarray = arr  # shape: (n_periods, n_strategies)
        self.n_periods: int = arr.shape[0]
        self.n_strategies: int = arr.shape[1]
        self.periods_per_year: int | None = periods_per_year

    @property
    def is_single(self) -> bool:
        """True if the input represents a single strategy."""
        return self.n_strategies == 1

    def __repr__(self) -> str:
        return (
            f"ReturnsInput(n_periods={self.n_periods}, "
            f"n_strategies={self.n_strategies}, "
            f"periods_per_year={self.periods_per_year})"
        )


class ExposureInput:
    """Wraps returns + positions/weights data.

    Not yet implemented — placeholder for Phase 5.
    """

    def __init__(self, returns: Any, positions: Any, periods_per_year: int | None = None):
        raise NotImplementedError("ExposureInput not yet implemented.")


class TradeInput:
    """Wraps returns + trade/transaction log.

    Not yet implemented — placeholder for Phase 5.
    """

    def __init__(
        self,
        returns: Any,
        trades: Any,
        positions: Any | None = None,
        periods_per_year: int | None = None,
    ):
        raise NotImplementedError("TradeInput not yet implemented.")
