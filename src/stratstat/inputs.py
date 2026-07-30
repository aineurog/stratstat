"""Input containers — validate and normalize user-provided data.

Each container accepts pandas.Series/DataFrame, polars.Series/DataFrame,
or numpy.ndarray, normalizes to numpy internally, and exposes what tiers
of metrics are computable given what was actually provided.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _to_numpy(data: Any) -> NDArray[np.floating]:
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

        self.values: NDArray[np.floating] = arr  # shape: (n_periods, n_strategies)
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
    """Wraps positions/weights data for exposure-tier metrics.

    The core input is a positions (weights) matrix of shape
    ``(n_periods, n_assets)`` where each element :math:`w_{i,t}` is the
    weight of asset *i* at period *t* — defined as
    ``position_value_{i,t} / portfolio_value_t``.  Weights may be
    negative (short positions) and need not sum to 1.

    Several exposure metrics require additional data:

    * **Asset-level returns** (``returns=``) — needed for long/short
      book-return metrics (§6.6–§6.7) and beta metrics (§6.8–§6.9).
    * **Benchmark returns** (``benchmark=``) — needed for long/short
      beta (§6.8–§6.9).
    * **Portfolio equity** (``equity=``) — needed for leverage (§6.3).
      If omitted but *returns* is provided, the equity curve is
      computed from the portfolio's cumulative return
      (:math:`\\sum_i w_{i,t-1} r_{i,t}`).

    Parameters
    ----------
    positions: array-like
        Position weights of shape ``(n_periods, n_assets)``.
        Accepts numpy, pandas, or polars.
    returns: array-like, optional
        Asset-level returns of shape ``(n_periods, n_assets)``.
    benchmark: array-like, optional
        Benchmark returns of shape ``(n_periods,)``.
    equity: array-like, optional
        Portfolio equity of shape ``(n_periods,)``.
    periods_per_year: int, optional
        Annualization factor (e.g. 252 for daily). Required only
        by turnover.
    """

    def __init__(
        self,
        positions: Any,
        returns: Any | None = None,
        benchmark: Any | None = None,
        equity: Any | None = None,
        periods_per_year: int | None = None,
    ) -> None:
        # -- positions (always required) ---------------------------------
        pos = _to_numpy(positions)
        if pos.ndim == 0:
            pos = pos.reshape(1, 1)
        elif pos.ndim == 1:
            pos = pos.reshape(-1, 1)
        elif pos.ndim > 2:
            raise ValueError(
                f"Positions must be 1-D or 2-D, got {pos.ndim}-D array."
            )
        self.positions: NDArray[np.floating] = pos  # (n_periods, n_assets)
        self.n_periods: int = pos.shape[0]
        self.n_assets: int = pos.shape[1]

        # -- asset-level returns (optional) ------------------------------
        if returns is not None:
            ret = _to_numpy(returns)
            if ret.ndim == 1:
                ret = ret.reshape(-1, 1)
            if ret.shape != (self.n_periods, self.n_assets):
                raise ValueError(
                    f"Returns shape {ret.shape} must match positions shape "
                    f"{(self.n_periods, self.n_assets)}."
                )
            self.returns: NDArray[np.floating] | None = ret
        else:
            self.returns = None

        # -- benchmark returns (optional) ---------------------------------
        if benchmark is not None:
            bench = _to_numpy(benchmark)
            bench = bench.ravel()
            if bench.shape[0] != self.n_periods:
                raise ValueError(
                    f"Benchmark length {bench.shape[0]} must match "
                    f"n_periods {self.n_periods}."
                )
            self.benchmark: NDArray[np.floating] | None = bench
        else:
            self.benchmark = None

        # -- equity (optional; can be derived from positions + returns) ---
        if equity is not None:
            eq = _to_numpy(equity).ravel()
            if eq.shape[0] != self.n_periods:
                raise ValueError(
                    f"Equity length {eq.shape[0]} must match "
                    f"n_periods {self.n_periods}."
                )
            self.equity: NDArray[np.floating] | None = eq
        elif self.returns is not None:
            # Compute portfolio returns from lagged weights and asset returns.
            # w_{i,t-1} for t >= 1; t=0 is NaN (unknown prior weights).
            w_lag = np.roll(self.positions, shift=1, axis=0)
            w_lag[0, :] = np.nan
            port_ret = np.nansum(w_lag * self.returns, axis=1)  # (n_periods,)
            # Equity starts at 1.0; cumprod of (1 + return).
            self.equity = np.nancumprod(1.0 + port_ret)
            self.equity[0] = np.nan  # first period's equity is undefined
            # If all first-period returns are NaN, fill the first equity
            # entry with 1.0 so the rest of the curve is usable.
            if np.isnan(self.equity[0]):
                # Find first non-NaN equity value and back-fill.
                valid = np.where(~np.isnan(self.equity))[0]
                if len(valid) > 0:
                    self.equity[: valid[0]] = self.equity[valid[0]]
                else:
                    self.equity = None
        else:
            self.equity = None

        self.periods_per_year: int | None = periods_per_year

    # -- convenience predicates ------------------------------------------
    @property
    def has_returns(self) -> bool:
        """True if asset-level returns were provided."""
        return self.returns is not None

    @property
    def has_benchmark(self) -> bool:
        """True if benchmark returns were provided."""
        return self.benchmark is not None

    @property
    def has_equity(self) -> bool:
        """True if portfolio equity is available."""
        return self.equity is not None

    def __repr__(self) -> str:
        return (
            f"ExposureInput(n_periods={self.n_periods}, "
            f"n_assets={self.n_assets}, "
            f"has_returns={self.has_returns}, "
            f"has_benchmark={self.has_benchmark}, "
            f"has_equity={self.has_equity}, "
            f"periods_per_year={self.periods_per_year})"
        )


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
