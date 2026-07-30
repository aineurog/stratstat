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
    """Wraps returns + trade/transaction log for trade-tier metrics.

    The trade log is a table where each row represents one completed
    (round-trip) trade.  It can be provided as a ``dict`` of arrays, a
    ``pandas.DataFrame``, or a ``polars.DataFrame``.

    **Required field:**

    * ``pnl`` — per-trade profit/loss expressed as a return fraction
      (e.g. 0.02 = 2 % gain, −0.015 = 1.5 % loss).

    **Optional fields** (some metrics raise ``ValueError`` if the
    required field is absent — see individual metric docs):

    * ``side`` — trade direction: ``"long"``/``"short"`` (strings,
      case-insensitive) or ``+1``/``-1`` (int).  Needed for long/short
      win rates (§7.3–§7.4) and long/short breakdowns (§7.31–§7.37).
    * ``duration`` — holding period duration in the same units as the
      return frequency.  Needed for holding-period metrics (§7.10,
      §7.11, §7.18, §7.19, §7.23, §7.34).  If absent but ``entry_time``
      and ``exit_time`` are present, duration is computed from them.
    * ``fill_price``, ``decision_price`` — needed for implementation
      shortfall (§7.15).
    * ``intratrade_prices`` — a sequence of per-trade price paths
      (list of arrays).  Needed for MFE/MAE (§7.28–§7.29).

    Parameters
    ----------
    returns: array-like, optional
        Portfolio-level returns of shape ``(n_periods,)``.  Not required
        by core trade metrics; can be ``None``.
    trades: dict, pandas.DataFrame, or polars.DataFrame
        Trade log with required ``pnl`` column and optional columns.
    positions: array-like, optional
        Position data for cross-referencing (not required by current
        trade metrics).
    periods_per_year: int, optional
        Annualization factor (e.g. 252 for daily).
    """

    def __init__(
        self,
        returns: Any = None,
        trades: Any = None,
        positions: Any | None = None,
        periods_per_year: int | None = None,
    ) -> None:
        # -- returns (optional) -----------------------------------------
        if returns is not None:
            ret = _to_numpy(returns)
            if ret.ndim == 0:
                ret = ret.reshape(1)
            elif ret.ndim == 1:
                ret = ret.reshape(-1, 1)
            elif ret.ndim > 2:
                raise ValueError(
                    f"Returns data must be 1-D or 2-D, got {ret.ndim}-D array."
                )
            self.returns: NDArray[np.floating] | None = ret
            self.n_periods: int = ret.shape[0]
        else:
            self.returns = None
            self.n_periods = 0

        # -- trades log -------------------------------------------------
        self._trades = self._normalize_trades(trades)

        # -- positions (optional) ---------------------------------------
        if positions is not None:
            pos = _to_numpy(positions)
            if pos.ndim == 1:
                pos = pos.reshape(-1, 1)
            self.positions: NDArray[np.floating] | None = pos
        else:
            self.positions = None

        self.periods_per_year: int | None = periods_per_year

    # ------------------------------------------------------------------
    # Trade-log normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_trades(trades: Any) -> dict[str, Any]:
        """Normalize trade log to a dict of numpy arrays.

        Accepts dict, pandas.DataFrame, or polars.DataFrame.
        Validates that the required ``pnl`` field is present.
        Normalises ``side`` to a boolean ``is_long`` array.
        Computes ``duration`` from ``entry_time``/``exit_time`` if
        not provided directly.
        """
        # -- convert to dict of lists/arrays ---------------------------
        if isinstance(trades, dict):
            raw: dict[str, Any] = dict(trades)
        else:
            # Try pandas
            try:
                import pandas as pd
            except ImportError:
                pass
            else:
                if isinstance(trades, pd.DataFrame):
                    raw = {
                        col: trades[col].to_numpy()
                        for col in trades.columns
                    }
                    return TradeInput._validate_and_augment(raw)

            # Try polars
            try:
                import polars as pl
            except ImportError:
                pass
            else:
                if isinstance(trades, pl.DataFrame):
                    raw = {
                        col: trades[col].to_numpy()
                        for col in trades.columns
                    }
                    return TradeInput._validate_and_augment(raw)

            raise TypeError(
                f"Unsupported trades type: {type(trades).__name__}. "
                f"Expected dict, pandas.DataFrame, or polars.DataFrame."
            )

        return TradeInput._validate_and_augment(raw)

    @staticmethod
    def _validate_and_augment(
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate required fields and normalize optional fields."""
        # -- required: pnl ----------------------------------------------
        if "pnl" not in raw:
            raise ValueError(
                "Trade log must contain a 'pnl' column with per-trade "
                "profit/loss values."
            )
        pnl = np.asarray(raw["pnl"], dtype=np.float64).ravel()
        result: dict[str, Any] = {"pnl": pnl}
        n_trades = len(pnl)

        # -- optional: side -> is_long ---------------------------------
        if "side" in raw:
            result["is_long"] = TradeInput._normalize_side(
                raw["side"], n_trades
            )

        # -- optional: duration (or entry_time/exit_time) ---------------
        if "duration" in raw:
            result["duration"] = np.asarray(
                raw["duration"], dtype=np.float64
            ).ravel()
        elif "entry_time" in raw and "exit_time" in raw:
            entry = np.asarray(raw["entry_time"], dtype=np.float64).ravel()
            exit_ = np.asarray(raw["exit_time"], dtype=np.float64).ravel()
            result["duration"] = exit_ - entry

        # -- optional: fill_price, decision_price -----------------------
        for field in ("fill_price", "decision_price"):
            if field in raw:
                result[field] = np.asarray(
                    raw[field], dtype=np.float64
                ).ravel()

        # -- optional: intratrade_prices --------------------------------
        if "intratrade_prices" in raw:
            itp = raw["intratrade_prices"]
            if isinstance(itp, (list, tuple)):
                result["intratrade_prices"] = [
                    np.asarray(p, dtype=np.float64).ravel() for p in itp
                ]
            else:
                # Single array — wrap per trade? Assume one per row.
                result["intratrade_prices"] = [
                    np.asarray(itp, dtype=np.float64).ravel()
                ]

        return result

    @staticmethod
    def _normalize_side(
        side: Any, n_trades: int
    ) -> NDArray[np.bool_]:
        """Normalize the ``side`` field to a boolean ``is_long`` array.

        Accepts:
        * strings: ``"long"`` / ``"short"`` (case-insensitive)
        * ints: ``+1`` (long) / ``-1`` (short)
        * bools: ``True`` (long) / ``False`` (short)
        """
        raw = np.asarray(side).ravel()
        if raw.dtype.kind in ("U", "S", "O"):
            # String / object — lower-case and compare
            is_long = np.array(
                [str(s).strip().lower() == "long" for s in raw],
                dtype=bool,
            )
        elif raw.dtype.kind == "b":
            is_long = raw.astype(bool)
        elif raw.dtype.kind in ("i", "f"):
            # +1 = long, -1 = short
            is_long = raw > 0
        else:
            raise ValueError(
                f"Cannot interpret 'side' values of dtype {raw.dtype}. "
                f"Use 'long'/'short' strings or +1/-1 integers."
            )
        return is_long

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def pnl(self) -> NDArray[np.floating]:
        """Per-trade P&L array (n_trades,)."""
        arr: NDArray[np.floating] = self._trades["pnl"]
        return arr

    @property
    def n_trades(self) -> int:
        """Number of completed round-trip trades."""
        return len(self._trades["pnl"])

    @property
    def has_side(self) -> bool:
        """True if ``side`` information is available."""
        return "is_long" in self._trades

    @property
    def has_duration(self) -> bool:
        """True if trade duration is available."""
        return "duration" in self._trades

    @property
    def has_prices(self) -> bool:
        """True if fill/decision prices are available (implementation shortfall)."""
        return (
            "fill_price" in self._trades
            and "decision_price" in self._trades
        )

    @property
    def has_intratrade(self) -> bool:
        """True if intra-trade price paths are available (MFE/MAE)."""
        return "intratrade_prices" in self._trades

    # -- field accessors (return None if absent) -----------------------

    @property
    def is_long(self) -> NDArray[np.bool_] | None:
        """Boolean array: True for long trades, False for short."""
        return self._trades.get("is_long")

    @property
    def duration(self) -> NDArray[np.floating] | None:
        """Holding period duration array."""
        return self._trades.get("duration")

    @property
    def fill_price(self) -> NDArray[np.floating] | None:
        """Fill price array."""
        return self._trades.get("fill_price")

    @property
    def decision_price(self) -> NDArray[np.floating] | None:
        """Decision price array."""
        return self._trades.get("decision_price")

    @property
    def intratrade_prices(self) -> list[NDArray[np.floating]] | None:
        """List of intra-trade price arrays (one per trade)."""
        return self._trades.get("intratrade_prices")

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        extras = []
        if self.has_side:
            extras.append("side")
        if self.has_duration:
            extras.append("duration")
        if self.has_prices:
            extras.append("prices")
        if self.has_intratrade:
            extras.append("intratrade")
        fields = ", ".join(extras) if extras else "pnl-only"
        return (
            f"TradeInput(n_trades={self.n_trades}, "
            f"n_periods={self.n_periods}, "
            f"fields=[{fields}], "
            f"periods_per_year={self.periods_per_year})"
        )
