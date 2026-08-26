"""Input containers — validate and normalize user-provided data.

Each container accepts pandas.Series/DataFrame, polars.Series/DataFrame,
or numpy.ndarray, normalizes to numpy internally, and exposes what tiers
of metrics are computable given what was actually provided.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from stratstat.exceptions import MetricNotApplicableError


def _select_column(data: Any, name: str | None) -> Any:
    """Select column *name* from a DataFrame, exactly.

    Returns *data* unchanged when no name is given or the input is not a
    DataFrame, so callers can apply a schema unconditionally.

    Raises:
        KeyError: If the column is absent.  Selecting silently from the wrong
            column, or falling back to the whole frame, would produce a
            confident wrong number.
    """
    if name is None:
        return data

    for module, frame_attr in (("pandas", "DataFrame"), ("polars", "DataFrame")):
        try:
            mod = __import__(module)
        except ImportError:
            continue
        if isinstance(data, getattr(mod, frame_attr)):
            if name not in data.columns:
                raise KeyError(
                    f"Schema names column {name!r}, which is not in the data. "
                    f"Available columns: {list(data.columns)}."
                )
            return data[name]
    return data


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
            return cast("NDArray[np.floating]", data.to_numpy())
        if isinstance(data, pl.DataFrame):
            return cast("NDArray[np.floating]", data.to_numpy())

    raise TypeError(
        f"Unsupported input type: {type(data).__name__}. "
        f"Expected numpy.ndarray, pandas.Series/DataFrame, or polars.Series/DataFrame."
    )


def _column_labels(data: Any) -> list[str] | None:
    """Return the column labels of *data*, or None when it carries none.

    Pandas and polars DataFrames contribute their column names; a Series
    contributes its own name when it has one; numpy arrays have no labels and
    yield None.  This is how a DataFrame's column names survive the numpy
    conversion and reach the metric results, so per-strategy and per-asset
    outputs keep something to key on (issue I7).
    """
    if data is None or isinstance(data, np.ndarray):
        return None

    # pandas
    try:
        import pandas as pd
    except ImportError:
        pass
    else:
        if isinstance(data, pd.DataFrame):
            return [str(c) for c in data.columns]
        if isinstance(data, pd.Series):
            return [str(data.name)] if data.name is not None else None

    # polars
    try:
        import polars as pl
    except ImportError:
        pass
    else:
        if isinstance(data, pl.DataFrame):
            return [str(c) for c in data.columns]
        if isinstance(data, pl.Series):
            name: str | None = getattr(data, "name", None)
            return [str(name)] if name is not None else None

    return None


#: Annualization factor used when the caller does not provide one.  Daily bars
#: are the default assumption (issue I13); every container records where its
#: factor came from in ``ppy_source``.
_DEFAULT_PPY = 252


def _resolve_ppy(periods_per_year: int | None) -> tuple[int, str]:
    """Resolve the annualization factor, defaulting to 252.

    Returns ``(periods_per_year, ppy_source)`` where ``ppy_source`` is
    ``"default"`` when the 252 default applied and ``"user"`` when the
    caller supplied a value.
    """
    if periods_per_year is None:
        return _DEFAULT_PPY, "default"
    return periods_per_year, "user"


def deannualize_rf(rf: float, periods_per_year: int) -> float:
    """Convert an annual risk-free rate to its per-period equivalent.

    ``rf`` is an annual rate everywhere (resolved decision 6).  A metric that
    compares against period returns needs the per-period rate, obtained
    geometrically as ``(1 + rf) ** (1 / periods_per_year) - 1``.  This matches
    QuantStats' ``to_excess_returns`` and is more correct than simple division
    because a rate compounds.
    """
    return float((1.0 + rf) ** (1.0 / periods_per_year) - 1.0)


# Well-known engine column names that are not canonical but are common enough
# that seeing one in place of its canonical name should draw a hint rather than
# a silent drop.  Anything else that still looks close to a canonical name is
# caught by the fuzzy match in :func:`_warn_unmapped_trade_columns`.
_COMMON_TRADE_ALIASES: dict[str, str] = {
    "direction": "side",
    "profit": "pnl",
    "holding_periods": "duration",
}


def _warn_unmapped_trade_columns(raw: dict[str, Any]) -> None:
    """Warn about trade-log columns that would map under a canonical name.

    A column that is neither canonical nor mapped is dropped without a word,
    which is how a ``direction`` column in place of ``side`` silently costs the
    side-dependent metrics (issue I1).  Flag any such column that is a known
    alias or a close match to a canonical name, so the caller can add a
    ``columns=`` mapping rather than wondering why eleven metrics vanished.
    """
    from difflib import get_close_matches

    from stratstat.schema import CANONICAL_TRADE_COLUMNS

    canonical = CANONICAL_TRADE_COLUMNS
    hints: list[str] = []
    for col in raw:
        if col in canonical:
            continue
        target = _COMMON_TRADE_ALIASES.get(col)
        if target is None:
            matches = get_close_matches(col, sorted(canonical), n=1, cutoff=0.8)
            target = matches[0] if matches else None
        if target is not None:
            hints.append(f"{col!r} means {target!r} (add columns={{{target!r}: {col!r}}})")
    if hints:
        warnings.warn(
            "Unrecognized trade columns that look like a canonical column and "
            "will be ignored: " + "; ".join(hints) + ".",
            UserWarning,
            stacklevel=3,
        )


class ReturnsInput:
    """Wraps a returns series or matrix.

    Accepts a single strategy (1-D) or multiple strategies (2-D columns).
    All inputs are normalized to numpy arrays.
    """

    def __init__(
        self,
        data: Any,
        periods_per_year: int | None = None,
        schema: Any = None,
        columns: Any = None,
    ):
        from stratstat.schema import _coerce

        self.schema = _coerce(schema, columns, tier=None)
        if self.schema is not None:
            data = _select_column(data, self.schema.returns)
        self._raw = data
        self.labels: list[str] | None = _column_labels(data)
        arr = _to_numpy(data)

        # Ensure at least 1-D
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)  # single-column for uniform batch handling
        elif arr.ndim > 2:
            raise ValueError(f"Returns data must be 1-D or 2-D, got {arr.ndim}-D array.")

        self.values: NDArray[np.floating] = arr  # shape: (n_periods, n_strategies)
        self.n_periods: int = arr.shape[0]
        self.n_strategies: int = arr.shape[1]
        resolved_ppy, resolved_source = _resolve_ppy(periods_per_year)
        self.periods_per_year: int | None = resolved_ppy
        self.ppy_source: str = resolved_source

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

    * **Asset-level returns** (``asset_returns=``) — needed for long/short
      book-return metrics (§6.6–§6.7) and beta metrics (§6.8–§6.9).
    * **Benchmark returns** (``benchmark=``) — needed for long/short
      beta (§6.8–§6.9).
    * **Benchmark constituent weights** (``benchmark_weights=``) —
      needed for active share (§6.23).  Shape ``(n_assets,)`` for
      static weights or ``(n_periods, n_assets)`` for time-varying.
    * **Portfolio equity** (``equity=``) — needed for leverage (§6.3).
      If omitted, equity is derived in this order of precedence: from
      strategy *returns* as ``cumprod(1 + r)``, then from *asset_returns*
      and lagged *positions* as the portfolio's cumulative return.  The
      route taken is recorded in ``equity_source``.

    Parameters
    ----------
    positions: array-like
        Position weights of shape ``(n_periods, n_assets)``.
        Accepts numpy, pandas, or polars.
    returns: array-like, optional
        Strategy-level returns of shape ``(n_periods,)``.  Used only to
        derive the equity curve for leverage.
    asset_returns: array-like, optional
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
        asset_returns: Any | None = None,
        benchmark: Any | None = None,
        benchmark_weights: Any | None = None,
        equity: Any | None = None,
        periods_per_year: int | None = None,
        schema: Any = None,
        columns: Any = None,
    ) -> None:
        from stratstat.schema import _coerce

        self.schema = _coerce(schema, columns, tier=None)
        if self.schema is not None:
            positions = _select_column(positions, self.schema.positions)
            returns = _select_column(returns, self.schema.returns)
            asset_returns = _select_column(asset_returns, self.schema.asset_returns)
            benchmark = _select_column(benchmark, self.schema.benchmark)
            equity = _select_column(equity, self.schema.equity)
        self.labels: list[str] | None = _column_labels(positions)
        # -- positions (always required) ---------------------------------
        pos = _to_numpy(positions)
        if pos.ndim == 0:
            pos = pos.reshape(1, 1)
        elif pos.ndim == 1:
            pos = pos.reshape(-1, 1)
        elif pos.ndim > 2:
            raise ValueError(f"Positions must be 1-D or 2-D, got {pos.ndim}-D array.")
        self.positions: NDArray[np.floating] = pos  # (n_periods, n_assets)
        self.n_periods: int = pos.shape[0]
        self.n_assets: int = pos.shape[1]

        # -- asset-level returns (optional) ------------------------------
        if asset_returns is not None:
            ret = _to_numpy(asset_returns)
            if ret.ndim == 1:
                ret = ret.reshape(-1, 1)
            if ret.shape != (self.n_periods, self.n_assets):
                raise ValueError(
                    f"Returns shape {ret.shape} must match positions shape "
                    f"{(self.n_periods, self.n_assets)}."
                )
            self.asset_returns: NDArray[np.floating] | None = ret
        else:
            self.asset_returns = None

        # -- benchmark returns (optional) ---------------------------------
        if benchmark is not None:
            bench = _to_numpy(benchmark)
            bench = bench.ravel()
            if bench.shape[0] != self.n_periods:
                raise ValueError(
                    f"Benchmark length {bench.shape[0]} must match n_periods {self.n_periods}."
                )
            self.benchmark: NDArray[np.floating] | None = bench
        else:
            self.benchmark = None

        # -- benchmark constituent weights (optional) ---------------------
        if benchmark_weights is not None:
            bw = _to_numpy(benchmark_weights)
            if bw.ndim == 1:
                if bw.shape[0] != self.n_assets:
                    raise ValueError(
                        f"benchmark_weights length {bw.shape[0]} must match "
                        f"n_assets {self.n_assets}."
                    )
                self.benchmark_weights: NDArray[np.floating] | None = bw
            elif bw.ndim == 2:
                if bw.shape != (self.n_periods, self.n_assets):
                    raise ValueError(
                        f"benchmark_weights shape {bw.shape} must match "
                        f"(n_periods={self.n_periods}, n_assets={self.n_assets})."
                    )
                self.benchmark_weights = bw
            else:
                raise ValueError(f"benchmark_weights must be 1-D or 2-D, got {bw.ndim}-D array.")
        else:
            self.benchmark_weights = None

        # -- strategy returns (optional; used only to derive equity) -----
        if returns is not None:
            sret = _to_numpy(returns)
            if sret.ndim == 2:
                if sret.shape[1] != 1:
                    raise ValueError(
                        "ExposureInput needs a single strategy returns series to "
                        f"derive equity, got {sret.shape[1]} columns."
                    )
                sret = sret.ravel()
            elif sret.ndim > 2:
                raise ValueError("Strategy returns must be 1-D or 2-D.")
            sret = np.asarray(sret, dtype=np.float64).ravel()
            if sret.shape[0] != self.n_periods:
                raise ValueError(
                    f"Strategy returns length {sret.shape[0]} must match "
                    f"n_periods {self.n_periods}."
                )
            self.returns: NDArray[np.floating] | None = sret
        else:
            self.returns = None

        # -- equity (three-route precedence) -----------------------------
        # Route 1: user supplied equity.  Route 2: strategy returns via
        # cumprod(1 + r).  Route 3: positions + asset returns.  The route is
        # recorded in ``equity_source`` so the leverage metric can state where
        # its equity curve came from (issue I10).
        if equity is not None:
            eq = _to_numpy(equity).ravel()
            if eq.shape[0] != self.n_periods:
                raise ValueError(
                    f"Equity length {eq.shape[0]} must match n_periods {self.n_periods}."
                )
            self.equity: NDArray[np.floating] | None = eq
            self.equity_source: str | None = "user"
        elif self.returns is not None:
            self.equity = np.cumprod(1.0 + self.returns)
            self.equity_source = "strategy_returns"
        elif self.asset_returns is not None:
            # Compute portfolio returns from lagged weights and asset returns.
            # w_{i,t-1} for t >= 1; t=0 is NaN (unknown prior weights).
            w_lag = np.roll(self.positions, shift=1, axis=0)
            w_lag[0, :] = np.nan
            port_ret = np.nansum(w_lag * self.asset_returns, axis=1)  # (n_periods,)
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
            self.equity_source = "positions"
        else:
            self.equity = None
            self.equity_source = None

        resolved_ppy, resolved_source = _resolve_ppy(periods_per_year)
        self.periods_per_year: int | None = resolved_ppy
        self.ppy_source: str = resolved_source

    # -- convenience predicates ------------------------------------------
    @property
    def has_asset_returns(self) -> bool:
        """True if asset-level returns were provided."""
        return self.asset_returns is not None

    @property
    def has_benchmark(self) -> bool:
        """True if benchmark returns were provided."""
        return self.benchmark is not None

    @property
    def has_benchmark_weights(self) -> bool:
        """True if benchmark constituent weights were provided."""
        return self.benchmark_weights is not None

    @property
    def has_equity(self) -> bool:
        """True if portfolio equity is available."""
        return self.equity is not None

    def __repr__(self) -> str:
        return (
            f"ExposureInput(n_periods={self.n_periods}, "
            f"n_assets={self.n_assets}, "
            f"has_asset_returns={self.has_asset_returns}, "
            f"has_benchmark={self.has_benchmark}, "
            f"has_benchmark_weights={self.has_benchmark_weights}, "
            f"has_equity={self.has_equity}, "
            f"periods_per_year={self.periods_per_year})"
        )


def _is_datetime_like(values: Any) -> bool:
    """True if *values* hold timestamps or timedeltas rather than numbers.

    A datetime/timedelta column cast straight to float turns into nanoseconds
    (issue I2).  Detecting the dtype before the cast lets the trade boundary
    reject a datetime ``duration`` or keep entry/exit timestamps for bar
    derived excursions, instead of silently producing a confident wrong number.
    """
    arr = np.asarray(values)
    if arr.dtype.kind in ("M", "m"):
        return True
    if arr.dtype.kind == "O":
        for v in arr.ravel():
            if v is not None and not isinstance(v, (int, float, np.integer, np.floating, bool)):
                return True
    return False


def _coerce_duration_periods(values: Any) -> NDArray[np.floating]:
    """Coerce a ``duration`` column to a numeric count of periods.

    ``duration`` is defined in periods (resolved decision 3).  A timestamp
    cannot be turned into a period count without a bar calendar, so a
    datetime/timedelta column is rejected outright rather than cast to float
    and turned into nanoseconds (issue I2).
    """
    if _is_datetime_like(values):
        raise ValueError(
            "Trade 'duration' must be a numeric count of periods, not a "
            "datetime or timedelta. A timestamp cannot be converted to a "
            "period count without the bar calendar; compute the period count "
            "yourself (for daily bars, e.g. (exit_time - entry_time) in days) "
            "and pass it as 'duration'."
        )
    return np.asarray(values, dtype=np.float64).ravel()


def _normalize_prices(prices: Any) -> dict[str, Any] | None:
    """Normalize the bar ``prices`` input to time/close/high/low arrays.

    The ``prices`` input supplies bars for deriving excursions when the trade
    log carries ``entry_time``/``exit_time`` but no precomputed excursion or
    extreme price columns (section 3.3, priority 3).  ``close`` and a time
    axis are required; ``high`` and ``low`` are optional, and their absence
    means the derived extremes use close only (recorded as ``"close_only"``).

    Accepted forms: a dict with ``time`` and ``close`` keys (plus optional
    ``high``/``low``), or a pandas/polars DataFrame whose index is the time
    axis and whose columns are ``close``/``high``/``low``.
    """
    if prices is None:
        return None

    raw: dict[str, Any]
    time: Any = None
    if isinstance(prices, dict):
        raw = dict(prices)
        time = raw.get("time", raw.get("timestamp"))
    else:
        raw = {}
        for module, frame_attr in (("pandas", "DataFrame"), ("polars", "DataFrame")):
            try:
                mod = __import__(module)
            except ImportError:
                continue
            frame_cls = getattr(mod, frame_attr)
            if isinstance(prices, frame_cls):
                time = np.asarray(prices.index)
                raw = {col: np.asarray(prices[col]) for col in prices.columns}
                break
        if time is None:
            raise TypeError(
                f"Unsupported prices type: {type(prices).__name__}. "
                f"Expected a dict with 'time'/'close' keys or a DataFrame "
                f"with a time index and a 'close' column."
            )

    if time is None:
        raise ValueError("The 'prices' input requires a 'time' axis.")
    if not _is_datetime_like(time):
        raise ValueError(
            "The 'prices' time axis must be datetime, so entry/exit timestamps "
            "can be matched against it; got a non-timestamp axis."
        )
    if "close" not in raw:
        raise ValueError("The 'prices' input requires a 'close' column.")

    return {
        "time": np.asarray(time),
        "close": np.asarray(raw["close"], dtype=np.float64).ravel(),
        "high": np.asarray(raw["high"], dtype=np.float64).ravel() if "high" in raw else None,
        "low": np.asarray(raw["low"], dtype=np.float64).ravel() if "low" in raw else None,
    }


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
    * ``duration`` — holding period in periods.  Needed for
      holding-period metrics (§7.10, §7.11, §7.18, §7.19, §7.23,
      §7.34).  A datetime/timedelta value is rejected, never cast to
      float.  If absent but numeric ``entry_time`` and ``exit_time``
      are present, duration is computed from them.
    * ``entry_time``, ``exit_time`` — timestamps.  When datetime, kept
      for bar-derived excursions; when numeric, used to derive
      ``duration``.
    * ``fill_price``, ``decision_price`` — needed for implementation
      shortfall (§7.15); ``fill_price`` is also the entry price for
      ``max_price``/``min_price`` based excursions.
    * ``position_size`` — fraction of the account committed to each
      trade.  Converts between trade basis and account basis pnl.
    * ``max_price``, ``min_price`` — extreme price reached while the
      trade was open (side neutral).  Used for MFE/MAE.
    * ``mfe``, ``mae`` — precomputed favorable/adverse excursion, as a
      fraction.  Highest priority MFE/MAE source.
    * ``price_path`` — a sequence of per-trade price paths (list of
      arrays).  Lowest priority MFE/MAE source.  ``intratrade_prices``
      is accepted as an alias.

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
    pnl_basis: str, optional
        Capital base the ``pnl`` column is measured against:
        ``"trade"`` (default) or ``"account"``.
    pnl_unit: str, optional
        Unit of the ``pnl`` column: ``"fraction"`` (default) or
        ``"currency"``.  Gates the metrics that require a fraction.
    prices: dict or DataFrame, optional
        Bar data with a ``time`` axis and ``close`` (required) plus
        optional ``high``/``low`` columns, for deriving excursions.
    """

    def __init__(
        self,
        returns: Any = None,
        trades: Any = None,
        positions: Any | None = None,
        periods_per_year: int | None = None,
        schema: Any = None,
        columns: Any = None,
        pnl_basis: str = "trade",
        pnl_unit: str = "fraction",
        prices: Any = None,
    ) -> None:
        from stratstat.schema import _coerce

        resolved = _coerce(schema, columns, tier="trades")
        # -- returns (optional) -----------------------------------------
        if returns is not None:
            ret = _to_numpy(returns)
            if ret.ndim == 0:
                ret = ret.reshape(1)
            elif ret.ndim == 1:
                ret = ret.reshape(-1, 1)
            elif ret.ndim > 2:
                raise ValueError(f"Returns data must be 1-D or 2-D, got {ret.ndim}-D array.")
            self.returns: NDArray[np.floating] | None = ret
            self.n_periods: int = ret.shape[0]
        else:
            self.returns = None
            self.n_periods = 0

        # -- trade log conventions (B2, B3) ------------------------------
        if pnl_basis not in ("trade", "account"):
            raise ValueError(f"pnl_basis must be 'trade' or 'account', got {pnl_basis!r}")
        if pnl_unit not in ("fraction", "currency"):
            raise ValueError(f"pnl_unit must be 'fraction' or 'currency', got {pnl_unit!r}")
        self.pnl_basis: str = pnl_basis
        self.pnl_unit: str = pnl_unit

        # -- trades log -------------------------------------------------
        self._trades = self._normalize_trades(trades, schema=resolved)
        self.schema = resolved
        self.labels: list[str] | None = sorted(self._trades)

        # -- bar prices for excursion derivation (B7) --------------------
        self.prices: dict[str, Any] | None = _normalize_prices(prices)

        # -- positions (optional) ---------------------------------------
        if positions is not None:
            pos = _to_numpy(positions)
            if pos.ndim == 1:
                pos = pos.reshape(-1, 1)
            self.positions: NDArray[np.floating] | None = pos
        else:
            self.positions = None

        resolved_ppy, resolved_source = _resolve_ppy(periods_per_year)
        self.periods_per_year: int | None = resolved_ppy
        self.ppy_source: str = resolved_source

    # ------------------------------------------------------------------
    # Trade-log normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_trades(trades: Any, schema: Any = None) -> dict[str, Any]:
        """Normalize trade log to a dict of numpy arrays.

        Accepts dict, pandas.DataFrame, or polars.DataFrame.
        Validates that the required ``pnl`` field is present.
        Normalises ``side`` to a boolean ``is_long`` array.
        Computes ``duration`` from ``entry_time``/``exit_time`` if
        not provided directly.

        When *schema* is given, columns are rekeyed to their canonical names
        first, so validation downstream only ever sees canonical names.  This
        is the single boundary at which mapping happens for this tier.
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
                    raw = {col: trades[col].to_numpy() for col in trades.columns}
                    return TradeInput._validate_and_augment(TradeInput._apply_schema(raw, schema))

            # Try polars
            try:
                import polars as pl
            except ImportError:
                pass
            else:
                if isinstance(trades, pl.DataFrame):
                    raw = {col: trades[col].to_numpy() for col in trades.columns}
                    return TradeInput._validate_and_augment(TradeInput._apply_schema(raw, schema))

            raise TypeError(
                f"Unsupported trades type: {type(trades).__name__}. "
                f"Expected dict, pandas.DataFrame, or polars.DataFrame."
            )

        return TradeInput._validate_and_augment(TradeInput._apply_schema(raw, schema))

    @staticmethod
    def _apply_schema(raw: dict[str, Any], schema: Any) -> dict[str, Any]:
        """Rekey *raw* to canonical column names, if a schema was supplied."""
        if schema is None:
            return raw
        return dict(schema.apply_to_trades(raw))

    @staticmethod
    def _validate_and_augment(
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate required fields and normalize optional fields."""
        # A column that looks like a canonical name but is neither canonical
        # nor mapped would be dropped in silence (I1).  Warn first.
        _warn_unmapped_trade_columns(raw)

        # -- required: pnl ----------------------------------------------
        if "pnl" not in raw:
            raise ValueError(
                "Trade log must contain a 'pnl' column with per-trade profit/loss values."
            )
        pnl = np.asarray(raw["pnl"], dtype=np.float64).ravel()
        result: dict[str, Any] = {"pnl": pnl}
        n_trades = len(pnl)

        # -- optional: side -> is_long ---------------------------------
        if "side" in raw:
            result["is_long"] = TradeInput._normalize_side(raw["side"], n_trades)

        # -- optional: duration (or numeric entry_time/exit_time) -------
        if "duration" in raw:
            result["duration"] = _coerce_duration_periods(raw["duration"])
        elif "entry_time" in raw and "exit_time" in raw:
            entry = raw["entry_time"]
            exit_ = raw["exit_time"]
            if not _is_datetime_like(entry) and not _is_datetime_like(exit_):
                entry_n = np.asarray(entry, dtype=np.float64).ravel()
                exit_n = np.asarray(exit_, dtype=np.float64).ravel()
                result["duration"] = exit_n - entry_n

        # -- optional: entry_time/exit_time kept for bar slicing ---------
        # Datetime entry/exit times are stored (not cast to float) so the
        # bar-derived excursion route can index into the ``prices`` axis.
        for field in ("entry_time", "exit_time"):
            if field in raw and _is_datetime_like(raw[field]):
                result[field] = np.asarray(raw[field])

        # -- optional: fill_price, decision_price -----------------------
        for field in ("fill_price", "decision_price"):
            if field in raw:
                result[field] = np.asarray(raw[field], dtype=np.float64).ravel()

        # -- optional: position_size, max_price, min_price, mfe, mae ----
        for field in ("position_size", "max_price", "min_price", "mfe", "mae"):
            if field in raw:
                result[field] = np.asarray(raw[field], dtype=np.float64).ravel()

        # -- optional: price_path (alias intratrade_prices) -------------
        itp = raw.get("price_path")
        if itp is None:
            itp = raw.get("intratrade_prices")
        if itp is not None:
            if isinstance(itp, (list, tuple)):
                result["price_path"] = [np.asarray(p, dtype=np.float64).ravel() for p in itp]
            else:
                arr = np.asarray(itp, dtype=np.float64)
                if arr.ndim == 2:
                    # 2D array: each row is one trade's price path.
                    result["price_path"] = [arr[i] for i in range(arr.shape[0])]
                else:
                    # 1D array: a single intra-trade price path.
                    result["price_path"] = [arr.ravel()]

        return result

    @staticmethod
    def _normalize_side(side: Any, n_trades: int) -> NDArray[np.bool_]:
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
        return "fill_price" in self._trades and "decision_price" in self._trades

    @property
    def has_price_path(self) -> bool:
        """True if intra-trade price paths are available (MFE/MAE)."""
        return "price_path" in self._trades

    @property
    def has_intratrade(self) -> bool:
        """Backward-compatible alias for :attr:`has_price_path`."""
        return self.has_price_path

    @property
    def has_excursions(self) -> bool:
        """True if any excursion source is available (MFE/MAE)."""
        return (
            self.has_price_path
            or self.max_price is not None
            or self.mfe is not None
            or (self.prices is not None and self.entry_time is not None)
        )

    # -- field accessors (return None if absent) -----------------------

    @property
    def is_long(self) -> NDArray[np.bool_] | None:
        """Boolean array: True for long trades, False for short."""
        return self._trades.get("is_long")

    @property
    def duration(self) -> NDArray[np.floating] | None:
        """Holding period duration array (in periods)."""
        return self._trades.get("duration")

    @property
    def entry_time(self) -> NDArray[np.datetime64] | None:
        """Entry timestamps (kept only when datetime, for bar slicing)."""
        return self._trades.get("entry_time")

    @property
    def exit_time(self) -> NDArray[np.datetime64] | None:
        """Exit timestamps (kept only when datetime, for bar slicing)."""
        return self._trades.get("exit_time")

    @property
    def fill_price(self) -> NDArray[np.floating] | None:
        """Fill price array."""
        return self._trades.get("fill_price")

    @property
    def decision_price(self) -> NDArray[np.floating] | None:
        """Decision price array."""
        return self._trades.get("decision_price")

    @property
    def position_size(self) -> NDArray[np.floating] | None:
        """Fraction of account committed to each trade (n_trades,)."""
        return self._trades.get("position_size")

    @property
    def max_price(self) -> NDArray[np.floating] | None:
        """Highest price reached while the trade was open (n_trades,)."""
        return self._trades.get("max_price")

    @property
    def min_price(self) -> NDArray[np.floating] | None:
        """Lowest price reached while the trade was open (n_trades,)."""
        return self._trades.get("min_price")

    @property
    def mfe(self) -> NDArray[np.floating] | None:
        """Precomputed maximum favorable excursion, as a fraction."""
        return self._trades.get("mfe")

    @property
    def mae(self) -> NDArray[np.floating] | None:
        """Precomputed maximum adverse excursion, as a fraction."""
        return self._trades.get("mae")

    @property
    def price_path(self) -> list[NDArray[np.floating]] | None:
        """List of intra-trade price arrays (one per trade)."""
        return self._trades.get("price_path")

    @property
    def intratrade_prices(self) -> list[NDArray[np.floating]] | None:
        """Backward-compatible alias for :attr:`price_path`."""
        return self.price_path

    # ------------------------------------------------------------------
    # Basis conversion (B2)
    # ------------------------------------------------------------------

    def pnl_account_basis(self) -> tuple[NDArray[np.floating], bool]:
        """P&L on account basis, converting from trade basis when needed.

        Returns ``(pnl, converted)``.  When ``pnl_basis`` is ``"account"`` the
        column is already account basis and is returned unchanged.  When it is
        ``"trade"`` and a ``position_size`` column is present, each trade's pnl
        is multiplied by its size; without a size column the bases coincide
        (size defaults to 1.0) and ``converted`` is ``False``.
        """
        pnl = self.pnl
        if self.pnl_basis == "account":
            return pnl, False
        size = self.position_size
        if size is None:
            return pnl, False
        return pnl * size, True

    def pnl_trade_basis(self) -> tuple[NDArray[np.floating], bool]:
        """P&L on trade basis, converting from account basis when needed.

        Returns ``(pnl, converted)``.  When ``pnl_basis`` is ``"trade"`` the
        column is already trade basis and is returned unchanged.  When it is
        ``"account"`` and a ``position_size`` column is present, each trade's
        pnl is divided by its size; without a size column the bases coincide
        (size defaults to 1.0) and ``converted`` is ``False``.
        """
        pnl = self.pnl
        if self.pnl_basis == "trade":
            return pnl, False
        size = self.position_size
        if size is None:
            return pnl, False
        return pnl / size, True

    # ------------------------------------------------------------------
    # Excursion derivation (B5, B6, B7, B8, B9)
    # ------------------------------------------------------------------

    def excursions(self) -> tuple[NDArray[np.floating], NDArray[np.floating], str]:
        """Per-trade favorable and adverse excursion, as fractions of entry.

        Returns ``(mfe, mae, source)`` where each of ``mfe``/``mae`` has length
        ``n_trades`` (NaN where a trade cannot be resolved) and ``source`` names
        the route that produced them, per the precedence in section 3.3:
        ``"mfe_mae"`` (precomputed columns), ``"max_min_price"`` (extreme price
        columns), ``"prices"`` or ``"prices_close_only"`` (sliced bars), or
        ``"price_path"`` (intra-trade path).
        """
        n = self.n_trades
        is_long = self.is_long
        if is_long is None:
            raise MetricNotApplicableError(
                "MFE/MAE require 'side' in the trade log to know which "
                "direction is favorable."
            )

        # 1. Precomputed excursion fractions (highest priority).
        if self.mfe is not None and self.mae is not None:
            return self.mfe.copy(), self.mae.copy(), "mfe_mae"

        # 2. Extreme price columns, relative to the fill (entry) price.
        if (
            self.max_price is not None
            and self.min_price is not None
            and self.fill_price is not None
        ):
            entry = self.fill_price
            mfe: NDArray[np.floating] = np.full(n, np.nan)
            mae: NDArray[np.floating] = np.full(n, np.nan)
            for j in range(n):
                e = entry[j]
                if not np.isfinite(e) or e == 0.0:
                    continue
                if is_long[j]:
                    mfe[j] = (self.max_price[j] - e) / e
                    mae[j] = (e - self.min_price[j]) / e
                else:
                    mfe[j] = (e - self.min_price[j]) / e
                    mae[j] = (self.max_price[j] - e) / e
            return mfe, mae, "max_min_price"

        # 3. Slice bars between entry and exit.
        if self.prices is not None and self.entry_time is not None and self.exit_time is not None:
            close_only = self.prices["high"] is None or self.prices["low"] is None
            if close_only:
                warnings.warn(
                    "The 'prices' input has no 'high' or 'low' column, so MFE/MAE "
                    "are derived from close prices only and understate the true "
                    "range. Supply high and low bars to capture the full excursion.",
                    UserWarning,
                    stacklevel=2,
                )
            mfe, mae = self._derive_from_bars(close_only)
            return mfe, mae, "prices_close_only" if close_only else "prices"

        # 4. Intra-trade price path (lowest priority).
        paths = self.price_path
        if paths is not None:
            mfe = np.full(n, np.nan)
            mae = np.full(n, np.nan)
            for j in range(n):
                if j >= len(paths):
                    break
                path = paths[j]
                if len(path) < 2 or not np.isfinite(path[0]):
                    continue
                entry = path[0]
                if is_long[j]:
                    mfe[j] = (np.nanmax(path) - entry) / entry
                    mae[j] = (entry - np.nanmin(path)) / entry
                else:
                    mfe[j] = (entry - np.nanmin(path)) / entry
                    mae[j] = (np.nanmax(path) - entry) / entry
            return mfe, mae, "price_path"

        raise MetricNotApplicableError(
            "MFE/MAE require one of: 'mfe'/'mae' columns, 'max_price'/'min_price' "
            "with 'fill_price', a 'prices' input with 'entry_time'/'exit_time', "
            "or a 'price_path' column."
        )

    def _derive_from_bars(
        self, close_only: bool,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Derive per-trade excursions by slicing the ``prices`` bars."""
        n = self.n_trades
        is_long = self.is_long
        assert is_long is not None
        assert self.prices is not None
        assert self.entry_time is not None and self.exit_time is not None

        times = self.prices["time"]
        close = self.prices["close"]
        high = self.prices["high"]
        low = self.prices["low"]

        mfe = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        entry = self.entry_time
        exit_ = self.exit_time
        for j in range(n):
            t0, t1 = entry[j], exit_[j]
            i0 = int(np.searchsorted(times, t0, side="left"))
            i1 = int(np.searchsorted(times, t1, side="right"))
            if i1 <= i0:
                continue
            seg_close = close[i0:i1]
            if seg_close.size == 0 or not np.isfinite(seg_close[0]):
                continue
            ref = seg_close[0]  # entry bar close
            if high is not None and low is not None:
                hi = np.nanmax(high[i0:i1])
                lo = np.nanmin(low[i0:i1])
            else:
                hi = np.nanmax(seg_close)
                lo = np.nanmin(seg_close)
            if is_long[j]:
                mfe[j] = (hi - ref) / ref
                mae[j] = (ref - lo) / ref
            else:
                mfe[j] = (ref - lo) / ref
                mae[j] = (hi - ref) / ref
        return mfe, mae

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        extras = []
        if self.has_side:
            extras.append("side")
        if self.has_duration:
            extras.append("duration")
        if self.has_prices:
            extras.append("prices")
        if self.has_price_path:
            extras.append("price_path")
        if self.position_size is not None:
            extras.append("position_size")
        if self.has_excursions:
            extras.append("excursions")
        fields = ", ".join(extras) if extras else "pnl-only"
        return (
            f"TradeInput(n_trades={self.n_trades}, "
            f"n_periods={self.n_periods}, "
            f"fields=[{fields}], "
            f"periods_per_year={self.periods_per_year})"
        )


class BenchmarkInput:
    """Wraps strategy returns + benchmark returns for benchmark-tier metrics.

    Parameters
    ----------
    returns: array-like
        Strategy returns of shape ``(n_periods,)`` or
        ``(n_periods, n_strategies)``.
    benchmark: array-like
        Benchmark returns of shape ``(n_periods,)``.
    periods_per_year: int, optional
        Annualization factor (e.g. 252 for daily).  Required by
        most benchmark metrics for annualization.
    rf: float, optional
        Annual risk-free rate (default 0.0).  Used by alpha
        (§8.1) and Treynor ratio (§8.12).
    """

    def __init__(
        self,
        returns: Any,
        benchmark: Any | None = None,
        periods_per_year: int | None = None,
        rf: float = 0.0,
        schema: Any = None,
        columns: Any = None,
    ) -> None:
        from stratstat.schema import _coerce

        self.schema = _coerce(schema, columns, tier=None)
        # Support tuple shortcut: BenchmarkInput((returns, benchmark))
        if benchmark is None and isinstance(returns, (tuple, list)):
            if len(returns) == 2:
                returns, benchmark = returns[0], returns[1]
            else:
                raise ValueError(
                    "Expected (returns, benchmark) tuple for "
                    f"BenchmarkInput, got sequence of length {len(returns)}."
                )

        if benchmark is None:
            raise ValueError(
                "BenchmarkInput requires benchmark returns. "
                "Provide benchmark= to BenchmarkInput, or pass a "
                "(returns, benchmark) tuple."
            )

        # Applied after the tuple shortcut, so both forms map identically.
        if self.schema is not None:
            returns = _select_column(returns, self.schema.returns)
            benchmark = _select_column(benchmark, self.schema.benchmark)
        self.labels: list[str] | None = _column_labels(returns)

        # -- strategy returns ---------------------------------------------
        ret = _to_numpy(returns)
        if ret.ndim == 0:
            ret = ret.reshape(1)
        elif ret.ndim == 1:
            ret = ret.reshape(-1, 1)
        elif ret.ndim > 2:
            raise ValueError(f"Returns data must be 1-D or 2-D, got {ret.ndim}-D array.")
        self.returns: NDArray[np.floating] = ret  # (n_periods, n_strategies)
        self.n_periods: int = ret.shape[0]
        self.n_strategies: int = ret.shape[1]

        # -- benchmark returns --------------------------------------------
        bench = _to_numpy(benchmark).ravel()
        if bench.shape[0] != self.n_periods:
            raise ValueError(
                f"Benchmark length {bench.shape[0]} must match n_periods {self.n_periods}."
            )
        self.benchmark: NDArray[np.floating] = bench  # (n_periods,)

        resolved_ppy, resolved_source = _resolve_ppy(periods_per_year)
        self.periods_per_year: int | None = resolved_ppy
        self.ppy_source: str = resolved_source
        self.rf: float = rf  # annual
        self.rf_period: float = deannualize_rf(rf, resolved_ppy)

    @property
    def is_single(self) -> bool:
        """True if the input represents a single strategy."""
        return self.n_strategies == 1

    def __repr__(self) -> str:
        return (
            f"BenchmarkInput(n_periods={self.n_periods}, "
            f"n_strategies={self.n_strategies}, "
            f"periods_per_year={self.periods_per_year}, "
            f"rf={self.rf})"
        )


class CompareInput:
    """Wraps multiple strategy return series for compare-tier metrics.

    Compare metrics operate on two or more strategies simultaneously:
    correlation, diversification, pairwise Sharpe-difference tests,
    White's Reality Check, PBO, marginal risk contributions, and
    component VaR.

    Parameters
    ----------
    returns: array-like
        Strategy returns of shape ``(n_periods, n_strategies)``.
        Must have at least 2 strategy columns.
    weights: array-like, optional
        Strategy weights of shape ``(n_strategies,)``.  Defaults to
        equal weight when needed by a metric (diversification ratio,
        MCR, component VaR).
    benchmark: array-like, optional
        Benchmark returns of shape ``(n_periods,)``.  Required only by
        White's Reality Check (§9.4).
    periods_per_year: int, optional
        Annualization factor (e.g. 252 for daily).  Required by metrics
        that annualise Sharpe ratios (JK test, PBO).
    rf: float, optional
        Annual risk-free rate (default 0.0).  Used in Sharpe-ratio
        computations inside JK test and PBO.
    """

    def __init__(
        self,
        returns: Any,
        weights: Any | None = None,
        benchmark: Any | None = None,
        periods_per_year: int | None = None,
        rf: float = 0.0,
        schema: Any = None,
        columns: Any = None,
    ) -> None:
        from stratstat.schema import _coerce

        self.schema = _coerce(schema, columns, tier=None)
        if self.schema is not None:
            benchmark = _select_column(benchmark, self.schema.benchmark)
        self.labels: list[str] | None = _column_labels(returns)
        # -- strategy returns -----------------------------------------------
        ret = _to_numpy(returns)
        if ret.ndim == 0:
            ret = ret.reshape(1, 1)
        elif ret.ndim == 1:
            ret = ret.reshape(-1, 1)
        elif ret.ndim > 2:
            raise ValueError(f"Returns data must be 1-D or 2-D, got {ret.ndim}-D array.")
        self.returns: NDArray[np.floating] = ret  # (n_periods, n_strategies)
        self.n_periods: int = ret.shape[0]
        self.n_strategies: int = ret.shape[1]

        # -- weights (optional, defaults to equal weight) -------------------
        if weights is not None:
            w = _to_numpy(weights).ravel()
            if w.shape[0] != self.n_strategies:
                raise ValueError(
                    f"Weights length {w.shape[0]} must match n_strategies {self.n_strategies}."
                )
            self.weights: NDArray[np.floating] | None = w
        else:
            self.weights = None

        # -- benchmark returns (optional) -----------------------------------
        if benchmark is not None:
            bench = _to_numpy(benchmark).ravel()
            if bench.shape[0] != self.n_periods:
                raise ValueError(
                    f"Benchmark length {bench.shape[0]} must match n_periods {self.n_periods}."
                )
            self.benchmark: NDArray[np.floating] | None = bench
        else:
            self.benchmark = None

        resolved_ppy, resolved_source = _resolve_ppy(periods_per_year)
        self.periods_per_year: int | None = resolved_ppy
        self.ppy_source: str = resolved_source
        self.rf: float = rf  # annual
        self.rf_period: float = deannualize_rf(rf, resolved_ppy)

    # -- convenience predicates ------------------------------------------

    @property
    def has_weights(self) -> bool:
        """True if explicit strategy weights were provided."""
        return self.weights is not None

    @property
    def has_benchmark(self) -> bool:
        """True if benchmark returns were provided."""
        return self.benchmark is not None

    def get_weights(self) -> NDArray[np.floating]:
        """Return strategy weights, defaulting to equal weight."""
        if self.weights is not None:
            return self.weights
        return np.full(self.n_strategies, 1.0 / self.n_strategies, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"CompareInput(n_periods={self.n_periods}, "
            f"n_strategies={self.n_strategies}, "
            f"periods_per_year={self.periods_per_year}, "
            f"rf={self.rf}, "
            f"has_benchmark={self.has_benchmark})"
        )
