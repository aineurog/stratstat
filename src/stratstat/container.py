"""Strategy and Comparison containers.

A :class:`Strategy` holds one strategy's inputs once and caches the derived
quantities that several metrics would otherwise recompute: the equity curve,
the running maximum, the drawdown series, and the drawdown episodes.  A
:class:`Comparison` stacks several strategies' period aligned returns into a
single ``(n_periods, n_strategies)`` matrix and runs the vectorized engine
once, then slices per strategy.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from stratstat.inputs import CompareInput, ExposureInput, ReturnsInput, TradeInput
from stratstat.results import MetricSet
from stratstat.schema import _coerce

# (equity, running_max, drawdown_series, episodes) produced by the risk module.
_Drawdowns = tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    list[list[dict[str, Any]]],
]


def _read_position_size(trades: Any) -> NDArray[np.floating] | None:
    """Read an optional ``position_size`` column from a raw trade log.

    ``position_size`` is not yet a formal contract column (it lands with
    Package B); read it opportunistically here so the reconciliation uses
    account basis whenever the caller happens to supply it, and falls back to
    full account (1.0) otherwise.
    """
    if isinstance(trades, dict):
        col = trades.get("position_size")
        if col is None:
            return None
        return np.asarray(col, dtype=np.float64).ravel()

    columns = getattr(trades, "columns", None)
    if columns is not None and "position_size" in columns:
        return np.asarray(trades["position_size"], dtype=np.float64).ravel()
    return None


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class Strategy:
    """One strategy: inputs built once, derived quantities cached once.

    Parameters mirror the container input contract.  ``returns``, ``equity``,
    ``trades``, ``positions``, ``asset_returns``, ``prices``, ``benchmark``,
    ``periods_per_year``, ``rf``, and ``schema`` are all optional.  Each
    provided input is turned into its :mod:`stratstat.inputs` container once
    at construction and reused by every ``compute`` call.  The equity curve,
    running maximum, drawdown series, and drawdown episodes are derived once
    from the strategy returns and cached, so a report or a later computation
    never redoes that work.

    When both ``returns`` and ``trades`` are present, construction also runs a
    reconciliation check: each trade's pnl is converted to account basis with
    ``position_size`` (full account when absent), compounded, and compared with
    the equity curve's total return.  A disagreement raises a warning and the
    figures are recorded on :attr:`reconciliation`.

    The container is treated as immutable after construction; its inputs and
    cached quantities are not reassigned.
    """

    def __init__(
        self,
        returns: Any = None,
        *,
        equity: Any = None,
        trades: Any = None,
        positions: Any = None,
        asset_returns: Any = None,
        prices: Any = None,
        benchmark: Any = None,
        periods_per_year: int | None = None,
        rf: float = 0.0,
        schema: Any = None,
        columns: Any = None,
        pnl_basis: str = "trade",
        pnl_unit: str = "fraction",
    ) -> None:
        self.schema = _coerce(schema, columns, tier=None)
        self.periods_per_year: int | None = periods_per_year
        self.rf: float = rf

        # Raw inputs, kept so report() and the benchmark tier can rebuild
        # exactly what was handed in.
        self.returns: Any = returns
        self.trades: Any = trades
        self.positions: Any = positions
        self.asset_returns: Any = asset_returns
        self.equity: Any = equity
        self.benchmark: Any = benchmark
        self.prices: Any = prices

        # Input containers, built once.
        self.returns_input: ReturnsInput | None = (
            ReturnsInput(returns, periods_per_year=periods_per_year, schema=self.schema)
            if returns is not None
            else None
        )
        self.trades_input: TradeInput | None = (
            TradeInput(
                trades=trades,
                periods_per_year=periods_per_year,
                schema=self.schema,
                pnl_basis=pnl_basis,
                pnl_unit=pnl_unit,
                prices=prices,
            )
            if trades is not None
            else None
        )
        self.exposure_input: ExposureInput | None = (
            ExposureInput(
                positions,
                returns=returns,
                asset_returns=asset_returns,
                benchmark=benchmark,
                equity=equity,
                periods_per_year=periods_per_year,
                schema=self.schema,
            )
            if positions is not None
            else None
        )

        self._drawdowns: _Drawdowns | None = None

        # Reconciliation of the trade log against the equity curve, run only
        # when both inputs are present.  Recorded even when the figures agree
        # so the check is observable.
        self.reconciliation: dict[str, Any] | None = (
            self._reconcile_trades()
            if returns is not None and trades is not None
            else None
        )

    def _reconcile_trades(self) -> dict[str, Any] | None:
        """Reconcile the trade log against the equity curve.

        With both strategy returns and a trade log present, compounding each
        trade's account basis pnl should reproduce the equity curve's total
        return.  Returns a small dict recording the two figures and whether
        ``position_size`` was applied, so the check is observable even when the
        numbers agree; warns when they do not.
        """
        assert self.returns_input is not None and self.trades_input is not None

        returns = self.returns_input.values
        if returns.shape[1] != 1:
            return None  # no single equity curve to reconcile a matrix against

        strat_returns = returns[:, 0]
        pnl = np.asarray(self.trades_input.pnl, dtype=np.float64)
        if np.any(np.isnan(strat_returns)) or np.any(np.isnan(pnl)):
            return None

        size = _read_position_size(self.trades)
        if size is not None and size.shape[0] == pnl.shape[0]:
            converted = True
        else:
            size = np.ones_like(pnl)
            converted = False
        account_pnl = pnl * size
        if np.any(account_pnl <= -1.0):
            return None

        compounded_pnl = float(np.prod(1.0 + account_pnl) - 1.0)
        equity_total = float(np.prod(1.0 + strat_returns) - 1.0)

        info: dict[str, Any] = {
            "compounded_account_pnl": compounded_pnl,
            "equity_total_return": equity_total,
            "converted_to_account_basis": converted,
        }

        close = bool(np.isclose(compounded_pnl, equity_total, rtol=1e-2, atol=1e-4))
        if not close:
            warnings.warn(
                "Trade pnl does not reconcile with the equity curve: compounded "
                f"account pnl is {compounded_pnl:.2%} but strategy returns "
                f"compound to {equity_total:.2%}. This usually means the trade "
                "log omits trades, or the returns include components outside "
                "the trades such as fees, dividends, or cash drag.",
                UserWarning,
                stacklevel=2,
            )

        return info

    # -- derived quantities (cached) -------------------------------------

    def _ensure_drawdowns(self) -> _Drawdowns:
        """Compute the cached drawdown analysis once."""
        if self._drawdowns is None:
            if self.returns_input is None:
                raise ValueError("drawdown quantities require strategy returns")
            from stratstat.core.returns.risk import _analyse_drawdowns

            self._drawdowns = cast(_Drawdowns, _analyse_drawdowns(self.returns_input.values))
        return self._drawdowns

    @property
    def equity_curve(self) -> NDArray[np.floating]:
        """Equity curve derived from returns, cached.

        Shape ``(n_periods + 1, n_strategies)`` with a prepended 1.0 row so
        the drawdown from start is measured.
        """
        return self._ensure_drawdowns()[0]

    @property
    def running_max(self) -> NDArray[np.floating]:
        """Running maximum of the equity curve, cached."""
        return self._ensure_drawdowns()[1]

    @property
    def drawdown_series(self) -> NDArray[np.floating]:
        """Drawdown series (values at or below zero), cached."""
        return self._ensure_drawdowns()[2]

    @property
    def drawdown_episodes(self) -> list[list[dict[str, Any]]]:
        """Drawdown episodes per strategy column, cached."""
        return self._ensure_drawdowns()[3]

    # -- computation ------------------------------------------------------

    def compute_all(self, **kwargs: Any) -> MetricSet:
        """Compute every metric for which this strategy has data.

        Forwards to :func:`stratstat.compute_all` with the containers built at
        construction, so inputs are never rebuilt.  Keyword arguments pass
        through to the metric functions (for example ``confidence=0.95``).
        """
        from stratstat.registry import _compute_all

        return _compute_all(
            returns=self.returns_input,
            trades=self.trades_input,
            benchmark=self.benchmark,
            exposure=self.exposure_input,
            periods_per_year=self.periods_per_year,
            rf=self.rf,
            schema=self.schema,
            **kwargs,
        )

    def compute(self, category: str, **kwargs: Any) -> MetricSet:
        """Compute metrics in one primary statistical category.

        *category* is a primary tag such as ``"risk"``, ``"descriptive"``, or
        ``"trades"``.  Keyword arguments pass through to the metric functions.
        """
        return self.compute_all(category=category, **kwargs)

    def report(
        self,
        output_path: str | Path,
        *,
        metrics: MetricSet | None = None,
        **kwargs: Any,
    ) -> None:
        """Generate an HTML (or PDF) report for this strategy.

        Requires ``plotly``.  *metrics* lets you hand in a precomputed
        :class:`MetricSet`; otherwise one is computed with :meth:`compute_all`.
        Keyword arguments pass through to
        :func:`stratstat.report.generate_report`.
        """
        from stratstat.report import generate_report

        if self.returns is None:
            raise ValueError("Strategy.report() requires strategy returns")
        ms = metrics if metrics is not None else self.compute_all()
        generate_report(
            self.returns,
            output_path,
            benchmark=self.benchmark,
            positions=self.positions,
            asset_returns=self.asset_returns,
            trades=self.trades,
            periods_per_year=self.periods_per_year,
            metrics=ms,
            **kwargs,
        )

    def __repr__(self) -> str:
        n = self.returns_input.n_periods if self.returns_input is not None else 0
        return f"Strategy(n_periods={n}, periods_per_year={self.periods_per_year})"


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


class Comparison:
    """Several strategies, stacked and run through the vectorized engine once.

    *strategies* is either a ``{name: returns}`` mapping or a 2-D array-like
    whose columns are strategies.  Every series must be period aligned (the
    same length).  The container stacks them into one
    ``(n_periods, n_strategies)`` matrix, builds a :class:`CompareInput` and a
    :class:`ReturnsInput` once, and runs the returns, benchmark, and compare
    tiers in a single vectorized pass.  Use :attr:`strategies` or
    ``comparison[name]`` to slice one strategy out for per-strategy work.

    The container is treated as immutable after construction.
    """

    def __init__(
        self,
        strategies: Any,
        *,
        weights: Any = None,
        benchmark: Any = None,
        periods_per_year: int | None = None,
        rf: float = 0.0,
        schema: Any = None,
        columns: Any = None,
    ) -> None:
        from stratstat.inputs import _column_labels, _to_numpy

        self.schema = _coerce(schema, columns, tier=None)
        self.periods_per_year: int | None = periods_per_year
        self.rf: float = rf
        self.benchmark: Any = benchmark
        self.weights: Any = weights

        matrix: NDArray[np.floating]
        if isinstance(strategies, dict):
            self.names = [str(k) for k in strategies]
            cols = [np.asarray(v, dtype=np.float64).ravel() for v in strategies.values()]
            if not cols:
                raise ValueError("Comparison requires at least one strategy")
            n_periods = cols[0].shape[0]
            for col in cols[1:]:
                if col.shape[0] != n_periods:
                    raise ValueError(
                        "all strategy return series must be period aligned (the same length)"
                    )
            matrix = np.column_stack(cols) if len(cols) > 1 else cols[0].reshape(-1, 1)
        else:
            labels = _column_labels(strategies)
            matrix = _to_numpy(strategies)
            if matrix.ndim == 1:
                matrix = matrix.reshape(-1, 1)
            self.names = (
                [str(x) for x in labels]
                if labels is not None and len(labels) == matrix.shape[1]
                else [f"s{i}" for i in range(matrix.shape[1])]
            )

        self.returns: NDArray[np.floating] = matrix  # (n_periods, n_strategies)
        self.compare_input = CompareInput(
            matrix,
            weights=weights,
            benchmark=benchmark,
            periods_per_year=periods_per_year,
            rf=rf,
            schema=self.schema,
        )
        self.returns_input = ReturnsInput(
            matrix, periods_per_year=periods_per_year, schema=self.schema
        )

    # -- slicing ----------------------------------------------------------

    @property
    def labels(self) -> list[str]:
        """Strategy names, in column order."""
        return list(self.names)

    @property
    def strategies(self) -> dict[str, Strategy]:
        """Slice the stacked matrix into one :class:`Strategy` per column.

        Each :class:`Strategy` carries that column's returns plus the shared
        benchmark, so ``comparison[name].compute_all()`` yields that
        strategy's returns and benchmark metrics without redoing the
        cross-strategy work.
        """
        return {
            name: Strategy(
                self.returns[:, i],
                benchmark=self.benchmark,
                periods_per_year=self.periods_per_year,
                rf=self.rf,
                schema=self.schema,
            )
            for i, name in enumerate(self.names)
        }

    def __getitem__(self, key: int | str) -> Strategy:
        if isinstance(key, str):
            if key not in self.names:
                raise KeyError(key)
            i = self.names.index(key)
        else:
            i = key
        return Strategy(
            self.returns[:, i],
            benchmark=self.benchmark,
            periods_per_year=self.periods_per_year,
            rf=self.rf,
            schema=self.schema,
        )

    def __len__(self) -> int:
        return self.returns.shape[1]

    def __iter__(self) -> Any:
        return iter(self.strategies)

    # -- computation ------------------------------------------------------

    def compute_all(self, **kwargs: Any) -> MetricSet:
        """Run the returns, benchmark, and compare tiers once on the stack.

        Returns-tier metrics are computed vectorized across all strategies and
        come back as per-strategy array values.  Keyword arguments pass through
        to the metric functions.
        """
        from stratstat.registry import _compute_all

        return _compute_all(
            returns=self.returns_input,
            benchmark=self.benchmark,
            compare=self.compare_input,
            periods_per_year=self.periods_per_year,
            rf=self.rf,
            schema=self.schema,
            **kwargs,
        )

    def compute(self, category: str, **kwargs: Any) -> MetricSet:
        """Compute metrics in one primary statistical category.

        *category* is a primary tag such as ``"risk"`` or ``"relative"``.
        Keyword arguments pass through to the metric functions.
        """
        return self.compute_all(category=category, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Comparison(n_strategies={self.returns.shape[1]}, "
            f"n_periods={self.returns.shape[0]}, "
            f"names={self.names})"
        )
