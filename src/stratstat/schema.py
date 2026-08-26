"""Column mapping: say once what your columns are called.

StratStat expects canonical column names.  Real data rarely uses them, and
until now an unrecognised column was skipped in silence, so a trade log with
``direction`` instead of ``side`` quietly lost eleven metrics with nothing
reported at call time.

A :class:`Schema` states the mapping explicitly::

    schema = ss.Schema(trades={"side": "direction", "pnl": "profit"})
    ss.compute_trades(df, schema=schema)

The key is always the canonical name and the value is the column in your data,
so the mapping reads in the direction you would say it out loud: "side is
called direction here".

Matching is exact.  Nothing is inferred, case folded, or fuzzy matched.  A
column already carrying its canonical name needs no entry.

Set it once for a session with :func:`set_schema`, or per call with
``schema=`` or the inline ``columns=`` shorthand.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

# Canonical trade log columns, per the data contract.
#
# ``price_path`` is the contract name; ``intratrade_prices`` is the pre-rename
# internal name and is accepted as an alias for one release so existing callers
# do not break silently.
CANONICAL_TRADE_COLUMNS: frozenset[str] = frozenset(
    {
        "pnl",
        "side",
        "duration",
        "entry_time",
        "exit_time",
        "fill_price",
        "decision_price",
        "price_path",
        "intratrade_prices",
        "position_size",
        "max_price",
        "min_price",
        "mfe",
        "mae",
        "asset",
    }
)

# Legacy name -> contract name.
_TRADE_ALIASES: dict[str, str] = {"intratrade_prices": "price_path"}

# Which trade metrics each canonical column unlocks.  Mirrors the
# ``_require_field`` calls in ``core/trades.py``: when a column is absent the
# metrics listed here become unavailable.  ``pnl`` is handled specially because
# every trade metric needs it (described from the registry at call time, so the
# list cannot drift).  ``entry_time``/``exit_time`` do not appear because they
# unlock ``duration`` by derivation, handled separately.
_TRADE_COLUMN_METRICS: dict[str, tuple[str, ...]] = {
    "side": (
        "win_rate_long",
        "win_rate_short",
        "implementation_shortfall",
        "mfe",
        "mae",
        "long_short_trade_count",
        "long_short_trade_pct",
        "long_short_winning_losing",
        "long_short_avg_duration",
        "long_short_total_pnl",
        "long_short_avg_pnl",
        "long_short_best_worst",
    ),
    "duration": (
        "avg_holding_period",
        "holding_period_distribution",
        "avg_winning_duration",
        "avg_losing_duration",
        "trade_duration_std",
        "long_short_avg_duration",
    ),
    "fill_price": ("implementation_shortfall", "mfe", "mae"),
    "decision_price": ("implementation_shortfall",),
    "price_path": ("mfe", "mae"),
    "intratrade_prices": ("mfe", "mae"),
    "max_price": ("mfe", "mae"),
    "min_price": ("mfe", "mae"),
    "mfe": ("mfe",),
    "mae": ("mae",),
}

# A canonical column derivable from another pair of canonical columns.
_TRADE_DERIVED: dict[str, tuple[str, str]] = {"duration": ("entry_time", "exit_time")}


@dataclass(frozen=True)
class Schema:
    """A mapping from canonical names to the names your data actually uses.

    Every field is optional.  The single series fields name a column to select
    when the corresponding input is a DataFrame; ``trades`` maps canonical
    trade log columns to their names in your trade log.

    Args:
        returns: Column holding strategy returns.
        equity: Column holding the equity curve.
        benchmark: Column holding benchmark returns.
        positions: Column holding position weights.
        asset_returns: Column holding per asset returns.
        prices: Column holding price bars.
        trades: Canonical trade column -> column name in your trade log.

    Raises:
        ValueError: If ``trades`` names a canonical column that does not exist.
            A typo there would otherwise map nothing and be indistinguishable
            from not having asked.
    """

    returns: str | None = None
    equity: str | None = None
    benchmark: str | None = None
    positions: str | None = None
    asset_returns: str | None = None
    prices: str | None = None
    trades: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.trades) - CANONICAL_TRADE_COLUMNS)
        if unknown:
            from difflib import get_close_matches

            parts = []
            for name in unknown:
                close = get_close_matches(name, sorted(CANONICAL_TRADE_COLUMNS), n=1)
                parts.append(f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
            raise ValueError(
                "Schema(trades=...) keys must be canonical column names. "
                f"Not recognised: {', '.join(parts)}. "
                "The key is the canonical name and the value is your column, "
                "so a mapping reads as {'side': 'direction'}."
            )
        # Normalise to a plain dict so the frozen instance cannot be mutated
        # through a caller's reference to the original mapping.
        object.__setattr__(self, "trades", dict(self.trades))

    @property
    def is_empty(self) -> bool:
        """True when this schema would rename nothing."""
        return not self.trades and all(
            getattr(self, f) is None
            for f in ("returns", "equity", "benchmark", "positions", "asset_returns", "prices")
        )

    def merge(self, other: Schema | None) -> Schema:
        """Return this schema overlaid with the non-empty parts of *other*.

        Used to let a per call schema override the session default without the
        caller having to restate the parts they did not change.
        """
        if other is None or other.is_empty:
            return self
        changes: dict[str, Any] = {}
        for name in ("returns", "equity", "benchmark", "positions", "asset_returns", "prices"):
            value = getattr(other, name)
            if value is not None:
                changes[name] = value
        if other.trades:
            changes["trades"] = {**self.trades, **other.trades}
        return replace(self, **changes)

    def apply_to_trades(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Rekey a raw trade log to canonical names.

        Columns already carrying a canonical name pass through untouched, so a
        partial mapping only has to name what differs.  An explicit mapping
        wins over a column that happens to already use the canonical name.

        A mapped column that is absent from the data draws a warning rather
        than an error, because one schema is meant to serve several trade logs
        and an optional column may genuinely be missing from some of them.  A
        missing ``pnl`` still fails hard downstream, where it is required.
        """
        if not self.trades:
            return dict(raw)

        missing = [(c, s) for c, s in self.trades.items() if s not in raw]
        if missing:
            detail = ", ".join(f"{c!r} <- {s!r}" for c, s in sorted(missing))
            warnings.warn(
                f"Schema maps columns that are not in the data: {detail}. "
                f"Available columns: {sorted(raw)}. "
                "Those mappings had no effect.",
                UserWarning,
                stacklevel=3,
            )

        renamed = {s: c for c, s in self.trades.items() if s in raw}
        out: dict[str, Any] = {}
        for key, value in raw.items():
            if key in renamed:
                continue  # re-added below under its canonical name
            out[key] = value
        for source, canonical in renamed.items():
            out[_TRADE_ALIASES.get(canonical, canonical)] = raw[source]
        return out


def _coerce(
    schema: Schema | None,
    columns: Mapping[str, Any] | None,
    *,
    tier: str | None,
) -> Schema | None:
    """Build one Schema from the ``schema=`` and ``columns=`` arguments.

    ``columns=`` is shorthand.  On a trades-only entry point it is the trade
    mapping itself, which is how a caller would naturally write it; elsewhere
    it mirrors the Schema fields.  Both build a Schema, so there is a single
    code path downstream.

    Raises:
        TypeError: If both *schema* and *columns* are given.  They express the
            same thing, and silently preferring one would hide the other.
    """
    if schema is not None and columns is not None:
        raise TypeError(
            "Pass either schema= or columns=, not both. columns= is shorthand that builds a Schema."
        )
    if columns is not None:
        schema = Schema(trades=dict(columns)) if tier == "trades" else Schema(**dict(columns))

    session = get_schema()
    if session is None:
        return schema
    return session.merge(schema)


# ----------------------------------------------------------------------
# Session default
# ----------------------------------------------------------------------
#
# Deliberately not routed through ``conventions.py``.  That module validates
# ``"param=value"`` strings against a per-metric registry and cannot carry a
# structured object.  Same pattern, separate state.

_session_schema: Schema | None = None


def set_schema(schema: Schema | Mapping[str, Any] | None) -> None:
    """Set the session wide column mapping.

    A per call ``schema=`` or ``columns=`` is overlaid on top of this, so the
    session default supplies whatever the call does not.

    Args:
        schema: A :class:`Schema`, a mapping of its fields, or None to clear.
    """
    global _session_schema
    if schema is None:
        _session_schema = None
        return
    _session_schema = schema if isinstance(schema, Schema) else Schema(**dict(schema))


def get_schema() -> Schema | None:
    """Return the session wide column mapping, or None if unset."""
    return _session_schema


def clear_schema() -> None:
    """Remove the session wide column mapping."""
    global _session_schema
    _session_schema = None


def describe_columns(data: Any, schema: Schema | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Report how a trade log's columns map to the canonical contract.

    Answers the two questions that matter before calling ``compute_trades``:
    which of your columns are recognised, and which metrics you are giving up
    by leaving a canonical column out.

    Args:
        data: A trade log.  Either a mapping of column name to values or a
            pandas/polars DataFrame.
        schema: Optional :class:`Schema` (or a ``trades`` mapping) to account
            for, so a column you have mapped under a non canonical name is
            reported as recognised rather than ignored.

    Returns:
        A dict with three keys:

        * ``"recognized"``: canonical column names available in *data*, either
          under their canonical name, via *schema*, or by derivation
          (``duration`` from ``entry_time`` and ``exit_time``).
        * ``"ignored"``: data columns that are neither canonical nor mapped, so
          they are dropped from the input.
        * ``"missing"``: canonical columns that are not recognised, each mapped
          to the metric names that become unavailable as a result.  A missing
          ``pnl`` lists every trade metric, since all of them need it.

    ``recognized`` and ``missing`` use the canonical names from
    :data:`CANONICAL_TRADE_COLUMNS`; the metric lists mirror the
    ``_require_field`` calls in ``core/trades.py``.
    """
    if schema is not None and not isinstance(schema, Schema):
        schema = Schema(trades=dict(schema))

    if isinstance(data, dict):
        data_cols = set(data)
    elif hasattr(data, "columns"):
        data_cols = {str(c) for c in data.columns}
    else:
        raise TypeError(
            "describe_columns expects a mapping of columns or a pandas/polars "
            f"DataFrame, got {type(data).__name__}."
        )

    mapped_values: set[str] = set(schema.trades.values()) if schema is not None else set()

    recognized: set[str] = {c for c in data_cols if c in CANONICAL_TRADE_COLUMNS}
    if schema is not None:
        for canonical, source in schema.trades.items():
            if source in data_cols:
                recognized.add(canonical)

    for derived, (left, right) in _TRADE_DERIVED.items():
        if left in recognized and right in recognized:
            recognized.add(derived)

    ignored = sorted(
        c for c in data_cols if c not in CANONICAL_TRADE_COLUMNS and c not in mapped_values
    )

    missing: dict[str, list[str]] = {}
    for column in sorted(CANONICAL_TRADE_COLUMNS - recognized):
        if column == "pnl":
            from stratstat.registry import list_metrics

            names = [m["name"] for m in list_metrics(requires="trades")]
        else:
            names = list(_TRADE_COLUMN_METRICS.get(column, ()))
        if names:
            missing[column] = names

    return {
        "recognized": sorted(recognized),
        "ignored": ignored,
        "missing": missing,
    }
