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
# ``intratrade_prices`` is the current internal name for what the contract
# calls ``price_path``.  Both are accepted; ``price_path`` is translated to the
# internal name on the way in, so callers can use the contract name before the
# rename lands.
#
# ``position_size``, ``max_price``, ``min_price``, ``mfe``, ``mae`` and
# ``asset`` are part of the contract but not yet consumed by any metric.  They
# are accepted here so a schema written against the contract does not raise,
# and they pass through untouched until the trade convention work reads them.
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

# Contract name -> current internal name.
_TRADE_ALIASES: dict[str, str] = {"price_path": "intratrade_prices"}


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
            "Pass either schema= or columns=, not both. "
            "columns= is shorthand that builds a Schema."
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
