"""Metric result types — the universal return shape for all StratStat computations.

MetricResult: a single value with metadata.
MetricSet: an ordered collection of MetricResult with serialization methods.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Display order for category sections
# ---------------------------------------------------------------------------

_CATEGORY_ORDER: dict[str, int] = {
    "descriptive": 0,
    "risk": 1,
    "risk_adjusted": 2,
    "inference": 3,
    "benchmark": 4,
    "exposure": 5,
    "trades": 6,
    "relative": 7,
}

_CATEGORY_LABELS: dict[str, str] = {
    "descriptive": "Descriptive",
    "risk": "Risk",
    "risk_adjusted": "Risk-Adjusted",
    "inference": "Inference",
    "benchmark": "Benchmark",
    "exposure": "Exposure",
    "trades": "Trades",
    "relative": "Relative",
}


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    """Format a metric value for display.

    * float / numpy scalar → 6 significant digits
    * small 1-D numpy array → compact list repr
    * large numpy array → ``array{shape}``
    * NaN → ``"N/A"``
    * ±Inf → ``"∞"`` / ``"-∞"``
    * dict → compact JSON
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 1 and value.size <= 5:
            inner = ", ".join(_format_scalar(float(v)) for v in value)
            return f"[{inner}]"
        return f"array{value.shape}"
    if isinstance(value, (np.floating, np.integer)):
        return _format_scalar(float(value))
    if isinstance(value, float):
        return _format_scalar(value)
    if isinstance(value, dict):
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return str(value)
    if value is None:
        return "N/A"
    return str(value)


def _format_scalar(v: float) -> str:
    """Format a single numeric value."""
    if np.isnan(v):
        return "N/A"
    if np.isposinf(v):
        return "∞"
    if np.isneginf(v):
        return "-∞"
    return f"{v:.6g}"


# ---------------------------------------------------------------------------
# Category grouping
# ---------------------------------------------------------------------------


def _group_by_category(
    results: list[MetricResult],
) -> dict[str, list[MetricResult]]:
    """Group results by primary category tag, in display order.

    Categories not in ``_CATEGORY_ORDER`` are placed in an ``"Other"``
    group at the end.
    """
    groups: dict[str, list[MetricResult]] = {}
    for r in results:
        primary = r.category[0] if r.category else "other"
        groups.setdefault(primary, []).append(r)

    # Sort within each group alphabetically by name
    for g in groups.values():
        g.sort(key=lambda m: m.name)

    # Order groups
    def _order_key(item: tuple[str, list[MetricResult]]) -> int:
        return _CATEGORY_ORDER.get(item[0], 99)

    return dict(sorted(groups.items(), key=_order_key))


def _requires_of(name: str) -> str | None:
    """Return the input tier a metric requires, or None if unregistered.

    Lazily imports the registry to avoid an import cycle (``registry``
    imports ``results`` only under ``TYPE_CHECKING`` and inside functions).
    """
    try:
        from stratstat.registry import requires_of
    except ImportError:  # pragma: no cover - registry is always present
        return None
    return requires_of(name)


# ===================================================================
# MetricResult
# ===================================================================

# Union of every value shape a metric may return. Kept as an alias so the
# dataclass field reads clearly while still being honest about the full range
# of scalar, array, and dict results.
MetricValue: TypeAlias = float | int | bool | str | np.ndarray[Any, Any] | dict[str, Any] | None


@dataclass
class MetricResult:
    """The result of a single metric computation.

    Attributes:
        name: Metric name (e.g. "sharpe_ratio").
        value: The computed value — a scalar (float/int/bool/str), a numpy
            array for batch results, or a dict for composite metrics such as
            ``period_counts``.
        category: Tuple of classification tags (e.g. ("risk_adjusted", "returns")).
        periods_per_year: Annualization factor used, or None if not applicable.
        meta: Dict of metadata — formula reference, convention used, ddof, etc.
    """

    name: str
    value: MetricValue
    category: tuple[str, ...] = ()
    periods_per_year: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        val: Any = self.value
        if isinstance(val, np.ndarray):
            val = f"array{val.shape}"
        elif isinstance(val, float):
            val = f"{val:.6g}"
        return f"MetricResult(name={self.name!r}, value={val}, category={self.category})"

    def __str__(self) -> str:
        """User-friendly single-line display."""
        return f"{self.name}: {_format_value(self.value)}"


# ===================================================================
# MetricSet
# ===================================================================


@dataclass
class MetricSet:
    """An ordered collection of MetricResult objects.

    Produced by compute_all() or batch computation. Supports serialization
    to dict, DataFrame, JSON, CSV, and markdown.  The ``__str__`` and
    ``_repr_html_`` methods produce sectioned, grouped output suitable
    for terminal and Jupyter display respectively.

    ``meta`` carries batch-level information; ``compute_all()`` records the
    names of metrics it skipped (because they were inapplicable to the input)
    under ``meta["skipped"]``.
    """

    results: list[MetricResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- Container protocol -----------------------------------------------

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[MetricResult]:
        return iter(self.results)

    def __getitem__(self, key: int | str) -> MetricResult:
        """Access a result by integer position or by metric name.

        ``ms[0]`` returns the first result; ``ms["sharpe_ratio"]`` returns
        the result whose ``name`` matches.  A missing name raises
        ``KeyError``.
        """
        if isinstance(key, str):
            for r in self.results:
                if r.name == key:
                    return r
            raise KeyError(key)
        return self.results[key]

    def __contains__(self, item: object) -> bool:
        """True if *item* is a result name or a ``MetricResult`` present here."""
        if isinstance(item, str):
            return any(r.name == item for r in self.results)
        return item in self.results

    def get(self, name: str, default: Any = None) -> MetricResult | Any:
        """Return the result with *name*, or *default* if absent."""
        for r in self.results:
            if r.name == name:
                return r
        return default

    def by_tier(self) -> dict[str, MetricSet]:
        """Group results by input tier (``requires``).

        Returns a ``{tier: MetricSet}`` dict keyed by the registry input tier
        (``"returns"``, ``"exposure"``, ``"trades"``, ``"benchmark"``,
        ``"compare"``).  Only tiers that are actually present appear as keys.
        Metrics not found in the registry are grouped under ``"unknown"``.
        """
        groups: dict[str, list[MetricResult]] = {}
        for r in self.results:
            tier = _requires_of(r.name) or "unknown"
            groups.setdefault(tier, []).append(r)
        return {tier: MetricSet(results=grp, meta=dict(self.meta)) for tier, grp in groups.items()}

    def by_category(self) -> dict[str, MetricSet]:
        """Group results by primary statistical category tag.

        Returns a ``{category: MetricSet}`` dict keyed by the first element
        of each result's ``category`` tuple (e.g. ``"risk"``,
        ``"descriptive"``).  Results with an empty category are grouped under
        ``"other"``.
        """
        groups: dict[str, list[MetricResult]] = {}
        for r in self.results:
            primary = r.category[0] if r.category else "other"
            groups.setdefault(primary, []).append(r)
        return {cat: MetricSet(results=grp, meta=dict(self.meta)) for cat, grp in groups.items()}

    # -- Omission reporting ----------------------------------------------

    @property
    def skipped(self) -> list[str]:
        """Metric names that could not run, in order they were encountered.

        Populated by ``compute_all()`` when a metric is missing a required
        keyword argument or raises
        :class:`~stratstat.exceptions.MetricNotApplicableError`.  Empty for a
        hand built :class:`MetricSet`.
        """
        return list(self.meta.get("skipped", []))

    @property
    def excluded(self) -> list[str]:
        """Metric names deliberately left out of the run.

        The union of resampling backend metrics (always excluded from
        ``compute_all`` because they are expensive and need their own
        parameters) and deduplicated aliases (dropped because their canonical
        ``alias_of`` metric also ran).  Empty for a hand built
        :class:`MetricSet`.
        """
        return list(self.meta.get("excluded_resampling", [])) + list(
            self.meta.get("deduplicated", [])
        )

    @property
    def excluded_tiers(self) -> list[str]:
        """Tier names that did not run (data absent or ``include_*`` false)."""
        return list(self.meta.get("excluded_tiers", []))

    def summary(self) -> str:
        """One line reporting what ran and what was omitted.

        Complements the :attr:`skipped`, :attr:`excluded` and
        :attr:`excluded_tiers` properties with a single readable count, so a
        caller can glance at a batch result and see whether anything was left
        out rather than assuming everything ran.
        """
        parts = [f"{len(self.results)} metrics computed"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped ({', '.join(self.skipped)})")
        if self.excluded:
            parts.append(f"{len(self.excluded)} excluded ({', '.join(self.excluded)})")
        if self.excluded_tiers:
            parts.append(f"tiers not run: {', '.join(self.excluded_tiers)}")
        return "; ".join(parts)

    # -- Display ----------------------------------------------------------

    def __str__(self) -> str:
        """Sectioned plain-text display grouped by primary category."""
        if not self.results:
            return "MetricSet (empty)"

        groups = _group_by_category(self.results)
        lines: list[str] = []
        for primary, metrics in groups.items():
            label = _CATEGORY_LABELS.get(primary, primary.title())
            # Section header
            lines.append(f"═══ {label} ═══")
            # Compute column widths
            max_name = max((len(m.name) for m in metrics), default=0)
            for m in metrics:
                formatted = _format_value(m.value)
                lines.append(f"  {m.name:<{max_name}}  {formatted}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """IPython / Jupyter rich display — sectioned HTML tables."""
        if not self.results:
            return "<p><em>MetricSet (empty)</em></p>"

        groups = _group_by_category(self.results)
        parts: list[str] = []
        for primary, metrics in groups.items():
            label = _CATEGORY_LABELS.get(primary, primary.title())
            parts.append(f"<h3>{label}</h3>")
            parts.append("<table>")
            parts.append("<thead><tr><th>Metric</th><th>Value</th><th>Citation</th></tr></thead>")
            parts.append("<tbody>")
            for i, m in enumerate(metrics):
                bg = ' style="background:#f8fafc"' if i % 2 == 0 else ""
                ref = m.meta.get("ref", "")
                # Truncate long refs for tooltip display
                ref_title = ref.replace('"', "&quot;").replace("<", "&lt;")
                ref_short = (ref[:120] + "…") if len(ref) > 120 else ref
                parts.append(
                    f"<tr{bg}>"
                    f"<td><code>{m.name}</code></td>"
                    f"<td><code>{_format_value(m.value)}</code></td>"
                    f'<td style="color:#a0aec0;font-size:11px;'
                    f"max-width:400px;overflow:hidden;"
                    f'text-overflow:ellipsis;white-space:nowrap"'
                    f' title="{ref_title}">{ref_short}</td>'
                    f"</tr>"
                )
            parts.append("</tbody></table>")
        return "\n".join(parts)

    # -- Serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return results as {name: value} dict."""
        return {r.name: r.value for r in self.results}

    def to_frame(self, explode: bool = True) -> pd.DataFrame:
        """Return results as a pandas DataFrame.

        Columns: ``name``, ``value``, ``tier``, ``category``,
        ``periods_per_year``, plus each key in ``meta`` expanded as its own
        column.  ``tier`` is the registry input tier (``requires``); it is
        ``None`` for metrics not present in the registry.

        Args:
            explode: When True (default), a metric whose value is a 1-D array
                and whose ``meta`` carries an ``output_index`` of matching
                length is expanded into one row per element, with the index
                label in an ``output_index`` column.  This is how per-strategy
                outputs (``component_var``,
                ``marginal_contribution_to_risk``) and per-level outputs
                (``percentiles``) become sortable and filterable instead of
                landing as an array in a single cell.  Metrics without a
                matching index (scalars, matrices such as
                ``correlation_matrix``) stay in one row regardless.
        """
        records: list[dict[str, Any]] = []
        for r in self.results:
            base: dict[str, Any] = {
                "name": r.name,
                "tier": _requires_of(r.name),
                "category": r.category,
                "periods_per_year": r.periods_per_year,
                "degenerate": bool(r.meta.get("degenerate", False)),
                "degenerate_reason": r.meta.get("degenerate_reason"),
            }
            idx = r.meta.get("output_index")
            value = r.value
            if (
                explode
                and isinstance(value, np.ndarray)
                and value.ndim == 1
                and isinstance(idx, (list, tuple, np.ndarray))
                and len(idx) == value.shape[0]
            ):
                scalar_meta = {k: v for k, v in r.meta.items() if k != "output_index"}
                for label, v in zip(idx, value, strict=True):
                    rec = dict(base)
                    rec["value"] = v
                    rec["output_index"] = label
                    rec.update(scalar_meta)
                    records.append(rec)
                continue
            rec = dict(base)
            rec["value"] = value
            rec.update(r.meta)
            records.append(rec)
        return pd.DataFrame(records)

    def to_json(self, indent: int = 2) -> str:
        """Return results as a JSON string."""
        return json.dumps(
            [asdict(r) for r in self.results],
            indent=indent,
            default=str,
        )

    def to_markdown(self) -> str:
        """Return results as a markdown table."""
        lines = ["| Metric | Value |", "|--------|-------|"]
        for r in self.results:
            val = _format_value(r.value)
            lines.append(f"| {r.name} | {val} |")
        return "\n".join(lines)

    def to_csv(self, path: str | Path) -> None:
        """Write results to a CSV file.

        Parameters
        ----------
        path: Output file path (``.csv`` suffix recommended).
        """
        p = Path(path) if isinstance(path, str) else path
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(p, index=False)

    def to_clipboard(self) -> None:
        """Copy results to the system clipboard as a tab-separated table.

        Delegates to :meth:`pandas.DataFrame.to_clipboard`.
        """
        self.to_frame().to_clipboard(index=False)
