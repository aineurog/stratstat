"""Metric result types — the universal return shape for all StratStat computations.

MetricResult: a single value with metadata.
MetricSet: an ordered collection of MetricResult with serialization methods.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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


# ===================================================================
# MetricResult
# ===================================================================


@dataclass
class MetricResult:
    """The result of a single metric computation.

    Attributes:
        name: Metric name (e.g. "sharpe_ratio").
        value: The computed value — a float or a numpy array for batch results.
        category: Tuple of classification tags (e.g. ("risk_adjusted", "returns")).
        periods_per_year: Annualization factor used, or None if not applicable.
        meta: Dict of metadata — formula reference, convention used, ddof, etc.
    """

    name: str
    value: float | Any  # float | np.ndarray, but avoid numpy import at module level
    category: tuple[str, ...] = ()
    periods_per_year: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        val = self.value
        if hasattr(val, "shape"):
            val = f"array{val.shape}"
        else:
            val = f"{val:.6g}" if isinstance(val, float) else val
        return (
            f"MetricResult(name={self.name!r}, value={val}, "
            f"category={self.category})"
        )

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
    """

    results: list[MetricResult] = field(default_factory=list)

    # -- Container protocol -----------------------------------------------

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[MetricResult]:
        return iter(self.results)

    def __getitem__(self, index: int) -> MetricResult:
        return self.results[index]

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
            parts.append(
                "<thead><tr>"
                "<th>Metric</th><th>Value</th><th>Citation</th>"
                "</tr></thead>"
            )
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
                    f'max-width:400px;overflow:hidden;'
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

    def to_frame(self) -> pd.DataFrame:
        """Return results as a pandas DataFrame.

        Columns: ``name``, ``value``, ``category``, ``periods_per_year``,
        plus each key in ``meta`` expanded as its own column.
        """
        records = []
        for r in self.results:
            rec = {
                "name": r.name,
                "value": r.value,
                "category": r.category,
                "periods_per_year": r.periods_per_year,
            }
            rec.update(r.meta)
            records.append(rec)
        df: pd.DataFrame = pd.DataFrame(records)
        return df

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
