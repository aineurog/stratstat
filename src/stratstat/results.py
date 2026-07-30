"""Metric result types — the universal return shape for all StratStat computations.

MetricResult: a single value with metadata.
MetricSet: an ordered collection of MetricResult with serialization methods.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


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


@dataclass
class MetricSet:
    """An ordered collection of MetricResult objects.

    Produced by compute_all() or batch computation. Supports serialization
    to dict, DataFrame, JSON, and markdown.
    """

    results: list[MetricResult] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, index: int) -> MetricResult:
        return self.results[index]

    def to_dict(self) -> dict[str, Any]:
        """Return results as {name: value} dict."""
        return {r.name: r.value for r in self.results}

    def to_frame(self):
        """Return results as a pandas DataFrame with category columns."""
        import pandas as pd

        records = []
        for r in self.results:
            rec = {"name": r.name, "value": r.value}
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
            val = r.value
            if isinstance(val, float):
                val = f"{val:.6g}"
            lines.append(f"| {r.name} | {val} |")
        return "\n".join(lines)
