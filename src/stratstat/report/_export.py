"""Export utilities for report figures and metric sets.

Supports writing figures to HTML (interactive), PNG, and SVG, and
MetricSet objects to Markdown, LaTeX, and JSON files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _path(path: str | Path) -> Path:
    """Coerce to Path."""
    return Path(path) if isinstance(path, str) else path


# ---------------------------------------------------------------------------
# Figure exports (plotly)
# ---------------------------------------------------------------------------


def to_html(fig: Any, path: str | Path) -> None:
    """Write a plotly Figure to an interactive HTML file.

    Parameters
    ----------
    fig: A ``plotly.graph_objects.Figure``.
    path: Output file path (``.html`` suffix recommended).
    """
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(p))


def to_image(
    fig: Any,
    path: str | Path,
    format: str = "png",
    width: int | None = None,
    height: int | None = None,
    scale: int = 2,
) -> None:
    """Write a plotly Figure to a static image file.

    Requires ``kaleido`` (``pip install kaleido``).

    Parameters
    ----------
    fig: A ``plotly.graph_objects.Figure``.
    path: Output file path.  The extension determines the format if
        *format* is not given.
    format: One of ``"png"``, ``"svg"``, ``"pdf"``, ``"jpeg"``, ``"webp"``.
    width: Optional width in pixels.
    height: Optional height in pixels.
    scale: Image scale factor (default 2 for retina-quality PNG).
    """
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(p), format=format, width=width, height=height,
                    scale=scale)


# ---------------------------------------------------------------------------
# MetricSet / table exports
# ---------------------------------------------------------------------------


def to_markdown(metric_set: Any, path: str | Path) -> None:
    """Write a MetricSet to a Markdown table file.

    Parameters
    ----------
    metric_set: A ``MetricSet`` instance.
    path: Output file path (``.md`` suffix recommended).
    """
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(metric_set.to_markdown())


def to_latex(metric_set: Any, path: str | Path) -> None:
    """Write a MetricSet to a LaTeX table file.

    Parameters
    ----------
    metric_set: A ``MetricSet`` instance.
    path: Output file path (``.tex`` suffix recommended).
    """
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
    ]
    for r in metric_set:
        name = r.name.replace("_", r"\_")
        val = r.value
        if isinstance(val, float):
            val_str = f"{val:.6g}"
        elif hasattr(val, "shape"):
            val_str = f"array{val.shape}"
        else:
            val_str = str(val)
        lines.append(f"{name} & {val_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    p.write_text("\n".join(lines) + "\n")


def to_json(metric_set: Any, path: str | Path, indent: int = 2) -> None:
    """Write a MetricSet to a JSON file.

    Parameters
    ----------
    metric_set: A ``MetricSet`` instance.
    path: Output file path (``.json`` suffix recommended).
    indent: JSON indentation level.
    """
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(metric_set.to_json(indent=indent))
