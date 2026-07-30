"""Reporting module — Plotly-based visualizations.

This module depends on core but never the reverse. It is an optional extra
(``pip install stratstat[report]``). Importing stratstat must not require plotly.

All imports of plotly are lazy (inside function bodies) so that the report
subpackage can be imported without plotly installed — only calling a
visualization function triggers the import check.
"""


def _ensure_plotly() -> None:
    """Check that plotly is installed; raise a helpful error if not."""
    try:
        import plotly  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "plotly is required for the report module. "
            "Install it with: pip install stratstat[report]"
        ) from err


# Visualization functions will be added in Phase 7.
# Each will call _ensure_plotly() at the top of its body.
