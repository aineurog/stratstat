"""Session-level metric convention overrides.

For metrics with genuinely competing real-world definitions, this module
provides set_default() / get_default() so users can set session-wide
preferences that all metric calls will consult.
"""

from __future__ import annotations

_defaults: dict[str, str] = {}


def set_default(metric: str, convention: str) -> None:
    """Set the default convention for a metric session-wide.

    Args:
        metric: Metric name (e.g. "sharpe_ratio").
        convention: Convention value (e.g. "ddof=1").

    Raises:
        ConventionError: If the metric or convention is not recognized.
    """
    _defaults[metric] = convention


def get_default(metric: str) -> str | None:
    """Get the session-wide default convention for a metric, if set.

    Args:
        metric: Metric name.

    Returns:
        The convention string, or None if no override is set.
    """
    return _defaults.get(metric)


def clear_defaults() -> None:
    """Reset all session-wide convention overrides."""
    _defaults.clear()
