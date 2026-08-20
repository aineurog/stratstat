"""Session-level metric convention overrides.

For metrics with genuinely competing real-world definitions (Sharpe ddof,
Sortino denominator, max drawdown return type, VaR/CVaR estimator choice,
beta variant, drawdown duration units, tail cutoffs), this module provides
``set_default()`` / ``get_default()`` so users can set session-wide
preferences that every metric call consults as a fallback for its convention
parameter.

Conventions are expressed in ``"param=value"`` form (e.g. ``"ddof=0"``),
which is unambiguous even for metrics that expose more than one convention
parameter (VaR/CVaR expose both ``method`` and ``confidence``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from stratstat.exceptions import ConventionError

_T = TypeVar("_T")

_defaults: dict[str, str] = {}


@dataclass(frozen=True)
class _ParamSpec:
    """Validation + coercion rules for a single convention parameter."""

    validate: Callable[[str], bool]
    coerce: Callable[[str], Any]
    description: str


def _choice(*options: str) -> _ParamSpec:
    """A categorical parameter whose value must be one of *options*."""
    return _ParamSpec(
        validate=lambda v: v in options,
        coerce=lambda v: v,
        description="/".join(repr(o) for o in options),
    )


def _int_choice(*options: str) -> _ParamSpec:
    """A categorical parameter whose value must be one of *options*, coerced to int."""
    return _ParamSpec(
        validate=lambda v: v in options,
        coerce=int,
        description="/".join(repr(o) for o in options),
    )


def _float_range(low: float, high: float, *, inclusive_high: bool = False) -> _ParamSpec:
    """A numeric parameter that must lie within (low, high)."""

    def validate(value: str) -> bool:
        try:
            f = float(value)
        except ValueError:
            return False
        return (low < f <= high) if inclusive_high else (low < f < high)

    end = "]" if inclusive_high else ")"
    return _ParamSpec(
        validate=validate,
        coerce=float,
        description=f"float in ({low}, {high}{end}",
    )


#: Convention vocabulary, mirroring the table in docs/formula-reference.md.
#: Only metrics with genuinely competing definitions are listed (build
#: instructions §3.6 — do not add configuration surface to metrics with a
#: single legitimate definition).
_CONVENTIONS: dict[str, dict[str, _ParamSpec]] = {
    "sharpe_ratio": {"ddof": _int_choice("0", "1")},
    "sortino_ratio": {"denominator": _choice("full_downside", "downside_only")},
    "max_drawdown": {"return_type": _choice("simple", "log")},
    "longest_drawdown_duration": {"units": _choice("periods", "years")},
    "var": {
        "method": _choice("historical", "parametric", "cornish_fisher"),
        "confidence": _float_range(0.0, 1.0),
    },
    "cvar": {
        "method": _choice("historical", "parametric"),
        "confidence": _float_range(0.0, 1.0),
    },
    "tail_ratio": {"tail_cutoff": _float_range(0.0, 0.5)},
    "hill_tail_index": {"tail_fraction": _float_range(0.0, 0.5, inclusive_high=True)},
}


def set_default(metric: str, convention: str) -> None:
    """Set the default convention for a metric session-wide.

    Args:
        metric: Metric name (e.g. ``"sharpe_ratio"``).
        convention: Convention in ``"param=value"`` form (e.g. ``"ddof=0"``).

    Raises:
        ConventionError: If the metric, parameter, or value is not recognized.
    """
    params = _CONVENTIONS.get(metric)
    if params is None:
        raise ConventionError(
            f"Unknown convention-bearing metric: {metric!r}. Supported metrics: "
            f"{', '.join(sorted(_CONVENTIONS))}."
        )
    param, sep, value = convention.partition("=")
    if not sep:
        raise ConventionError(
            f"Convention for {metric!r} must be 'param=value', got {convention!r}"
        )
    spec = params.get(param)
    if spec is None:
        raise ConventionError(
            f"Unknown parameter {param!r} for {metric!r}. Valid parameters: "
            f"{', '.join(sorted(params))}."
        )
    if not spec.validate(value):
        raise ConventionError(
            f"Invalid value {value!r} for {metric!r}.{param} "
            f"(expected {spec.description})."
        )
    _defaults[metric] = f"{param}={value}"


def get_default(metric: str) -> str | None:
    """Get the session-wide default convention for a metric, if set.

    Args:
        metric: Metric name.

    Returns:
        The canonical ``"param=value"`` convention string, or None if no
        override is set.
    """
    return _defaults.get(metric)


def resolve_default(metric: str, param: str) -> Any | None:
    """Coerce and return the session override for a metric's parameter.

    Used internally by convention-bearing metrics to consult the session
    default as a fallback for their explicit parameter.

    Args:
        metric: Metric name (e.g. ``"sharpe_ratio"``).
        param: Parameter name (e.g. ``"ddof"``).

    Returns:
        The coerced override value (typed: int/float/str as appropriate), or
        ``None`` if no override is set for that metric's parameter.
    """
    convention = _defaults.get(metric)
    if convention is None:
        return None
    p, sep, value = convention.partition("=")
    if not sep or p != param:
        return None
    return _CONVENTIONS[metric][param].coerce(value)


def resolve_convention(value: _T | None, metric: str, param: str, builtin: _T) -> _T:
    """Resolve a convention parameter to its effective value.

    Precedence: an explicit *value* (not ``None``) wins; otherwise the
    session default set via :func:`set_default` is used; otherwise *builtin*
    (the cited default).

    Convention-bearing metrics accept ``None`` for their convention parameter
    and call this helper at the top of their body.
    """
    if value is not None:
        return value
    override = resolve_default(metric, param)
    return builtin if override is None else override


def clear_defaults() -> None:
    """Reset all session-wide convention overrides."""
    _defaults.clear()
