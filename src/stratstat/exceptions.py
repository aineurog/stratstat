"""Custom exception types for StratStat.

Users can catch these specifically rather than relying on generic ValueError/KeyError.
"""


class StratStatError(Exception):
    """Base exception for all StratStat errors."""


class InvalidInputError(StratStatError, ValueError):
    """Raised when input data fails validation (wrong shape, type, or content)."""


class InsufficientDataError(StratStatError, ValueError):
    """Raised when input does not contain enough data for a requested metric."""


class UnknownMetricError(StratStatError, KeyError):
    """Raised when a metric name is not found in the registry."""


class MetricNotApplicableError(StratStatError, ValueError):
    """Raised when a metric cannot be computed for the given input.

    Signals that a metric is legitimately inapplicable — e.g. it requires data
    the input does not provide (exposure equity, trade ``side``, an
    annualization factor, etc.).  It subclasses :class:`ValueError` so that
    direct ``compute()`` calls still surface it as a normal error, while batch
    wrappers (``compute_all``, ``rolling``, ``by_regime``) catch it and record
    the metric as skipped rather than failing the whole batch.
    """


class ConventionError(StratStatError, ValueError):
    """Raised when an invalid convention/method value is specified for a metric."""
