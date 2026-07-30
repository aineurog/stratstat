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


class ConventionError(StratStatError, ValueError):
    """Raised when an invalid convention/method value is specified for a metric."""
