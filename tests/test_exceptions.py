"""Tests for custom exception types."""

import pytest

from stratstat.exceptions import (
    ConventionError,
    InsufficientDataError,
    InvalidInputError,
    StratStatError,
    UnknownMetricError,
)


def test_stratstat_error_is_base():
    """All custom exceptions should inherit from StratStatError."""
    assert issubclass(InvalidInputError, StratStatError)
    assert issubclass(InsufficientDataError, StratStatError)
    assert issubclass(UnknownMetricError, StratStatError)
    assert issubclass(ConventionError, StratStatError)


def test_invalid_input_error_is_valueerror():
    """Users should be able to catch ValueError too."""
    assert issubclass(InvalidInputError, ValueError)


def test_unknown_metric_error_is_keyerror():
    """Users should be able to catch KeyError too."""
    assert issubclass(UnknownMetricError, KeyError)


def test_exceptions_can_be_raised():
    """Sanity check that exceptions can be raised and caught specifically."""
    with pytest.raises(InvalidInputError):
        raise InvalidInputError("bad input")

    with pytest.raises(InsufficientDataError):
        raise InsufficientDataError("not enough data")

    with pytest.raises(UnknownMetricError):
        raise UnknownMetricError("no such metric")

    with pytest.raises(ConventionError):
        raise ConventionError("bad convention")
