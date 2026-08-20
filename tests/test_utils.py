"""Tests for core._utils — annualization, NaN handling, numba dispatch."""

import numpy as np
import pytest

from stratstat.core._utils import (
    annualization_factor,
    is_numba_available,
    nanmean,
    nanstd,
    numba_worthwhile,
)


class TestAnnualizationFactor:
    def test_daily(self):
        assert annualization_factor(252) == 252.0

    def test_monthly(self):
        assert annualization_factor(12) == 12.0

    def test_weekly(self):
        assert annualization_factor(52) == 52.0

    def test_none_returns_one(self):
        assert annualization_factor(None) == 1.0

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            annualization_factor(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            annualization_factor(-1)

    def test_returns_float(self):
        result = annualization_factor(252)
        assert isinstance(result, float)


class TestIsNumbaAvailable:
    def test_returns_bool(self):
        result = is_numba_available()
        assert isinstance(result, bool)


class TestNumbaWorthwhile:
    def test_zero_is_not_worthwhile(self):
        assert numba_worthwhile(0) is False

    def test_small_workload_is_not_worthwhile(self):
        assert numba_worthwhile(999_999) is False

    def test_at_threshold_is_worthwhile(self):
        assert numba_worthwhile(1_000_000) is True

    def test_large_workload_is_worthwhile(self):
        assert numba_worthwhile(10_000_000) is True


class TestNanMean:
    def test_basic(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert nanmean(arr) == 2.0

    def test_with_nans(self):
        arr = np.array([1.0, np.nan, 3.0])
        assert nanmean(arr) == 2.0

    def test_all_nans(self):
        arr = np.array([np.nan, np.nan])
        result = nanmean(arr)
        assert np.isnan(result)


class TestNanStd:
    def test_basic(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert nanstd(arr, ddof=0) == pytest.approx(np.sqrt(2 / 3))

    def test_with_nans(self):
        arr = np.array([1.0, np.nan, 3.0])
        result = nanstd(arr, ddof=0)
        assert result == 1.0  # std of [1, 3] with ddof=0

    def test_ddof_default(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = nanstd(arr)
        expected = np.nanstd(arr, ddof=1)
        assert result == pytest.approx(expected)
