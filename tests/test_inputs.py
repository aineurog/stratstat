"""Tests for input containers — validation and normalization."""

import numpy as np
import pytest

from stratstat.inputs import ReturnsInput, _to_numpy


class TestToNumpy:
    """Tests for _to_numpy normalization helper."""

    def test_numpy_ndarray_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_numpy(arr)
        assert result is arr  # same object, no copy

    def test_pandas_series(self):
        pd = pytest.importorskip("pandas")
        s = pd.Series([0.01, -0.02, 0.015])
        result = _to_numpy(s)
        assert isinstance(result, np.ndarray)
        assert result.dtype == float

    def test_pandas_dataframe(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [0.01, -0.02], "b": [0.03, 0.01]})
        result = _to_numpy(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_polars_series(self):
        pl = pytest.importorskip("polars")
        s = pl.Series("returns", [0.01, -0.02, 0.015])
        result = _to_numpy(s)
        assert isinstance(result, np.ndarray)

    def test_polars_dataframe(self):
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"a": [0.01, -0.02], "b": [0.03, 0.01]})
        result = _to_numpy(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_unsupported_input_type(self):
        with pytest.raises(TypeError, match="Unsupported input type"):
            _to_numpy([1, 2, 3])  # plain list is not accepted


class TestReturnsInput:
    """Tests for ReturnsInput container."""

    def test_single_strategy_1d(self):
        data = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
        ri = ReturnsInput(data)
        assert ri.n_periods == 5
        assert ri.n_strategies == 1
        assert ri.is_single is True
        assert ri.values.shape == (5, 1)

    def test_multi_strategy_2d(self):
        data = np.array([[0.01, 0.02], [-0.02, 0.01], [0.015, -0.005]])
        ri = ReturnsInput(data)
        assert ri.n_periods == 3
        assert ri.n_strategies == 2
        assert ri.is_single is False
        assert ri.values.shape == (3, 2)

    def test_periods_per_year(self):
        ri = ReturnsInput(np.array([0.01, -0.02]), periods_per_year=252)
        assert ri.periods_per_year == 252

    def test_periods_per_year_none(self):
        ri = ReturnsInput(np.array([0.01, -0.02]))
        assert ri.periods_per_year is None

    def test_repr(self):
        ri = ReturnsInput(np.array([0.01, -0.02]), periods_per_year=252)
        rep = repr(ri)
        assert "ReturnsInput" in rep
        assert "n_periods=2" in rep
        assert "periods_per_year=252" in rep

    def test_3d_input_raises(self):
        data = np.zeros((2, 3, 4))
        with pytest.raises(ValueError, match="1-D or 2-D"):
            ReturnsInput(data)
