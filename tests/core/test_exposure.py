"""Tests for exposure-tier metrics.

Covers all 23 registered exposure metrics, edge cases, input types,
and registry integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Import triggers @register_metric decorators
import stratstat.core.exposure  # noqa: F401
from stratstat.core.exposure import active_share
from stratstat.inputs import ExposureInput

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_positions():
    """A 130/30 portfolio: 5 assets, 10 periods.

    Positions are weights (position_value / portfolio_value).
    """
    return np.array(
        [
            [0.30, 0.30, 0.20, 0.10, 0.10],
            [0.35, 0.25, 0.20, 0.15, 0.05],
            [0.40, 0.30, 0.20, 0.10, -0.10],
            [0.45, 0.35, 0.25, 0.00, -0.20],
            [0.50, 0.30, 0.20, -0.10, -0.20],
            [0.40, 0.30, 0.10, -0.10, -0.10],
            [0.35, 0.25, 0.10, 0.00, 0.00],
            [0.30, 0.20, 0.10, 0.10, 0.00],
            [0.25, 0.15, 0.10, 0.10, 0.00],
            [0.20, 0.10, 0.05, 0.05, 0.00],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def simple_returns():
    """Asset-level returns matching the shape of simple_positions."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 0.01, size=(10, 5)).astype(np.float64)


@pytest.fixture
def simple_benchmark():
    """Benchmark returns matching n_periods."""
    rng = np.random.default_rng(43)
    return rng.normal(0.0, 0.008, size=10).astype(np.float64)


@pytest.fixture
def inp_no_ret(simple_positions):
    """ExposureInput with positions only."""
    return ExposureInput(positions=simple_positions, periods_per_year=252)


@pytest.fixture
def inp_with_ret(simple_positions, simple_returns, simple_benchmark):
    """ExposureInput with positions, returns, and benchmark."""
    return ExposureInput(
        positions=simple_positions,
        returns=simple_returns,
        benchmark=simple_benchmark,
        periods_per_year=252,
    )


# ---------------------------------------------------------------------------
# Helper computations for known-value tests
# ---------------------------------------------------------------------------


def _compute_ge(p):
    """Gross exposure at each period."""
    return np.sum(np.abs(p), axis=1)


def _compute_ne(p):
    """Net exposure at each period."""
    return np.sum(p, axis=1)


# ===================================================================
# §6.1  Gross Exposure
# ===================================================================


class TestGrossExposure:
    """Tests for gross_exposure metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "gross_exposure")
        ge = _compute_ge(inp_no_ret.positions)
        np.testing.assert_array_equal(
            result.value,
            np.array([ge[-1], np.max(ge), np.mean(ge)]),
        )
        assert result.meta["output_index"] == ["current", "max", "mean"]

    def test_all_zero_positions(self):
        positions = np.zeros((5, 3))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "gross_exposure")
        arr = result.value
        assert arr[0] == 0.0  # current
        assert arr[1] == 0.0  # max
        assert arr[2] == 0.0  # mean

    def test_single_asset(self):
        positions = np.array([[0.5], [0.6], [0.4]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "gross_exposure")
        arr = result.value
        assert arr[0] == 0.4  # current
        assert arr[1] == 0.6  # max
        np.testing.assert_allclose(arr[2], 0.5)  # mean

    def test_single_period(self):
        positions = np.array([[0.3, 0.5]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "gross_exposure")
        arr = result.value
        assert arr[0] == 0.8  # current
        assert arr[1] == 0.8  # max
        assert arr[2] == 0.8  # mean

    def test_nan_in_positions(self):
        positions = np.array([[0.3, np.nan], [0.5, 0.2]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "gross_exposure")
        arr = result.value
        # NaN is treated as 0 by nansum
        assert arr[0] == 0.7  # current: 0.5 + 0.2
        assert arr[1] == 0.7  # max
        np.testing.assert_allclose(arr[2], (0.3 + 0.7) / 2)  # mean

    def test_pandas_input(self, simple_positions):
        df = pd.DataFrame(simple_positions)
        inp = ExposureInput(positions=df)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "gross_exposure")
        ge = _compute_ge(simple_positions)
        np.testing.assert_array_equal(
            result.value,
            np.array([ge[-1], np.max(ge), np.mean(ge)]),
        )

    def test_polars_input(self, simple_positions):
        pl = pytest.importorskip("polars")
        df = pl.from_numpy(simple_positions)
        inp = ExposureInput(positions=df)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "gross_exposure")
        ge = _compute_ge(simple_positions)
        np.testing.assert_array_equal(
            result.value,
            np.array([ge[-1], np.max(ge), np.mean(ge)]),
        )


# ===================================================================
# §6.2  Net Exposure
# ===================================================================


class TestNetExposure:
    """Tests for net_exposure metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "net_exposure")
        ne = _compute_ne(inp_no_ret.positions)
        expected = np.array(
            [ne[-1], np.max(ne), np.min(ne), np.mean(ne),
             np.max(ne) - np.min(ne)]
        )
        np.testing.assert_array_equal(result.value, expected)
        assert result.meta["output_index"] == [
            "current", "max", "min", "mean", "range",
        ]

    def test_all_zero_positions(self):
        positions = np.zeros((5, 3))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "net_exposure")
        arr = result.value
        assert arr[0] == 0.0  # current
        assert arr[1] == 0.0  # max
        assert arr[2] == 0.0  # min
        assert arr[3] == 0.0  # mean
        assert arr[4] == 0.0  # range

    def test_negative_net_exposure(self):
        """Portfolio that stays net short throughout."""
        positions = np.array(
            [[-0.3, -0.2], [-0.4, -0.1], [-0.5, -0.2]]
        )
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "net_exposure")
        ne = _compute_ne(positions)
        assert result.value[0] == ne[-1]  # current
        assert result.value[1] == np.max(ne)  # max
        assert result.value[2] == np.min(ne)  # min

    def test_single_asset(self):
        positions = np.array([[0.5], [-0.3], [0.2]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "net_exposure")
        arr = result.value
        assert arr[0] == 0.2  # current
        assert arr[1] == 0.5  # max
        assert arr[2] == -0.3  # min


# ===================================================================
# §6.3  Leverage
# ===================================================================


class TestLeverage:
    """Tests for leverage metric."""

    def test_known_value(self, inp_with_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_with_ret, "leverage")
        ge = _compute_ge(inp_with_ret.positions)
        eq = inp_with_ret.equity
        assert eq is not None
        lev = ge / eq
        expected = np.nanmean(lev)
        np.testing.assert_allclose(result.value, expected)

    def test_no_equity_no_returns_raises(self, inp_no_ret):
        from stratstat.registry import _compute_one

        assert inp_no_ret.equity is None
        with pytest.raises(ValueError, match="leverage requires equity"):
            _compute_one(inp_no_ret, "leverage")

    def test_with_explicit_equity(self, simple_positions):
        equity = np.array(
            [1.0, 1.01, 0.99, 1.02, 1.05, 1.03, 1.06, 1.08, 1.07, 1.10]
        )
        inp = ExposureInput(positions=simple_positions, equity=equity)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "leverage")
        ge = _compute_ge(simple_positions)
        lev = ge / equity
        expected = np.nanmean(lev)
        np.testing.assert_allclose(result.value, expected)

    def test_zero_equity_period(self):
        """Periods with zero equity should produce NaN in leverage."""
        positions = np.array([[0.5, 0.3], [0.6, 0.2]])
        equity = np.array([1.0, 0.0])
        inp = ExposureInput(positions=positions, equity=equity)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "leverage")
        # Period 0: 0.8/1.0=0.8; Period 1: 0.8/0.0=NaN; mean=0.8
        np.testing.assert_allclose(result.value, 0.8)


# ===================================================================
# §6.4  Long Exposure %
# ===================================================================


class TestLongExposurePct:
    """Tests for long_exposure_pct metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "long_exposure_pct")
        le = np.sum(
            np.where(inp_no_ret.positions > 0, inp_no_ret.positions, 0),
            axis=1,
        )
        np.testing.assert_allclose(result.value, np.nanmean(le))

    def test_all_short(self):
        positions = np.array([[-0.3, -0.2], [-0.1, -0.4]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "long_exposure_pct")
        assert result.value == 0.0

    def test_all_long(self):
        positions = np.array([[0.3, 0.2], [0.1, 0.4]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "long_exposure_pct")
        assert result.value == 0.5  # mean of [0.5, 0.5] = 0.5


# ===================================================================
# §6.5  Short Exposure %
# ===================================================================


class TestShortExposurePct:
    """Tests for short_exposure_pct metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "short_exposure_pct")
        se = np.sum(
            np.where(
                inp_no_ret.positions < 0,
                np.abs(inp_no_ret.positions),
                0,
            ),
            axis=1,
        )
        np.testing.assert_allclose(result.value, np.nanmean(se))

    def test_all_long(self):
        positions = np.array([[0.3, 0.2], [0.1, 0.4]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "short_exposure_pct")
        assert result.value == 0.0

    def test_all_short(self):
        positions = np.array([[-0.3, -0.2], [-0.1, -0.4]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "short_exposure_pct")
        assert result.value == 0.5  # mean of [0.5, 0.5] = 0.5


# ===================================================================
# §6.6  Long-Book Return
# ===================================================================


class TestLongBookReturn:
    """Tests for long_book_return metric."""

    def test_known_value(self, inp_with_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_with_ret, "long_book_return")
        assert isinstance(result.value, float)
        assert np.isfinite(result.value)

    def test_independent(self):
        """Independently compute long-book return to verify."""
        positions = np.array([[0.5, 0.3], [0.6, 0.2], [0.4, 0.1]])
        returns = np.array([[0.01, -0.02], [0.02, 0.01], [-0.01, 0.03]])
        inp = ExposureInput(positions=positions, returns=returns)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "long_book_return")
        # t=0: w_lag=[nan,nan], long_only=[0,0] -> contrib=0 (nansum of zeros)
        # t=1: w_lag=[0.5,0.3] -> 0.5*0.02 + 0.3*0.01 = 0.013
        # t=2: w_lag=[0.6,0.2] -> 0.6*(-0.01) + 0.2*0.03 = 0.0
        # mean = (0 + 0.013 + 0.0) / 3 = 0.013 / 3
        np.testing.assert_allclose(result.value, 0.013 / 3)

    def test_no_returns_raises(self, inp_no_ret):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="long_book_return requires"):
            _compute_one(inp_no_ret, "long_book_return")

    def test_all_short(self):
        """When all positions are short, long-book return = 0."""
        positions = np.array([[-0.5, -0.3], [-0.6, -0.2]])
        returns = np.array([[0.01, 0.02], [0.02, -0.01]])
        inp = ExposureInput(positions=positions, returns=returns)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "long_book_return")
        assert result.value == 0.0


# ===================================================================
# §6.7  Short-Book Return
# ===================================================================


class TestShortBookReturn:
    """Tests for short_book_return metric."""

    def test_known_value(self, inp_with_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_with_ret, "short_book_return")
        assert isinstance(result.value, float)
        assert np.isfinite(result.value)

    def test_independent(self):
        """Independently compute short-book return to verify."""
        positions = np.array([[0.5, -0.3], [0.6, -0.2], [0.4, -0.1]])
        returns = np.array([[0.01, -0.02], [0.02, 0.01], [-0.01, 0.03]])
        inp = ExposureInput(positions=positions, returns=returns)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "short_book_return")
        # t=0: w_lag=[nan,nan], short_only=[0,0] -> contrib=0 (nansum of zeros)
        # t=1: short_only=[0,-0.3] -> -0.3*0.01 = -0.003
        # t=2: short_only=[0,-0.2] -> -0.2*0.03 = -0.006
        # mean = (0 - 0.003 - 0.006) / 3 = -0.003
        np.testing.assert_allclose(result.value, (-0.003 - 0.006) / 3)

    def test_no_returns_raises(self, inp_no_ret):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="short_book_return requires"):
            _compute_one(inp_no_ret, "short_book_return")

    def test_all_long(self):
        """When all positions are long, short-book return = 0."""
        positions = np.array([[0.5, 0.3], [0.6, 0.2]])
        returns = np.array([[0.01, 0.02], [0.02, -0.01]])
        inp = ExposureInput(positions=positions, returns=returns)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "short_book_return")
        assert result.value == 0.0


# ===================================================================
# §6.8  Long Beta
# ===================================================================


class TestLongBeta:
    """Tests for long_beta metric."""

    def test_known_value(self, inp_with_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_with_ret, "long_beta")
        assert isinstance(result.value, float)

    def test_independent(self):
        """Verify long beta against a known proportional relationship.

        Single long asset, weight = 1, asset return = 1.5 * benchmark.
        Then r_long ≈ 1.5 * benchmark for t >= 1, so beta ≈ 1.5.
        n=50 dilutes the t=0 zero pad to ~2 %.
        """
        n = 50
        rng = np.random.default_rng(77)
        benchmark = rng.normal(0.0, 0.01, size=n)
        positions = np.ones((n, 1))
        returns = 1.5 * benchmark.reshape(-1, 1)

        inp = ExposureInput(
            positions=positions, returns=returns, benchmark=benchmark,
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "long_beta")
        # Beta should be close to 1.5 (t=0 zero pad dilutes by ≈1/n).
        assert abs(float(result.value) - 1.5) < 0.03

    def test_no_returns_raises(self, inp_no_ret):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="long_beta requires asset-level"):
            _compute_one(inp_no_ret, "long_beta")

    def test_no_benchmark_raises(self, simple_positions, simple_returns):
        inp = ExposureInput(positions=simple_positions, returns=simple_returns)
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="long_beta requires benchmark"):
            _compute_one(inp, "long_beta")

    def test_too_few_periods(self):
        """With < 3 valid periods, beta returns NaN."""
        positions = np.array([[0.5, 0.3], [0.6, 0.2]])
        returns = np.array([[0.01, -0.02], [0.02, 0.01]])
        benchmark = np.array([0.005, 0.010])
        inp = ExposureInput(
            positions=positions, returns=returns, benchmark=benchmark,
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "long_beta")
        assert np.isnan(result.value)


# ===================================================================
# §6.9  Short Beta
# ===================================================================


class TestShortBeta:
    """Tests for short_beta metric."""

    def test_known_value(self, inp_with_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_with_ret, "short_beta")
        assert isinstance(result.value, float)

    def test_independent(self):
        """Verify short beta against a known proportional relationship.

        Construct a case where r_short ≈ 2 * benchmark for t >= 1
        (single short asset, weight = -1, asset return = -2*benchmark).
        t=0 gets a zero pad (no prior weights), so with n=50 periods
        the beta should be very close to 2.0 — the zero pad contributes
        at most 2% mis-estimation.  This is a genuine known-value test:
        the expected value is derived from the data-generating process,
        not from reimplementing the implementation's internal steps.
        """
        n = 50
        rng = np.random.default_rng(99)
        benchmark = rng.normal(0.0, 0.01, size=n)
        positions = -np.ones((n, 1))
        returns = -2.0 * benchmark.reshape(-1, 1)

        inp = ExposureInput(
            positions=positions, returns=returns, benchmark=benchmark,
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "short_beta")
        # Beta should be very close to 2.0 (the t=0 zero pad dilutes
        # it by at most 1/n = 2 %).
        assert abs(float(result.value) - 2.0) < 0.04

    def test_no_returns_raises(self, inp_no_ret):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="short_beta requires asset-level"):
            _compute_one(inp_no_ret, "short_beta")

    def test_no_benchmark_raises(self, simple_positions, simple_returns):
        inp = ExposureInput(positions=simple_positions, returns=simple_returns)
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="short_beta requires benchmark"):
            _compute_one(inp, "short_beta")

    def test_all_long_short_beta_zero(self):
        """When all positions are long, short-book return = 0, beta = 0.

        The short-book return is all zeros, which has zero covariance
        with the benchmark, giving beta = 0 (not NaN — constant series
        has defined beta of zero).
        """
        positions = np.array([[0.5, 0.3], [0.6, 0.2], [0.4, 0.1]])
        returns = np.array([[0.01, -0.02], [0.02, 0.01], [-0.01, 0.03]])
        benchmark = np.array([0.005, 0.010, -0.005])
        inp = ExposureInput(
            positions=positions, returns=returns, benchmark=benchmark,
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "short_beta")
        np.testing.assert_allclose(result.value, 0.0, atol=1e-10)


# ===================================================================
# §6.10  Position Concentration (HHI)
# ===================================================================


class TestPositionConcentration:
    """Tests for position_concentration metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "position_concentration")
        positions = inp_no_ret.positions
        ge = np.sum(np.abs(positions), axis=1)
        hhi_t = np.sum((positions / ge[:, None]) ** 2, axis=1)
        hhi_t = np.where(ge > 0, hhi_t, np.nan)
        np.testing.assert_allclose(result.value, np.nanmean(hhi_t))

    def test_single_asset(self):
        """Single asset always has HHI = 1.0."""
        positions = np.array([[0.5], [0.6], [0.4]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "position_concentration")
        np.testing.assert_allclose(result.value, 1.0)

    def test_equal_weights(self):
        """N assets with equal weight -> HHI = 1/N."""
        n = 4
        positions = np.tile(np.array([0.25] * n), (5, 1))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "position_concentration")
        np.testing.assert_allclose(result.value, 1.0 / n)

    def test_zero_exposure_period(self):
        """Period with all zeros should not affect the mean HHI."""
        positions = np.array([[0.5, 0.5], [0.0, 0.0], [0.3, 0.7]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "position_concentration")
        # t=0: [0.5,0.5], HHI=0.5^2+0.5^2=0.5
        # t=1: GE=0, excluded (NaN)
        # t=2: [0.3,0.7], HHI=0.09+0.49=0.58
        # mean = (0.5+0.58)/2 = 0.54
        np.testing.assert_allclose(result.value, (0.5 + 0.58) / 2.0)


# ===================================================================
# §6.11  Effective N Positions
# ===================================================================


class TestEffectiveNPositions:
    """Tests for effective_n_positions metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "effective_n_positions")
        positions = inp_no_ret.positions
        ge = np.sum(np.abs(positions), axis=1)
        norm = np.where(ge[:, None] > 0, positions / ge[:, None], 0)
        hhi = np.sum(norm**2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            n_eff = np.where(hhi > 0, 1.0 / hhi, np.nan)
        n_eff = np.where(ge > 0, n_eff, np.nan)
        np.testing.assert_allclose(result.value, np.nanmean(n_eff))

    def test_single_asset(self):
        """Single asset -> effective N = 1."""
        positions = np.array([[0.5], [0.6], [0.4]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "effective_n_positions")
        np.testing.assert_allclose(result.value, 1.0)

    def test_equal_weights(self):
        """N assets with equal weight -> effective N = N."""
        n = 4
        positions = np.tile(np.array([0.25] * n), (5, 1))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "effective_n_positions")
        np.testing.assert_allclose(result.value, float(n))


# ===================================================================
# §6.12  Turnover
# ===================================================================


class TestTurnover:
    """Tests for turnover metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "turnover")
        positions = inp_no_ret.positions
        delta = np.diff(positions, axis=0)
        to_t = 0.5 * np.sum(np.abs(delta), axis=1)
        expected = np.mean(to_t) * 252.0
        np.testing.assert_allclose(result.value, expected)

    def test_no_turnover(self):
        """Static weights produce zero turnover."""
        positions = np.tile(np.array([0.3, 0.4, 0.3]), (10, 1))
        inp = ExposureInput(positions=positions, periods_per_year=252)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "turnover")
        np.testing.assert_allclose(result.value, 0.0, atol=1e-10)

    def test_full_turnover(self):
        """Selling everything and buying new -> 100% turnover per period."""
        positions = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        inp = ExposureInput(positions=positions, periods_per_year=252)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "turnover")
        # delta: t0->t1: |(-1,1)|=2, TO=1.0; t1->t2: |(1,-1)|=2, TO=1.0
        # mean TO=1.0, annualized=252
        np.testing.assert_allclose(result.value, 252.0)

    def test_single_period(self):
        """Single period -> cannot compute delta, returns NaN."""
        positions = np.array([[0.5, 0.5]])
        inp = ExposureInput(positions=positions, periods_per_year=252)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "turnover")
        assert np.isnan(result.value)

    def test_no_periods_per_year_raises(self, simple_positions):
        inp = ExposureInput(positions=simple_positions)
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="turnover requires periods_per_year"):
            _compute_one(inp, "turnover")

    def test_monthly_turnover(self):
        """Test with monthly periods_per_year."""
        positions = np.array([[0.5, 0.5], [0.3, 0.7], [0.6, 0.4]])
        inp = ExposureInput(positions=positions, periods_per_year=12)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "turnover")
        delta = np.diff(positions, axis=0)
        to_t = 0.5 * np.sum(np.abs(delta), axis=1)
        expected = np.mean(to_t) * 12.0
        np.testing.assert_allclose(result.value, expected)


# ===================================================================
# §6.13  Average Holding Weight
# ===================================================================


class TestAvgHoldingWeight:
    """Tests for avg_holding_weight metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "avg_holding_weight")
        positions = inp_no_ret.positions
        abs_pos = np.abs(positions)
        n_active = np.sum(~np.isclose(abs_pos, 0.0), axis=1)
        sum_abs = np.sum(abs_pos, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_w_t = np.where(n_active > 0, sum_abs / n_active, np.nan)
        expected = np.nanmean(avg_w_t)
        np.testing.assert_allclose(result.value, expected)

    def test_all_equal_weights(self):
        positions = np.array([[0.2] * 5, [0.2] * 5, [0.2] * 5])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "avg_holding_weight")
        np.testing.assert_allclose(result.value, 0.2)

    def test_with_zero_positions(self):
        """Periods with some zero positions should not affect the mean."""
        positions = np.array(
            [[0.5, 0.5, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "avg_holding_weight")
        # t=0: (0.5+0.5)/2=0.5; t=1: 0.3/1=0.3; t=2: NaN
        # mean = (0.5+0.3)/2 = 0.4
        np.testing.assert_allclose(result.value, 0.4)

    def test_all_zero_period(self):
        """When all periods have zero positions, returns NaN."""
        positions = np.zeros((3, 2))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "avg_holding_weight")
        assert np.isnan(result.value)


# ===================================================================
# §6.14  Position Coverage
# ===================================================================


class TestPositionCoverage:
    """Tests for position_coverage metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "position_coverage")
        positions = inp_no_ret.positions
        has_position = np.any(~np.isclose(np.abs(positions), 0.0), axis=1)
        expected = np.sum(has_position) / inp_no_ret.n_periods
        np.testing.assert_allclose(result.value, expected)

    def test_always_covered(self):
        positions = np.ones((5, 3))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "position_coverage")
        assert result.value == 1.0

    def test_never_covered(self):
        positions = np.zeros((5, 3))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "position_coverage")
        assert result.value == 0.0

    def test_partial_coverage(self):
        positions = np.array([[0.5, 0.0], [0.0, 0.0], [0.3, 0.0]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "position_coverage")
        # Periods 0 and 2 have positions -> 2/3
        np.testing.assert_allclose(result.value, 2.0 / 3.0)

    def test_empty_input(self):
        positions = np.empty((0, 3))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "position_coverage")
        assert np.isnan(result.value)


# ===================================================================
# §6.15  Long/Short Position Coverage
# ===================================================================


class TestLongPositionCoverage:
    """Tests for long_position_coverage metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "long_position_coverage")
        positions = inp_no_ret.positions
        has_long = np.any(positions > 0.0, axis=1)
        expected = np.sum(has_long) / inp_no_ret.n_periods
        np.testing.assert_allclose(result.value, expected)

    def test_all_long(self):
        positions = np.ones((5, 3))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "long_position_coverage")
        assert result.value == 1.0

    def test_all_short(self):
        positions = -np.ones((5, 3))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "long_position_coverage")
        assert result.value == 0.0


class TestShortPositionCoverage:
    """Tests for short_position_coverage metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "short_position_coverage")
        positions = inp_no_ret.positions
        has_short = np.any(positions < 0.0, axis=1)
        expected = np.sum(has_short) / inp_no_ret.n_periods
        np.testing.assert_allclose(result.value, expected)

    def test_all_short(self):
        positions = -np.ones((5, 3))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "short_position_coverage")
        assert result.value == 1.0

    def test_all_long(self):
        positions = np.ones((5, 3))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "short_position_coverage")
        assert result.value == 0.0


# ===================================================================
# §6.16  Exposure Volatility
# ===================================================================


class TestExposureVolatility:
    """Tests for exposure_volatility metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "exposure_volatility")
        ge = _compute_ge(inp_no_ret.positions)
        expected = np.nanstd(ge, ddof=1)
        np.testing.assert_allclose(result.value, expected)

    def test_constant_exposure(self):
        positions = np.tile(np.array([0.3, 0.4, 0.3]), (10, 1))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "exposure_volatility")
        assert result.value == 0.0

    def test_single_period(self):
        positions = np.array([[0.5, 0.3]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "exposure_volatility")
        assert np.isnan(result.value)  # std of single value with ddof=1


# ===================================================================
# §6.17  Net Exposure Volatility
# ===================================================================


class TestNetExposureVolatility:
    """Tests for net_exposure_volatility metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "net_exposure_volatility")
        ne = _compute_ne(inp_no_ret.positions)
        expected = np.nanstd(ne, ddof=1)
        np.testing.assert_allclose(result.value, expected)

    def test_constant_net_exposure(self):
        # Each row sums to 1.0, so net exposure is identically 1.0
        # and its volatility is zero.
        positions = np.array([[0.5, 0.5], [0.3, 0.7], [0.6, 0.4]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "net_exposure_volatility")
        assert result.value == 0.0

    def test_varying_net_exposure(self):
        """Net exposure changes across periods → nonzero volatility."""
        positions = np.array([[0.5, 0.5], [0.3, 0.3], [0.7, 0.1]])
        # NE = [1.0, 0.6, 0.8]; std(ddof=1) = std([1, 0.6, 0.8], ddof=1)
        # = sqrt(((1-0.8)^2 + (0.6-0.8)^2 + (0.8-0.8)^2) / 2)
        # = sqrt((0.04 + 0.04 + 0) / 2) = sqrt(0.04) = 0.2
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "net_exposure_volatility")
        np.testing.assert_allclose(result.value, 0.2)


# ===================================================================
# §6.18  Exposure CV
# ===================================================================


class TestExposureCV:
    """Tests for exposure_cv metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "exposure_cv")
        ge = _compute_ge(inp_no_ret.positions)
        expected = np.nanstd(ge, ddof=1) / np.abs(np.nanmean(ge))
        np.testing.assert_allclose(result.value, expected)

    def test_zero_mean_exposure(self):
        """When mean exposure is zero, CV is inf or NaN."""
        positions = np.zeros((5, 3))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "exposure_cv")
        assert np.isnan(result.value) or np.isinf(result.value)


# ===================================================================
# §6.19  Exposure Utilization
# ===================================================================


class TestExposureUtilization:
    """Tests for exposure_utilization metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "exposure_utilization")
        ge = _compute_ge(inp_no_ret.positions)
        expected = np.nanmean(ge) / np.nanmax(ge)
        np.testing.assert_allclose(result.value, expected)

    def test_constant_exposure(self):
        positions = np.tile(np.array([0.3, 0.4, 0.3]), (5, 1))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "exposure_utilization")
        np.testing.assert_allclose(result.value, 1.0)

    def test_perfect_utilization(self):
        """When max == mean, utilization = 1.0."""
        positions = np.array([[0.5, 0.5], [0.5, 0.5]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "exposure_utilization")
        assert result.value == 1.0


# ===================================================================
# §6.20  Exposure Directional Bias
# ===================================================================


class TestExposureDirectionalBias:
    """Tests for exposure_directional_bias metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "exposure_directional_bias")
        ne = _compute_ne(inp_no_ret.positions)
        ge = _compute_ge(inp_no_ret.positions)
        expected = np.abs(np.nanmean(ne)) / np.nanmean(ge)
        np.testing.assert_allclose(result.value, expected)

    def test_pure_long(self):
        """Long-only portfolio -> bias = 1.0."""
        positions = np.array([[0.3, 0.3, 0.4], [0.2, 0.5, 0.3]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "exposure_directional_bias")
        np.testing.assert_allclose(result.value, 1.0)

    def test_market_neutral(self):
        """Perfectly market-neutral -> bias = 0.0."""
        positions = np.array([[0.5, -0.5], [0.3, -0.3], [0.4, -0.4]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "exposure_directional_bias")
        np.testing.assert_allclose(result.value, 0.0, atol=1e-10)


# ===================================================================
# §6.21  Exposure Percentiles
# ===================================================================


class TestExposurePercentiles:
    """Tests for exposure_percentiles metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "exposure_percentiles")
        ge = _compute_ge(inp_no_ret.positions)
        # Implementation uses np.isfinite (includes zeros, excludes NaN/inf).
        expected = np.percentile(ge[np.isfinite(ge)], [25, 50, 75, 90, 95])
        np.testing.assert_allclose(result.value, expected)
        assert result.meta["output_index"] == ["p25", "p50", "p75", "p90", "p95"]

    def test_single_value(self):
        positions = np.ones((1, 3))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "exposure_percentiles")
        # All percentiles equal the single value (3.0)
        np.testing.assert_allclose(result.value, 3.0)

    def test_all_zero(self):
        positions = np.zeros((5, 3))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "exposure_percentiles")
        # All percentiles = 0
        np.testing.assert_allclose(result.value, 0.0)

    def test_monotonic(self, inp_no_ret):
        """Percentiles should be monotonically increasing."""
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "exposure_percentiles")
        arr = result.value
        for i in range(len(arr) - 1):
            assert arr[i] <= arr[i + 1]


# ===================================================================
# §6.22  Period Counts
# ===================================================================


class TestPeriodCounts:
    """Tests for period_counts metric."""

    def test_known_value(self, inp_no_ret):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_no_ret, "period_counts")
        assert result.meta["output_index"] == [
            "total", "position", "long", "short", "idle",
        ]
        arr = result.value
        assert arr[0] == inp_no_ret.n_periods
        assert arr[1] + arr[4] == arr[0]  # position + idle = total

    def test_long_only(self):
        positions = np.array([[0.3, 0.2], [0.1, 0.4], [0.5, 0.0]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "period_counts")
        arr = result.value
        assert arr[0] == 3
        assert arr[1] == 3
        assert arr[2] == 3
        assert arr[3] == 0
        assert arr[4] == 0

    def test_short_only(self):
        positions = np.array([[-0.3, -0.2], [-0.1, -0.4]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "period_counts")
        arr = result.value
        assert arr[2] == 0  # long
        assert arr[3] == 2  # short

    def test_idle(self):
        positions = np.array([[0.0, 0.0], [0.0, 0.0]])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "period_counts")
        arr = result.value
        assert arr[1] == 0  # position
        assert arr[4] == 2  # idle

    def test_mixed(self):
        positions = np.array([
            [0.3, 0.2],     # long
            [-0.1, -0.4],   # short
            [0.5, -0.1],    # long+short
            [0.0, 0.0],     # idle
            [0.0, 0.2],     # long
        ])
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "period_counts")
        arr = result.value
        assert arr[0] == 5   # total
        assert arr[1] == 4   # position
        assert arr[2] == 2   # long
        assert arr[3] == 1   # short
        assert arr[4] == 1   # idle


# ===================================================================
# Empty and edge-case tests
# ===================================================================


class TestEmptyInput:
    """Edge case tests for exposure metrics."""

    def test_empty_positions(self):
        positions = np.empty((0, 3))
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        metrics = ["gross_exposure", "net_exposure", "position_concentration"]
        for name in metrics:
            result = _compute_one(inp, name)
            assert result is not None

    def test_single_asset_single_period(self):
        positions = np.array([[0.5]])
        inp = ExposureInput(positions=positions, periods_per_year=252)
        from stratstat.registry import _compute_one

        for name in [
            "gross_exposure",
            "net_exposure",
            "position_concentration",
            "effective_n_positions",
            "avg_holding_weight",
            "position_coverage",
            "exposure_volatility",
        ]:
            result = _compute_one(inp, name)
            assert result is not None, f"Failed on {name}"

    def test_all_nan_positions(self):
        positions = np.full((5, 3), np.nan)
        inp = ExposureInput(positions=positions)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "gross_exposure")
        # All NaN -> gross_exposure = 0 at each period
        assert result.value[0] == 0.0
        assert result.value[1] == 0.0
        assert result.value[2] == 0.0


# ===================================================================
# Input type tests
# ===================================================================


class TestInputTypes:
    """Tests for various input types."""

    def test_pandas_dataframe(self, simple_positions):
        df = pd.DataFrame(simple_positions)
        inp = ExposureInput(positions=df, periods_per_year=252)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "gross_exposure")
        ge = _compute_ge(simple_positions)
        np.testing.assert_array_equal(
            result.value,
            np.array([ge[-1], np.max(ge), np.mean(ge)]),
        )

    def test_pandas_series(self, simple_positions):
        """Pandas Series is treated as a single asset."""
        series = pd.Series(simple_positions[:, 0])
        inp = ExposureInput(positions=series)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "gross_exposure")
        assert result is not None

    def test_polars_dataframe(self, simple_positions):
        pl = pytest.importorskip("polars")
        df = pl.from_numpy(simple_positions)
        inp = ExposureInput(positions=df)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "gross_exposure")
        ge = _compute_ge(simple_positions)
        np.testing.assert_array_equal(
            result.value,
            np.array([ge[-1], np.max(ge), np.mean(ge)]),
        )

    def test_polars_series(self, simple_positions):
        pl = pytest.importorskip("polars")
        series = pl.Series(simple_positions[:, 0])
        inp = ExposureInput(positions=series)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "gross_exposure")
        assert result is not None

    def test_numpy_1d(self, simple_positions):
        """1-D numpy array is treated as a single-asset portfolio."""
        arr_1d = simple_positions[:, 0]
        inp = ExposureInput(positions=arr_1d)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "gross_exposure")
        assert result is not None

    def test_3d_positions_raises(self):
        positions = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="must be 1-D or 2-D"):
            ExposureInput(positions=positions)

    def test_returns_shape_mismatch_raises(self, simple_positions):
        returns_wrong = np.ones((5, 3))  # wrong n_periods
        with pytest.raises(ValueError, match="must match positions"):
            ExposureInput(positions=simple_positions, returns=returns_wrong)

    def test_benchmark_length_mismatch_raises(self, simple_positions):
        benchmark_wrong = np.ones(5)  # wrong length
        with pytest.raises(ValueError, match="must match n_periods"):
            ExposureInput(
                positions=simple_positions, benchmark=benchmark_wrong,
            )

    def test_equity_length_mismatch_raises(self, simple_positions):
        equity_wrong = np.ones(5)
        with pytest.raises(ValueError, match="must match n_periods"):
            ExposureInput(positions=simple_positions, equity=equity_wrong)


# ===================================================================
# Registry integration
# ===================================================================


class TestRegistry:
    """Registry integration tests."""

    def test_all_exposure_metrics_registered(self):
        from stratstat.registry import list_metrics

        metrics = list_metrics(requires="exposure")
        names = {m["name"] for m in metrics}
        expected = {
            "gross_exposure",
            "net_exposure",
            "leverage",
            "long_exposure_pct",
            "short_exposure_pct",
            "long_book_return",
            "short_book_return",
            "long_beta",
            "short_beta",
            "position_concentration",
            "effective_n_positions",
            "turnover",
            "avg_holding_weight",
            "position_coverage",
            "long_position_coverage",
            "short_position_coverage",
            "exposure_volatility",
            "net_exposure_volatility",
            "exposure_cv",
            "exposure_utilization",
            "exposure_directional_bias",
            "exposure_percentiles",
            "period_counts",
            "active_share",
        }
        assert names == expected
        assert len(metrics) == 24

    def test_compute_single(self, inp_no_ret):
        from stratstat import compute

        result = compute(inp_no_ret, "gross_exposure")
        assert result.name == "gross_exposure"
        assert result.category == ("exposure",)

    def test_compute_all_exposure_category(self, inp_with_ret):
        from stratstat import compute_all

        results = compute_all(inp_with_ret, category="exposure")
        # Primary "exposure" category metrics (12 of them)
        assert len(results) >= 12

    def test_compute_all_concentration(self, inp_no_ret):
        from stratstat import compute_all

        results = compute_all(inp_no_ret, category="exposure")
        names = {r.name for r in results}
        assert {"position_concentration", "effective_n_positions"} <= names

    def test_compute_all_turnover(self, inp_no_ret):
        from stratstat import compute_all

        results = compute_all(inp_no_ret, category="exposure")
        names = {r.name for r in results}
        assert "turnover" in names

    def test_raw_array_auto_wraps(self):
        """Passing a raw 2D array auto-wraps to ExposureInput."""
        from stratstat.registry import _compute_one

        positions = np.array([[0.3, 0.5], [0.6, 0.2]])
        result = _compute_one(positions, "gross_exposure")
        assert result.name == "gross_exposure"
        ge = _compute_ge(positions)
        np.testing.assert_array_equal(
            result.value,
            np.array([ge[-1], np.max(ge), np.mean(ge)]),
        )


# ===================================================================
# ExposureInput feature tests
# ===================================================================


class TestExposureInputFeatures:
    """Tests for ExposureInput features and convenience properties."""

    def test_auto_equity_from_positions_returns(
        self, simple_positions, simple_returns,
    ):
        inp = ExposureInput(positions=simple_positions, returns=simple_returns)
        assert inp.has_equity
        assert inp.equity is not None
        assert inp.equity.shape == (inp.n_periods,)

    def test_no_equity_without_returns(self, simple_positions):
        inp = ExposureInput(positions=simple_positions)
        assert not inp.has_equity
        assert inp.equity is None

    def test_explicit_equity_overrides(self, simple_positions, simple_returns):
        explicit_eq = np.ones(10) * 100.0
        inp = ExposureInput(
            positions=simple_positions,
            returns=simple_returns,
            equity=explicit_eq,
        )
        np.testing.assert_array_equal(inp.equity, explicit_eq)

    def test_has_returns(self, inp_no_ret, inp_with_ret):
        assert not inp_no_ret.has_returns
        assert inp_with_ret.has_returns

    def test_has_benchmark(self, inp_no_ret, inp_with_ret):
        assert not inp_no_ret.has_benchmark
        assert inp_with_ret.has_benchmark

    def test_repr(self, inp_no_ret):
        r = repr(inp_no_ret)
        assert "ExposureInput" in r
        assert "n_periods" in r
        assert "n_assets" in r

    def test_n_assets(self, simple_positions):
        inp = ExposureInput(positions=simple_positions)
        assert inp.n_assets == simple_positions.shape[1]

    def test_n_periods(self, simple_positions):
        inp = ExposureInput(positions=simple_positions)
        assert inp.n_periods == simple_positions.shape[0]

    def test_returns_1d_broadcasts(self):
        """1-D returns array for single-asset should work."""
        positions = np.array([[0.5], [0.6], [0.4]])
        returns = np.array([0.01, 0.02, -0.01])
        inp = ExposureInput(positions=positions, returns=returns)
        assert inp.has_returns
        assert inp.returns.shape == (3, 1)

    def test_benchmark_weights_1d(self):
        """1-D benchmark weights should be stored correctly."""
        positions = np.array([[0.3, 0.3, 0.4], [0.25, 0.35, 0.4]])
        bw = np.array([0.4, 0.3, 0.3])
        inp = ExposureInput(positions=positions, benchmark_weights=bw)
        assert inp.has_benchmark_weights
        assert inp.benchmark_weights.shape == (3,)

    def test_benchmark_weights_2d(self):
        """2-D benchmark weights should be stored correctly."""
        positions = np.array([[0.3, 0.3, 0.4], [0.25, 0.35, 0.4]])
        bw = np.array([[0.4, 0.3, 0.3], [0.35, 0.35, 0.3]])
        inp = ExposureInput(positions=positions, benchmark_weights=bw)
        assert inp.has_benchmark_weights
        assert inp.benchmark_weights.shape == (2, 3)

    def test_benchmark_weights_wrong_shape(self):
        """Mismatched benchmark_weights shape should raise."""
        positions = np.array([[0.3, 0.3, 0.4]])
        bw = np.array([0.4, 0.3])  # wrong length
        with pytest.raises(ValueError, match="benchmark_weights"):
            ExposureInput(positions=positions, benchmark_weights=bw)


# ---------------------------------------------------------------------------
# Active Share
# ---------------------------------------------------------------------------


class TestActiveShare:
    """Tests for active_share metric."""

    def test_perfect_match(self):
        """Active Share should be 0 when portfolio matches benchmark."""
        positions = np.array([[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]])
        bw = np.array([0.4, 0.3, 0.3])
        inp = ExposureInput(positions=positions, benchmark_weights=bw)
        result = active_share(inp)
        assert result.value == pytest.approx(0.0, abs=1e-10)

    def test_no_overlap(self):
        """Active Share should be 1.0 when there is zero overlap."""
        positions = np.array([[1.0, 0.0], [1.0, 0.0]])
        bw = np.array([0.0, 1.0])
        inp = ExposureInput(positions=positions, benchmark_weights=bw)
        result = active_share(inp)
        assert result.value == pytest.approx(1.0, abs=1e-10)

    def test_partial_overlap(self):
        """Active Share for a known partial-overlap case."""
        # Portfolio: [0.6, 0.4], Benchmark: [0.5, 0.5]
        # AS = 0.5 * (|0.6-0.5| + |0.4-0.5|) = 0.5 * (0.1 + 0.1) = 0.1
        positions = np.array([[0.6, 0.4]])
        bw = np.array([0.5, 0.5])
        inp = ExposureInput(positions=positions, benchmark_weights=bw)
        result = active_share(inp)
        assert result.value == pytest.approx(0.1, abs=1e-10)

    def test_time_varying_benchmark(self):
        """Active Share with time-varying benchmark weights."""
        positions = np.array([
            [0.6, 0.4],
            [0.5, 0.5],
            [0.3, 0.7],
        ])
        bw = np.array([
            [0.5, 0.5],
            [0.6, 0.4],
            [0.4, 0.6],
        ])
        inp = ExposureInput(positions=positions, benchmark_weights=bw)
        result = active_share(inp)
        assert 0.0 <= result.value <= 1.0
        assert "series" in result.meta

    def test_series_in_meta(self):
        """Per-period active share series should be stored in meta."""
        positions = np.array([[0.6, 0.4], [0.3, 0.7]])
        bw = np.array([0.5, 0.5])
        inp = ExposureInput(positions=positions, benchmark_weights=bw)
        result = active_share(inp)

        series = result.meta["series"]
        assert series.shape == (2,)
        # First period: 0.5 * (|0.6-0.5| + |0.4-0.5|) = 0.5 * 0.2 = 0.1
        assert series[0] == pytest.approx(0.1, abs=1e-10)
        # Second period: 0.5 * (|0.3-0.5| + |0.7-0.5|) = 0.5 * 0.4 = 0.2
        assert series[1] == pytest.approx(0.2, abs=1e-10)
        # Mean: (0.1 + 0.2) / 2 = 0.15
        assert result.value == pytest.approx(0.15, abs=1e-10)

    def test_requires_benchmark_weights(self):
        """Active Share should raise when benchmark_weights is missing."""
        positions = np.array([[0.6, 0.4]])
        inp = ExposureInput(positions=positions)
        with pytest.raises(ValueError, match="benchmark_weights"):
            active_share(inp)
