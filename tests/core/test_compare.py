"""Tests for compare-tier metrics.

Covers all 7 registered compare metrics, edge cases, input types,
and registry integration.
"""

from __future__ import annotations

import numpy as np
import pytest

# Import triggers @register_metric decorators
import stratstat.core.compare  # noqa: F401
from stratstat.inputs import CompareInput

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_returns():
    """Three-strategy returns — 252 periods with known properties."""
    rng = np.random.default_rng(42)
    n = 252
    # Strategy 1: moderate positive returns
    s1 = rng.normal(0.001, 0.02, size=n)
    # Strategy 2: slightly lower returns, correlated with s1
    s2 = 0.7 * s1 + rng.normal(0.0005, 0.015, size=n)
    # Strategy 3: uncorrelated, higher vol
    s3 = rng.normal(0.0008, 0.03, size=n)
    return np.column_stack([s1, s2, s3])


@pytest.fixture
def two_strategy_returns():
    """Two uncorrelated strategies for pairwise tests."""
    rng = np.random.default_rng(99)
    n = 500
    s1 = rng.normal(0.001, 0.02, size=n)
    s2 = rng.normal(0.0005, 0.025, size=n)
    return np.column_stack([s1, s2])


@pytest.fixture
def inp_basic(simple_returns):
    """CompareInput with 3 strategies."""
    return CompareInput(returns=simple_returns, periods_per_year=252)


@pytest.fixture
def inp_two(two_strategy_returns):
    """CompareInput with exactly 2 strategies."""
    return CompareInput(returns=two_strategy_returns, periods_per_year=252)


@pytest.fixture
def inp_with_benchmark(simple_returns):
    """CompareInput with benchmark returns for White's RC."""
    rng = np.random.default_rng(123)
    n_periods = simple_returns.shape[0]
    bench = rng.normal(0.0005, 0.015, size=n_periods)
    return CompareInput(
        returns=simple_returns,
        benchmark=bench,
        periods_per_year=252,
    )


@pytest.fixture
def inp_with_weights(simple_returns):
    """CompareInput with custom strategy weights."""
    return CompareInput(
        returns=simple_returns,
        weights=np.array([0.5, 0.3, 0.2]),
        periods_per_year=252,
    )


@pytest.fixture
def inp_empty():
    """CompareInput with minimal data (2 strategies, 3 periods)."""
    return CompareInput(
        returns=np.array(
            [
                [0.01, 0.005],
                [0.02, 0.01],
                [-0.01, 0.0],
            ]
        ),
        periods_per_year=252,
    )


# ===================================================================
# §9.1  Correlation Matrix
# ===================================================================


class TestCorrelationMatrix:
    def test_basic(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "correlation_matrix")
        corr = result.value
        assert corr.shape == (3, 3)
        # Diagonal should be near 1.0
        for i in range(3):
            assert np.abs(corr[i, i] - 1.0) < 1e-10
        # Symmetric
        assert np.allclose(corr, corr.T, equal_nan=True)
        # Off-diagonal values should be in [-1, 1]
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert -1.0 <= corr[i, j] <= 1.0

    def test_known_correlation(self):
        """Perfectly correlated strategies."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 0.01, size=100)
        y = 2.0 * x  # perfect positive correlation
        z = -x  # perfect negative correlation
        returns = np.column_stack([x, y, z])
        inp = CompareInput(returns=returns)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "correlation_matrix")
        corr = result.value
        # x-y: +1; x-z: -1; y-z: -1
        assert corr[0, 1] == pytest.approx(1.0, abs=1e-10)
        assert corr[0, 2] == pytest.approx(-1.0, abs=1e-10)
        assert corr[1, 2] == pytest.approx(-1.0, abs=1e-10)

    def test_requires_two_strategies(self):
        """Single strategy should raise ValueError."""
        inp = CompareInput(returns=np.random.randn(100, 1))
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="at least 2"):
            _compute_one(inp, "correlation_matrix")

    def test_nan_handling(self):
        """NaN in some periods should not break the result."""
        rng = np.random.default_rng(42)
        r = rng.normal(0, 0.01, size=(100, 3))
        r[0:10, 0] = np.nan  # first 10 periods of s1 are NaN
        inp = CompareInput(returns=r)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "correlation_matrix")
        corr = result.value
        assert corr.shape == (3, 3)
        # Diagonal should still be 1.0
        assert corr[1, 1] == pytest.approx(1.0)

    def test_registry_integration(self):
        """Test via top-level compute()."""
        from stratstat import compute

        rng = np.random.default_rng(42)
        r = rng.normal(0, 0.01, size=(100, 2))
        inp = CompareInput(returns=r)
        result = compute(inp, "correlation_matrix")
        assert result.name == "correlation_matrix"
        assert result.value.shape == (2, 2)


# ===================================================================
# §9.2  Diversification Ratio
# ===================================================================


class TestDiversificationRatio:
    def test_basic(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "diversification_ratio")
        # DR should be >= 1.0 for any diversification benefit
        assert result.value >= 1.0
        assert result.name == "diversification_ratio"

    def test_perfect_correlation(self):
        """With perfectly correlated strategies, DR ≈ 1.0."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 0.01, size=100)
        returns = np.column_stack([x, x, x])  # identical strategies
        inp = CompareInput(returns=returns)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "diversification_ratio")
        assert result.value == pytest.approx(1.0, abs=0.05)

    def test_low_correlation(self):
        """With uncorrelated strategies, DR > 1.0."""
        rng = np.random.default_rng(42)
        s1 = rng.normal(0, 0.02, size=500)
        s2 = rng.normal(0, 0.02, size=500)
        returns = np.column_stack([s1, s2])
        inp = CompareInput(returns=returns)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "diversification_ratio")
        # DR should be noticeably > 1 with uncorrelated assets
        assert result.value > 1.1

    def test_custom_weights(self, inp_with_weights):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_with_weights, "diversification_ratio")
        assert result.value >= 1.0
        assert result.meta["weights"] == [0.5, 0.3, 0.2]

    def test_requires_two_strategies(self):
        inp = CompareInput(returns=np.random.randn(100, 1))
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="at least 2"):
            _compute_one(inp, "diversification_ratio")


# ===================================================================
# §9.3  Pairwise Sharpe-Difference Test (Jobson-Korkie, Memmel)
# ===================================================================


class TestSharpeDifferenceTest:
    def test_basic(self, inp_two):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_two, "sharpe_difference_test")
        z, p_value, sr_diff = result.value
        assert isinstance(z, float)
        assert 0.0 <= p_value <= 1.0
        # sr_diff should match sign of z
        assert np.sign(sr_diff) == np.sign(z) or (sr_diff == 0 and z == 0)

    def test_identical_strategies(self):
        """When SRs are equal and rho=1, variance → 0 so z → NaN.

        Uses deterministic alternating returns — both series identical.
        """
        r = np.array(
            [[0.01], [-0.005], [0.02], [-0.01], [0.015], [0.01], [-0.005], [0.02], [-0.01], [0.015]]
        )
        returns = np.column_stack([r.ravel(), r.ravel()])
        inp = CompareInput(returns=returns)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "sharpe_difference_test")
        z, p_value, sr_diff = result.value
        # Identical strategies → SR1=SR2, rho=1 → variance=0 → z=NaN
        assert np.isnan(z)
        assert np.isnan(p_value)
        assert sr_diff == pytest.approx(0.0)

    def test_requires_exactly_two(self, inp_basic):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="exactly 2"):
            _compute_one(inp_basic, "sharpe_difference_test")

    def test_insufficient_data(self):
        """Fewer than 3 valid overlapping observations."""
        r = np.array([[0.01, 0.02], [np.nan, 0.01], [0.03, np.nan]])
        inp = CompareInput(returns=r)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "sharpe_difference_test")
        assert np.all(np.isnan(result.value))

    def test_rf_parameter(self, two_strategy_returns):
        """Higher rf should affect SR values and thus the test."""
        inp1 = CompareInput(returns=two_strategy_returns, rf=0.0)
        inp2 = CompareInput(returns=two_strategy_returns, rf=0.01)
        from stratstat.registry import _compute_one

        r1 = _compute_one(inp1, "sharpe_difference_test")
        r2 = _compute_one(inp2, "sharpe_difference_test")
        # Different rf should give different results
        assert r1.value[2] != r2.value[2]  # sr_diff should differ

    def test_meta_includes_details(self, inp_two):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_two, "sharpe_difference_test")
        assert "sr1" in result.meta
        assert "sr2" in result.meta
        assert "rho" in result.meta
        assert "T" in result.meta or "n_valid" in result.meta
        assert result.meta["output_index"] == ["z", "p_value", "sr_diff"]


# ===================================================================
# §9.4  White's Reality Check
# ===================================================================


class TestWhitesRealityCheck:
    def test_basic(self, inp_with_benchmark):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_with_benchmark, "whites_reality_check")
        stat, p_value = result.value
        assert isinstance(stat, float)
        assert 0.0 <= p_value <= 1.0

    def test_requires_benchmark(self, inp_basic):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="benchmark"):
            _compute_one(inp_basic, "whites_reality_check")

    def test_seed_reproducibility(self, inp_with_benchmark):
        from stratstat.registry import _compute_one

        r1 = _compute_one(inp_with_benchmark, "whites_reality_check", seed=42)
        r2 = _compute_one(inp_with_benchmark, "whites_reality_check", seed=42)
        assert r1.value[0] == r2.value[0]
        assert r1.value[1] == r2.value[1]

    def test_single_strategy(self):
        """White's RC works with a single strategy (min 1)."""
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.02, size=(100, 1))
        bench = rng.normal(0.0005, 0.015, size=100)
        inp = CompareInput(returns=r, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "whites_reality_check")
        stat, p_value = result.value
        assert isinstance(stat, float)

    def test_all_underperform(self):
        """When all strategies strictly underperform the benchmark,
        p_value should be large (fail to reject H0).

        Uses deterministic data: both strategies lag benchmark by a
        fixed amount every period, so the null is certainly true.
        """
        rng = np.random.default_rng(42)
        bench = rng.normal(0.0, 0.01, size=200)
        s1 = bench - 0.005  # always 50bp worse
        s2 = bench - 0.010  # always 100bp worse
        inp = CompareInput(
            returns=np.column_stack([s1, s2]),
            benchmark=bench,
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "whites_reality_check", n_boot=500, seed=1)
        stat, p_value = result.value
        # Both strategies are strictly worse than benchmark,
        # so the test statistic should be negative and p_value
        # should indicate no rejection of the null.
        assert stat < 0.0
        assert p_value > 0.5


# ===================================================================
# §9.5  PBO (Probability of Backtest Overfitting)
# ===================================================================


class TestPBO:
    def test_basic(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "pbo", n_splits=50, seed=42)
        pbo_val, n_used = result.value
        assert isinstance(pbo_val, float)
        assert n_used > 0  # should generate some valid splits

    def test_output_index(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "pbo", n_splits=20, seed=42)
        assert result.meta["output_index"] == ["pbo", "n_splits"]

    def test_seed_reproducibility(self, inp_basic):
        from stratstat.registry import _compute_one

        r1 = _compute_one(inp_basic, "pbo", n_splits=30, seed=42)
        r2 = _compute_one(inp_basic, "pbo", n_splits=30, seed=42)
        assert r1.value[0] == r2.value[0]

    def test_requires_two_strategies(self):
        inp = CompareInput(returns=np.random.randn(100, 1))
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="at least 2"):
            _compute_one(inp, "pbo")

    def test_purge_embargo_params(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(
            inp_basic, "pbo", n_splits=30, purge_pct=0.05, embargo_pct=0.02, seed=42
        )
        assert result.meta["purge_pct"] == 0.05
        assert result.meta["embargo_pct"] == 0.02

    def test_small_data(self):
        """Very small data should return NaN for pbo."""
        r = np.random.randn(20, 3)
        inp = CompareInput(returns=r)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "pbo", n_splits=10, seed=42)
        pbo_val, n_used = result.value
        # With 20 observations, splits are hard to generate
        # pbo may be NaN if no splits could be generated
        assert isinstance(pbo_val, float)


# ===================================================================
# §9.6  Marginal Contribution to Portfolio Risk
# ===================================================================


class TestMarginalContributionToRisk:
    def test_basic(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "marginal_contribution_to_risk")
        mcr = result.value
        assert mcr.shape == (3,)
        # MCR values should sum to portfolio vol
        assert "portfolio_vol" in result.meta

    def test_sum_to_portfolio_vol(self, inp_basic):
        """MCR_i values should approximately sum to portfolio vol."""
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "marginal_contribution_to_risk")
        mcr = result.value
        port_vol = result.meta["portfolio_vol"]
        assert np.sum(mcr) == pytest.approx(port_vol, rel=0.01)

    def test_custom_weights(self, inp_with_weights):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_with_weights, "marginal_contribution_to_risk")
        assert result.meta["weights"] == [0.5, 0.3, 0.2]

    def test_requires_two_strategies(self):
        inp = CompareInput(returns=np.random.randn(100, 1))
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="at least 2"):
            _compute_one(inp, "marginal_contribution_to_risk")

    def test_equal_weight_default(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "marginal_contribution_to_risk")
        assert result.meta["weights"] == [1.0 / 3, 1.0 / 3, 1.0 / 3]


# ===================================================================
# §9.7  Component VaR
# ===================================================================


class TestComponentVar:
    def test_basic(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "component_var")
        cvar = result.value
        assert cvar.shape == (3,)
        assert "total_var" in result.meta

    def test_custom_confidence(self, inp_basic):
        from stratstat.registry import _compute_one

        r1 = _compute_one(inp_basic, "component_var", confidence=0.95)
        r2 = _compute_one(inp_basic, "component_var", confidence=0.99)
        # Higher confidence → higher total VaR
        assert r2.meta["total_var"] >= r1.meta["total_var"]

    def test_requires_two_strategies(self):
        inp = CompareInput(returns=np.random.randn(100, 1))
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="at least 2"):
            _compute_one(inp, "component_var")

    def test_custom_weights(self, inp_with_weights):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_with_weights, "component_var")
        assert result.meta["weights"] == [0.5, 0.3, 0.2]

    def test_nan_handling(self):
        """NaN in returns should not crash component VaR."""
        rng = np.random.default_rng(42)
        r = rng.normal(0, 0.01, size=(100, 3))
        r[0:5, 0] = np.nan
        inp = CompareInput(returns=r)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "component_var")
        assert result.value.shape == (3,)


# ===================================================================
# CompareInput tests
# ===================================================================


class TestCompareInput:
    def test_basic_construction(self, simple_returns):
        inp = CompareInput(returns=simple_returns)
        assert inp.n_periods == 252
        assert inp.n_strategies == 3
        assert inp.periods_per_year is None
        assert inp.rf == 0.0

    def test_with_all_params(self, simple_returns):
        bench = np.random.randn(252)
        w = np.array([0.5, 0.3, 0.2])
        inp = CompareInput(
            returns=simple_returns,
            weights=w,
            benchmark=bench,
            periods_per_year=252,
            rf=0.02,
        )
        assert inp.periods_per_year == 252
        assert inp.rf == 0.02
        assert inp.has_benchmark is True
        assert inp.has_weights is True

    def test_default_weights(self, inp_basic):
        w = inp_basic.get_weights()
        assert w.shape == (3,)
        assert np.sum(w) == pytest.approx(1.0)
        assert np.allclose(w, [1.0 / 3, 1.0 / 3, 1.0 / 3])

    def test_weight_length_mismatch_raises(self, simple_returns):
        with pytest.raises(ValueError, match="Weights length"):
            CompareInput(returns=simple_returns, weights=np.array([0.5, 0.5]))

    def test_benchmark_length_mismatch_raises(self, simple_returns):
        with pytest.raises(ValueError, match="Benchmark length"):
            CompareInput(returns=simple_returns, benchmark=np.array([0.01]))

    def test_repr(self, inp_basic):
        r = repr(inp_basic)
        assert "CompareInput" in r
        assert "n_periods=252" in r
        assert "n_strategies=3" in r

    def test_pandas_input(self, simple_returns):
        """CompareInput should accept pandas DataFrame."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(simple_returns)
        inp = CompareInput(returns=df)
        assert inp.n_strategies == 3
        assert inp.n_periods == 252

    def test_1d_input_reshaped(self):
        """1-D input should be reshaped to (n, 1)."""
        r = np.random.randn(100)
        inp = CompareInput(returns=r)
        assert inp.n_strategies == 1
        assert inp.returns.shape == (100, 1)

    def test_3d_input_raises(self):
        with pytest.raises(ValueError, match="must be 1-D or 2-D"):
            CompareInput(returns=np.random.randn(10, 5, 3))


# ===================================================================
# Registry integration tests
# ===================================================================


class TestRegistryIntegration:
    def test_list_compare_metrics(self):
        from stratstat import list_metrics

        metrics = list_metrics(requires="compare")
        names = {m["name"] for m in metrics}
        assert "correlation_matrix" in names
        assert "diversification_ratio" in names
        assert "sharpe_difference_test" in names
        assert "whites_reality_check" in names
        assert "pbo" in names
        assert "marginal_contribution_to_risk" in names
        assert "component_var" in names
        assert len(metrics) == 7

    def test_list_compare_by_category(self):
        from stratstat import list_metrics

        metrics = list_metrics(category="relative")
        names = {m["name"] for m in metrics}
        assert "correlation_matrix" in names

    def test_auto_wrap_raw_array(self):
        """Passing a raw array to _compute_one auto-creates CompareInput."""
        from stratstat.registry import _compute_one

        rng = np.random.default_rng(42)
        r = rng.normal(0, 0.01, size=(100, 2))
        result = _compute_one(r, "correlation_matrix")
        assert result.name == "correlation_matrix"
        assert result.value.shape == (2, 2)

    def test_compute_all_compare(self):
        """compute_all with a 2-strategy input + benchmark runs the vectorized
        compare metrics; resampling metrics are excluded."""
        from stratstat import compute_all

        rng = np.random.default_rng(42)
        n = 252
        s1 = rng.normal(0.001, 0.02, size=n)
        s2 = rng.normal(0.0005, 0.025, size=n)
        bench = rng.normal(0.0003, 0.015, size=n)
        inp = CompareInput(
            returns=np.column_stack([s1, s2]),
            benchmark=bench,
            periods_per_year=252,
        )
        result_set = compute_all(compare=inp)
        # 5 vectorized compare metrics; the 2 resampling metrics are excluded.
        assert len(result_set) == 5
        assert set(result_set.meta.get("excluded_resampling", [])) == {
            "whites_reality_check",
            "pbo",
        }

    def test_compute_top_level(self, inp_basic):
        from stratstat import compute

        result = compute(inp_basic, "correlation_matrix")
        assert result.name == "correlation_matrix"


# ---------------------------------------------------------------------------
# Numba vs pure-numpy agreement
# ---------------------------------------------------------------------------


class TestNumbaAgreement:
    def test_stationary_bootstrap_agree(self):
        """Numba and pure-numpy stationary bootstrap means agree."""
        from stratstat.core.compare import _HAS_NUMBA

        if not _HAS_NUMBA:
            pytest.skip("numba not installed")

        from stratstat.core.compare import (
            _stationary_bootstrap_fallback,
            _stationary_bootstrap_numba,
        )

        rng = np.random.default_rng(11)
        data = rng.normal(0.0003, 0.008, size=(80, 2))
        data[3, 0] = np.nan

        n_boot, block_size, n_periods = 30, 1.0, data.shape[0]
        draw_rng = np.random.default_rng(13)
        starts = draw_rng.integers(0, n_periods, size=(n_boot, n_periods))
        blens = draw_rng.geometric(1.0 / block_size, size=(n_boot, n_periods))

        numba_means = _stationary_bootstrap_numba(data, starts, blens)
        pure_means = _stationary_bootstrap_fallback(data, starts, blens)
        assert np.allclose(numba_means, pure_means, rtol=1e-10, equal_nan=True)

    def test_pbo_overfit_agree(self):
        """Numba and pure-numpy PBO overfit counts agree."""
        from stratstat.core.compare import _HAS_NUMBA

        if not _HAS_NUMBA:
            pytest.skip("numba not installed")

        from stratstat.core.compare import (
            _pbo_overfit_fallback,
            _pbo_overfit_numba,
        )

        rng = np.random.default_rng(17)
        data = rng.normal(0.0002, 0.01, size=(120, 3))
        data[7, 1] = np.nan

        split_points = np.array([40, 50, 60, 70], dtype=np.int64)
        numba_count = _pbo_overfit_numba(data, split_points, purge=2, embargo=1, rf=0.0)
        pure_count = _pbo_overfit_fallback(data, split_points, purge=2, embargo=1, rf=0.0)
        assert numba_count == pure_count
