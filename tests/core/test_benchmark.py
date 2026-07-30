"""Tests for benchmark-tier metrics.

Covers all 18 registered benchmark metrics, edge cases, input types,
and registry integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Import triggers @register_metric decorators
import stratstat.core.benchmark  # noqa: F401
from stratstat.inputs import BenchmarkInput

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_benchmark():
    """Benchmark returns — 10 periods with known values."""
    return np.array(
        [0.01, 0.02, -0.01, 0.005, -0.02, 0.015, 0.03, -0.005, 0.01, -0.015],
        dtype=np.float64,
    )


@pytest.fixture
def simple_returns(simple_benchmark):
    """Strategy returns with beta=1.2 vs benchmark.

    r = 0.001 + 1.2 * r_m  (alpha_per_period = 0.001)
    """
    return 0.001 + 1.2 * simple_benchmark


@pytest.fixture
def inp_basic(simple_returns, simple_benchmark):
    """BenchmarkInput with returns, benchmark, and periods_per_year."""
    return BenchmarkInput(
        returns=simple_returns,
        benchmark=simple_benchmark,
        periods_per_year=252,
    )


@pytest.fixture
def inp_no_pp(simple_returns, simple_benchmark):
    """BenchmarkInput without periods_per_year."""
    return BenchmarkInput(returns=simple_returns, benchmark=simple_benchmark)


@pytest.fixture
def inp_empty():
    """BenchmarkInput with minimal data."""
    return BenchmarkInput(
        returns=np.array([0.01, 0.02, 0.03]),
        benchmark=np.array([0.005, 0.01, 0.015]),
        periods_per_year=252,
    )


# ===================================================================
# §8.1  Alpha (Jensen's)
# ===================================================================


class TestAlpha:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "alpha")
        # With r = 0.001 + 1.2 * r_m, beta = 1.2 (exact).
        # Hand-computed (see benchmark fixture):
        #   CAGR_r = exp(252 * mean(log(1+r))) - 1 ≈ 3.1181
        #   CAGR_m = exp(252 * mean(log(1+bench))) - 1 ≈ 1.6558
        #   alpha = CAGR_r - beta * CAGR_m ≈ 1.1312
        assert result.value == pytest.approx(1.131172, rel=0.02)
        assert result.meta["annualized"] is True

    def test_requires_periods_per_year(self, inp_no_pp):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="periods_per_year"):
            _compute_one(inp_no_pp, "alpha")

    def test_alpha_zero_when_beta_one_and_no_excess(self):
        # r = r_m → beta=1, no alpha
        rng = np.random.default_rng(42)
        bench = rng.normal(0.0, 0.01, size=100)
        rets = bench.copy()  # identical
        inp = BenchmarkInput(returns=rets, benchmark=bench, periods_per_year=252)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "alpha")
        assert abs(result.value) < 0.05  # close to zero


# ===================================================================
# §8.2  Beta
# ===================================================================


class TestBeta:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "beta")
        assert result.value == pytest.approx(1.2, rel=0.05)
        assert result.meta["variant"] == "least_squares"

    def test_exact_proportional(self):
        # r = 0.5 * r_m — exact proportional, no noise
        bench = np.array([0.01, 0.02, -0.01, 0.005, -0.02, 0.015])
        rets = 0.5 * bench
        inp = BenchmarkInput(returns=rets, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "beta")
        assert result.value == pytest.approx(0.5)

    def test_few_observations_returns_nan(self):
        inp = BenchmarkInput(
            returns=np.array([0.01, 0.02]),
            benchmark=np.array([0.005, 0.01]),
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "beta")
        assert np.isnan(result.value)

    def test_negative_beta(self):
        bench = np.array([0.01, 0.02, -0.01, 0.005, -0.02, 0.015])
        rets = -0.8 * bench
        inp = BenchmarkInput(returns=rets, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "beta")
        assert result.value == pytest.approx(-0.8)


# ===================================================================
# §8.3  R²
# ===================================================================


class TestRSquared:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "r_squared")
        # r = 0.001 + 1.2*r_m is a perfect linear fit (R² = 1.0)
        assert result.value == pytest.approx(1.0, abs=0.01)

    def test_noisy_relationship(self):
        rng = np.random.default_rng(42)
        bench = rng.normal(0.0, 0.01, size=252)
        noise = rng.normal(0.0, 0.005, size=252)
        rets = 0.8 * bench + noise
        inp = BenchmarkInput(returns=rets, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "r_squared")
        assert 0.0 < result.value < 1.0

    def test_constant_returns(self):
        bench = np.array([0.01, 0.02, -0.01, 0.005, -0.02])
        rets = np.array([0.005, 0.005, 0.005, 0.005, 0.005])
        inp = BenchmarkInput(returns=rets, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "r_squared")
        assert np.isnan(result.value)  # zero variance in returns


# ===================================================================
# §8.4  Tracking Error
# ===================================================================


class TestTrackingError:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "tracking_error")
        # active = returns - benchmark = 0.001 + 0.2 * benchmark
        active = inp_basic.returns[:, 0] - inp_basic.benchmark
        te_period = np.std(active, ddof=1)
        expected = te_period * np.sqrt(252)
        assert result.value == pytest.approx(expected)
        assert result.meta["annualized"] is True

    def test_requires_periods_per_year(self, inp_no_pp):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="periods_per_year"):
            _compute_one(inp_no_pp, "tracking_error")

    def test_zero_tracking_error(self):
        bench = np.array([0.01, 0.02, -0.01])
        rets = bench.copy()
        inp = BenchmarkInput(returns=rets, benchmark=bench, periods_per_year=252)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "tracking_error")
        assert result.value == 0.0


# ===================================================================
# §8.5  Information Ratio
# ===================================================================


class TestInformationRatio:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "information_ratio")
        active = inp_basic.returns[:, 0] - inp_basic.benchmark
        p = 252.0
        ir = (np.mean(active) * p) / (np.std(active, ddof=1) * np.sqrt(p))
        assert result.value == pytest.approx(ir)

    def test_zero_te_returns_nan(self):
        bench = np.array([0.01, 0.02, -0.01])
        rets = bench.copy()
        inp = BenchmarkInput(returns=rets, benchmark=bench, periods_per_year=252)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "information_ratio")
        assert np.isnan(result.value)


# ===================================================================
# §8.6  Up-Capture Ratio
# ===================================================================


class TestUpCapture:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "up_capture")
        bench = inp_basic.benchmark
        rets = inp_basic.returns[:, 0]
        up_mask = bench > 0.0
        expected = np.mean(rets[up_mask]) / np.mean(bench[up_mask])
        assert result.value == pytest.approx(expected, rel=0.05)

    def test_no_up_periods_returns_nan(self):
        bench = np.array([-0.01, -0.02, -0.005])
        rets = np.array([0.01, 0.02, 0.005])
        inp = BenchmarkInput(returns=rets, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "up_capture")
        assert np.isnan(result.value)


# ===================================================================
# §8.7  Down-Capture Ratio
# ===================================================================


class TestDownCapture:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "down_capture")
        bench = inp_basic.benchmark
        rets = inp_basic.returns[:, 0]
        down_mask = bench < 0.0
        expected = np.mean(rets[down_mask]) / np.mean(bench[down_mask])
        assert result.value == pytest.approx(expected, rel=0.05)

    def test_no_down_periods_returns_nan(self):
        bench = np.array([0.01, 0.02, 0.005])
        rets = np.array([0.01, 0.02, 0.005])
        inp = BenchmarkInput(returns=rets, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "down_capture")
        assert np.isnan(result.value)


# ===================================================================
# §8.8  Up/Down Capture Ratio
# ===================================================================


class TestUpDownCapture:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "up_down_capture")
        uc = _compute_one(inp_basic, "up_capture").value
        dc = _compute_one(inp_basic, "down_capture").value
        expected = uc / abs(dc)
        assert result.value == pytest.approx(expected)

    def test_no_down_capture_returns_nan(self):
        bench = np.array([0.01, 0.02, 0.005])
        rets = np.array([0.01, 0.02, 0.005])
        inp = BenchmarkInput(returns=rets, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "up_down_capture")
        assert np.isnan(result.value)


# ===================================================================
# §8.9  Correlation
# ===================================================================


class TestCorrelation:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "correlation")
        rets = inp_basic.returns[:, 0]
        bench = inp_basic.benchmark
        mask = np.isfinite(rets) & np.isfinite(bench)
        expected = np.corrcoef(rets[mask], bench[mask])[0, 1]
        assert result.value == pytest.approx(expected)

    def test_perfect_positive(self):
        bench = np.array([0.01, 0.02, -0.01, 0.005, -0.02])
        rets = 0.5 * bench
        inp = BenchmarkInput(returns=rets, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "correlation")
        assert result.value == pytest.approx(1.0)

    def test_few_obs_returns_nan(self):
        inp = BenchmarkInput(
            returns=np.array([0.01, 0.02]),
            benchmark=np.array([0.005, 0.01]),
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "correlation")
        assert np.isnan(result.value)


# ===================================================================
# §8.10  Active Return
# ===================================================================


class TestActiveReturn:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "active_return")
        active = inp_basic.returns[:, 0] - inp_basic.benchmark
        expected = np.mean(active) * 252.0
        assert result.value == pytest.approx(expected)

    def test_requires_periods_per_year(self, inp_no_pp):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="periods_per_year"):
            _compute_one(inp_no_pp, "active_return")


# ===================================================================
# §8.11  Batting Average vs Benchmark
# ===================================================================


class TestBattingAverage:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "batting_average")
        rets = inp_basic.returns[:, 0]
        bench = inp_basic.benchmark
        expected = np.sum(rets > bench) / len(rets)
        assert result.value == pytest.approx(expected)

    def test_always_beats(self):
        bench = np.array([0.01, 0.02, -0.01])
        rets = bench + 0.1  # always above
        inp = BenchmarkInput(returns=rets, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "batting_average")
        assert result.value == 1.0

    def test_never_beats(self):
        bench = np.array([0.01, 0.02, -0.01])
        rets = bench - 0.1  # always below
        inp = BenchmarkInput(returns=rets, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "batting_average")
        assert result.value == 0.0

    def test_all_nan_returns_nan(self):
        inp = BenchmarkInput(
            returns=np.array([np.nan, np.nan]),
            benchmark=np.array([0.01, 0.02]),
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "batting_average")
        assert np.isnan(result.value)


# ===================================================================
# §8.12  Treynor Ratio
# ===================================================================


class TestTreynorRatio:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "treynor_ratio")
        rets = inp_basic.returns[:, 0]
        excess = rets - inp_basic.rf
        mean_excess_ann = np.mean(excess) * 252.0
        beta_val = 1.2  # approximate, from construction
        expected = mean_excess_ann / beta_val
        assert result.value == pytest.approx(expected, rel=0.05)

    def test_zero_beta_positive_excess(self):
        bench = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
        urelated = np.array([0.005, 0.003, -0.002, 0.001, 0.004])
        # Construct returns uncorrelated with benchmark
        inp = BenchmarkInput(
            returns=urelated,
            benchmark=bench,
            periods_per_year=252,
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "treynor_ratio")
        # beta close to zero → large ratio
        assert abs(result.value) > 1.0


# ===================================================================
# §8.13  Outperformance
# ===================================================================


class TestOutperformance:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "outperformance")
        rets = inp_basic.returns[:, 0]
        bench = inp_basic.benchmark
        r_cum = np.prod(1.0 + rets) - 1.0
        m_cum = np.prod(1.0 + bench) - 1.0
        expected = r_cum - m_cum
        assert result.value == pytest.approx(expected)

    def test_identical_returns(self):
        bench = np.array([0.01, 0.02, -0.01])
        inp = BenchmarkInput(returns=bench, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "outperformance")
        assert result.value == pytest.approx(0.0)


# ===================================================================
# §8.14  Outperformance Ratio
# ===================================================================


class TestOutperformanceRatio:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "outperformance_ratio")
        rets = inp_basic.returns[:, 0]
        bench = inp_basic.benchmark
        r_cum = np.prod(1.0 + rets)
        m_cum = np.prod(1.0 + bench)
        expected = r_cum / m_cum
        assert result.value == pytest.approx(expected)


# ===================================================================
# §8.15  Underperforming Periods / %
# ===================================================================


class TestUnderperformingPeriods:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "underperforming_periods")
        rets = inp_basic.returns[:, 0]
        bench = inp_basic.benchmark
        n_under = int(np.sum(rets < bench))
        expected = np.array([n_under, n_under / len(rets)])
        np.testing.assert_array_equal(result.value, expected)
        assert result.meta["output_index"] == ["count", "pct"]


# ===================================================================
# §8.16  Max Outperformance
# ===================================================================


class TestMaxOutperformance:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "max_outperformance")
        rets = inp_basic.returns[:, 0]
        bench = inp_basic.benchmark
        active = rets - bench
        cum_active = np.cumsum(active)
        assert result.value == pytest.approx(np.max(cum_active))

    def test_always_outperforming(self):
        bench = np.array([0.01, 0.02, -0.01])
        rets = bench + 0.01  # always +1%
        inp = BenchmarkInput(returns=rets, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "max_outperformance")
        active = rets - bench
        assert result.value == pytest.approx(np.sum(active))


# ===================================================================
# §8.17  Max Underperformance
# ===================================================================


class TestMaxUnderperformance:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "max_underperformance")
        rets = inp_basic.returns[:, 0]
        bench = inp_basic.benchmark
        active = rets - bench
        cum_active = np.cumsum(active)
        expected = np.abs(min(0.0, np.min(cum_active)))
        assert result.value == pytest.approx(expected)

    def test_never_underperforming(self):
        bench = np.array([0.01, 0.02, -0.01])
        rets = bench + 0.01  # always above
        inp = BenchmarkInput(returns=rets, benchmark=bench)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "max_underperformance")
        assert result.value == 0.0


# ===================================================================
# §8.18  Benchmark Volatility
# ===================================================================


class TestBenchmarkVolatility:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "benchmark_volatility")
        bench = inp_basic.benchmark
        expected = np.std(bench, ddof=1) * np.sqrt(252.0)
        assert result.value == pytest.approx(expected)
        assert result.meta["annualized"] is True

    def test_requires_periods_per_year(self, inp_no_pp):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="periods_per_year"):
            _compute_one(inp_no_pp, "benchmark_volatility")


# ===================================================================
# BenchmarkInput feature tests
# ===================================================================


class TestBenchmarkInputFeatures:
    def test_tuple_shortcut(self, simple_returns, simple_benchmark):
        inp = BenchmarkInput((simple_returns, simple_benchmark))
        assert inp.n_periods == 10
        assert inp.n_strategies == 1

    def test_benchmark_required(self):
        with pytest.raises(ValueError, match="requires benchmark"):
            BenchmarkInput(returns=np.array([0.01, 0.02]))

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="must match"):
            BenchmarkInput(
                returns=np.array([0.01, 0.02, 0.03]),
                benchmark=np.array([0.005, 0.01]),
            )

    def test_tuple_len_3_raises(self):
        with pytest.raises(ValueError, match="length 3"):
            BenchmarkInput((np.array([0.01]), np.array([0.02]), np.array([0.03])))

    def test_repr(self, inp_basic):
        r = repr(inp_basic)
        assert "BenchmarkInput" in r
        assert "n_periods=10" in r

    def test_is_single(self, inp_basic):
        assert inp_basic.is_single is True

    def test_rf_default(self, simple_returns, simple_benchmark):
        inp = BenchmarkInput((simple_returns, simple_benchmark))
        assert inp.rf == 0.0

    def test_rf_custom(self, simple_returns, simple_benchmark):
        inp = BenchmarkInput(
            returns=simple_returns,
            benchmark=simple_benchmark,
            rf=0.02 / 252,
        )
        assert inp.rf == pytest.approx(0.02 / 252)

    def test_pandas_input(self, simple_returns, simple_benchmark):
        rets_df = pd.DataFrame({"strat": simple_returns})
        bench_series = pd.Series(simple_benchmark)
        inp = BenchmarkInput(returns=rets_df, benchmark=bench_series)
        assert inp.n_periods == 10

    def test_polars_input(self, simple_returns, simple_benchmark):
        pl = pytest.importorskip("polars")
        rets_df = pl.from_dict({"strat": simple_returns})
        bench_series = pl.Series(simple_benchmark)
        inp = BenchmarkInput(returns=rets_df, benchmark=bench_series)
        assert inp.n_periods == 10

    def test_periods_per_year_passthrough(self, inp_basic):
        assert inp_basic.periods_per_year == 252

    def test_multi_strategy_raises(self, simple_benchmark):
        # 2-D returns with 3 strategies
        rets_multi = np.column_stack(
            [simple_benchmark, simple_benchmark * 1.2, simple_benchmark * 0.8]
        )
        inp = BenchmarkInput(
            returns=rets_multi,
            benchmark=simple_benchmark,
            periods_per_year=252,
        )
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="single strategy"):
            _compute_one(inp, "alpha")


# ===================================================================
# Registry integration tests
# ===================================================================


class TestRegistryIntegration:
    def test_all_benchmark_metrics_registered(self):
        from stratstat.registry import list_metrics

        metrics = list_metrics(requires="benchmark")
        names = {m["name"] for m in metrics}
        assert len(names) == 18

    def test_compute_single_auto_wrap_tuple(self, simple_returns, simple_benchmark):
        from stratstat import compute

        result = compute((simple_returns, simple_benchmark), "beta")
        assert result.name == "beta"
        assert result.value == pytest.approx(1.2, rel=0.05)

    def test_compute_all_benchmark_category(self, simple_returns, simple_benchmark):
        from stratstat import compute_all

        inp = BenchmarkInput(
            returns=simple_returns,
            benchmark=simple_benchmark,
            periods_per_year=252,
        )
        results = compute_all(inp, category="benchmark")
        names = {r.name for r in results}
        assert "beta" in names
        assert "alpha" in names
        assert "correlation" in names

    def test_compute_all_capture_category(self, simple_returns, simple_benchmark):
        from stratstat import compute_all

        inp = BenchmarkInput((simple_returns, simple_benchmark))
        results = compute_all(inp, category="capture")
        names = {r.name for r in results}
        assert "up_capture" in names
        assert "down_capture" in names
        assert "up_down_capture" in names

    def test_unknown_metric_raises(self, simple_returns, simple_benchmark):
        from stratstat import compute
        from stratstat.exceptions import UnknownMetricError

        inp = BenchmarkInput((simple_returns, simple_benchmark), periods_per_year=252)
        with pytest.raises(UnknownMetricError):
            compute(inp, "nonexistent_metric")

    def test_compute_with_benchmarkinput_instance(self, inp_basic):
        from stratstat import compute

        result = compute(inp_basic, "correlation")
        assert result.name == "correlation"
