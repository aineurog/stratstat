"""Tests for descriptive returns metrics.

Each test validates the computed value against a known reference:
  - Hand-computed values from small, worked datasets
  - Cross-validation against numpy/pandas for consistency
  - Edge cases: empty, all-zero, single-point, NaN, short series

References are noted in each test per the build instructions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stratstat.core.returns.descriptive import (
    annualized_volatility,
    arithmetic_mean_return,
    autocorrelation,
    best_period,
    cagr,
    coefficient_of_variation,
    cumulative_return,
    excess_kurtosis,
    geometric_mean_return,
    outlier_iqr,
    percentiles,
    positive_period_ratio,
    return_range,
    skewness,
    variance,
    worst_period,
)
from stratstat.inputs import ReturnsInput

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_pp():
    """Standard daily periods_per_year."""
    return 252


@pytest.fixture
def sample_returns():
    """Small deterministic returns series for known-value computation.

    Values: [0.01, -0.02, 0.015, 0.03, -0.01, 0.005, -0.015, 0.02, -0.005, 0.01]
    10 daily returns with known statistical properties.
    """
    return np.array([0.01, -0.02, 0.015, 0.03, -0.01, 0.005, -0.015, 0.02, -0.005, 0.01])


@pytest.fixture
def sample_input(sample_returns, daily_pp):
    """ReturnsInput wrapping the sample returns with daily periods_per_year."""
    return ReturnsInput(sample_returns, periods_per_year=daily_pp)


@pytest.fixture
def sample_input_no_pp(sample_returns):
    """ReturnsInput without periods_per_year (for metrics that don't need it)."""
    return ReturnsInput(sample_returns)


# ---------------------------------------------------------------------------
# 1.1 CAGR
# Reference: Damodaran (2012, Investment Valuation, 3rd ed., Ch. 3)
# ---------------------------------------------------------------------------


class TestCAGR:
    def test_known_value(self, sample_input):
        """CAGR computed from known returns.

        Sample: [0.01, -0.02, 0.015, 0.03, -0.01, 0.005, -0.015, 0.02, -0.005, 0.01]
        Total return = prod(1+r) - 1
        = 1.01*0.98*1.015*1.03*0.99*1.005*0.985*1.02*0.995*1.01 - 1
        = 1.039357... - 1 = 0.039357...
        CAGR = (1 + total_return)^(252/10) - 1
        """
        result = cagr(sample_input)
        # Total return
        total_ret = np.prod(1.0 + sample_input.values.squeeze()) - 1
        expected = (1.0 + total_ret) ** (252.0 / 10.0) - 1.0
        assert result.name == "cagr"
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_requires_periods_per_year(self):
        """CAGR raises ValueError when periods_per_year is None."""
        inp = ReturnsInput(np.array([0.01, 0.02, -0.01]))
        with pytest.raises(ValueError, match="periods_per_year"):
            cagr(inp)

    def test_all_positive_returns(self, daily_pp):
        """Positive-only returns produce positive CAGR."""
        r = np.array([0.01, 0.02, 0.015, 0.005])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = cagr(inp)
        assert result.value > 0.0

    def test_all_negative_returns(self, daily_pp):
        """Negative-only returns produce negative CAGR."""
        r = np.array([-0.01, -0.02, -0.015])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = cagr(inp)
        assert result.value < 0.0

    def test_single_period(self, daily_pp):
        """Single-period CAGR equals the period return annualized."""
        r = np.array([0.05])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        expected = (1.05) ** daily_pp - 1.0
        result = cagr(inp)
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_multi_strategy(self, sample_returns, daily_pp):
        """Multi-strategy input produces array result."""
        multi = np.column_stack([sample_returns, sample_returns * 0.5])
        inp = ReturnsInput(multi, periods_per_year=daily_pp)
        result = cagr(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)

    def test_nan_handling(self, sample_returns, daily_pp):
        """NaN values are ignored in computation."""
        r = sample_returns.copy().astype(float)
        r[2] = np.nan
        r[7] = np.nan
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = cagr(inp)
        # Should match the CAGR on the clean subset
        assert not np.isnan(result.value)


# ---------------------------------------------------------------------------
# 1.2 Annualized Volatility
# Reference: CFA Institute, Quantitative Methods (CFA Program Curriculum, Level I, Vol. 1)
# ---------------------------------------------------------------------------


class TestAnnualizedVolatility:
    def test_known_value(self, sample_input):
        """Annualized vol = std(ddof=1) * sqrt(252).

        std of sample = 0.0156... (computed via numpy)
        """
        r = sample_input.values.squeeze()
        std = np.std(r, ddof=1)
        expected = std * np.sqrt(252)
        result = annualized_volatility(sample_input)
        assert result.name == "annualized_volatility"
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_requires_periods_per_year(self):
        """Raises when periods_per_year is None."""
        inp = ReturnsInput(np.array([0.01, -0.02]))
        with pytest.raises(ValueError, match="periods_per_year"):
            annualized_volatility(inp)

    def test_constant_returns(self, daily_pp):
        """Constant returns have zero volatility."""
        r = np.full(100, 0.001)
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = annualized_volatility(inp)
        assert result.value == pytest.approx(0.0, abs=1e-12)

    def test_single_observation(self, daily_pp):
        """Single observation gives NaN (undefined variance)."""
        r = np.array([0.01])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = annualized_volatility(inp)
        assert np.isnan(result.value)


# ---------------------------------------------------------------------------
# 1.3 Cumulative Return
# Reference: Bacon (2008, Practical Portfolio Perf. Measurement, 2nd ed., Sec. 2.1)
# ---------------------------------------------------------------------------


class TestCumulativeReturn:
    def test_known_value(self, sample_input):
        """Cumulative return = prod(1+r) - 1."""
        r = sample_input.values.squeeze()
        expected = np.prod(1.0 + r) - 1.0
        result = cumulative_return(sample_input)
        assert result.name == "cumulative_return"
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_zero_return(self):
        """Single zero-return period gives cumulative return of 0."""
        inp = ReturnsInput(np.array([0.0]))
        result = cumulative_return(inp)
        assert result.value == 0.0

    def test_double_your_money(self):
        """100% return gives cumulative return of 1.0."""
        inp = ReturnsInput(np.array([1.0]))
        result = cumulative_return(inp)
        assert result.value == 1.0

    def test_lose_everything(self):
        """-100% return gives cumulative return of -1.0."""
        inp = ReturnsInput(np.array([-1.0]))
        result = cumulative_return(inp)
        assert result.value == -1.0

    def test_multi_strategy(self, sample_returns):
        """Multi-strategy returns produce array output."""
        multi = np.column_stack([sample_returns, -sample_returns])
        inp = ReturnsInput(multi)
        result = cumulative_return(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)

    def test_nan_handling(self):
        """NaN values are skipped — validate exact value.

        Array [0.01, NaN, 0.02, NaN, -0.01]: valid returns are 0.01, 0.02, -0.01.
        prod(1 + valid) - 1 = 1.01 * 1.02 * 0.99 - 1.
        """
        r = np.array([0.01, np.nan, 0.02, np.nan, -0.01])
        inp = ReturnsInput(r)
        result = cumulative_return(inp)
        expected = 1.01 * 1.02 * 0.99 - 1.0
        assert result.value == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 1.4 Arithmetic Mean Return
# Reference: Casella & Berger (2002, Statistical Inference, Sec. 5.2)
# ---------------------------------------------------------------------------


class TestArithmeticMeanReturn:
    def test_known_value(self, sample_input_no_pp):
        """Arithmetic mean = sum(r)/n."""
        r = sample_input_no_pp.values.squeeze()
        expected = np.mean(r)
        result = arithmetic_mean_return(sample_input_no_pp)
        assert result.name == "arithmetic_mean_return"
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_single_value(self):
        """Single value: mean = the value itself."""
        inp = ReturnsInput(np.array([0.05]))
        result = arithmetic_mean_return(inp)
        assert result.value == 0.05

    def test_positive_and_negative(self):
        """Mixed returns produce intermediate mean."""
        r = np.array([0.10, -0.10])
        inp = ReturnsInput(r)
        result = arithmetic_mean_return(inp)
        assert result.value == 0.0

    def test_multi_strategy(self):
        """Multi-strategy input."""
        multi = np.column_stack([[0.01, 0.02, -0.01], [0.05, -0.02, 0.03]])
        inp = ReturnsInput(multi)
        result = arithmetic_mean_return(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value[0] == pytest.approx(np.mean(multi[:, 0]))
        assert result.value[1] == pytest.approx(np.mean(multi[:, 1]))

    def test_nan_handling(self):
        """NaN values are ignored — nanmean behaviour."""
        r = np.array([0.01, np.nan, 0.03, np.nan, 0.02])
        inp = ReturnsInput(r)
        result = arithmetic_mean_return(inp)
        expected = np.nanmean(r)
        assert result.value == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 1.5 Geometric Mean Return
# Reference: Campbell, Lo & MacKinlay (1997, The Econometrics of Financial Markets, Sec. 1.4)
# ---------------------------------------------------------------------------


class TestGeometricMeanReturn:
    def test_known_value(self, sample_input_no_pp):
        """Geometric mean = exp(mean(log(1+r))) - 1."""
        r = sample_input_no_pp.values.squeeze()
        expected = np.exp(np.mean(np.log(1.0 + r))) - 1.0
        result = geometric_mean_return(sample_input_no_pp)
        assert result.name == "geometric_mean_return"
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_arithmetic_greater_than_geometric(self, sample_input_no_pp):
        """Jensen's inequality: arithmetic >= geometric for non-constant returns."""
        arith = arithmetic_mean_return(sample_input_no_pp).value
        geom = geometric_mean_return(sample_input_no_pp).value
        assert arith >= geom

    def test_constant_returns(self):
        """For constant returns, arithmetic = geometric."""
        r = np.full(10, 0.01)
        inp = ReturnsInput(r)
        arith = arithmetic_mean_return(inp).value
        geom = geometric_mean_return(inp).value
        assert arith == pytest.approx(geom)

    def test_single_period(self):
        """Single-period geometric mean = the return itself."""
        inp = ReturnsInput(np.array([0.05]))
        result = geometric_mean_return(inp)
        assert result.value == pytest.approx(0.05, rel=1e-12)

    def test_multi_strategy(self):
        """Multi-strategy returns produce array."""
        multi = np.column_stack([[0.01, -0.02, 0.03], [0.05, 0.01, -0.01]])
        inp = ReturnsInput(multi)
        result = geometric_mean_return(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)


# ---------------------------------------------------------------------------
# 1.6 Skewness
# Reference: Fisher (1930); Cramer (1946, Mathematical Methods of Statistics, Sec. 27.4)
# ---------------------------------------------------------------------------


class TestSkewness:
    def test_symmetric_returns(self):
        """Symmetric returns have near-zero skewness."""
        r = np.array([-0.03, -0.01, -0.005, 0.005, 0.01, 0.03])
        inp = ReturnsInput(r)
        result = skewness(inp)
        # Small sample, but symmetry gives approximately zero
        assert abs(result.value) < 0.5

    def test_positive_skew(self):
        """Positively skewed: large right tail."""
        r = np.array([-0.01, -0.01, -0.005, 0.0, 0.0, 0.005, 0.01, 0.02, 0.05])
        inp = ReturnsInput(r)
        result = skewness(inp)
        assert result.value > 0.0

    def test_negative_skew(self):
        """Negatively skewed: large left tail."""
        r = np.array([-0.05, -0.02, -0.01, -0.005, 0.0, 0.0, 0.005, 0.01, 0.01])
        inp = ReturnsInput(r)
        result = skewness(inp)
        assert result.value < 0.0

    def test_cross_validate_pandas(self, sample_input_no_pp):
        """Cross-validate against pandas skew (also bias-corrected)."""
        r = sample_input_no_pp.values.squeeze()
        pd_skew = float(pd.Series(r).skew())
        result = skewness(sample_input_no_pp)
        assert result.value == pytest.approx(pd_skew, rel=1e-12)

    def test_insufficient_data(self):
        """Fewer than 3 observations gives NaN."""
        r = np.array([0.01, 0.02])
        inp = ReturnsInput(r)
        result = skewness(inp)
        assert np.isnan(result.value)

    def test_constant_returns(self):
        """Constant returns have undefined skewness (NaN)."""
        r = np.full(10, 0.01)
        inp = ReturnsInput(r)
        result = skewness(inp)
        assert np.isnan(result.value)

    def test_multi_strategy(self):
        """Multi-strategy skewness."""
        multi = np.column_stack([
            [0.01, -0.02, 0.015, 0.03, -0.01],
            [-0.01, 0.02, -0.015, -0.03, 0.01],
        ])
        inp = ReturnsInput(multi)
        result = skewness(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)

    def test_nan_handling(self):
        """NaN values are ignored."""
        r = np.array([0.01, np.nan, -0.02, 0.015, np.nan, 0.03, -0.01, 0.005])
        inp = ReturnsInput(r)
        result = skewness(inp)
        clean = r[~np.isnan(r)]
        clean_inp = ReturnsInput(clean)
        expected = skewness(clean_inp).value
        assert result.value == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 1.7 Excess Kurtosis
# Reference: Fisher (1930); Cramer (1946, Mathematical Methods of Statistics, Sec. 27.4)
# ---------------------------------------------------------------------------


class TestExcessKurtosis:
    def test_normal_like_returns(self):
        """Returns from a normal-ish distribution should have near-zero excess kurtosis."""
        rng = np.random.default_rng(12345)
        r = rng.normal(0.001, 0.02, size=1000)
        inp = ReturnsInput(r)
        result = excess_kurtosis(inp)
        assert abs(result.value) < 1.0  # loose bound for small sample

    def test_heavy_tailed(self):
        """Fat-tailed returns have positive excess kurtosis."""
        # Mix of normal and extreme values
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.02, size=500)
        # Add some extreme outliers
        r = np.concatenate([r, np.array([0.15, -0.12, 0.10, -0.14, 0.13])])
        inp = ReturnsInput(r)
        result = excess_kurtosis(inp)
        assert result.value > 0.0

    def test_cross_validate_pandas(self, sample_input_no_pp):
        """Cross-validate against pandas kurtosis (fisher=True, bias-corrected)."""
        r = sample_input_no_pp.values.squeeze()
        pd_kurt = float(pd.Series(r).kurtosis())  # fisher=True by default
        result = excess_kurtosis(sample_input_no_pp)
        assert result.value == pytest.approx(pd_kurt, rel=1e-12)

    def test_insufficient_data(self):
        """Fewer than 4 observations gives NaN."""
        r = np.array([0.01, 0.02, -0.01])
        inp = ReturnsInput(r)
        result = excess_kurtosis(inp)
        assert np.isnan(result.value)

    def test_constant_returns(self):
        """Constant returns have undefined excess kurtosis (NaN)."""
        r = np.full(10, 0.01)
        inp = ReturnsInput(r)
        result = excess_kurtosis(inp)
        assert np.isnan(result.value)

    def test_nan_handling(self):
        """NaN values are ignored."""
        r = np.array([0.01, np.nan, -0.02, 0.015, np.nan, 0.03, -0.01, 0.005, 0.02])
        inp = ReturnsInput(r)
        result = excess_kurtosis(inp)
        clean = r[~np.isnan(r)]
        clean_inp = ReturnsInput(clean)
        expected = excess_kurtosis(clean_inp).value
        assert result.value == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 1.8 Best Period / 1.9 Worst Period
# Reference: Bacon (2008, Practical Portfolio Perf. Measurement, 2nd ed., Sec. 3.10)
# ---------------------------------------------------------------------------


class TestBestWorstPeriod:
    def test_best_period(self, sample_input):
        """Best period = max return."""
        r = sample_input.values.squeeze()
        expected = np.max(r)
        result = best_period(sample_input)
        assert result.name == "best_period"
        assert result.value == pytest.approx(expected)

    def test_worst_period(self, sample_input):
        """Worst period = min return."""
        r = sample_input.values.squeeze()
        expected = np.min(r)
        result = worst_period(sample_input)
        assert result.name == "worst_period"
        assert result.value == pytest.approx(expected)

    def test_single_period(self):
        """Single period: best = worst = the return."""
        inp = ReturnsInput(np.array([0.03]))
        assert best_period(inp).value == 0.03
        assert worst_period(inp).value == 0.03

    def test_all_positive(self):
        """All positive returns: best > 0, worst > 0."""
        r = np.array([0.01, 0.02, 0.005])
        inp = ReturnsInput(r)
        assert best_period(inp).value > 0
        assert worst_period(inp).value > 0

    def test_nan_handling(self):
        """NaN values are skipped in max/min."""
        r = np.array([0.01, np.nan, 0.03, np.nan, -0.01])
        inp = ReturnsInput(r)
        assert best_period(inp).value == 0.03
        assert worst_period(inp).value == -0.01


# ---------------------------------------------------------------------------
# 1.10 Positive-Period Ratio
# Reference: Bacon (2008, Practical Portfolio Perf. Measurement, 2nd ed., Sec. 3.11)
# ---------------------------------------------------------------------------


class TestPositivePeriodRatio:
    def test_known_value(self, sample_input_no_pp):
        """PPR = fraction of positive returns (strictly > 0)."""
        r = sample_input_no_pp.values.squeeze()
        expected = np.mean(r > 0)
        result = positive_period_ratio(sample_input_no_pp)
        assert result.name == "positive_period_ratio"
        assert result.value == pytest.approx(expected)

    def test_all_positive(self):
        """All positive: ratio = 1.0."""
        r = np.array([0.01, 0.02, 0.03])
        inp = ReturnsInput(r)
        assert positive_period_ratio(inp).value == 1.0

    def test_all_negative(self):
        """All negative: ratio = 0.0."""
        r = np.array([-0.01, -0.02, -0.03])
        inp = ReturnsInput(r)
        assert positive_period_ratio(inp).value == 0.0

    def test_zero_not_positive(self):
        """Zero returns are not counted as positive."""
        r = np.array([0.0, 0.01, -0.01, 0.0])
        inp = ReturnsInput(r)
        result = positive_period_ratio(inp)
        assert result.value == 0.25  # only 0.01 is positive

    def test_nan_handling(self):
        """NaN values are excluded from denominator."""
        r = np.array([0.01, 0.02, np.nan, -0.01, np.nan])
        inp = ReturnsInput(r)
        result = positive_period_ratio(inp)
        expected = 2.0 / 3.0  # 2 positive out of 3 non-NaN
        assert result.value == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 1.11 Autocorrelation (Lag-1)
# Reference: Campbell, Lo & MacKinlay (1997, The Econometrics of Financial Markets, Sec. 2.4)
# ---------------------------------------------------------------------------


class TestAutocorrelation:
    def test_known_value(self):
        """Lag-1 autocorrelation per Campbell-Lo-MacKinlay formula.

        rho_1 = sum_{t=2..n} (r_t - r_bar)(r_{t-1} - r_bar)
               / sum_{t=1..n} (r_t - r_bar)^2

        where r_bar is the full-sample mean of all n observations.
        """
        r = np.array([0.01, 0.02, -0.01, 0.005, -0.015, 0.03, 0.01, -0.02])
        inp = ReturnsInput(r)
        result = autocorrelation(inp)

        # Compute expected manually using the formula
        r_bar = np.mean(r)
        numerator = np.sum((r[1:] - r_bar) * (r[:-1] - r_bar))
        denominator = np.sum((r - r_bar) ** 2)
        expected = numerator / denominator
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_positive_autocorrelation(self):
        """Trending returns: positive autocorrelation."""
        r = np.linspace(0.001, 0.02, 20)
        inp = ReturnsInput(r)
        result = autocorrelation(inp)
        assert result.value > 0.0

    def test_negative_autocorrelation(self):
        """Perfectly alternating returns: unambiguous negative autocorrelation.

        For r = [0.10, -0.10, 0.10, -0.10, 0.10, -0.10]:
          r_bar = 0.0
          numerator = 5 * (-0.01) = -0.05
          denominator = 6 * 0.01 = 0.06
          rho_1 = -5/6 ≈ -0.8333
        """
        r = np.array([0.10, -0.10, 0.10, -0.10, 0.10, -0.10])
        inp = ReturnsInput(r)
        result = autocorrelation(inp)
        assert result.value == pytest.approx(-5.0 / 6.0, rel=1e-12)

    def test_insufficient_data(self):
        """Fewer than 2 observations gives NaN."""
        inp = ReturnsInput(np.array([0.01]))
        result = autocorrelation(inp)
        assert np.isnan(result.value)

    def test_nan_handling(self):
        """NaN values excluded pairwise — only consecutive non-NaN pairs count.

        Array [0.01, NaN, -0.01, 0.03]:
          Non-NaN positions: idx 0, 2, 3.
          Consecutive non-NaN pairs: only (idx 2, idx 3) = (-0.01, 0.03).
          The pair (0, 2) = (0.01, -0.01) is NOT consecutive (NaN at idx 1).

        r_bar = mean([0.01, -0.01, 0.03]) = 0.01
        Numerator: (-0.01 - 0.01)*(0.03 - 0.01) = (-0.02)*(0.02) = -0.0004
        Denominator: (0.01-0.01)^2 + (-0.01-0.01)^2 + (0.03-0.01)^2
                    = 0 + 0.0004 + 0.0004 = 0.0008
        rho_1 = -0.0004 / 0.0008 = -0.5
        """
        r = np.array([0.01, np.nan, -0.01, 0.03])
        inp = ReturnsInput(r)
        result = autocorrelation(inp)
        assert result.value == pytest.approx(-0.5, rel=1e-12)

    def test_constant_returns(self):
        """Constant returns have undefined autocorrelation (NaN)."""
        r = np.full(10, 0.01)
        inp = ReturnsInput(r)
        result = autocorrelation(inp)
        assert np.isnan(result.value)

    def test_multi_strategy(self):
        """Multi-strategy autocorrelation."""
        multi = np.column_stack([
            [0.01, 0.02, -0.01, 0.005, -0.015],
            [-0.01, -0.02, 0.01, -0.005, 0.015],
        ])
        inp = ReturnsInput(multi)
        result = autocorrelation(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)


# ---------------------------------------------------------------------------
# 1.12 Variance
# Reference: Fisher (1925, Statistical Methods for Research Workers)
# ---------------------------------------------------------------------------


class TestVariance:
    def test_known_value(self, sample_input_no_pp):
        """Sample variance = sum((r - mean)^2) / (n-1)."""
        r = sample_input_no_pp.values.squeeze()
        expected = np.var(r, ddof=1)
        result = variance(sample_input_no_pp)
        assert result.name == "variance"
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_constant_returns(self):
        """Constant returns: variance = 0."""
        r = np.full(5, 0.02)
        inp = ReturnsInput(r)
        assert variance(inp).value == 0.0

    def test_single_observation(self):
        """Single observation: undefined variance (NaN)."""
        inp = ReturnsInput(np.array([0.01]))
        result = variance(inp)
        assert np.isnan(result.value)

    def test_nan_handling(self):
        """NaN values are excluded."""
        r = np.array([0.01, np.nan, -0.02, 0.015])
        inp = ReturnsInput(r)
        result = variance(inp)
        clean = r[~np.isnan(r)]
        expected = np.var(clean, ddof=1)
        assert result.value == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 1.13 Return Range
# Reference: Bacon (2008, Practical Portfolio Perf. Measurement, 2nd ed., Sec. 3.10)
# ---------------------------------------------------------------------------


class TestReturnRange:
    def test_known_value(self, sample_input):
        """Range = max - min."""
        r = sample_input.values.squeeze()
        expected = np.max(r) - np.min(r)
        result = return_range(sample_input)
        assert result.name == "return_range"
        assert result.value == pytest.approx(expected)

    def test_positive_range(self):
        """Range is always non-negative."""
        inp = ReturnsInput(np.array([-0.05, -0.03, -0.01]))
        result = return_range(inp)
        assert result.value >= 0.0

    def test_single_period(self):
        """Single period: range = 0."""
        inp = ReturnsInput(np.array([0.03]))
        assert return_range(inp).value == 0.0


# ---------------------------------------------------------------------------
# 1.14 Percentiles
# Reference: Hyndman & Fan (1996), "Sample Quantiles in Statistical Packages"
# ---------------------------------------------------------------------------


class TestPercentiles:
    def test_known_value(self, sample_input_no_pp):
        """Percentiles match numpy percentile with method='linear'."""
        r = sample_input_no_pp.values.squeeze()
        levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        expected = np.percentile(r, levels, method="linear")
        result = percentiles(sample_input_no_pp)
        assert result.name == "percentiles"
        assert result.value == pytest.approx(expected)

    def test_levels_in_meta(self, sample_input_no_pp):
        """Meta field records the percentile levels."""
        result = percentiles(sample_input_no_pp)
        assert "levels" in result.meta
        assert result.meta["levels"] == [1, 5, 10, 25, 50, 75, 90, 95, 99]

    def test_median_is_50th(self, sample_input_no_pp):
        """The 50th percentile (index 4) is the median."""
        r = sample_input_no_pp.values.squeeze()
        result = percentiles(sample_input_no_pp)
        assert result.value[4] == pytest.approx(np.median(r))

    def test_monotonic(self, sample_input_no_pp):
        """Percentiles should be monotonically increasing."""
        result = percentiles(sample_input_no_pp)
        assert np.all(np.diff(result.value) >= 0)

    def test_multi_strategy(self):
        """Multi-strategy returns 2-D array."""
        multi = np.column_stack([
            [0.01, -0.02, 0.03, -0.01, 0.005],
            [0.02, -0.01, 0.01, -0.03, 0.015],
        ])
        inp = ReturnsInput(multi)
        result = percentiles(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (9, 2)

    def test_nan_handling(self):
        """NaN values are excluded."""
        r = np.array([0.01, np.nan, -0.02, 0.015, np.nan, 0.03])
        inp = ReturnsInput(r)
        result = percentiles(inp)
        assert not np.any(np.isnan(result.value))


# ---------------------------------------------------------------------------
# 1.15 Coefficient of Variation
# Reference: Pearson (1896); Everitt & Skrondal (2010, The Cambridge Dictionary of Statistics)
# ---------------------------------------------------------------------------


class TestCoefficientOfVariation:
    def test_known_value(self, sample_input_no_pp):
        """CV = std / |mean|."""
        r = sample_input_no_pp.values.squeeze()
        std = np.std(r, ddof=1)
        mean = np.mean(r)
        expected = std / abs(mean)
        result = coefficient_of_variation(sample_input_no_pp)
        assert result.name == "coefficient_of_variation"
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_zero_mean(self):
        """When mean is zero, CV is NaN."""
        r = np.array([-0.02, -0.01, 0.01, 0.02])
        inp = ReturnsInput(r)
        result = coefficient_of_variation(inp)
        assert np.isnan(result.value)

    def test_positive_cv(self):
        """CV is always non-negative."""
        r = np.array([-0.03, -0.02, -0.01])
        inp = ReturnsInput(r)
        result = coefficient_of_variation(inp)
        assert result.value > 0.0

    def test_nan_handling(self):
        """NaN values are excluded."""
        r = np.array([0.01, np.nan, 0.02, -0.01, np.nan, 0.015])
        inp = ReturnsInput(r)
        result = coefficient_of_variation(inp)
        clean = r[~np.isnan(r)]
        clean_inp = ReturnsInput(clean)
        expected = coefficient_of_variation(clean_inp).value
        assert result.value == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 1.16 Outlier Count & % (IQR Method)
# Reference: Tukey (1977, Exploratory Data Analysis)
# ---------------------------------------------------------------------------


class TestOutlierIQR:
    def test_known_value(self, sample_input_no_pp):
        """Outlier detection via IQR method."""
        r = sample_input_no_pp.values.squeeze()
        q1 = np.percentile(r, 25)
        q3 = np.percentile(r, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        expected_count = int(np.sum((r < lower) | (r > upper)))
        expected_pct = expected_count / len(r) * 100.0

        result = outlier_iqr(sample_input_no_pp)
        assert result.name == "outlier_iqr"
        # Value is ndarray([count, percentage]) per output_index in meta
        assert result.value[0] == expected_count
        assert result.value[1] == pytest.approx(expected_pct)
        assert result.meta["output_index"] == ["count", "percentage"]

    def test_no_outliers(self):
        """Tight symmetric data: no outliers."""
        r = np.array([-0.005, -0.002, 0.0, 0.002, 0.005])
        inp = ReturnsInput(r)
        result = outlier_iqr(inp)
        assert result.value[0] == 0  # count
        assert result.value[1] == 0.0  # percentage

    def test_has_outliers(self):
        """Data with clear outliers."""
        r = np.array([-0.01, -0.005, 0.0, 0.005, 0.01, 0.10, -0.08])
        inp = ReturnsInput(r)
        result = outlier_iqr(inp)
        assert result.value[0] > 0  # count
        assert result.value[1] > 0.0  # percentage

    def test_multi_strategy(self):
        """Multi-strategy outlier detection."""
        multi = np.column_stack([
            [0.01, -0.02, 0.03, -0.01, 0.005],
            [0.02, -0.01, 0.01, -0.03, 0.015],
        ])
        inp = ReturnsInput(multi)
        result = outlier_iqr(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2, 2)

    def test_nan_handling(self):
        """NaN values are excluded from outlier detection."""
        r = np.array([0.01, np.nan, -0.02, 0.03, np.nan, -0.01, 0.005])
        inp = ReturnsInput(r)
        result = outlier_iqr(inp)

        # Verify value matches manual computation on clean data.
        clean = r[~np.isnan(r)]
        q1 = np.percentile(clean, 25)
        q3 = np.percentile(clean, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        expected_count = int(np.sum((clean < lower) | (clean > upper)))
        expected_pct = expected_count / len(clean) * 100.0

        assert result.value[0] == expected_count
        assert result.value[1] == pytest.approx(expected_pct)

    def test_method_in_meta(self, sample_input_no_pp):
        """Meta records the detection method."""
        result = outlier_iqr(sample_input_no_pp)
        assert "method" in result.meta
        assert "IQR" in result.meta["method"]


# ---------------------------------------------------------------------------
# Input type variations
# ---------------------------------------------------------------------------


class TestInputTypes:
    """Verify that all accepted input types (ndarray, Series, DataFrame, polars) work."""

    @pytest.fixture
    def data_1d(self):
        return np.array([0.01, -0.02, 0.015, 0.03, -0.01])

    @pytest.fixture
    def data_2d(self):
        return np.column_stack([
            [0.01, -0.02, 0.015, 0.03, -0.01],
            [0.02, -0.01, 0.01, -0.03, 0.005],
        ])

    def test_ndarray_1d(self, data_1d, daily_pp):
        """numpy 1-D array works."""
        inp = ReturnsInput(data_1d, periods_per_year=daily_pp)
        result = arithmetic_mean_return(inp)
        assert result.value == pytest.approx(np.mean(data_1d))

    def test_ndarray_2d(self, data_2d, daily_pp):
        """numpy 2-D array works (multi-strategy)."""
        inp = ReturnsInput(data_2d, periods_per_year=daily_pp)
        result = arithmetic_mean_return(inp)
        assert isinstance(result.value, np.ndarray)
        assert len(result.value) == 2

    def test_pandas_series(self, data_1d, daily_pp):
        """pandas Series works."""
        s = pd.Series(data_1d)
        inp = ReturnsInput(s, periods_per_year=daily_pp)
        result = arithmetic_mean_return(inp)
        assert result.value == pytest.approx(np.mean(data_1d))

    def test_pandas_dataframe(self, data_2d, daily_pp):
        """pandas DataFrame works (multi-strategy)."""
        df = pd.DataFrame(data_2d)
        inp = ReturnsInput(df, periods_per_year=daily_pp)
        result = arithmetic_mean_return(inp)
        assert isinstance(result.value, np.ndarray)
        assert len(result.value) == 2

    def test_polars_series(self, data_1d, daily_pp):
        """polars Series works."""
        pl = pytest.importorskip("polars")
        s = pl.Series("returns", data_1d)
        inp = ReturnsInput(s, periods_per_year=daily_pp)
        result = arithmetic_mean_return(inp)
        assert result.value == pytest.approx(np.mean(data_1d))

    def test_polars_dataframe(self, data_2d, daily_pp):
        """polars DataFrame works (multi-strategy)."""
        pl = pytest.importorskip("polars")
        df = pl.DataFrame(data_2d)
        inp = ReturnsInput(df, periods_per_year=daily_pp)
        result = arithmetic_mean_return(inp)
        assert isinstance(result.value, np.ndarray)
        assert len(result.value) == 2


# ---------------------------------------------------------------------------
# Edge case: empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_array(self):
        """Empty array: nanmean-based metrics should return NaN."""
        r = np.array([])
        inp = ReturnsInput(r)
        result = arithmetic_mean_return(inp)
        assert np.isnan(result.value)

    def test_all_nan(self):
        """All-NaN returns should produce NaN output."""
        r = np.full(10, np.nan)
        inp = ReturnsInput(r)
        result = arithmetic_mean_return(inp)
        assert np.isnan(result.value)


# ---------------------------------------------------------------------------
# Edge case: all-zero returns
# ---------------------------------------------------------------------------


class TestAllZeroReturns:
    def test_all_zero_volatility(self, daily_pp):
        """All-zero returns have zero volatility."""
        r = np.zeros(100)
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = annualized_volatility(inp)
        assert result.value == 0.0

    def test_all_zero_variance(self):
        """All-zero returns have zero variance."""
        r = np.zeros(10)
        inp = ReturnsInput(r)
        result = variance(inp)
        assert result.value == 0.0

    def test_all_zero_autocorrelation(self):
        """All-zero returns have undefined autocorrelation (NaN)."""
        r = np.zeros(10)
        inp = ReturnsInput(r)
        result = autocorrelation(inp)
        assert np.isnan(result.value)

    def test_all_zero_range(self):
        """All-zero: range = 0."""
        r = np.zeros(5)
        inp = ReturnsInput(r)
        assert return_range(inp).value == 0.0

    def test_all_zero_positive_ratio(self):
        """All-zero returns: zero is not positive."""
        r = np.zeros(5)
        inp = ReturnsInput(r)
        assert positive_period_ratio(inp).value == 0.0

    def test_all_zero_cv(self):
        """All-zero returns: CV is NaN (mean is zero)."""
        r = np.zeros(5)
        inp = ReturnsInput(r)
        result = coefficient_of_variation(inp)
        assert np.isnan(result.value)

    def test_all_zero_outlier(self):
        """All-zero: no outliers since IQR = 0."""
        r = np.zeros(10)
        inp = ReturnsInput(r)
        result = outlier_iqr(inp)
        assert result.value[0] == 0  # count
        assert result.value[1] == 0.0  # percentage


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_all_metrics_registered(self):
        """All 16 descriptive metrics appear in the registry."""
        from stratstat.registry import list_metrics

        desc_metrics = list_metrics(requires="returns", category="descriptive")
        names = {m["name"] for m in desc_metrics}
        expected = {
            "cagr",
            "annualized_volatility",
            "cumulative_return",
            "arithmetic_mean_return",
            "geometric_mean_return",
            "skewness",
            "excess_kurtosis",
            "best_period",
            "worst_period",
            "positive_period_ratio",
            "autocorrelation",
            "variance",
            "return_range",
            "percentiles",
            "coefficient_of_variation",
            "outlier_iqr",
        }
        assert names == expected

    def test_compute_via_public_api(self, sample_input):
        """Metrics are reachable via stratstat.compute()."""
        from stratstat import compute

        result = compute(sample_input, "cagr")
        assert result.name == "cagr"
        assert isinstance(result.value, float)

    def test_compute_all_descriptive(self, sample_input):
        """compute_all with category='descriptive' returns all 16 metrics."""
        from stratstat import compute_all

        result_set = compute_all(sample_input, category="descriptive")
        assert len(result_set) == 16
        names = {r.name for r in result_set}
        assert "cagr" in names
        assert "skewness" in names
