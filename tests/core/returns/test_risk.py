"""Tests for risk metrics.

Validates against known values from hand-computed examples and cross-validation
against numpy computations. References noted per build instructions.
"""

from __future__ import annotations

import numpy as np
import pytest

from stratstat.core.returns.risk import (
    average_drawdown,
    average_drawdown_duration,
    common_sense_ratio,
    current_drawdown,
    current_drawdown_duration,
    cvar,
    downside_deviation,
    downside_semivariance,
    drawdown_periods_count,
    drawdown_total_duration,
    drawdown_volatility,
    gpd_tail_fit,
    hill_tail_index,
    longest_drawdown_duration,
    max_drawdown,
    modified_var,
    pain_index,
    prospect_ratio,
    risk_of_ruin,
    tail_ratio,
    time_to_recovery,
    ulcer_index,
    upside_deviation,
    var,
)
from stratstat.inputs import ReturnsInput

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_pp():
    return 252


@pytest.fixture
def sample_returns():
    """10 daily returns used in descriptive tests."""
    return np.array(
        [0.01, -0.02, 0.015, 0.03, -0.01, 0.005, -0.015, 0.02, -0.005, 0.01]
    )


@pytest.fixture
def sample_input(sample_returns, daily_pp):
    return ReturnsInput(sample_returns, periods_per_year=daily_pp)


# ---------------------------------------------------------------------------
# 2.1 Max Drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_no_loss(self):
        """All positive returns: max drawdown = 0.0."""
        r = np.array([0.01, 0.02, 0.005, 0.03])
        inp = ReturnsInput(r)
        result = max_drawdown(inp)
        assert result.value == pytest.approx(0.0, abs=1e-15)

    def test_known_drawdown(self):
        """Simple case with a known max drawdown.

        Returns: [0.10, -0.20, 0.05]
        Equity: [1.0, 1.10, 0.88, 0.924]
        Running max: [1.0, 1.10, 1.10, 1.10]
        DD: [0.0, 0.0, -0.20, -0.16]
        Max DD = -0.20
        """
        r = np.array([0.10, -0.20, 0.05])
        inp = ReturnsInput(r)
        result = max_drawdown(inp)
        assert result.value == pytest.approx(-0.20, rel=1e-10)

    def test_log_return_type(self):
        """Log returns: equity via exp(cumsum(r))."""
        r = np.array([0.10, -0.20, 0.05])
        inp = ReturnsInput(r)
        result = max_drawdown(inp, return_type="log")
        expected = (np.exp(-0.10) - np.exp(0.10)) / np.exp(0.10)
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_invalid_return_type(self):
        """Invalid return_type raises ValueError."""
        inp = ReturnsInput(np.array([0.01, -0.01]))
        with pytest.raises(ValueError, match="return_type"):
            max_drawdown(inp, return_type="invalid")

    def test_multi_strategy(self):
        """Multi-strategy returns array output."""
        multi = np.column_stack([[0.10, -0.20, 0.05], [0.05, 0.03, -0.01]])
        inp = ReturnsInput(multi)
        result = max_drawdown(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)


# ---------------------------------------------------------------------------
# 2.2 Longest Drawdown Duration
# ---------------------------------------------------------------------------


class TestLongestDrawdownDuration:
    def test_no_drawdown(self):
        """No drawdown periods so zero longest duration."""
        r = np.array([0.01, 0.02, 0.01])
        inp = ReturnsInput(r)
        result = longest_drawdown_duration(inp)
        assert result.value == 0.0

    def test_known_duration(self):
        """Returns: [0.10, -0.05, -0.03, 0.20] so one drawdown of 2 periods."""
        r = np.array([0.10, -0.05, -0.03, 0.20])
        inp = ReturnsInput(r)
        result = longest_drawdown_duration(inp)
        assert result.value == 2.0

    def test_units_years(self, daily_pp):
        """units='years' divides by periods_per_year."""
        r = np.array([0.10, -0.05, -0.03, 0.20, -0.01, -0.02, 0.30])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result_periods = longest_drawdown_duration(inp, units="periods")
        result_years = longest_drawdown_duration(inp, units="years")
        assert result_years.value == pytest.approx(
            result_periods.value / daily_pp
        )

    def test_units_years_requires_pp(self):
        """units='years' without periods_per_year raises ValueError."""
        inp = ReturnsInput(np.array([0.10, -0.05, 0.20]))
        with pytest.raises(ValueError, match="periods_per_year"):
            longest_drawdown_duration(inp, units="years")


# ---------------------------------------------------------------------------
# 2.3 Time to Recovery
# ---------------------------------------------------------------------------


class TestTimeToRecovery:
    def test_known_recovery(self):
        """Returns: [0.10, -0.05, -0.03, 0.20].
        One episode: 2 periods underwater, recovers on 4th period.
        """
        r = np.array([0.10, -0.05, -0.03, 0.20])
        inp = ReturnsInput(r)
        result = time_to_recovery(inp)
        # Value is ndarray([mean, median, max]) per output_index
        assert result.value[0] == 2.0  # mean
        assert result.value[1] == 2.0  # median
        assert result.value[2] == 2.0  # max
        assert result.meta["output_index"] == ["mean", "median", "max"]

    def test_unrecovered_episode(self):
        """Ending underwater: unrecovered episodes excluded from stats."""
        r = np.array([0.10, -0.20, -0.05])
        inp = ReturnsInput(r)
        result = time_to_recovery(inp)
        assert np.isnan(result.value[0])  # mean is NaN


# ---------------------------------------------------------------------------
# 2.4 Average Drawdown
# ---------------------------------------------------------------------------


class TestAverageDrawdown:
    def test_no_drawdown(self):
        """No drawdowns so ADD = 0.0."""
        r = np.array([0.01, 0.02, 0.03])
        inp = ReturnsInput(r)
        result = average_drawdown(inp)
        assert result.value == 0.0

    def test_with_drawdowns(self):
        """Two episodes: both negative depths, average is negative."""
        r = np.array([0.10, -0.05, 0.10, -0.03, 0.10])
        inp = ReturnsInput(r)
        result = average_drawdown(inp)
        assert result.value < 0.0


# ---------------------------------------------------------------------------
# 2.5 Average Drawdown Duration
# ---------------------------------------------------------------------------


class TestAverageDrawdownDuration:
    def test_known_value(self):
        """Two drawdowns: 1 period and 2 periods so avg = 1.5."""
        r = np.array([0.10, -0.05, 0.10, -0.03, -0.01, 0.10])
        inp = ReturnsInput(r)
        result = average_drawdown_duration(inp)
        assert result.value == 1.5

    def test_no_drawdown(self):
        """No drawdowns so zero duration."""
        r = np.array([0.01, 0.02, 0.01])
        inp = ReturnsInput(r)
        assert average_drawdown_duration(inp).value == 0.0


# ---------------------------------------------------------------------------
# 2.6 Ulcer Index
# ---------------------------------------------------------------------------


class TestUlcerIndex:
    def test_known_value(self):
        """UI from simple drawdown series.

        Returns: [0.10, -0.20, 0.05]
        Equity: [1.0, 1.1, 0.88, 0.924]
        DD: [0.0, 0.0, -0.20, -0.16]
        UI = sqrt(mean([0, 0, 0.04, 0.0256])) = sqrt(0.0164)
        """
        r = np.array([0.10, -0.20, 0.05])
        inp = ReturnsInput(r)
        result = ulcer_index(inp)
        expected = np.sqrt(np.mean([0.0, 0.0, 0.04, 0.0256]))
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_no_drawdown(self):
        """All gains gives UI = 0."""
        r = np.array([0.01, 0.02, 0.01])
        inp = ReturnsInput(r)
        assert ulcer_index(inp).value == 0.0

    def test_non_negative(self):
        """UI is always non-negative."""
        r = np.array([-0.10, -0.20, -0.05, 0.01])
        inp = ReturnsInput(r)
        assert ulcer_index(inp).value >= 0.0


# ---------------------------------------------------------------------------
# 2.7 Downside Deviation
# ---------------------------------------------------------------------------


class TestDownsideDeviation:
    def test_known_value(self):
        """Returns [0.02, -0.03, 0.01, -0.01], mar=0.
        Below-mar squared: [0, 0.0009, 0, 0.0001].
        Mean = 0.0010/4 = 0.00025, sqrt = 0.015811...
        """
        r = np.array([0.02, -0.03, 0.01, -0.01])
        inp = ReturnsInput(r)
        result = downside_deviation(inp)
        expected = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_all_positive(self):
        """No returns below MAR so DD = 0."""
        r = np.array([0.01, 0.02, 0.03])
        inp = ReturnsInput(r)
        assert downside_deviation(inp).value == 0.0

    def test_custom_mar(self):
        """Non-zero MAR shifts threshold."""
        r = np.array([0.05, 0.01, -0.02, 0.03])
        inp = ReturnsInput(r)
        mar = 0.02
        result = downside_deviation(inp, mar=mar)
        below = np.minimum(r - mar, 0.0)
        expected = np.sqrt(np.mean(below**2))
        assert result.value == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 2.8 Upside Deviation
# ---------------------------------------------------------------------------


class TestUpsideDeviation:
    def test_known_value(self):
        """Returns [0.02, -0.03, 0.01, -0.01], mar=0."""
        r = np.array([0.02, -0.03, 0.01, -0.01])
        inp = ReturnsInput(r)
        result = upside_deviation(inp)
        expected = np.sqrt(np.mean(np.maximum(r, 0) ** 2))
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_all_negative(self):
        """No returns above MAR so UD = 0."""
        r = np.array([-0.01, -0.02, -0.03])
        inp = ReturnsInput(r)
        assert upside_deviation(inp).value == 0.0


# ---------------------------------------------------------------------------
# 2.9 VaR (Value at Risk)
# ---------------------------------------------------------------------------


class TestVaR:
    def test_historical_known_value(self):
        """Historical VaR at 95% for 100 returns.
        VaR = -(5th percentile of returns).
        """
        rng = np.random.default_rng(123)
        r = rng.normal(0.0, 0.02, 100)
        inp = ReturnsInput(r)
        result = var(inp, method="historical", confidence=0.95)
        expected = -np.percentile(r, 5.0)
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_var_larger_than_cvar(self, sample_input):
        """CVaR should be >= VaR."""
        var_val = var(sample_input, method="historical", confidence=0.95).value
        cvar_val = cvar(sample_input, method="historical", confidence=0.95).value
        assert cvar_val >= var_val

    def test_invalid_method(self):
        """Invalid method raises ValueError."""
        inp = ReturnsInput(np.array([0.01, -0.01]))
        with pytest.raises(ValueError, match="method"):
            var(inp, method="invalid")

    def test_confidence_bounds(self):
        """confidence must be in (0, 1)."""
        inp = ReturnsInput(np.array([0.01, -0.01]))
        with pytest.raises(ValueError):
            var(inp, confidence=1.5)

    def test_cornish_fisher(self):
        """Cornish-Fisher VaR with well-behaved data."""
        r = np.array(
            [0.01, -0.02, 0.015, -0.01, 0.005, 0.02, -0.015, 0.01] * 30
        )
        inp = ReturnsInput(r)
        result = var(inp, method="cornish_fisher", confidence=0.95)
        assert result.value > 0.0

    def test_cornish_fisher_insufficient_data(self):
        """CF VaR with too few observations returns NaN."""
        r = np.array([0.01, -0.02])
        inp = ReturnsInput(r)
        result = var(inp, method="cornish_fisher", confidence=0.95)
        assert np.isnan(result.value)


# ---------------------------------------------------------------------------
# 2.10 CVaR / Expected Shortfall
# ---------------------------------------------------------------------------


class TestCVaR:
    def test_historical_positive(self):
        """CVaR is a positive loss magnitude for mixed returns."""
        r = np.array(
            [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        )
        inp = ReturnsInput(r)
        result = cvar(inp, method="historical", confidence=0.80)
        assert result.value > 0.0

    def test_parametric_basic(self, sample_input):
        """Parametric CVaR for known data."""
        result = cvar(sample_input, method="parametric", confidence=0.95)
        assert result.value > 0.0


# ---------------------------------------------------------------------------
# 2.11 Tail Ratio
# ---------------------------------------------------------------------------


class TestTailRatio:
    def test_positive_value(self, sample_input):
        """Tail ratio should be positive."""
        result = tail_ratio(sample_input)
        assert result.value > 0.0

    def test_invalid_cutoff(self):
        """tail_cutoff must be in (0, 0.5)."""
        inp = ReturnsInput(np.array([0.01, -0.01]))
        with pytest.raises(ValueError):
            tail_ratio(inp, tail_cutoff=0.6)

    def test_symmetric_returns(self):
        """Symmetric returns have tail_ratio close to 1."""
        rng = np.random.default_rng(42)
        r = rng.normal(0.0, 0.02, 1000)
        inp = ReturnsInput(r)
        result = tail_ratio(inp, tail_cutoff=0.10)
        assert 0.5 < result.value < 2.0


# ---------------------------------------------------------------------------
# 2.12 Common-Sense Ratio
# ---------------------------------------------------------------------------


class TestCommonSenseRatio:
    def test_positive_value(self, sample_input):
        """CSR should be positive for mixed returns."""
        result = common_sense_ratio(sample_input)
        assert result.value > 0.0


# ---------------------------------------------------------------------------
# 2.13 Hill Tail Index
# ---------------------------------------------------------------------------


class TestHillTailIndex:
    def test_positive_heavy_tail(self):
        """Heavy-tailed positive returns give Hill > 0."""
        rng = np.random.default_rng(99)
        r = rng.standard_t(df=3, size=500) * 0.02
        inp = ReturnsInput(r)
        result = hill_tail_index(inp, tail_fraction=0.10)
        assert result.value > 0.0

    def test_insufficient_data(self):
        """Too few observations for k >= 2 gives NaN."""
        r = np.array([0.01, 0.02])
        inp = ReturnsInput(r)
        result = hill_tail_index(inp, tail_fraction=0.10)
        assert np.isnan(result.value)

    def test_invalid_fraction(self):
        """tail_fraction must be in (0, 0.5]."""
        inp = ReturnsInput(np.array([0.01, -0.01]))
        with pytest.raises(ValueError):
            hill_tail_index(inp, tail_fraction=0.6)


# ---------------------------------------------------------------------------
# 2.14 GPD Tail Fit
# ---------------------------------------------------------------------------


class TestGPDTailFit:
    def test_returns_shape_scale(self, sample_input):
        """GPD fit returns shape and scale as ndarray([shape, scale])."""
        result = gpd_tail_fit(sample_input)
        assert result.meta["output_index"] == ["shape", "scale"]
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)

    def test_insufficient_data(self):
        """Too few observations gives NaN parameters."""
        r = np.array([0.01, -0.02, 0.01])
        inp = ReturnsInput(r)
        result = gpd_tail_fit(inp)
        assert np.isnan(result.value[0])  # shape
        assert np.isnan(result.value[1])  # scale


# ---------------------------------------------------------------------------
# 2.15 Risk of Ruin
# ---------------------------------------------------------------------------


class TestRiskOfRuin:
    def test_probability_range(self, sample_input):
        """Risk of ruin is between 0 and 1."""
        result = risk_of_ruin(sample_input)
        assert 0.0 <= result.value <= 1.0

    def test_requires_periods_per_year(self):
        """Raises when periods_per_year is None."""
        inp = ReturnsInput(np.array([0.01, -0.02]))
        with pytest.raises(ValueError, match="periods_per_year"):
            risk_of_ruin(inp)

    def test_positive_mean_low_ruin(self, daily_pp):
        """Strong positive returns give low risk of ruin."""
        r = np.array([0.01, 0.02, 0.015, 0.02, 0.01] * 50)
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = risk_of_ruin(inp)
        assert result.value < 0.5

    def test_warning_in_meta(self, sample_input):
        """Docstring warning about normality is in meta."""
        result = risk_of_ruin(sample_input)
        assert "warning" in result.meta
        assert "normality" in result.meta["warning"].lower()


# ---------------------------------------------------------------------------
# 2.16-2.20 Drawdown Family
# ---------------------------------------------------------------------------


class TestDrawdownFamily:
    def test_count(self):
        """Count of drawdown episodes."""
        r = np.array([0.10, -0.05, 0.10, -0.03, 0.05])
        inp = ReturnsInput(r)
        result = drawdown_periods_count(inp)
        assert result.value == 2.0

    def test_no_drawdown_count(self):
        """No drawdown gives zero count."""
        r = np.array([0.01, 0.02, 0.03])
        inp = ReturnsInput(r)
        assert drawdown_periods_count(inp).value == 0.0

    def test_current_drawdown_underwater(self):
        """Ending underwater: current drawdown < 0."""
        r = np.array([0.10, -0.20, -0.05])
        inp = ReturnsInput(r)
        result = current_drawdown(inp)
        assert result.value < 0.0

    def test_current_drawdown_at_peak(self):
        """Ending at new high: current drawdown = 0.0."""
        r = np.array([-0.01, 0.05])
        inp = ReturnsInput(r)
        result = current_drawdown(inp)
        assert result.value == pytest.approx(0.0, abs=1e-14)

    def test_current_duration_at_peak(self):
        """Ending at peak gives 0 duration."""
        r = np.array([0.10, 0.05])
        inp = ReturnsInput(r)
        assert current_drawdown_duration(inp).value == 0.0

    def test_current_duration_underwater(self):
        """Ending underwater gives positive duration."""
        r = np.array([0.10, -0.20, -0.05])
        inp = ReturnsInput(r)
        result = current_drawdown_duration(inp)
        assert result.value > 0.0

    def test_total_duration(self):
        """Total underwater duration sums across episodes."""
        r = np.array([0.10, -0.05, 0.10, -0.03, -0.01, 0.10])
        inp = ReturnsInput(r)
        result = drawdown_total_duration(inp)
        assert result.value == 3.0


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_all_risk_metrics_registered(self):
        """All 24 risk metrics appear in the registry."""
        from stratstat.registry import list_metrics

        risk_metrics = list_metrics(requires="returns", category="risk")
        names = {m["name"] for m in risk_metrics}
        expected = {
            "max_drawdown",
            "longest_drawdown_duration",
            "time_to_recovery",
            "average_drawdown",
            "average_drawdown_duration",
            "ulcer_index",
            "downside_deviation",
            "downside_semivariance",
            "upside_deviation",
            "var",
            "modified_var",
            "cvar",
            "tail_ratio",
            "common_sense_ratio",
            "hill_tail_index",
            "gpd_tail_fit",
            "risk_of_ruin",
            "drawdown_volatility",
            "drawdown_periods_count",
            "current_drawdown",
            "current_drawdown_duration",
            "drawdown_total_duration",
            "pain_index",
            "prospect_ratio",
        }
        assert names == expected

    def test_compute_via_public_api(self, sample_input):
        """Metrics reachable via stratstat.compute()."""
        from stratstat import compute

        result = compute(sample_input, "max_drawdown")
        assert result.name == "max_drawdown"
        assert result.value <= 0.0


# ---------------------------------------------------------------------------
# Edge case: NaN handling
# ---------------------------------------------------------------------------


class TestNaNHandling:
    def test_downside_deviation_with_nan(self):
        """NaN values are excluded."""
        r = np.array([0.02, np.nan, -0.03, 0.01])
        inp = ReturnsInput(r)
        result = downside_deviation(inp)
        clean = r[~np.isnan(r)]
        expected = np.sqrt(np.mean(np.minimum(clean, 0) ** 2))
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_ulcer_index_with_nan(self):
        """NaN values are excluded from drawdown series."""
        r = np.array([0.02, np.nan, -0.03, 0.01])
        inp = ReturnsInput(r)
        result = ulcer_index(inp)
        assert not np.isnan(result.value)

    def test_max_drawdown_with_nan(self):
        """NaN in returns: drawdown computed correctly on non-NaN periods."""
        r = np.array([0.10, np.nan, -0.20, 0.05])
        inp = ReturnsInput(r)
        result = max_drawdown(inp)
        assert result.value <= 0.0
        assert not np.isnan(result.value)

    def test_var_historical_with_nan(self):
        """VaR historical ignores NaN values."""
        r = np.array([0.02, np.nan, -0.03, 0.01, -0.01])
        inp = ReturnsInput(r)
        result = var(inp, method="historical", confidence=0.95)
        assert not np.isnan(result.value)

    def test_cvar_historical_with_nan(self):
        """CVaR historical ignores NaN values."""
        r = np.array([0.02, np.nan, -0.03, 0.01, -0.01])
        inp = ReturnsInput(r)
        result = cvar(inp, method="historical", confidence=0.95)
        assert not np.isnan(result.value)

    def test_longest_drawdown_duration_with_nan(self):
        """Longest DD duration handles NaN returns."""
        r = np.array([0.10, np.nan, -0.05, -0.03, 0.20])
        inp = ReturnsInput(r)
        result = longest_drawdown_duration(inp)
        assert not np.isnan(result.value)


# ---------------------------------------------------------------------------
# Deterministic known-value tests (review findings)
# ---------------------------------------------------------------------------


class TestVaRKnownValues:
    """VaR tests with deterministic, hand-computed expected values."""

    def test_historical_deterministic(self):
        """Historical VaR at 90% confidence on 10-element array.

        Returns: [0.02, -0.03, 0.01, -0.01, -0.02, 0.04, -0.01, 0.005, 0.015, -0.005]
        Sorted: [-0.03, -0.02, -0.01, -0.01, -0.005, 0.005, 0.01, 0.015, 0.02, 0.04]
        10th percentile (alpha=0.10): linear interpolation between index 0 and 1:
        index 0 = -0.03, index 1 = -0.02. percentile = -0.03 + 0.9*(-0.02 + 0.03) = -0.021.
        Actually, numpy method='linear' (type 7): pct at 10% = -0.029 (interpolation).
        VaR = -percentile.
        """
        r = np.array(
            [0.02, -0.03, 0.01, -0.01, -0.02, 0.04, -0.01, 0.005, 0.015, -0.005]
        )
        inp = ReturnsInput(r)
        result = var(inp, method="historical", confidence=0.90)
        expected = -np.percentile(r, 10.0)
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_parametric_deterministic(self):
        """Parametric VaR at 95% on 5 returns.

        Returns: [0.01, -0.02, 0.015, -0.01, 0.005]
        Mean = 0.0, Std (ddof=1) = sqrt(0.000825/4) = sqrt(0.00020625) = 0.014361...
        z_0.05 = -1.64485...
        VaR = -(0 + (-1.645) * 0.01436) ≈ 0.02364
        """
        r = np.array([0.01, -0.02, 0.015, -0.01, 0.005])
        inp = ReturnsInput(r)
        result = var(inp, method="parametric", confidence=0.95)
        mean = np.mean(r)
        std = np.std(r, ddof=1)
        # z_0.05 = norm.ppf(0.05) ≈ -1.64485...
        expected = -(mean - 1.6448536269514729 * std)
        assert result.value == pytest.approx(expected, rel=1e-8)
        assert result.value > 0.0

    def test_cornish_fisher_deterministic(self):
        """Cornish-Fisher VaR at 95% on 30 repeated values.

        Uses the 5-element pattern repeated 6 times for enough observations
        for skewness/kurtosis estimation. Checks that CF VaR differs from
        parametric (because skew/kurt are non-zero) and is positive.
        """
        base = np.array([0.01, -0.02, 0.015, -0.01, 0.005])
        r = np.tile(base, 12)  # 60 obs
        inp = ReturnsInput(r)
        cf_result = var(inp, method="cornish_fisher", confidence=0.95)
        param_result = var(inp, method="parametric", confidence=0.95)
        # CF should differ from parametric for non-normal data
        assert cf_result.value != pytest.approx(param_result.value, abs=1e-6)
        assert cf_result.value > 0.0


class TestTailRatioDeterministic:
    """Tail ratio with deterministic, hand-computed expected values."""

    def test_symmetric_deterministic(self):
        """Symmetric returns: [-0.03, -0.01, -0.005, 0.005, 0.01, 0.03], cutoff=1/3.

        tail_cutoff ≈ 0.1667 (2 out of 12 in each tail, or 1 out of 6).
        Actually with cutoff=1/3 ≈ 0.333, lower: <= 33rd pct, upper: >= 67th pct.
        With 6 values at cutoff=1/3: lower 2 = [-0.03, -0.01], upper 2 = [0.01, 0.03].
        TR = mean([0.01, 0.03]) / |mean([-0.03, -0.01])| = 0.02/0.02 = 1.0.
        """
        r = np.array([-0.03, -0.01, -0.005, 0.005, 0.01, 0.03])
        inp = ReturnsInput(r)
        result = tail_ratio(inp, tail_cutoff=1.0 / 3.0)
        assert result.value == pytest.approx(1.0, rel=1e-10)


class TestCommonSenseRatioDeterministic:
    """CSR with deterministic known values."""

    def test_known_value(self):
        """Returns: [0.02, -0.01, 0.03, -0.02], cutoff=0.25.

        Tail ratio at 0.25: 25th pct lower = -0.02, 75th pct upper = 0.03.
        TR = 0.03 / 0.02 = 1.5.
        Gain-to-pain = (0.02 + 0.03) / |(-0.01 + -0.02)| = 0.05 / 0.03 = 1.666...
        CSR = 1.5 * 1.666... = 2.5.
        """
        r = np.array([0.02, -0.01, 0.03, -0.02])
        inp = ReturnsInput(r)
        result = common_sense_ratio(inp, tail_cutoff=0.25)
        assert result.value == pytest.approx(2.5, rel=1e-10)


class TestDrawdownVolatility:
    """Drawdown volatility — previously untested."""

    def test_known_value(self):
        """DD vol = std of drawdown series.
        Returns: [0.10, -0.20, 0.05]
        Equity: [1.0, 1.1, 0.88, 0.924]
        DD: [0.0, 0.0, -0.20, -0.16]
        Std of DD (ddof=1): mean = -0.09, var = ((0.09^2)+(0.09^2)+(0.11^2)+(-0.07^2))/3
        = (0.0081+0.0081+0.0121+0.0049)/3 = 0.0332/3 = 0.011067
        std = sqrt(0.011067) ≈ 0.1052
        """
        r = np.array([0.10, -0.20, 0.05])
        inp = ReturnsInput(r)
        result = drawdown_volatility(inp)
        dd_series = np.array([0.0, 0.0, -0.20, -0.16])
        expected = np.std(dd_series, ddof=1)
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_no_drawdown(self):
        """All gains gives DD vol = 0.0."""
        r = np.array([0.01, 0.02, 0.01])
        inp = ReturnsInput(r)
        result = drawdown_volatility(inp)
        assert result.value == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Multi-strategy tests (review finding)
# ---------------------------------------------------------------------------


class TestMultiStrategy:
    def test_var_multi(self):
        """VaR on multi-strategy returns."""
        multi = np.column_stack([
            [0.01, -0.02, 0.03, -0.01, 0.005],
            [0.02, -0.01, 0.01, -0.03, 0.015],
        ])
        inp = ReturnsInput(multi)
        result = var(inp, method="historical", confidence=0.95)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)

    def test_cvar_multi(self):
        """CVaR on multi-strategy returns."""
        multi = np.column_stack([
            [0.01, -0.02, 0.03, -0.01, 0.005],
            [0.02, -0.01, 0.01, -0.03, 0.015],
        ])
        inp = ReturnsInput(multi)
        result = cvar(inp, method="historical", confidence=0.95)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)

    def test_ulcer_index_multi(self):
        """Ulcer index on multi-strategy returns."""
        multi = np.column_stack([
            [0.10, -0.20, 0.05],
            [0.05, -0.10, 0.02],
        ])
        inp = ReturnsInput(multi)
        result = ulcer_index(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)


# ---------------------------------------------------------------------------
# Numba vs pure-numpy agreement (review finding)
# ---------------------------------------------------------------------------


class TestNumbaAgreement:
    def test_drawdown_episodes_agree(self):
        """Numba and pure-numpy paths produce identical episode lists."""
        from stratstat.core.returns.risk import (
            _drawdown_episodes,
            _drawdown_episodes_numba,
            _drawdown_series,
            _equity_curve,
        )

        r = np.array([0.01, -0.02, 0.015, 0.03, -0.01, 0.005, -0.015, 0.02, -0.005, 0.01])
        eq = _equity_curve(r.reshape(-1, 1), "simple")
        rm, dd_ser = _drawdown_series(eq)
        eq1d, rm1d, dd1d = eq[:, 0], rm[:, 0], dd_ser[:, 0]

        pure = _drawdown_episodes(eq1d, rm1d, dd1d)
        numba_result = _drawdown_episodes_numba(eq1d, rm1d, dd1d)

        assert len(pure) == len(numba_result)
        for pe, ne in zip(pure, numba_result, strict=True):
            assert pe["start"] == ne["start"]
            assert pe["end"] == ne["end"]
            assert pe["trough_idx"] == ne["trough_idx"]
            assert pe["depth"] == pytest.approx(ne["depth"])
            assert pe["duration"] == ne["duration"]
            assert pe["recovered"] == ne["recovered"]


# ---------------------------------------------------------------------------
# Pain Index
# ---------------------------------------------------------------------------


class TestPainIndex:
    """Tests for pain_index metric."""

    def test_known_value(self):
        """Pain Index should be less negative than average drawdown since it
        includes zero-drawdown periods."""
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.005, 0.04, -0.03])
        inp = ReturnsInput(returns)
        result = pain_index(inp)
        # Pain Index = mean of all drawdowns (including zeros)
        assert result.value < 0  # negative (there are drawdowns)
        assert result.name == "pain_index"

    def test_pain_index_vs_average_drawdown(self):
        """Pain Index is period-level (includes zeros); Average Drawdown is
        episode-level (only underwater periods). Both should be negative
        for a strategy with drawdowns."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, size=252)
        inp = ReturnsInput(returns)

        pi_result = pain_index(inp)
        add_result = average_drawdown(inp)

        # Both are negative when there are drawdowns
        assert pi_result.value < 0
        assert add_result.value < 0
        # Pain Index should be ≤ 0 (could be 0 if no drawdowns)
        assert pi_result.value <= 0

    def test_no_drawdowns(self):
        """Pain Index should be exactly 0 for strictly increasing equity."""
        returns = np.array([0.01, 0.02, 0.01, 0.03, 0.02])
        inp = ReturnsInput(returns)
        result = pain_index(inp)
        assert result.value == pytest.approx(0.0)

    def test_multi_strategy(self):
        """Pain Index should return an array for multi-strategy input."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, size=(252, 3))
        inp = ReturnsInput(returns)
        result = pain_index(inp)
        assert hasattr(result.value, "shape")
        assert result.value.shape == (3,)


# ---------------------------------------------------------------------------
# Prospect Ratio
# ---------------------------------------------------------------------------


class TestProspectRatio:
    """Tests for prospect_ratio metric."""

    def test_symmetric_returns(self):
        """Prospect Ratio should be near 1 for symmetric return distributions."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, size=1000)
        inp = ReturnsInput(returns)
        result = prospect_ratio(inp)
        # Should be close to 1 for a symmetric distribution
        assert result.value == pytest.approx(1.0, rel=0.2)

    def test_positive_skew(self):
        """Prospect Ratio > 1 when upside semivariance > downside semivariance."""
        # Positive skew: large gains, small losses
        returns = np.array([0.05, 0.02, -0.005, 0.03, -0.01, 0.04, -0.005, -0.01])
        inp = ReturnsInput(returns)
        result = prospect_ratio(inp)
        assert result.value > 1.0

    def test_no_downside(self):
        """Prospect Ratio should be inf when there is no downside semivariance."""
        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02])
        inp = ReturnsInput(returns)
        result = prospect_ratio(inp)
        assert result.value == float("inf")

    def test_all_zero(self):
        """Prospect Ratio should be NaN for all-zero returns."""
        returns = np.zeros(100)
        inp = ReturnsInput(returns)
        result = prospect_ratio(inp)
        assert np.isnan(result.value)

    def test_with_mar(self):
        """Prospect Ratio with non-zero MAR."""
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        inp = ReturnsInput(returns)
        result = prospect_ratio(inp, mar=0.01)
        assert result.value > 0
        assert result.meta["mar"] == 0.01


# ---------------------------------------------------------------------------
# Downside Semi-Variance
# ---------------------------------------------------------------------------


class TestDownsideSemivariance:
    """Tests for downside_semivariance metric."""

    def test_sqrt_of_semivar_is_downside_dev(self):
        """sqrt(downside_semivariance) == downside_deviation."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, size=500)
        inp = ReturnsInput(returns)
        dsv_result = downside_semivariance(inp)
        dd_result = downside_deviation(inp)
        assert np.sqrt(dsv_result.value) == pytest.approx(dd_result.value, rel=1e-12)

    def test_known_values(self):
        """Hand-computed check."""
        returns = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
        inp = ReturnsInput(returns)
        # Only below-zero entries: -0.02^2=0.0004, -0.01^2=0.0001
        expected = (0.0004 + 0.0001) / 5.0  # 0.0001
        result = downside_semivariance(inp)
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_no_downside(self):
        """Zero when no periods fall below MAR."""
        returns = np.array([0.01, 0.02, 0.03])
        inp = ReturnsInput(returns)
        result = downside_semivariance(inp)
        assert result.value == 0.0

    def test_with_mar(self):
        """Respects MAR threshold."""
        returns = np.array([0.005, 0.02, -0.01, 0.03])
        inp = ReturnsInput(returns)
        result_default = downside_semivariance(inp, mar=0.0)
        result_high_mar = downside_semivariance(inp, mar=0.02)
        assert result_high_mar.value > result_default.value

    def test_all_nan(self):
        """Returns NaN for all-NaN input."""
        returns = np.full(100, np.nan)
        inp = ReturnsInput(returns)
        result = downside_semivariance(inp)
        assert np.isnan(result.value)


# ---------------------------------------------------------------------------
# Modified VaR
# ---------------------------------------------------------------------------


class TestModifiedVar:
    """Tests for modified_var metric."""

    def test_same_as_var_cf(self):
        """modified_var should equal var(method='cornish_fisher')."""
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.0002, 0.02, size=500)
        inp = ReturnsInput(returns)
        mvar_result = modified_var(inp, confidence=0.95)
        var_cf_result = var(inp, method="cornish_fisher", confidence=0.95)
        assert mvar_result.value == pytest.approx(var_cf_result.value, rel=1e-12)

    def test_finite_value(self):
        """Modified VaR should return a finite positive value for typical data."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, size=500)
        inp = ReturnsInput(returns)
        mvar95 = modified_var(inp, confidence=0.95)
        mvar99 = modified_var(inp, confidence=0.99)
        assert np.isfinite(mvar95.value)
        assert np.isfinite(mvar99.value)
        assert mvar95.value > 0
        assert mvar99.value > 0

    def test_negative_skew_increases_var(self):
        """Negatively skewed returns should have higher modified VaR."""
        rng = np.random.default_rng(42)
        normal_returns = rng.normal(0.0004, 0.01, size=500)
        # Negatively skewed: mix of small gains and occasional large losses
        skewed_returns = np.where(
            rng.random(500) < 0.1,
            rng.normal(-0.05, 0.02, size=500),
            rng.normal(0.005, 0.005, size=500),
        )
        inp_normal = ReturnsInput(normal_returns)
        inp_skewed = ReturnsInput(skewed_returns)
        mvar_normal = modified_var(inp_normal, confidence=0.95)
        mvar_skewed = modified_var(inp_skewed, confidence=0.95)
        # The skewed series has more tail risk → higher modified VaR
        assert mvar_skewed.value > mvar_normal.value

    def test_invalid_confidence(self):
        """Confidence outside (0, 1) raises ValueError."""
        returns = np.random.default_rng(42).normal(0.0, 0.01, size=50)
        inp = ReturnsInput(returns)
        with pytest.raises(ValueError):
            modified_var(inp, confidence=1.5)
        with pytest.raises(ValueError):
            modified_var(inp, confidence=-0.5)

    def test_too_few_periods(self):
        """Returns NaN for fewer than 4 valid observations."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, size=3)
        inp = ReturnsInput(returns)
        result = modified_var(inp)
        assert np.isnan(result.value)
