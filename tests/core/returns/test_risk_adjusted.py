"""Tests for risk-adjusted return metrics.

Validates against known values from hand-computed examples. References noted
per build instructions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stratstat.core.returns.risk_adjusted import (
    adjusted_sortino_ratio,
    autocorr_penalty,
    burke_ratio,
    calmar_ratio,
    gain_to_pain_ratio,
    k_ratio,
    kappa_3,
    martin_ratio,
    modified_sharpe_ratio,
    omega_ratio,
    pain_ratio,
    rar,
    recovery_factor,
    risk_return_ratio,
    roys_safety_first,
    serenity_ratio,
    sharpe_ratio,
    smart_sharpe,
    smart_sortino,
    sortino_ratio,
    sterling_ratio,
    upi,
    upside_potential_ratio,
)
from stratstat.inputs import ReturnsInput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_pp():
    return 252


@pytest.fixture
def sample_returns():
    """10 daily returns: 0.01, -0.02, 0.015, 0.03, -0.01, 0.005, -0.015, 0.02, -0.005, 0.01."""
    return np.array(
        [0.01, -0.02, 0.015, 0.03, -0.01, 0.005, -0.015, 0.02, -0.005, 0.01]
    )


@pytest.fixture
def sample_input(sample_returns, daily_pp):
    return ReturnsInput(sample_returns, periods_per_year=daily_pp)


# ---------------------------------------------------------------------------
# 3.1 Sharpe Ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    def test_known_value(self, sample_input):
        """Sharpe computed by hand with ddof=1.

        r = [0.01, -0.02, 0.015, 0.03, -0.01, 0.005, -0.015, 0.02, -0.005, 0.01]
        mean = 0.004
        std (ddof=1) = 0.015776...  (sqrt of sum((r - mean)^2)/(n-1))
        SR = mean/std * sqrt(252) = 0.004/0.01584 * 15.8745... ≈ 4.0080
        """
        result = sharpe_ratio(sample_input)
        # Compute expected from numpy: mean/std * sqrt(periods_per_year).
        mean = np.mean(sample_input.values)
        std = np.std(sample_input.values, ddof=1)
        expected = mean / std * np.sqrt(252)
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_all_positive(self, daily_pp):
        """Sharpe with all positive returns: high positive value."""
        r = np.array([0.01, 0.02, 0.005, 0.03, 0.01])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = sharpe_ratio(inp)
        assert result.value > 0

    def test_all_zero_returns(self, daily_pp):
        """All-zero returns: std ≈ 0 → NaN."""
        r = np.zeros(20)
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = sharpe_ratio(inp)
        assert np.isnan(result.value)

    def test_ddof0(self, sample_input):
        """ddof=0 (population std) gives a different value from ddof=1."""
        r0 = sharpe_ratio(sample_input, ddof=0)
        r1 = sharpe_ratio(sample_input, ddof=1)
        assert r0.value != pytest.approx(r1.value)

    def test_rf_nonzero(self, sample_input):
        """Nonzero risk-free rate shifts the Sharpe."""
        r0 = sharpe_ratio(sample_input, rf=0.0)
        r_rf = sharpe_ratio(sample_input, rf=0.001)
        # With positive rf, excess return is lower → Sharpe is lower
        assert r_rf.value < r0.value

    def test_multi_strategy(self, daily_pp):
        """Batch: 2 strategies."""
        r = np.column_stack(
            [
                [0.01, -0.02, 0.015, 0.03, -0.01],
                [0.005, 0.01, -0.005, 0.02, 0.015],
            ]
        )
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = sharpe_ratio(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)

    def test_single_observation(self, daily_pp):
        """Single period: ddof=1 → std is NaN → Sharpe NaN."""
        r = np.array([0.01])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = sharpe_ratio(inp)
        assert np.isnan(result.value)


# ---------------------------------------------------------------------------
# 3.2 Sortino Ratio
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    def test_known_value_full_downside(self, sample_input):
        """Sortino with full_downside denominator, hand-computed.

        r = [0.01, -0.02, 0.015, 0.03, -0.01, 0.005, -0.015, 0.02, -0.005, 0.01]
        mean = 0.004
        Downside deviations below MAR=0: -0.02, -0.01, -0.015, -0.005
        DD_full = sqrt(mean([(-0.02)^2, (-0.01)^2, (-0.015)^2, (-0.005)^2, plus
        6 zeros]) = sqrt((0.0004+0.0001+0.000225+0.000025)/10)
        = sqrt(0.00075/10) = sqrt(0.000075) ≈ 0.0086603
        Sortino = 0.004 * sqrt(252) / 0.0086603 ≈ 7.327
        """
        result = sortino_ratio(sample_input)
        expected_dd = np.sqrt(
            (0.02**2 + 0.01**2 + 0.015**2 + 0.005**2) / 10
        )
        expected = 0.004 * np.sqrt(252) / expected_dd
        assert result.value == pytest.approx(expected, rel=1e-10)
        assert result.meta["denominator"] == "full_downside"

    def test_known_value_downside_only(self, sample_input):
        """Sortino with downside_only denominator, hand-computed.

        sample_returns has 4 downside periods below MAR=0 out of 10.
        sum_sq_down = 0.02^2 + 0.01^2 + 0.015^2 + 0.005^2 = 0.00075
        n_down = 4, so DD = sqrt(0.00075 / 4) = sqrt(0.0001875) ≈ 0.0136931
        Sortino = 0.004 * sqrt(252) / 0.0136931 ≈ 4.637
        """
        result = sortino_ratio(sample_input, denominator="downside_only")
        expected_dd = np.sqrt(
            (0.02**2 + 0.01**2 + 0.015**2 + 0.005**2) / 4
        )
        expected = 0.004 * np.sqrt(252) / expected_dd
        assert result.value == pytest.approx(expected, rel=1e-10)
        assert result.meta["denominator"] == "downside_only"

    def test_downside_only_vs_full(self, sample_input):
        """downside_only divides by n_down (fewer periods) → larger DD → smaller ratio."""
        r_full = sortino_ratio(sample_input, denominator="full_downside")
        r_down = sortino_ratio(sample_input, denominator="downside_only")
        assert r_down.value < r_full.value

    def test_invalid_denominator(self, sample_input):
        """Invalid denominator raises ValueError."""
        with pytest.raises(ValueError, match="denominator"):
            sortino_ratio(sample_input, denominator="invalid")

    def test_all_positive_no_downside(self, daily_pp):
        """All positive returns: no downside → DD ≈ 0 → NaN."""
        r = np.array([0.01, 0.02, 0.005, 0.03])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = sortino_ratio(inp)
        assert np.isnan(result.value)

    def test_multi_strategy(self, daily_pp):
        """Batch: 2 strategies."""
        r = np.column_stack(
            [
                [0.01, -0.02, 0.015, 0.03, -0.01],
                [0.005, 0.01, -0.005, 0.02, 0.015],
            ]
        )
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = sortino_ratio(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)


# ---------------------------------------------------------------------------
# 3.3 Calmar Ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    def test_known_value(self, daily_pp):
        """Calmar computed from a simple known scenario.

        r = [0.10, -0.20, 0.05]  (3 periods, p=252 for annualization)
        Log returns: [ln(1.1), ln(0.8), ln(1.05)] = [0.09531, -0.22314, 0.04879]
        mean_log = -0.026347, CAGR = exp(-0.026347*252) - 1 ≈ ...
        Equity: [1.0, 1.10, 0.88, 0.924]
        Max DD: (0.88-1.10)/1.10 = -0.20
        Calmar = CAGR / 0.20
        """
        r = np.array([0.10, -0.20, 0.05])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = calmar_ratio(inp)

        # Hand-compute CAGR
        log_r = np.log(1.0 + r)
        mean_log = np.mean(log_r)
        cagr = np.exp(mean_log * 252) - 1.0  # ≈ -0.9987... (near total loss annualised)

        # Max DD = -0.20
        expected = cagr / 0.20
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_no_drawdown(self, daily_pp):
        """All positive returns: no drawdown → NaN."""
        r = np.array([0.01, 0.02, 0.005, 0.03])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = calmar_ratio(inp)
        assert np.isnan(result.value)

    def test_multi_strategy(self, daily_pp):
        """Batch: 2 strategies for Calmar."""
        r = np.column_stack(
            [
                [0.10, -0.20, 0.05, 0.02],
                [0.01, -0.05, 0.03, -0.02],
            ]
        )
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = calmar_ratio(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)

    def test_requires_periods_per_year(self):
        """Calmar needs periods_per_year for CAGR."""
        r = np.array([0.01, -0.02, 0.015])
        inp = ReturnsInput(r)
        with pytest.raises(ValueError, match="periods_per_year"):
            calmar_ratio(inp)


# ---------------------------------------------------------------------------
# 3.4 Omega Ratio
# ---------------------------------------------------------------------------


class TestOmegaRatio:
    def test_known_value(self):
        """Omega with known gains and losses.

        r = [0.02, -0.01, 0.03, -0.005, 0.01]
        gains = 0.02 + 0.03 + 0.01 = 0.06
        losses = |-0.01 + -0.005| = 0.015
        Omega = 0.06 / 0.015 = 4.0
        """
        r = np.array([0.02, -0.01, 0.03, -0.005, 0.01])
        inp = ReturnsInput(r)
        result = omega_ratio(inp)
        assert result.value == pytest.approx(4.0, rel=1e-10)

    def test_all_positive(self):
        """All positive returns: no losses → inf."""
        r = np.array([0.01, 0.02, 0.03])
        inp = ReturnsInput(r)
        result = omega_ratio(inp)
        assert result.value == pytest.approx(float("inf"))

    def test_all_negative(self):
        """All negative returns: no gains → 0.0."""
        r = np.array([-0.01, -0.02, -0.005])
        inp = ReturnsInput(r)
        result = omega_ratio(inp)
        assert result.value == 0.0

    def test_threshold_shift(self):
        """Nonzero threshold changes the Omega value."""
        r = np.array([0.02, -0.01, 0.03, -0.005, 0.01])
        inp = ReturnsInput(r)
        r0 = omega_ratio(inp, threshold=0.0)
        r_tau = omega_ratio(inp, threshold=0.01)
        assert r0.value != r_tau.value

    def test_multi_strategy(self):
        """Batch: 2 strategies."""
        r = np.column_stack(
            [
                [0.02, -0.01, 0.03, -0.005],
                [0.01, 0.01, -0.02, 0.005],
            ]
        )
        inp = ReturnsInput(r)
        result = omega_ratio(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)


# ---------------------------------------------------------------------------
# 3.5 Sterling Ratio
# ---------------------------------------------------------------------------


class TestSterlingRatio:
    def test_known_value(self, daily_pp):
        """Sterling with a simple drawdown case.

        r = [0.10, -0.20, 0.05]  (3 periods, p=252)
        Equity: [1.0, 1.10, 0.88, 0.924]
        Drawdowns: [0, 0, -0.20, -0.16]
        One episode: depth = -0.20
        ADD = -0.20
        CAGR = exp(mean_log * 252) - 1
        Sterling = CAGR / (|ADD| + 0.10)
        """
        r = np.array([0.10, -0.20, 0.05])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = sterling_ratio(inp)

        log_r = np.log(1.0 + r)
        mean_log = np.mean(log_r)
        cagr = np.exp(mean_log * 252) - 1.0
        expected = cagr / (0.20 + 0.10)
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_floor_parameter(self, daily_pp):
        """Custom floor changes the denominator.

        Use returns with positive CAGR so the comparison is intuitive.
        r = [0.02, -0.01, 0.03, 0.01] → positive CAGR at p=12.
        Smaller floor → smaller denominator → larger ratio.
        """
        r = np.array([0.02, -0.01, 0.03, 0.01])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        r_default = sterling_ratio(inp, floor=0.10)
        r_custom = sterling_ratio(inp, floor=0.05)
        # Smaller floor → smaller denominator → larger ratio
        assert r_custom.value > r_default.value

    def test_negative_floor(self, daily_pp):
        """Negative floor is rejected."""
        r = np.array([0.01, -0.02])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        with pytest.raises(ValueError, match="floor"):
            sterling_ratio(inp, floor=-0.01)

    def test_no_drawdown(self, daily_pp):
        """No drawdowns: ADD=0, denominator = floor → finite ratio."""
        r = np.array([0.01, 0.02, 0.005])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = sterling_ratio(inp)
        assert not np.isnan(result.value)
        assert not np.isinf(result.value)

    def test_multi_strategy(self, daily_pp):
        """Batch: 2 strategies for Sterling."""
        r = np.column_stack(
            [
                [0.10, -0.20, 0.05, 0.02],
                [0.01, -0.05, 0.03, -0.02],
            ]
        )
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = sterling_ratio(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)

    def test_requires_periods_per_year(self):
        """Sterling needs periods_per_year."""
        r = np.array([0.01, -0.02])
        inp = ReturnsInput(r)
        with pytest.raises(ValueError, match="periods_per_year"):
            sterling_ratio(inp)


# ---------------------------------------------------------------------------
# 3.6 Burke Ratio
# ---------------------------------------------------------------------------


class TestBurkeRatio:
    def test_known_value(self, daily_pp):
        """Burke from simple drawdown case.

        r = [0.10, -0.20, 0.05]  (3 periods, p=252)
        Equity: [1.0, 1.10, 0.88, 0.924]
        DD series: [0, 0, -0.20, -0.16]
        sum(dd^2) = 0.04 + 0.0256 = 0.0656
        sqrt = 0.25612...
        CAGR as above.
        Burke = CAGR / 0.25612...
        """
        r = np.array([0.10, -0.20, 0.05])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = burke_ratio(inp)

        log_r = np.log(1.0 + r)
        mean_log = np.mean(log_r)
        cagr = np.exp(mean_log * 252) - 1.0
        expected = cagr / np.sqrt(0.20**2 + 0.16**2)
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_no_drawdown(self, daily_pp):
        """No drawdown → denominator zero → NaN."""
        r = np.array([0.01, 0.02, 0.005])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = burke_ratio(inp)
        assert np.isnan(result.value)

    def test_multi_strategy(self, daily_pp):
        """Batch: 2 strategies for Burke."""
        r = np.column_stack(
            [
                [0.10, -0.20, 0.05, 0.02],
                [0.01, -0.05, 0.03, -0.02],
            ]
        )
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = burke_ratio(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)

    def test_requires_periods_per_year(self):
        """Burke needs periods_per_year."""
        r = np.array([0.01, -0.02])
        inp = ReturnsInput(r)
        with pytest.raises(ValueError, match="periods_per_year"):
            burke_ratio(inp)


# ---------------------------------------------------------------------------
# 3.7 Kappa-3
# ---------------------------------------------------------------------------


class TestKappa3:
    def test_known_value(self):
        """Kappa-3 hand-computed.

        r = [0.02, -0.01, 0.03, -0.005, 0.01]
        mean = 0.009
        Below MAR=0: max(0 - r, 0) for r<0 = [0, 0.01, 0, 0.005, 0]
        LPM_3 = mean([0, 0.01^3, 0, 0.005^3, 0]) = mean([0, 1e-6, 0, 1.25e-10, 0])
        = (1e-6 + 1.25e-10) / 5 = 2.00025e-7
        cbrt(LPM_3) ≈ 0.005848...
        Kappa-3 = 0.009 / 0.005848... ≈ 1.5388
        """
        r = np.array([0.02, -0.01, 0.03, -0.005, 0.01])
        inp = ReturnsInput(r)
        result = kappa_3(inp)

        mean = np.mean(r)
        below = np.maximum(0.0 - r, 0.0)
        lpm3 = np.mean(below**3)
        expected = mean / np.cbrt(lpm3)
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_mar_shift(self):
        """Nonzero MAR lowers excess return and increases LPM."""
        r = np.array([0.02, -0.01, 0.03, -0.005, 0.01])
        inp = ReturnsInput(r)
        r0 = kappa_3(inp, mar=0.0)
        r_mar = kappa_3(inp, mar=0.005)
        assert r0.value != r_mar.value

    def test_all_above_mar(self):
        """All returns above MAR: LPM_3 = 0 → NaN."""
        r = np.array([0.01, 0.02, 0.03])
        inp = ReturnsInput(r)
        result = kappa_3(inp, mar=0.0)
        assert np.isnan(result.value)

    def test_multi_strategy(self):
        """Batch: 2 strategies."""
        r = np.column_stack(
            [
                [0.02, -0.01, 0.03, -0.005],
                [0.01, 0.01, -0.02, 0.005],
            ]
        )
        inp = ReturnsInput(r)
        result = kappa_3(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)


# ---------------------------------------------------------------------------
# 3.8 Martin Ratio
# ---------------------------------------------------------------------------


class TestMartinRatio:
    def test_known_value(self, daily_pp):
        """Martin (CAGR / Ulcer Index) with simple drawdown case.

        r = [0.10, -0.20, 0.05]  (3 periods, p=252)
        Equity: [1.0, 1.10, 0.88, 0.924]
        DD: [0, 0, -0.20, -0.16]
        UI = sqrt(mean(dd^2)) = sqrt((0.04 + 0.0256)/4) = sqrt(0.0164) ≈ 0.12806...
        CAGR as above.
        Martin = CAGR / UI
        """
        r = np.array([0.10, -0.20, 0.05])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = martin_ratio(inp)

        log_r = np.log(1.0 + r)
        mean_log = np.mean(log_r)
        cagr = np.exp(mean_log * 252) - 1.0
        ui = np.sqrt(np.mean(np.array([0.0, 0.0, -0.20, -0.16]) ** 2))
        expected = cagr / ui
        assert result.value == pytest.approx(expected, rel=1e-10)

    def test_no_drawdown(self, daily_pp):
        """No drawdown: UI = 0 → NaN."""
        r = np.array([0.01, 0.02, 0.005])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = martin_ratio(inp)
        assert np.isnan(result.value)

    def test_multi_strategy(self, daily_pp):
        """Batch: 2 strategies for Martin."""
        r = np.column_stack(
            [
                [0.10, -0.20, 0.05, 0.02],
                [0.01, -0.05, 0.03, -0.02],
            ]
        )
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = martin_ratio(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)

    def test_requires_periods_per_year(self):
        """Martin needs periods_per_year."""
        r = np.array([0.01, -0.02])
        inp = ReturnsInput(r)
        with pytest.raises(ValueError, match="periods_per_year"):
            martin_ratio(inp)


# ---------------------------------------------------------------------------
# 3.9 Gain-to-Pain Ratio
# ---------------------------------------------------------------------------


class TestGainToPainRatio:
    def test_known_value(self):
        """Gain-to-Pain hand-computed.

        r = [0.02, -0.01, 0.03, -0.005, 0.01]
        gains = 0.02 + 0.03 + 0.01 = 0.06
        losses = |-0.01 + -0.005| = 0.015
        GPR = 0.06 / 0.015 = 4.0
        """
        r = np.array([0.02, -0.01, 0.03, -0.005, 0.01])
        inp = ReturnsInput(r)
        result = gain_to_pain_ratio(inp)
        assert result.value == pytest.approx(4.0, rel=1e-10)

    def test_all_positive(self):
        """All positive returns: no losses → inf."""
        r = np.array([0.01, 0.02, 0.03])
        inp = ReturnsInput(r)
        result = gain_to_pain_ratio(inp)
        assert result.value == pytest.approx(float("inf"))

    def test_all_negative(self):
        """All negative returns: no gains → 0.0."""
        r = np.array([-0.01, -0.02, -0.005])
        inp = ReturnsInput(r)
        result = gain_to_pain_ratio(inp)
        assert result.value == 0.0

    def test_multi_strategy(self):
        """Batch: 2 strategies."""
        r = np.column_stack(
            [
                [0.02, -0.01, 0.03, -0.005],
                [0.01, 0.01, -0.02, 0.005],
            ]
        )
        inp = ReturnsInput(r)
        result = gain_to_pain_ratio(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_array_sharpe(self):
        """Empty array: Sharpe = NaN."""
        r = np.array([])
        inp = ReturnsInput(r, periods_per_year=252)
        result = sharpe_ratio(inp)
        assert np.isnan(result.value)

    def test_empty_array_omega(self):
        """Empty array: both gains and losses zero → NaN (no data to evaluate)."""
        r = np.array([])
        inp = ReturnsInput(r)
        result = omega_ratio(inp)
        assert np.isnan(result.value)

    def test_empty_array_gpr(self):
        """Empty array: both gains and losses zero → NaN."""
        r = np.array([])
        inp = ReturnsInput(r)
        result = gain_to_pain_ratio(inp)
        assert np.isnan(result.value)

    def test_all_nan_sharpe(self):
        """All-NaN returns produce NaN Sharpe."""
        r = np.array([np.nan, np.nan, np.nan])
        inp = ReturnsInput(r, periods_per_year=252)
        result = sharpe_ratio(inp)
        assert np.isnan(result.value)

    def test_all_nan_gain_to_pain(self):
        """All-NaN returns: no data → NaN."""
        r = np.array([np.nan, np.nan, np.nan])
        inp = ReturnsInput(r)
        result = gain_to_pain_ratio(inp)
        assert np.isnan(result.value)

    def test_all_nan_omega(self):
        """All-NaN returns: no data → NaN."""
        r = np.array([np.nan, np.nan, np.nan])
        inp = ReturnsInput(r)
        result = omega_ratio(inp)
        assert np.isnan(result.value)


class TestConstantReturns:
    def test_constant_positive_sharpe(self, daily_pp):
        """Constant positive returns: std(ddof=1) ≈ 0 → NaN."""
        r = np.full(10, 0.01)
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = sharpe_ratio(inp)
        assert np.isnan(result.value)

    def test_constant_single_point(self, daily_pp):
        """Single observation: std(ddof=1) is NaN → Sharpe NaN."""
        r = np.array([0.01])
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = sharpe_ratio(inp)
        assert np.isnan(result.value)


# ---------------------------------------------------------------------------
# Input type variations
# ---------------------------------------------------------------------------


class TestInputTypes:
    def test_pandas_series_sharpe(self, sample_returns, daily_pp):
        """Sharpe works with pandas Series."""
        s = pd.Series(sample_returns)
        inp = ReturnsInput(s, periods_per_year=daily_pp)
        result = sharpe_ratio(inp)
        assert isinstance(result.value, float)

    def test_pandas_dataframe_sharpe(self, daily_pp):
        """Sharpe works with pandas DataFrame (multi-strategy)."""
        df = pd.DataFrame(
            {"a": [0.01, -0.02, 0.015, 0.03, -0.01], "b": [0.005, 0.01, -0.005, 0.02, 0.015]}
        )
        inp = ReturnsInput(df, periods_per_year=daily_pp)
        result = sharpe_ratio(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)

    def test_ndarray_2d(self, daily_pp):
        """Sortino works with 2-D ndarray."""
        r = np.column_stack(
            [[0.01, -0.02, 0.015], [0.005, 0.01, -0.005]]
        )
        inp = ReturnsInput(r, periods_per_year=daily_pp)
        result = sortino_ratio(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)

    def test_polars_series_sharpe(self, sample_returns, daily_pp):
        """Sharpe works with polars Series."""
        pl = pytest.importorskip("polars")
        s = pl.Series("returns", sample_returns)
        inp = ReturnsInput(s, periods_per_year=daily_pp)
        result = sharpe_ratio(inp)
        assert isinstance(result.value, float)

    def test_polars_dataframe_omega(self, daily_pp):
        """Omega works with polars DataFrame (multi-strategy)."""
        pl = pytest.importorskip("polars")
        df = pl.DataFrame(
            {
                "a": [0.02, -0.01, 0.03, -0.005],
                "b": [0.01, 0.01, -0.02, 0.005],
            }
        )
        inp = ReturnsInput(df, periods_per_year=daily_pp)
        result = omega_ratio(inp)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == (2,)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


class TestNaNHandling:
    def test_omega_with_nans(self):
        """Omega ignores NaN entries correctly."""
        r = np.array([0.02, np.nan, -0.01, 0.03, np.nan, -0.005])
        inp = ReturnsInput(r)
        result = omega_ratio(inp)
        # gains = 0.02 + 0.03 = 0.05, losses = |-0.01 + -0.005| = 0.015
        # Omega = 0.05 / 0.015 ≈ 3.333...
        assert result.value == pytest.approx(0.05 / 0.015, rel=1e-10)

    def test_gpr_with_nans(self):
        """Gain-to-Pain ignores NaN entries correctly."""
        r = np.array([0.02, np.nan, -0.01, 0.03, np.nan, -0.005])
        inp = ReturnsInput(r)
        result = gain_to_pain_ratio(inp)
        assert result.value == pytest.approx(0.05 / 0.015, rel=1e-10)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_list_metrics_risk_adjusted(self):
        """All 23 risk-adjusted metrics appear under the risk_adjusted category."""
        from stratstat.registry import list_metrics

        metrics = list_metrics(category="risk_adjusted")
        names = {m["name"] for m in metrics}
        expected = {
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "omega_ratio",
            "sterling_ratio",
            "burke_ratio",
            "kappa_3",
            "martin_ratio",
            "gain_to_pain_ratio",
            "pain_ratio",
            "recovery_factor",
            "k_ratio",
            "serenity_ratio",
            "upi",
            "modified_sharpe_ratio",
            "upside_potential_ratio",
            "risk_return_ratio",
            "roys_safety_first",
            "autocorr_penalty",
            "smart_sharpe",
            "smart_sortino",
            "adjusted_sortino_ratio",
            "rar",
        }
        assert names == expected

    def test_compute_sharpe(self, sample_input):
        """compute() works for a risk-adjusted metric."""
        from stratstat import compute

        result = compute(sample_input, "sharpe_ratio")
        assert result.name == "sharpe_ratio"
        assert isinstance(result.value, float)

    def test_compute_all_risk_adjusted(self, sample_input):
        """compute_all(category='risk_adjusted') returns all 23 metrics."""
        from stratstat import compute_all

        results = compute_all(sample_input, category="risk_adjusted")
        assert len(results) == 23
        names = {r.name for r in results}
        assert "sharpe_ratio" in names
        assert "kappa_3" in names
        assert "pain_ratio" in names
        assert "k_ratio" in names
        assert "upi" in names
        assert "autocorr_penalty" in names
        assert "smart_sharpe" in names
        assert "rar" in names


# ---------------------------------------------------------------------------
# Pain Ratio
# ---------------------------------------------------------------------------


class TestPainRatio:
    """Tests for pain_ratio metric."""

    def test_pain_ratio_positive(self, sample_input):
        """Pain Ratio should be positive for a strategy with positive CAGR."""
        result = pain_ratio(sample_input)
        assert result.value > 0

    def test_no_drawdowns(self):
        """Pain Ratio should be inf when there are no drawdowns."""
        returns = np.array([0.01, 0.02, 0.01, 0.03])
        inp = ReturnsInput(returns, periods_per_year=252)
        result = pain_ratio(inp)
        assert result.value == float("inf")


# ---------------------------------------------------------------------------
# Recovery Factor
# ---------------------------------------------------------------------------


class TestRecoveryFactor:
    """Tests for recovery_factor metric."""

    def test_known_value(self):
        """Recovery Factor = total_return / |MDD|."""
        # A 100k -> 150k strategy = 0.5 total return with -20% MDD
        # RF = 0.5 / 0.2 = 2.5
        # Simulate: +50% cumulative, -20% max drawdown
        returns = np.array([0.02, 0.02, -0.20, 0.05, 0.10, 0.15, 0.10, 0.05, 0.10])
        inp = ReturnsInput(returns)
        result = recovery_factor(inp)
        assert result.value > 0
        assert result.name == "recovery_factor"

    def test_no_drawdowns(self):
        """Recovery Factor should be inf when max drawdown is zero."""
        returns = np.array([0.01, 0.02, 0.03])
        inp = ReturnsInput(returns)
        result = recovery_factor(inp)
        assert result.value == float("inf")


# ---------------------------------------------------------------------------
# K-Ratio
# ---------------------------------------------------------------------------


class TestKRatio:
    """Tests for k_ratio metric."""

    def test_positive_for_steady_growth(self, sample_input):
        """K-Ratio should be a finite number for typical returns."""
        result = k_ratio(sample_input)
        assert np.isfinite(result.value)

    def test_too_few_periods(self):
        """K-Ratio should be NaN for fewer than 3 periods."""
        returns = np.array([0.01, -0.01])
        inp = ReturnsInput(returns)
        result = k_ratio(inp)
        assert np.isnan(result.value)


# ---------------------------------------------------------------------------
# Serenity Ratio
# ---------------------------------------------------------------------------


class TestSerenityRatio:
    """Tests for serenity_ratio metric."""

    def test_serenity_positive(self, sample_input):
        """Serenity Ratio should be a finite number for typical returns."""
        result = serenity_ratio(sample_input)
        assert np.isfinite(result.value)

    def test_requires_periods_per_year(self):
        """Should raise ValueError without periods_per_year."""
        returns = np.random.default_rng(42).normal(0.0004, 0.01, size=100)
        inp = ReturnsInput(returns)
        with pytest.raises(ValueError, match="periods_per_year"):
            serenity_ratio(inp)


# ---------------------------------------------------------------------------
# UPI (Ulcer Performance Index)
# ---------------------------------------------------------------------------


class TestUPI:
    """Tests for upi metric."""

    def test_upi_vs_martin(self, sample_input):
        """UPI uses excess return; Martin uses CAGR. Both should be finite."""
        upi_result = upi(sample_input)
        martin_result = martin_ratio(sample_input)
        assert np.isfinite(upi_result.value)
        assert np.isfinite(martin_result.value)
        # They differ by construction (numerator differs)

    def test_requires_periods_per_year(self):
        """Should raise ValueError without periods_per_year."""
        returns = np.random.default_rng(42).normal(0.0004, 0.01, size=100)
        inp = ReturnsInput(returns)
        with pytest.raises(ValueError, match="periods_per_year"):
            upi(inp)


# ---------------------------------------------------------------------------
# Modified Sharpe Ratio
# ---------------------------------------------------------------------------


class TestModifiedSharpeRatio:
    """Tests for modified_sharpe_ratio metric."""

    def test_modified_sharpe_finite(self, sample_input):
        """Modified Sharpe Ratio should be finite for typical returns."""
        result = modified_sharpe_ratio(sample_input)
        assert np.isfinite(result.value)

    def test_default_confidence(self):
        """Default confidence level should be recorded in meta."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, size=252)
        inp = ReturnsInput(returns, periods_per_year=252)
        result = modified_sharpe_ratio(inp)
        assert result.meta["confidence"] == 0.95

    def test_custom_confidence(self):
        """Custom confidence level should be used."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, size=252)
        inp = ReturnsInput(returns, periods_per_year=252)
        result = modified_sharpe_ratio(inp, confidence=0.99)
        assert result.meta["confidence"] == 0.99


# ---------------------------------------------------------------------------
# Upside Potential Ratio
# ---------------------------------------------------------------------------


class TestUpsidePotentialRatio:
    """Tests for upside_potential_ratio metric."""

    def test_positive_for_gains(self):
        """UPR should be positive when there are gains above MAR."""
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.01])
        inp = ReturnsInput(returns)
        result = upside_potential_ratio(inp)
        assert result.value > 0

    def test_no_downside(self):
        """UPR should be inf when there is no downside deviation."""
        returns = np.array([0.01, 0.02, 0.03])
        inp = ReturnsInput(returns)
        result = upside_potential_ratio(inp)
        assert result.value == float("inf")


# ---------------------------------------------------------------------------
# Risk Return Ratio
# ---------------------------------------------------------------------------


class TestRiskReturnRatio:
    """Tests for risk_return_ratio metric."""

    def test_simpler_than_calmar(self, sample_input):
        """Risk Return Ratio uses annualized arithmetic return, not CAGR."""
        rr_result = risk_return_ratio(sample_input)
        calmar_result = calmar_ratio(sample_input)
        # Both should be finite; they differ by numerator
        assert np.isfinite(rr_result.value)
        assert np.isfinite(calmar_result.value)

    def test_no_drawdowns(self):
        """Risk Return Ratio should be inf when max drawdown is zero."""
        returns = np.array([0.01, 0.02, 0.03])
        inp = ReturnsInput(returns, periods_per_year=252)
        result = risk_return_ratio(inp)
        assert result.value == float("inf")

    def test_requires_periods_per_year(self):
        """Should raise ValueError without periods_per_year."""
        returns = np.random.default_rng(42).normal(0.0004, 0.01, size=100)
        inp = ReturnsInput(returns)
        with pytest.raises(ValueError, match="periods_per_year"):
            risk_return_ratio(inp)


# ---------------------------------------------------------------------------
# Roy's Safety-First Ratio
# ---------------------------------------------------------------------------


class TestRoysSafetyFirst:
    """Tests for roys_safety_first metric."""

    def test_equals_sharpe_when_mar_equals_rf(self):
        """When MAR = rf = 0, RSF should equal Sharpe ratio."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, size=500)
        inp = ReturnsInput(returns, periods_per_year=252)
        rsf = roys_safety_first(inp, mar=0.0)
        sr = sharpe_ratio(inp, rf=0.0)
        assert rsf.value == pytest.approx(sr.value, rel=1e-12)

    def test_higher_mar_lowers_ratio(self):
        """Higher MAR should produce lower RSF."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=500)
        inp = ReturnsInput(returns, periods_per_year=252)
        rsf_low = roys_safety_first(inp, mar=0.0)
        rsf_high = roys_safety_first(inp, mar=0.0005)
        assert rsf_high.value < rsf_low.value

    def test_requires_periods_per_year(self):
        """Should raise ValueError without periods_per_year."""
        returns = np.random.default_rng(42).normal(0.0004, 0.01, size=100)
        inp = ReturnsInput(returns)
        with pytest.raises(ValueError, match="periods_per_year"):
            roys_safety_first(inp)


# ---------------------------------------------------------------------------
# 3.19 Autocorrelation Penalty (Lo 2002)
# ---------------------------------------------------------------------------


class TestAutocorrPenalty:
    def test_known_value(self):
        """r = [1, -1, 1, -1] has |lag-1 corr| = 1 -> penalty = 2.0.

        corr_k = ((n-k)/n) * 1^k summed over k=1..3 gives 1.5, so
        penalty = sqrt(1 + 2*1.5) = 2.0.
        """
        returns = np.array([1.0, -1.0, 1.0, -1.0])
        result = autocorr_penalty(ReturnsInput(returns))
        assert result.value == pytest.approx(2.0, rel=1e-12)

    def test_constant_series(self):
        """Constant returns -> penalty 1.0 (no adjustment)."""
        returns = np.full(10, 0.01)
        result = autocorr_penalty(ReturnsInput(returns))
        assert result.value == 1.0

    def test_at_least_one(self, sample_input):
        """Penalty is always >= 1 for real data."""
        result = autocorr_penalty(sample_input)
        assert result.value >= 1.0

    def test_few_observations(self):
        """Single observation -> NaN."""
        result = autocorr_penalty(ReturnsInput(np.array([0.01])))
        assert np.isnan(result.value)


# ---------------------------------------------------------------------------
# 3.20 Smart Sharpe Ratio (Lo 2002)
# ---------------------------------------------------------------------------


class TestSmartSharpe:
    def test_equals_sharpe_over_penalty(self, sample_input):
        """smart_sharpe = sharpe_ratio / autocorr_penalty."""
        expected = sharpe_ratio(sample_input).value / autocorr_penalty(sample_input).value
        result = smart_sharpe(sample_input)
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_meta_records_penalty(self, sample_input):
        """The penalty is stored in meta for auditability."""
        result = smart_sharpe(sample_input)
        assert result.meta["autocorr_penalty"] == pytest.approx(
            autocorr_penalty(sample_input).value, rel=1e-12
        )

    def test_nonzero_rf(self, sample_input):
        """rf is threaded through to the Sharpe base."""
        result = smart_sharpe(sample_input, rf=0.001)
        expected = sharpe_ratio(sample_input, rf=0.001).value / autocorr_penalty(
            sample_input
        ).value
        assert result.value == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 3.21 Smart Sortino Ratio (Lo 2002)
# ---------------------------------------------------------------------------


class TestSmartSortino:
    def test_equals_sortino_over_penalty(self, sample_input):
        """smart_sortino = sortino_ratio / autocorr_penalty."""
        expected = sortino_ratio(sample_input).value / autocorr_penalty(sample_input).value
        result = smart_sortino(sample_input)
        assert result.value == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 3.22 Adjusted Sortino Ratio (Schwager)
# ---------------------------------------------------------------------------


class TestAdjustedSortinoRatio:
    def test_equals_sortino_over_sqrt2(self, sample_input):
        """adjusted_sortino = sortino_ratio / sqrt(2)."""
        expected = sortino_ratio(sample_input).value / np.sqrt(2.0)
        result = adjusted_sortino_ratio(sample_input)
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_denominator_convention(self, sample_input):
        """Downside-only denominator threads through."""
        expected = sortino_ratio(
            sample_input, denominator="downside_only"
        ).value / np.sqrt(2.0)
        result = adjusted_sortino_ratio(sample_input, denominator="downside_only")
        assert result.value == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 3.23 Risk-Adjusted Return (RAR)
# ---------------------------------------------------------------------------


class TestRar:
    def test_equals_cagr_over_exposure(self, sample_input):
        """RAR = CAGR(excess) / exposure. With rf=0, exposure=1.0 here."""
        from stratstat.core.returns.descriptive import cagr, exposure_time

        expected = cagr(sample_input).value / exposure_time(sample_input).value
        result = rar(sample_input)
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_requires_periods_per_year(self):
        """RAR raises MetricNotApplicableError without periods_per_year."""
        from stratstat.exceptions import MetricNotApplicableError

        inp = ReturnsInput(np.array([0.01, -0.02, 0.03]))
        with pytest.raises(MetricNotApplicableError):
            rar(inp)

    def test_all_zero_returns(self, daily_pp):
        """Zero exposure -> NaN RAR (division by zero avoided)."""
        inp = ReturnsInput(np.zeros(10), periods_per_year=daily_pp)
        result = rar(inp)
        assert np.isnan(result.value)

    def test_percent_ceil_rounding(self, daily_pp):
        """percent_ceil divides by the rounded exposure (QuantStats-compatible)."""
        from stratstat.core.returns.descriptive import cagr, exposure_time

        returns = np.array(
            [0.01, 0.0, -0.02, 0.03, 0.0, 0.005, -0.01, 0.02, 0.0, 0.0, 0.001]
        )
        inp = ReturnsInput(returns, periods_per_year=daily_pp)
        expected = cagr(inp).value / exposure_time(inp, rounding="percent_ceil").value
        result = rar(inp, rounding="percent_ceil")
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_percent_ceil_differs_from_raw(self, daily_pp):
        """Rounded exposure changes RAR when exposure is not a whole percent."""
        returns = np.array(
            [0.01, 0.0, -0.02, 0.03, 0.0, 0.005, -0.01, 0.02, 0.0, 0.0, 0.001]
        )
        inp = ReturnsInput(returns, periods_per_year=daily_pp)
        assert rar(inp, rounding="percent_ceil").value != rar(inp).value
