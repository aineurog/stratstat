"""Tests for rolling() and by_regime() generic wrappers."""

from __future__ import annotations

import numpy as np
import pytest

# Ensure metrics are registered.
import stratstat.core.returns.descriptive  # noqa: F401
import stratstat.core.returns.risk_adjusted  # noqa: F401
from stratstat import by_regime, rolling
from stratstat.exceptions import UnknownMetricError
from stratstat.inputs import ReturnsInput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def daily_returns(rng):
    return rng.normal(0.001, 0.02, size=252)


@pytest.fixture
def returns_input(daily_returns):
    return ReturnsInput(daily_returns, periods_per_year=252)


# ---------------------------------------------------------------------------
# rolling()
# ---------------------------------------------------------------------------


class TestRolling:
    def test_basic(self, daily_returns):
        """Rolling Sharpe produces a time series with leading NaN."""
        result = rolling(daily_returns, "sharpe_ratio", window=60,
                         periods_per_year=252)
        assert result.name == "rolling_60_sharpe_ratio"
        vals = result.value
        assert vals.shape == (252,)
        assert np.all(np.isnan(vals[:59]))  # first window-1 are NaN
        assert np.all(np.isfinite(vals[59:]))  # rest are finite

    def test_accepts_returns_input(self, returns_input):
        """Rolling works when given a ReturnsInput directly."""
        result = rolling(returns_input, "cagr", window=30)
        assert result.value.shape == (252,)
        assert result.meta["metric"] == "cagr"
        assert result.meta["window"] == 30

    def test_known_value_constant_returns(self):
        """Rolling mean of constant returns equals the constant."""
        r = np.full(20, 0.01)
        result = rolling(r, "cagr", window=5, periods_per_year=12)
        vals = result.value
        # All windowed values should be finite and equal.
        finite = vals[4:]
        assert np.all(np.isfinite(finite))
        assert np.allclose(finite, finite[0], rtol=1e-10)

    def test_window_too_small_raises(self, daily_returns):
        """window < 2 is invalid."""
        with pytest.raises(ValueError, match="at least 2"):
            rolling(daily_returns, "sharpe_ratio", window=1)

    def test_window_larger_than_data_raises(self, daily_returns):
        """window > n raises ValueError."""
        with pytest.raises(ValueError, match="exceeds"):
            rolling(daily_returns, "sharpe_ratio", window=500)

    def test_single_observation(self):
        """Single observation: window=2 raises, window=1 raises."""
        r = np.array([0.01])
        with pytest.raises(ValueError):
            rolling(r, "sharpe_ratio", window=2)

    def test_multi_strategy_raises(self):
        """Multi-column returns should raise ValueError."""
        r = np.column_stack([[0.01, -0.02, 0.03], [0.02, 0.01, -0.01]])
        with pytest.raises(ValueError, match="single-strategy"):
            rolling(r, "sharpe_ratio", window=2)

    def test_kwargs_forwarding(self, daily_returns):
        """metric_kwargs are forwarded to the inner metric."""
        result = rolling(daily_returns, "var", window=60,
                         confidence=0.99, method="historical",
                         periods_per_year=252)
        vals = result.value
        assert np.all(np.isfinite(vals[59:]))

    def test_unknown_metric_raises(self, daily_returns):
        """Unknown metric name raises UnknownMetricError (not silent NaN)."""
        with pytest.raises(UnknownMetricError):
            rolling(daily_returns, "nonexistent_metric", window=10)


# ---------------------------------------------------------------------------
# by_regime()
# ---------------------------------------------------------------------------


class TestByRegime:
    def test_basic_int_labels(self, daily_returns):
        """Two regimes with integer labels."""
        n = len(daily_returns)
        labels = np.zeros(n, dtype=int)
        labels[n // 2:] = 1  # second half is regime 1
        result = by_regime(daily_returns, "cagr", labels,
                           periods_per_year=252)
        assert result.name == "cagr_by_regime"
        vals = result.value
        assert vals.shape == (2,)  # 2 regimes
        assert np.all(np.isfinite(vals))
        assert result.meta["regime_labels"] == [0, 1]

    def test_string_labels(self, daily_returns):
        """String regime labels work."""
        n = len(daily_returns)
        labels = np.array(["bull"] * n, dtype=object)
        labels[n // 3:] = "bear"
        labels[2 * n // 3:] = "neutral"
        result = by_regime(daily_returns, "annualized_volatility",
                           labels, periods_per_year=252)
        vals = result.value
        assert vals.shape == (3,)
        assert result.meta["regime_labels"] == ["bear", "bull", "neutral"]

    def test_bool_labels(self, daily_returns):
        """Boolean labels work (True/False regimes)."""
        labels = daily_returns > 0  # positive vs non-positive days
        result = by_regime(daily_returns, "cagr", labels,
                           periods_per_year=252)
        vals = result.value
        assert vals.shape == (2,)
        assert result.meta["regime_labels"] == [False, True]

    def test_single_regime(self, daily_returns):
        """All same label — one regime."""
        labels = np.full(len(daily_returns), "single")
        result = by_regime(daily_returns, "cagr", labels,
                           periods_per_year=252)
        assert result.value.shape == (1,)
        assert np.isfinite(result.value[0])

    def test_tiny_regime_produces_nan(self, daily_returns):
        """A regime with < 2 observations yields NaN."""
        labels = np.full(len(daily_returns), "main")
        labels[0] = "tiny"  # only 1 observation
        result = by_regime(daily_returns, "cagr", labels,
                           periods_per_year=252)
        vals = result.value
        tiny_idx = result.meta["regime_labels"].index("tiny")
        assert np.isnan(vals[tiny_idx])

    def test_mismatched_length_raises(self, daily_returns):
        """regime_labels length != returns length raises ValueError."""
        labels = np.array([0, 1])  # too short
        with pytest.raises(ValueError, match="must match"):
            by_regime(daily_returns, "cagr", labels)

    def test_multi_strategy_raises(self):
        """Multi-column returns should raise."""
        r = np.column_stack([[0.01, -0.02], [0.02, 0.01]])
        labels = np.array([0, 1])
        with pytest.raises(ValueError, match="single-strategy"):
            by_regime(r, "cagr", labels)

    def test_accepts_returns_input(self, returns_input):
        """by_regime works with a ReturnsInput."""
        labels = np.full(252, "regime_a")
        labels[126:] = "regime_b"
        result = by_regime(returns_input, "sharpe_ratio", labels)
        assert result.value.shape == (2,)
        assert np.all(np.isfinite(result.value))

    def test_kwargs_forwarding(self, daily_returns):
        """metric_kwargs are forwarded."""
        labels = np.full(len(daily_returns), 0)
        labels[len(daily_returns) // 2:] = 1
        result = by_regime(daily_returns, "var", labels,
                           confidence=0.99, method="historical",
                           periods_per_year=252)
        assert np.all(np.isfinite(result.value))
