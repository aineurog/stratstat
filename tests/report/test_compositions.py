"""Tests for tear sheet and dashboard compositions."""

from __future__ import annotations

import numpy as np
import pytest

# Trigger core metric registration
import stratstat.core.returns.descriptive  # noqa: F401
import stratstat.core.returns.risk  # noqa: F401
import stratstat.core.returns.risk_adjusted  # noqa: F401
from stratstat.inputs import ReturnsInput
from stratstat.report import dashboard, tear_sheet


@pytest.fixture
def daily_returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.001, 0.02, size=252)


@pytest.fixture
def multi_returns():
    rng = np.random.default_rng(42)
    s1 = rng.normal(0.001, 0.02, size=252)
    s2 = rng.normal(0.0005, 0.025, size=252)
    s3 = rng.normal(0.0008, 0.018, size=252)
    return np.column_stack([s1, s2, s3])


@pytest.fixture
def returns_input(daily_returns):
    return ReturnsInput(daily_returns, periods_per_year=252)


class TestTearSheet:
    def test_basic(self, daily_returns):
        fig = tear_sheet(daily_returns, periods_per_year=252)
        assert hasattr(fig, "data")

    def test_accepts_returns_input(self, returns_input):
        fig = tear_sheet(returns_input)
        assert hasattr(fig, "data")

    def test_with_benchmark(self, daily_returns):
        bench = np.random.default_rng(99).normal(0.0005, 0.015, size=252)
        fig = tear_sheet(daily_returns, benchmark=bench, periods_per_year=252)
        assert hasattr(fig, "data")

    def test_custom_title(self, daily_returns):
        fig = tear_sheet(daily_returns, title="My Tear Sheet",
                         periods_per_year=252)
        assert "My Tear Sheet" in fig.layout.title.text

    def test_short_series(self):
        """Short series should not crash tear sheet, and stats table
        should contain expected metric names."""
        r = np.array([0.01, -0.02, 0.015, 0.03, -0.01])
        fig = tear_sheet(r, periods_per_year=12)
        assert hasattr(fig, "data")

        # Find the Table trace (should be present in a tear sheet)
        import plotly.graph_objects as go
        table_trace = None
        for trace in fig.data:
            if isinstance(trace, go.Table):
                table_trace = trace
                break
        assert table_trace is not None, "Tear sheet should include a stats table"

        # Header cells contain the metric names
        header_texts = table_trace.header.values
        assert any("cagr" in t.lower() or "CAGR" in t for t in header_texts), (
            f"Stats table missing CAGR metric; headers: {header_texts}"
        )
        assert any("sharpe" in t.lower() for t in header_texts), (
            f"Stats table missing Sharpe metric; headers: {header_texts}"
        )
        assert any("excess" in t.lower() or "kurtosis" in t.lower()
                   for t in header_texts), (
            f"Stats table missing excess_kurtosis metric; headers: {header_texts}"
        )


class TestDashboard:
    def test_basic(self, multi_returns):
        fig = dashboard(multi_returns, periods_per_year=252)
        assert hasattr(fig, "data")

    def test_two_strategies(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.02, size=(100, 2))
        fig = dashboard(r, periods_per_year=252)
        assert hasattr(fig, "data")

    def test_custom_title(self, multi_returns):
        fig = dashboard(multi_returns, title="My Dashboard",
                        periods_per_year=252)
        assert "My Dashboard" in fig.layout.title.text

    def test_rolling_window_param(self, multi_returns):
        fig = dashboard(multi_returns, rolling_window=30,
                        periods_per_year=252)
        assert hasattr(fig, "data")
