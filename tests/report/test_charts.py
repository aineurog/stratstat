"""Tests for report chart functions.

Verifies figure generation, data transformations, input handling,
and edge cases.  Does NOT test visual rendering (pixel-level).
"""

from __future__ import annotations

import numpy as np
import pytest

# Trigger core metric registration needed by tear_sheet/dashboard
import stratstat.core.returns.descriptive  # noqa: F401
import stratstat.core.returns.risk  # noqa: F401
import stratstat.core.returns.risk_adjusted  # noqa: F401
from stratstat.inputs import ReturnsInput, TradeInput
from stratstat.report import (
    benchmark_overlay_chart,
    cumulative_return_chart,
    drawdown_chart,
    equity_curve,
    monthly_heatmap,
    rolling_metric_chart,
    trade_markers_chart,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_returns():
    """252 daily periods of single-strategy returns."""
    rng = np.random.default_rng(42)
    return rng.normal(0.001, 0.02, size=252)


@pytest.fixture
def multi_returns():
    """Three strategies, 252 periods."""
    rng = np.random.default_rng(42)
    s1 = rng.normal(0.001, 0.02, size=252)
    s2 = rng.normal(0.0005, 0.025, size=252)
    s3 = rng.normal(0.0008, 0.018, size=252)
    return np.column_stack([s1, s2, s3])


@pytest.fixture
def returns_input(daily_returns):
    return ReturnsInput(daily_returns, periods_per_year=252)


@pytest.fixture
def trade_input():
    """10-trade log with P&L and side."""
    return TradeInput(trades={
        "pnl": [0.02, -0.01, 0.03, -0.015, 0.01, -0.02, 0.04, -0.005, 0.015, -0.01],
        "side": ["long", "short", "long", "short", "long",
                  "short", "long", "long", "short", "short"],
    })


# ---------------------------------------------------------------------------
# Equity Curve
# ---------------------------------------------------------------------------


class TestEquityCurve:
    def test_returns_figure(self, daily_returns):
        fig = equity_curve(daily_returns)
        assert hasattr(fig, "data")
        assert len(fig.data) >= 1

    def test_accepts_returns_input(self, returns_input):
        fig = equity_curve(returns_input)
        assert len(fig.data) >= 1

    def test_multi_strategy(self, multi_returns):
        fig = equity_curve(multi_returns)
        assert len(fig.data) == 3  # 3 traces

    def test_custom_title(self, daily_returns):
        fig = equity_curve(daily_returns, title="My Equity Curve")
        assert "My Equity Curve" in fig.layout.title.text

    def test_cumulative_starts_at_one(self, daily_returns):
        """First value of cumulative return index should be 1.0 + r[0]."""
        from stratstat.report._charts import _cumulative_returns, _to_array

        r = _to_array(daily_returns)
        cum = _cumulative_returns(r)
        assert cum[0, 0] == pytest.approx(1.0 + daily_returns[0])

    def test_nan_handling(self):
        """NaN in returns should not break the equity curve."""
        r = np.array([0.01, np.nan, 0.02, -0.01, np.nan])
        fig = equity_curve(r)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# Drawdown Chart
# ---------------------------------------------------------------------------


class TestDrawdownChart:
    def test_returns_figure(self, daily_returns):
        fig = drawdown_chart(daily_returns)
        assert hasattr(fig, "data")

    def test_accepts_returns_input(self, returns_input):
        fig = drawdown_chart(returns_input)
        assert len(fig.data) >= 1

    def test_multi_strategy(self, multi_returns):
        fig = drawdown_chart(multi_returns)
        assert len(fig.data) >= 3  # at least one trace per strategy

    def test_drawdown_is_negative(self, daily_returns):
        """Drawdown values should be <= 0."""
        from stratstat.report._charts import _drawdown_series, _to_array

        r = _to_array(daily_returns)
        dd = _drawdown_series(r)
        assert np.all(dd <= 1e-10)  # floating-point tolerance

    def test_custom_title(self, daily_returns):
        fig = drawdown_chart(daily_returns, title="My Drawdown")
        assert "My Drawdown" in fig.layout.title.text


# ---------------------------------------------------------------------------
# Monthly Heatmap
# ---------------------------------------------------------------------------


class TestMonthlyHeatmap:
    def test_returns_figure(self, daily_returns):
        fig = monthly_heatmap(daily_returns)
        assert hasattr(fig, "data")

    def test_custom_title(self, daily_returns):
        fig = monthly_heatmap(daily_returns, title="My Heatmap")
        assert "My Heatmap" in fig.layout.title.text

    def test_empty_data(self):
        """Very short series should still produce a figure."""
        fig = monthly_heatmap(np.array([0.01, 0.02]))
        assert hasattr(fig, "data")

    def test_dates_param(self, daily_returns):
        import pandas as pd

        dates = pd.date_range("2020-01-31", periods=len(daily_returns), freq="ME")
        fig = monthly_heatmap(daily_returns, dates=dates)
        assert hasattr(fig, "data")

    def test_dates_length_mismatch(self, daily_returns):
        import pandas as pd

        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        with pytest.raises(ValueError, match="must match returns length"):
            monthly_heatmap(daily_returns, dates=dates)


# ---------------------------------------------------------------------------
# Rolling Metric Chart
# ---------------------------------------------------------------------------


class TestRollingMetricChart:
    def test_returns_figure(self, daily_returns):
        fig = rolling_metric_chart(daily_returns, "sharpe_ratio", window=60,
                                    periods_per_year=252)
        assert hasattr(fig, "data")
        assert len(fig.data) >= 1

    def test_accepts_returns_input(self, returns_input):
        fig = rolling_metric_chart(returns_input, "annualized_volatility",
                                    window=60)
        assert len(fig.data) >= 1

    def test_unknown_metric_produces_nan(self, daily_returns):
        """Unknown metric should raise UnknownMetricError rather than crash."""
        from stratstat.exceptions import UnknownMetricError

        with pytest.raises(UnknownMetricError):
            rolling_metric_chart(daily_returns, "nonexistent_metric",
                                 window=10)

    def test_window_larger_than_data(self):
        """Window > n_periods should produce all NaN values."""
        r = np.random.randn(20)
        fig = rolling_metric_chart(r, "sharpe_ratio", window=100,
                                    periods_per_year=252)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# Cumulative Return Chart
# ---------------------------------------------------------------------------


class TestCumulativeReturnChart:
    def test_basic(self, daily_returns):
        fig = cumulative_return_chart(daily_returns)
        assert len(fig.data) >= 1

    def test_with_benchmark(self, daily_returns):
        bench = np.random.default_rng(99).normal(0.0005, 0.015, size=252)
        fig = cumulative_return_chart(daily_returns, benchmark=bench)
        # Should have strategy + benchmark traces
        assert len(fig.data) >= 2

    def test_custom_title(self, daily_returns):
        fig = cumulative_return_chart(daily_returns, title="CumRet")
        assert "CumRet" in fig.layout.title.text


# ---------------------------------------------------------------------------
# Benchmark Overlay Chart
# ---------------------------------------------------------------------------


class TestBenchmarkOverlayChart:
    def test_basic(self, daily_returns):
        bench = np.random.default_rng(42).normal(0.0005, 0.015, size=252)
        fig = benchmark_overlay_chart(daily_returns, bench)
        assert hasattr(fig, "data")
        # Should have 3 traces: strategy cum, benchmark cum, active return
        assert len(fig.data) == 3

    def test_custom_title(self, daily_returns):
        bench = np.random.default_rng(42).normal(0.0005, 0.015, size=252)
        fig = benchmark_overlay_chart(daily_returns, bench,
                                       title="Bench Comp")
        assert "Bench Comp" in fig.layout.title.text


# ---------------------------------------------------------------------------
# Trade Markers Chart
# ---------------------------------------------------------------------------


class TestTradeMarkersChart:
    def test_basic(self, trade_input):
        fig = trade_markers_chart(trade_input)
        assert hasattr(fig, "data")

    def test_accepts_dict(self):
        fig = trade_markers_chart({
            "pnl": [0.01, -0.02, 0.03],
            "side": ["long", "short", "long"],
        })
        assert hasattr(fig, "data")

    def test_custom_title(self, trade_input):
        fig = trade_markers_chart(trade_input, title="Trade Viz")
        assert "Trade Viz" in fig.layout.title.text

    def test_equity_curve_when_returns(self, trade_input, daily_returns):
        fig = trade_markers_chart(trade_input, returns=daily_returns)
        names = [t.name for t in fig.data]
        assert "Equity Curve" in names

    def test_no_equity_without_returns(self, trade_input):
        fig = trade_markers_chart(trade_input)
        names = [t.name for t in fig.data]
        assert "Equity Curve" not in names
        assert "Cumulative P&L" in names

    def test_mfe_mae_overlay_when_intratrade(self):
        inp = TradeInput(trades={
            "pnl": [0.02, -0.01],
            "side": ["long", "short"],
            "intratrade_prices": [
                [100.0, 101.0, 102.5, 102.0],
                [100.0, 99.0, 98.0, 99.5],
            ],
        })
        fig = trade_markers_chart(inp)
        names = [t.name for t in fig.data]
        assert "MFE" in names
        assert "MAE" in names

    def test_no_mfe_mae_without_intratrade(self, trade_input):
        fig = trade_markers_chart(trade_input)
        names = [t.name for t in fig.data]
        assert "MFE" not in names
        assert "MAE" not in names
