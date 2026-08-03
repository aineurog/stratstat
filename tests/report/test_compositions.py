"""Tests for tear sheet, dashboard, and report compositions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Trigger core metric registration
import stratstat.core.returns.descriptive  # noqa: F401
import stratstat.core.returns.inference  # noqa: F401
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

        # Metric names are in the first column of cells (auto-discovered)
        cell_texts = table_trace.cells.values[0]
        all_text = " ".join(str(t).lower() for t in cell_texts)
        assert "cagr" in all_text, (
            f"Stats table missing CAGR; cells: {cell_texts}"
        )
        assert "sharpe" in all_text, (
            f"Stats table missing Sharpe; cells: {cell_texts}"
        )
        assert "kurtosis" in all_text or "excess" in all_text, (
            f"Stats table missing kurtosis; cells: {cell_texts}"
        )

    def test_tear_sheet_stats_are_dynamic(self):
        """Stats table includes metrics beyond the original hardcoded 10."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=252)
        fig = tear_sheet(returns, periods_per_year=252)

        import plotly.graph_objects as go
        table_trace = None
        for trace in fig.data:
            if isinstance(trace, go.Table):
                table_trace = trace
                break
        assert table_trace is not None
        cell_texts = table_trace.cells.values[0]
        all_text = " ".join(str(t).lower() for t in cell_texts)
        # Newer metrics that were NOT in the old hardcoded list of 10
        assert "stability" in all_text or "hurst" in all_text or \
            "fractal" in all_text or "upside" in all_text, (
            f"Dynamic metrics missing; cells: {cell_texts[:10]}..."
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

    def test_dashboard_rankings_dynamic(self):
        """Ranking table auto-discovers metrics from _RANKING_METRICS."""
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.02, size=(100, 3))
        fig = dashboard(r, periods_per_year=252)

        import plotly.graph_objects as go
        table_trace = None
        for trace in fig.data:
            if isinstance(trace, go.Table):
                table_trace = trace
                break
        assert table_trace is not None, "Dashboard should include a rankings table"
        header_vals = table_trace.header.values
        # Should include more than just the old 4 hardcoded columns
        assert len(header_vals) >= 5, (
            f"Expected >=5 ranking columns, got {len(header_vals)}: {header_vals}"
        )
        # Key metrics should be present
        header_text = " ".join(str(h).lower() for h in header_vals)
        assert "sharpe" in header_text
        assert "sortino" in header_text or "calmar" in header_text


# ---------------------------------------------------------------------------
# HTML Report generation
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for generate_report()."""

    def test_creates_file(self, daily_returns):
        """generate_report writes an HTML file."""
        from stratstat.report import generate_report

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            generate_report(daily_returns, path, periods_per_year=252)
            assert path.exists()
            content = path.read_text()
            assert "<!DOCTYPE html>" in content
            assert "Strategy Analysis Report" in content

    def test_custom_title(self, daily_returns):
        """Custom title appears in output."""
        from stratstat.report import generate_report

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            generate_report(daily_returns, path, periods_per_year=252,
                            title="My Custom Report")
            content = path.read_text()
            assert "My Custom Report" in content

    def test_contains_chart_divs(self, daily_returns):
        """Output contains plotly chart divs."""
        from stratstat.report import generate_report

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            generate_report(daily_returns, path, periods_per_year=252)
            content = path.read_text()
            assert "plotly-graph-div" in content

    def test_contains_stats_tables(self, daily_returns):
        """Output contains metric names in table cells."""
        from stratstat.report import generate_report

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            generate_report(daily_returns, path, periods_per_year=252)
            content = path.read_text()
            assert "sharpe_ratio" in content
            assert "max_drawdown" in content

    def test_contains_methodology(self, daily_returns):
        """Output contains methodology references."""
        from stratstat.report import generate_report

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            generate_report(daily_returns, path, periods_per_year=252)
            content = path.read_text()
            assert "Methodology" in content

    def test_with_benchmark(self, daily_returns):
        """Benchmark section included when benchmark provided."""
        from stratstat.report import generate_report

        bench = np.random.default_rng(99).normal(0.0005, 0.015, size=252)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            generate_report(daily_returns, path, benchmark=bench,
                            periods_per_year=252)
            content = path.read_text()
            assert "Benchmark" in content

    def test_creates_parent_dirs(self, daily_returns):
        """Output path parent directories are auto-created."""
        from stratstat.report import generate_report

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "deep" / "report.html"
            generate_report(daily_returns, path, periods_per_year=252)
            assert path.exists()

    def test_short_series_does_not_crash(self):
        """Very short return series should produce valid HTML."""
        from stratstat.report import generate_report

        r = np.array([0.01, -0.02, 0.03])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            generate_report(r, path, periods_per_year=12)
            assert path.exists()
            assert "<!DOCTYPE html>" in path.read_text()
