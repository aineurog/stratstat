"""Tests for report export utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from stratstat.results import MetricResult, MetricSet


class TestMetricSetExports:
    @pytest.fixture
    def metric_set(self):
        return MetricSet(results=[
            MetricResult(name="sharpe_ratio", value=1.5,
                         category=("risk_adjusted",)),
            MetricResult(name="cagr", value=0.12,
                         category=("descriptive",)),
            MetricResult(name="max_drawdown", value=np.array([-0.25, -0.30]),
                         category=("risk",)),
        ])

    def test_to_markdown_writes_file(self, metric_set):
        from stratstat.report import to_markdown

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.md"
            to_markdown(metric_set, path)
            content = path.read_text()
            assert "sharpe_ratio" in content
            assert "1.5" in content
            assert "cagr" in content

    def test_to_json_writes_file(self, metric_set):
        from stratstat.report import to_json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            to_json(metric_set, path)
            data = json.loads(path.read_text())
            assert len(data) == 3
            assert data[0]["name"] == "sharpe_ratio"

    def test_to_latex_writes_file(self, metric_set):
        from stratstat.report import to_latex

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.tex"
            to_latex(metric_set, path)
            content = path.read_text()
            assert r"\begin{tabular}" in content
            assert r"\end{tabular}" in content

    def test_export_creates_parent_dirs(self, metric_set):
        from stratstat.report import to_json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "deep" / "metrics.json"
            to_json(metric_set, path)
            assert path.exists()
