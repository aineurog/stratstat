"""Tests for MetricResult and MetricSet types."""

import pytest

from stratstat.results import MetricResult, MetricSet


def test_metric_result_creation():
    """MetricResult should store name, value, category, periods_per_year, and meta."""
    mr = MetricResult(
        name="sharpe_ratio",
        value=0.75,
        category=("risk_adjusted", "returns"),
        periods_per_year=252,
        meta={"convention": "ddof=1", "ref": "Sharpe (1966)"},
    )
    assert mr.name == "sharpe_ratio"
    assert mr.value == 0.75
    assert mr.category == ("risk_adjusted", "returns")
    assert mr.periods_per_year == 252
    assert mr.meta["convention"] == "ddof=1"


def test_metric_result_repr():
    """MetricResult repr should include name, value, and category."""
    mr = MetricResult(name="test", value=1.5, category=("risk",))
    rep = repr(mr)
    assert "test" in rep
    assert "1.5" in rep
    assert "risk" in rep


def test_metric_set_to_dict():
    """MetricSet.to_dict() should return {name: value} mapping."""
    ms = MetricSet(
        results=[
            MetricResult(name="a", value=1.0),
            MetricResult(name="b", value=2.0),
        ]
    )
    d = ms.to_dict()
    assert d == {"a": 1.0, "b": 2.0}


def test_metric_set_to_json():
    """MetricSet.to_json() should produce valid JSON."""
    ms = MetricSet(
        results=[MetricResult(name="a", value=1.0, meta={"ref": "Test (2024)"})]
    )
    json_str = ms.to_json()
    assert '"name"' in json_str
    assert '"value"' in json_str


def test_metric_set_to_markdown():
    """MetricSet.to_markdown() should produce a markdown table."""
    ms = MetricSet(
        results=[MetricResult(name="sharpe", value=0.75)]
    )
    md = ms.to_markdown()
    assert "| Metric | Value |" in md
    assert "sharpe" in md
    assert "0.75" in md


def test_metric_set_len_and_iter():
    """MetricSet should support len() and iteration."""
    results = [
        MetricResult(name="a", value=1.0),
        MetricResult(name="b", value=2.0),
    ]
    ms = MetricSet(results=results)
    assert len(ms) == 2
    names = [r.name for r in ms]
    assert names == ["a", "b"]


def test_metric_set_empty():
    """Empty MetricSet should have len 0."""
    ms = MetricSet()
    assert len(ms) == 0
    assert ms.to_dict() == {}
