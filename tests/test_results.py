"""Tests for MetricResult and MetricSet types."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from stratstat.results import (
    MetricResult,
    MetricSet,
    _format_value,
    _group_by_category,
)


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


def test_metric_result_str():
    """MetricResult __str__ should be clean name: value."""
    mr = MetricResult(name="sharpe_ratio", value=0.75)
    assert str(mr) == "sharpe_ratio: 0.75"


# ---------------------------------------------------------------------------
# MetricSet
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


class TestFormatValue:
    def test_float(self):
        assert _format_value(0.123456) == "0.123456"
        assert _format_value(1.0) == "1"

    def test_numpy_scalar(self):
        assert _format_value(np.float64(3.14159)) == "3.14159"

    def test_numpy_small_array(self):
        assert _format_value(np.array([1.0, 2.0, 3.0])) == "[1, 2, 3]"

    def test_numpy_large_array(self):
        arr = np.arange(10, dtype=np.float64)
        assert _format_value(arr) == "array(10,)"

    def test_nan(self):
        assert _format_value(float("nan")) == "N/A"
        assert _format_value(np.nan) == "N/A"

    def test_inf(self):
        assert _format_value(float("inf")) == "∞"
        assert _format_value(float("-inf")) == "-∞"

    def test_none(self):
        assert _format_value(None) == "N/A"

    def test_dict(self):
        assert _format_value({"key": "val"}) == '{"key": "val"}'


# ---------------------------------------------------------------------------
# Category grouping
# ---------------------------------------------------------------------------


class TestGroupByCategory:
    def test_known_categories_ordered(self):
        """Categories appear in _CATEGORY_ORDER sequence."""
        results = [
            MetricResult(name="z", value=1.0, category=("risk", "returns")),
            MetricResult(name="a", value=2.0, category=("descriptive", "returns")),
            MetricResult(name="m", value=3.0, category=("risk", "returns")),
        ]
        groups = _group_by_category(results)
        keys = list(groups.keys())
        # descriptive (0) before risk (1)
        assert keys[0] == "descriptive"
        assert keys[1] == "risk"

    def test_within_group_alphabetical(self):
        """Metrics within a section are alphabetized by name."""
        results = [
            MetricResult(name="c", value=1.0, category=("descriptive",)),
            MetricResult(name="a", value=2.0, category=("descriptive",)),
            MetricResult(name="b", value=3.0, category=("descriptive",)),
        ]
        groups = _group_by_category(results)
        names = [m.name for m in groups["descriptive"]]
        assert names == ["a", "b", "c"]

    def test_unknown_category_goes_to_other(self):
        """Unrecognized primary tag → 'Other' group at end."""
        results = [
            MetricResult(name="x", value=1.0, category=("descriptive",)),
            MetricResult(name="y", value=2.0, category=("unknown_tag",)),
        ]
        groups = _group_by_category(results)
        assert "descriptive" in groups
        assert "unknown_tag" in groups
        keys = list(groups.keys())
        assert keys[-1] == "unknown_tag"


# ---------------------------------------------------------------------------
# MetricSet display
# ---------------------------------------------------------------------------


class TestMetricSetStr:
    def test_empty(self):
        ms = MetricSet()
        assert "empty" in str(ms)

    def test_has_section_headers(self):
        """__str__ includes section headers."""
        ms = MetricSet(results=[
            MetricResult(name="cagr", value=0.12, category=("descriptive", "returns")),
            MetricResult(name="max_drawdown", value=-0.25, category=("risk", "returns")),
        ])
        s = str(ms)
        assert "═══ Descriptive ═══" in s
        assert "═══ Risk ═══" in s

    def test_sorts_within_section(self):
        """Within a section, metrics are alphabetized."""
        ms = MetricSet(results=[
            MetricResult(name="c", value=3.0, category=("descriptive",)),
            MetricResult(name="a", value=1.0, category=("descriptive",)),
        ])
        s = str(ms)
        # Find positions of the metric display lines (name + two spaces)
        pos_a = s.index("  a  ")
        pos_c = s.index("  c  ")
        assert pos_a < pos_c

    def test_handles_arrays(self):
        """Array values are formatted correctly."""
        ms = MetricSet(results=[
            MetricResult(name="arr_metric", value=np.array([1.0, 2.0, 3.0]),
                         category=("descriptive",)),
            MetricResult(name="big_arr", value=np.arange(10),
                         category=("descriptive",)),
        ])
        s = str(ms)
        assert "[1, 2, 3]" in s
        assert "array(10,)" in s

    def test_handles_nan(self):
        """NaN → N/A."""
        ms = MetricSet(results=[
            MetricResult(name="broken", value=float("nan"),
                         category=("risk",)),
        ])
        assert "N/A" in str(ms)


class TestMetricSetReprHtml:
    def test_empty(self):
        ms = MetricSet()
        html = ms._repr_html_()
        assert "empty" in html

    def test_has_table_elements(self):
        ms = MetricSet(results=[
            MetricResult(name="cagr", value=0.12, category=("descriptive", "returns"),
                         meta={"ref": "Test (2024)"}),
        ])
        html = ms._repr_html_()
        assert "<table>" in html
        assert "<h3>" in html
        assert "cagr" in html
        assert "0.12" in html

    def test_citation_included(self):
        ms = MetricSet(results=[
            MetricResult(name="sharpe", value=0.84,
                         category=("risk_adjusted",),
                         meta={"ref": "Sharpe (1966)"}),
        ])
        html = ms._repr_html_()
        assert "Sharpe (1966)" in html


# ---------------------------------------------------------------------------
# to_frame upgrade
# ---------------------------------------------------------------------------


class TestToFrame:
    def test_includes_category_column(self):
        ms = MetricSet(results=[
            MetricResult(name="cagr", value=0.12,
                         category=("descriptive", "returns"),
                         periods_per_year=252),
        ])
        df = ms.to_frame()
        assert "category" in df.columns
        assert df["category"].iloc[0] == ("descriptive", "returns")
        assert "periods_per_year" in df.columns
        assert df["periods_per_year"].iloc[0] == 252

    def test_meta_keys_expanded(self):
        ms = MetricSet(results=[
            MetricResult(name="sharpe", value=0.84,
                         meta={"rf": 0.02, "ddof": 1}),
        ])
        df = ms.to_frame()
        assert df["rf"].iloc[0] == 0.02
        assert df["ddof"].iloc[0] == 1


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------


class TestToCsv:
    def test_writes_file(self):
        ms = MetricSet(results=[
            MetricResult(name="cagr", value=0.12, category=("descriptive",)),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            ms.to_csv(path)
            content = path.read_text()
            assert "cagr" in content
            assert "0.12" in content

    def test_creates_parent_dirs(self):
        ms = MetricSet(results=[MetricResult(name="x", value=1.0)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "deep" / "out.csv"
            ms.to_csv(path)
            assert path.exists()


# ---------------------------------------------------------------------------
# to_clipboard
# ---------------------------------------------------------------------------


class TestToClipboard:
    def test_does_not_raise(self):
        """to_clipboard delegates to pandas — smoke test only."""
        ms = MetricSet(results=[
            MetricResult(name="a", value=1.0),
            MetricResult(name="b", value=2.0),
        ])
        try:
            ms.to_clipboard()
        except Exception:
            pytest.skip("clipboard not available in this environment")
