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
    ms = MetricSet(results=[MetricResult(name="a", value=1.0, meta={"ref": "Test (2024)"})])
    json_str = ms.to_json()
    assert '"name"' in json_str
    assert '"value"' in json_str


def test_metric_set_to_markdown():
    """MetricSet.to_markdown() should produce a markdown table."""
    ms = MetricSet(results=[MetricResult(name="sharpe", value=0.75)])
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
        ms = MetricSet(
            results=[
                MetricResult(name="cagr", value=0.12, category=("descriptive", "returns")),
                MetricResult(name="max_drawdown", value=-0.25, category=("risk", "returns")),
            ]
        )
        s = str(ms)
        assert "═══ Descriptive ═══" in s
        assert "═══ Risk ═══" in s

    def test_sorts_within_section(self):
        """Within a section, metrics are alphabetized."""
        ms = MetricSet(
            results=[
                MetricResult(name="c", value=3.0, category=("descriptive",)),
                MetricResult(name="a", value=1.0, category=("descriptive",)),
            ]
        )
        s = str(ms)
        # Find positions of the metric display lines (name + two spaces)
        pos_a = s.index("  a  ")
        pos_c = s.index("  c  ")
        assert pos_a < pos_c

    def test_handles_arrays(self):
        """Array values are formatted correctly."""
        ms = MetricSet(
            results=[
                MetricResult(
                    name="arr_metric", value=np.array([1.0, 2.0, 3.0]), category=("descriptive",)
                ),
                MetricResult(name="big_arr", value=np.arange(10), category=("descriptive",)),
            ]
        )
        s = str(ms)
        assert "[1, 2, 3]" in s
        assert "array(10,)" in s

    def test_handles_nan(self):
        """NaN → N/A."""
        ms = MetricSet(
            results=[
                MetricResult(name="broken", value=float("nan"), category=("risk",)),
            ]
        )
        assert "N/A" in str(ms)


class TestMetricSetReprHtml:
    def test_empty(self):
        ms = MetricSet()
        html = ms._repr_html_()
        assert "empty" in html

    def test_has_table_elements(self):
        ms = MetricSet(
            results=[
                MetricResult(
                    name="cagr",
                    value=0.12,
                    category=("descriptive", "returns"),
                    meta={"ref": "Test (2024)"},
                ),
            ]
        )
        html = ms._repr_html_()
        assert "<table>" in html
        assert "<h3>" in html
        assert "cagr" in html
        assert "0.12" in html

    def test_citation_included(self):
        ms = MetricSet(
            results=[
                MetricResult(
                    name="sharpe",
                    value=0.84,
                    category=("risk_adjusted",),
                    meta={"ref": "Sharpe (1966)"},
                ),
            ]
        )
        html = ms._repr_html_()
        assert "Sharpe (1966)" in html


# ---------------------------------------------------------------------------
# to_frame upgrade
# ---------------------------------------------------------------------------


class TestToFrame:
    def test_includes_category_column(self):
        ms = MetricSet(
            results=[
                MetricResult(
                    name="cagr",
                    value=0.12,
                    category=("descriptive", "returns"),
                    periods_per_year=252,
                ),
            ]
        )
        df = ms.to_frame()
        assert "category" in df.columns
        assert df["category"].iloc[0] == ("descriptive", "returns")
        assert "periods_per_year" in df.columns
        assert df["periods_per_year"].iloc[0] == 252

    def test_meta_keys_expanded(self):
        ms = MetricSet(
            results=[
                MetricResult(name="sharpe", value=0.84, meta={"rf": 0.02, "ddof": 1}),
            ]
        )
        df = ms.to_frame()
        assert df["rf"].iloc[0] == 0.02
        assert df["ddof"].iloc[0] == 1


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------


class TestToCsv:
    def test_writes_file(self):
        ms = MetricSet(
            results=[
                MetricResult(name="cagr", value=0.12, category=("descriptive",)),
            ]
        )
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
        ms = MetricSet(
            results=[
                MetricResult(name="a", value=1.0),
                MetricResult(name="b", value=2.0),
            ]
        )
        try:
            ms.to_clipboard()
        except Exception:
            pytest.skip("clipboard not available in this environment")


class TestToFrameExplode:
    def _set_with_array_metric(self):
        return MetricSet(
            results=[
                MetricResult(
                    name="component_var",
                    value=np.array([0.1, 0.2, 0.3]),
                    category=("relative", "compare"),
                    meta={"output_index": ["a", "b", "c"], "confidence": 0.95},
                ),
                MetricResult(name="cagr", value=0.12, category=("descriptive",)),
            ]
        )

    def test_explodes_array_with_matching_index(self):
        df = self._set_with_array_metric().to_frame()
        cv = df[df["name"] == "component_var"]
        assert len(cv) == 3
        assert list(cv["output_index"]) == ["a", "b", "c"]
        assert list(cv["value"]) == [0.1, 0.2, 0.3]
        assert list(cv["confidence"]) == [0.95, 0.95, 0.95]

    def test_scalar_metric_stays_single_row(self):
        df = self._set_with_array_metric().to_frame()
        cagr = df[df["name"] == "cagr"]
        assert len(cagr) == 1

    def test_explode_false_keeps_array_in_cell(self):
        df = self._set_with_array_metric().to_frame(explode=False)
        cv = df[df["name"] == "component_var"]
        assert len(cv) == 1
        assert "output_index" in df.columns  # carried as meta, not exploded

    def test_no_index_no_explode(self):
        ms = MetricSet(
            results=[MetricResult(name="x", value=np.array([1.0, 2.0, 3.0]), meta={})]
        )
        df = ms.to_frame()
        assert len(df) == 1  # no output_index, stays single row

    def test_matrix_stays_single_row(self):
        ms = MetricSet(
            results=[
                MetricResult(
                    name="correlation_matrix",
                    value=np.eye(2),
                    meta={"labels": ["a", "b"]},
                )
            ]
        )
        df = ms.to_frame()
        assert len(df) == 1  # 2-D matrix never explodes

    def test_mismatched_index_length_does_not_explode(self):
        ms = MetricSet(
            results=[
                MetricResult(
                    name="x",
                    value=np.array([1.0, 2.0]),
                    meta={"output_index": ["only-one-label"]},
                )
            ]
        )
        df = ms.to_frame()
        assert len(df) == 1


class TestOmissionReporting:
    def _set(self):
        return MetricSet(
            results=[MetricResult(name="cagr", value=0.12)],
            meta={
                "skipped": ["kelly_criterion"],
                "excluded_resampling": ["pbo", "white_reality_check"],
                "deduplicated": ["avg_up_period"],
                "excluded_tiers": ["benchmark"],
            },
        )

    def test_skipped_property(self):
        assert self._set().skipped == ["kelly_criterion"]

    def test_excluded_union(self):
        assert self._set().excluded == ["pbo", "white_reality_check", "avg_up_period"]

    def test_excluded_tiers_property(self):
        assert self._set().excluded_tiers == ["benchmark"]

    def test_empty_meta_defaults(self):
        ms = MetricSet(results=[MetricResult(name="a", value=1.0)])
        assert ms.skipped == []
        assert ms.excluded == []
        assert ms.excluded_tiers == []

    def test_summary_reports_counts(self):
        s = self._set().summary()
        assert "1 metrics computed" in s
        assert "1 skipped" in s
        assert "3 excluded" in s
        assert "tiers not run: benchmark" in s

    def test_summary_clean_when_nothing_omitted(self):
        ms = MetricSet(results=[MetricResult(name="a", value=1.0)])
        assert ms.summary() == "1 metrics computed"


class TestDegenerateColumns:
    def test_to_frame_always_has_degenerate_columns(self):
        ms = MetricSet(
            results=[
                MetricResult(name="ok", value=0.1),
                MetricResult(
                    name="degenerate",
                    value=1.0,
                    meta={"degenerate": True, "degenerate_reason": "single-asset input"},
                ),
            ]
        )
        df = ms.to_frame()
        assert "degenerate" in df.columns
        assert "degenerate_reason" in df.columns
        ok = df[df["name"] == "ok"].iloc[0]
        assert bool(ok["degenerate"]) is False
        assert ok["degenerate_reason"] is None
        deg = df[df["name"] == "degenerate"].iloc[0]
        assert bool(deg["degenerate"]) is True
        assert deg["degenerate_reason"] == "single-asset input"
