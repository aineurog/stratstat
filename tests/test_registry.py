"""Tests for stratstat.registry — registration, listing, and lookup."""

import pytest

from stratstat.exceptions import UnknownMetricError
from stratstat.registry import _registry, get_metric, list_metrics, register_metric


@pytest.fixture(scope="module", autouse=True)
def _cleanup_test_metric():
    """Remove the ``test_metric`` registered below once this module finishes.

    Prevents the synthetic metric (whose function returns a bare float, not a
    ``MetricResult``) from leaking into the global registry and breaking
    ``compute_all`` in later test modules.
    """
    yield
    _registry.pop("test_metric", None)


def test_register_metric_adds_to_registry():
    """A decorated function should appear in the registry with correct metadata."""

    @register_metric(
        name="test_metric",
        requires="returns",
        category=("descriptive", "returns"),
        backend="vectorized",
        ref="Test et al. (2024)",
    )
    def test_metric(input_data):
        return 42.0

    assert "test_metric" in _registry
    entry = _registry["test_metric"]
    assert entry["requires"] == "returns"
    assert entry["category"] == ("descriptive", "returns")
    assert entry["backend"] == "vectorized"
    assert entry["ref"] == "Test et al. (2024)"


def test_list_metrics_unfiltered():
    """list_metrics() with no filters should return all registered metrics."""
    results = list_metrics()
    assert isinstance(results, list)
    # At minimum, our test metric should be present
    names = [r["name"] for r in results]
    assert "test_metric" in names


def test_list_metrics_filter_by_requires():
    """Filtering by requires should return only matching metrics."""
    results = list_metrics(requires="returns")
    for r in results:
        assert r["requires"] == "returns"


def test_list_metrics_filter_by_category():
    """Filtering by category should return only metrics whose *primary* tag matches."""
    results = list_metrics(category="descriptive")
    assert results, "expected at least one descriptive metric"
    for r in results:
        assert r["category"][0] == "descriptive"


def test_list_metrics_filter_by_backend():
    """Filtering by backend should return only matching metrics."""
    results = list_metrics(backend="vectorized")
    for r in results:
        assert r["backend"] == "vectorized"


def test_get_metric_returns_entry():
    """get_metric() should return the full metadata dict for a registered metric."""
    entry = get_metric("test_metric")
    assert entry["requires"] == "returns"
    assert callable(entry["func"])


def test_get_metric_unknown_raises():
    """get_metric() should raise UnknownMetricError for an unregistered name."""
    with pytest.raises(UnknownMetricError):
        get_metric("nonexistent_metric")
