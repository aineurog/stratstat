"""Tests for stratstat.registry — registration, listing, and lookup."""

import pytest

from stratstat.registry import register_metric, list_metrics, get_metric, _registry
from stratstat.exceptions import UnknownMetricError


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
    """Filtering by category should return only metrics with that tag."""
    results = list_metrics(category="descriptive")
    for r in results:
        assert "descriptive" in r["category"]


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
