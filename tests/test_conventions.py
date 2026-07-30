"""Tests for session-level convention overrides."""

from stratstat.conventions import set_default, get_default, clear_defaults


def test_set_and_get_default():
    """set_default() should make a convention retrievable via get_default()."""
    set_default("sharpe_ratio", "ddof=1")
    assert get_default("sharpe_ratio") == "ddof=1"


def test_get_default_returns_none_for_unset():
    """get_default() should return None for metrics with no override."""
    assert get_default("nonexistent_metric") is None


def test_clear_defaults():
    """clear_defaults() should reset all overrides."""
    set_default("sharpe_ratio", "ddof=1")
    set_default("sortino_ratio", "full_downside")
    clear_defaults()
    assert get_default("sharpe_ratio") is None
    assert get_default("sortino_ratio") is None


def test_set_default_overwrites():
    """Setting the same metric twice should overwrite the previous value."""
    set_default("var", "historical")
    set_default("var", "parametric")
    assert get_default("var") == "parametric"

    clear_defaults()
