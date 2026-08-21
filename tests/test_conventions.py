"""Tests for session-level convention overrides."""

import numpy as np
import pytest

from stratstat.conventions import clear_defaults, get_default, set_default
from stratstat.core.returns.risk_adjusted import sharpe_ratio, sortino_ratio
from stratstat.exceptions import ConventionError
from stratstat.inputs import ReturnsInput


@pytest.fixture(autouse=True)
def _clean_defaults():
    """Isolate tests from session-wide convention state."""
    clear_defaults()
    yield
    clear_defaults()


def test_set_and_get_default():
    """set_default() should make a convention retrievable via get_default()."""
    set_default("sharpe_ratio", "ddof=0")
    assert get_default("sharpe_ratio") == "ddof=0"


def test_get_default_returns_none_for_unset():
    """get_default() should return None for metrics with no override."""
    assert get_default("nonexistent_metric") is None


def test_clear_defaults():
    """clear_defaults() should reset all overrides."""
    set_default("sharpe_ratio", "ddof=0")
    set_default("sortino_ratio", "denominator=downside_only")
    clear_defaults()
    assert get_default("sharpe_ratio") is None
    assert get_default("sortino_ratio") is None


def test_set_default_overwrites():
    """Setting the same metric twice should overwrite the previous value."""
    set_default("var", "method=historical")
    set_default("var", "method=parametric")
    assert get_default("var") == "method=parametric"


def test_set_default_unknown_metric_raises():
    """set_default() should reject metrics with no convention surface."""
    with pytest.raises(ConventionError):
        set_default("annual_return", "method=geometric")


def test_set_default_missing_equals_raises():
    """set_default() should require the 'param=value' form."""
    with pytest.raises(ConventionError):
        set_default("sharpe_ratio", "0")


def test_set_default_unknown_param_raises():
    """set_default() should reject an unknown parameter for a known metric."""
    with pytest.raises(ConventionError):
        set_default("sharpe_ratio", "denominator=full_downside")


def test_set_default_invalid_value_raises():
    """set_default() should reject a value outside the parameter's vocabulary."""
    with pytest.raises(ConventionError):
        set_default("sharpe_ratio", "ddof=2")
    with pytest.raises(ConventionError):
        set_default("var", "confidence=1.5")


def test_default_wires_into_sharpe():
    """A session default should be consulted as a fallback by the metric."""
    set_default("sharpe_ratio", "ddof=0")
    r = ReturnsInput(np.array([0.01, -0.02, 0.03, 0.01, -0.01]), periods_per_year=252)
    result = sharpe_ratio(r)
    assert result.meta["ddof"] == 0


def test_default_wires_into_sortino():
    """A session default should be consulted as a fallback by the metric."""
    set_default("sortino_ratio", "denominator=downside_only")
    r = ReturnsInput(np.array([0.01, -0.02, 0.03, 0.01, -0.01]), periods_per_year=252)
    result = sortino_ratio(r)
    assert result.meta["denominator"] == "downside_only"


def test_explicit_argument_beats_default():
    """An explicit keyword argument should take precedence over the default."""
    set_default("sharpe_ratio", "ddof=0")
    r = ReturnsInput(np.array([0.01, -0.02, 0.03, 0.01, -0.01]), periods_per_year=252)
    result = sharpe_ratio(r, ddof=1)
    assert result.meta["ddof"] == 1


def test_default_wires_into_exposure_time_rounding():
    """Session default should round exposure up to the nearest percent."""
    from stratstat.core.returns.descriptive import exposure_time

    set_default("exposure_time", "rounding=percent_ceil")
    returns = np.array(
        [0.01, 0.0, -0.02, 0.03, 0.0, 0.005, -0.01, 0.02, 0.0, 0.0, 0.001]
    )
    result = exposure_time(ReturnsInput(returns))
    assert result.value == pytest.approx(0.64, rel=1e-12)


def test_default_wires_into_rar_rounding():
    """Session default should select RAR's rounded exposure denominator."""
    from stratstat.core.returns.risk_adjusted import rar

    set_default("rar", "rounding=percent_ceil")
    returns = np.array(
        [0.01, 0.0, -0.02, 0.03, 0.0, 0.005, -0.01, 0.02, 0.0, 0.0, 0.001]
    )
    result = rar(ReturnsInput(returns, periods_per_year=252))
    assert result.meta["rounding"] == "percent_ceil"


def test_default_wires_into_psr_se_formula():
    """Session default should select the Lo standard-error formula."""
    from stratstat.core.returns.inference import psr

    set_default("psr", "se_formula=lo")
    r = ReturnsInput(np.array([0.01, -0.02, 0.03, 0.01, -0.01]), periods_per_year=252)
    result = psr(r)
    assert result.meta["se_formula"] == "lo"


def test_invalid_new_conventions_raise():
    """The new convention parameters reject out-of-vocabulary values."""
    with pytest.raises(ConventionError):
        set_default("exposure_time", "rounding=nearest")
    with pytest.raises(ConventionError):
        set_default("psr", "se_formula=lo2")
