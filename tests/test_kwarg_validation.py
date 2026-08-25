"""Regression tests: compute_all must not silently ignore what it was given.

``compute_all`` filters keywords per metric, because each metric takes
different parameters.  On its own that meant a typo vanished without a word:
``compute_all(returns=r, rff=0.04)`` ran 84 metrics and returned a result
identical to the run without the typo, so the caller's risk free rate was
discarded while the output looked authoritative.  Same failure class as the
``rf`` defect covered in ``test_rf_propagation.py``.

Two behaviours are asserted here, and the split between them matters:

* a keyword no metric and no container declares is a typo, and raises;
* a keyword that is recognised but that nothing which actually ran consumed is
  legal, so it must not raise, but it had no effect and is reported in
  ``meta["unused_kwargs"]`` rather than dropped.

The second case is why validation is not done against the metrics that ran.
Fifteen parameters belong only to resampling metrics, which ``compute_all``
always excludes, so a narrower check would reject ``target_metric`` and
fourteen others as typos.
"""

import numpy as np
import pytest

from stratstat import compute_all
from stratstat.inputs import (
    BenchmarkInput,
    CompareInput,
    ExposureInput,
    ReturnsInput,
    TradeInput,
)
from stratstat.registry import _container_params, _known_kwargs, _param_names, _registry


@pytest.fixture
def returns(rng):
    return rng.normal(0.10 / 252, 0.20 / np.sqrt(252), size=504)


def test_typo_raises(returns, daily_periods):
    """The exact call that motivated this: a misspelled rf ran clean."""
    with pytest.raises(TypeError, match="rff"):
        compute_all(returns=returns, periods_per_year=daily_periods, rff=0.04)


def test_typo_error_suggests_the_intended_keyword(returns, daily_periods):
    """The caller meant something. Say what."""
    with pytest.raises(TypeError, match="did you mean 'rf'"):
        compute_all(returns=returns, periods_per_year=daily_periods, rff=0.04)


def test_typo_with_no_close_match_still_raises_cleanly(returns, daily_periods):
    """No suggestion is available, so none should be invented."""
    with pytest.raises(TypeError) as excinfo:
        compute_all(returns=returns, periods_per_year=daily_periods, zzzz=1)
    assert "zzzz" in str(excinfo.value)
    assert "did you mean" not in str(excinfo.value)


def test_all_unknown_keywords_are_named(returns, daily_periods):
    """Reporting only the first would make fixing them a loop."""
    with pytest.raises(TypeError) as excinfo:
        compute_all(returns=returns, periods_per_year=daily_periods, zzzz=1, qqqq=2)
    message = str(excinfo.value)
    assert "zzzz" in message and "qqqq" in message


def test_resampling_only_keyword_does_not_raise(returns, daily_periods):
    """target_metric belongs to a resampling metric, which compute_all always
    excludes. Legal to pass, so it must not be mistaken for a typo."""
    result = compute_all(
        returns=returns, periods_per_year=daily_periods, target_metric="sharpe_ratio"
    )
    assert len(result.results) > 0


def test_recognised_but_unused_keyword_is_reported(returns, daily_periods):
    """Not an error, but it had no effect, and the caller should not have to
    guess that."""
    result = compute_all(
        returns=returns, periods_per_year=daily_periods, target_metric="sharpe_ratio"
    )
    assert result.meta.get("unused_kwargs") == ["target_metric"]


def test_clean_call_reports_no_unused_kwargs(returns, daily_periods):
    """The key must be absent rather than empty, so its presence is meaningful."""
    result = compute_all(returns=returns, periods_per_year=daily_periods)
    assert "unused_kwargs" not in result.meta


def test_rf_is_never_reported_unused(returns, daily_periods):
    """rf is a named parameter of the batch function, so it never reaches
    **kwargs and must not be flagged. Guards against a fix to one silent-drop
    defect creating a false report in another."""
    result = compute_all(returns=returns, periods_per_year=daily_periods, rf=0.04)
    assert "unused_kwargs" not in result.meta


def test_container_keywords_are_recognised():
    """Container parameters are legal keywords even though no metric declares
    them, so validation must span containers as well as metrics."""
    known = _known_kwargs()
    for cls in (ReturnsInput, TradeInput, BenchmarkInput, ExposureInput, CompareInput):
        for param in _container_params(cls, exclude=set()):
            assert param in known, f"{cls.__name__}.{param} not recognised"


def test_every_metric_parameter_is_recognised():
    """No registered metric may declare a parameter the validator would reject.
    If this fails, a legitimate call is being refused."""
    known = _known_kwargs()
    for name, entry in _registry.items():
        for param in _param_names(entry["func"]):
            assert param in known, f"{name} declares {param!r}, which validation rejects"
