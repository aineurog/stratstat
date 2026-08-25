"""Regression tests: the risk free rate must reach every metric that uses it.

These exist because of a defect where ``_build_input`` popped ``rf`` out of
kwargs and then built a ``ReturnsInput``, which has no ``rf`` parameter.  The
rate was destroyed on the way in, all twelve returns tier metrics that accept
``rf`` silently fell back to 0.0, and ``meta["rf"]`` reported 0.0 so the result
looked internally consistent while ignoring what the caller asked for.

The metric lists here are discovered from the registry rather than hardcoded,
so a metric that starts accepting ``rf`` later is covered automatically.  That
is the point: the original defect survived because nothing asserted the rate
had any effect.
"""

import numpy as np
import pytest

from stratstat import (
    BenchmarkInput,
    CompareInput,
    ReturnsInput,
    by_regime,
    compute,
    compute_all,
    compute_benchmark,
    compute_compare,
    compute_exposure,
    compute_returns,
    compute_trades,
    rolling,
)
from stratstat.registry import _param_names, _registry

# A rate that is meaningful read either as per period or as annual, so these
# tests keep their force when rf becomes an annual quantity.
RF = 0.04


def _rf_metrics(tier):
    """Registered non-resampling metrics on *tier* that declare ``rf``."""
    return sorted(
        name
        for name, entry in _registry.items()
        if entry["requires"] == tier
        and entry["backend"] != "resampling"
        and "rf" in _param_names(entry["func"])
    )


RETURNS_RF_METRICS = _rf_metrics("returns")


def _same(a, b):
    """Structural equality across scalars, arrays, dicts, tuples and None.

    Metric values are not uniformly numeric: some are arrays, some dicts of
    named components, so a bare ``==`` or ``np.allclose`` will not do.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if a is None or isinstance(a, str):
        return a == b
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return x.shape == y.shape and np.allclose(x, y, equal_nan=True)


def _changed(set_zero, set_rf):
    """Names of metrics whose value differs between two MetricSets."""
    a = {m.name: m.value for m in set_zero.results}
    b = {m.name: m.value for m in set_rf.results}
    return {k for k in a if k in b and not _same(a[k], b[k])}


@pytest.fixture
def returns(rng):
    """Two years of daily returns, long enough for the smart and probabilistic
    variants to have a stable sample."""
    return rng.normal(0.10 / 252, 0.20 / np.sqrt(252), size=504)


@pytest.fixture
def benchmark():
    return np.random.default_rng(7).normal(0.08 / 252, 0.18 / np.sqrt(252), size=504)


def test_returns_tier_rf_metrics_discovered():
    """Guard: the parametrized tests below must not be silently vacuous.

    If registry discovery breaks or these metrics lose their ``rf`` parameter,
    every parametrized case would pass by not running at all.

    The bound is ``>=`` rather than ``== 12``. Twelve is the count at the time
    of writing, but a metric that legitimately starts accepting ``rf`` is
    picked up by the parametrized tests automatically, so pinning the exact
    number would fail the suite for a change that is not a defect.
    """
    assert len(RETURNS_RF_METRICS) >= 12
    assert "sharpe_ratio" in RETURNS_RF_METRICS
    assert "sortino_ratio" in RETURNS_RF_METRICS


@pytest.mark.parametrize("name", RETURNS_RF_METRICS)
def test_compute_applies_rf(name, returns, daily_periods):
    """Every returns tier metric accepting rf must respond to it via compute()."""
    zero = compute(returns, name, periods_per_year=daily_periods, rf=0.0)
    rate = compute(returns, name, periods_per_year=daily_periods, rf=RF)
    assert not _same(zero.value, rate.value), f"{name} ignored rf"


@pytest.mark.parametrize("name", RETURNS_RF_METRICS)
def test_compute_reports_rf_actually_used(name, returns, daily_periods):
    """meta must report the rate used, not a default the caller never asked for.

    The original defect was worse than a wrong number because meta agreed with
    the wrong number.

    All twelve report it today, so this asserts unconditionally rather than
    skipping when the key is absent. A metric that stops reporting the rate it
    used is itself the regression.
    """
    result = compute(returns, name, periods_per_year=daily_periods, rf=RF)
    assert "rf" in result.meta, f"{name} does not report the rf it used"
    assert result.meta["rf"] == RF


def test_compute_returns_applies_rf(returns, daily_periods):
    """The whole returns tier, in one batch call."""
    changed = _changed(
        compute_returns(returns, periods_per_year=daily_periods, rf=0.0),
        compute_returns(returns, periods_per_year=daily_periods, rf=RF),
    )
    assert changed == set(RETURNS_RF_METRICS)


def test_compute_all_applies_rf(returns, daily_periods):
    """compute_all takes rf as a named parameter, so it never lands in **kwargs
    and has to be injected explicitly."""
    changed = _changed(
        compute_all(returns=returns, periods_per_year=daily_periods, rf=0.0),
        compute_all(returns=returns, periods_per_year=daily_periods, rf=RF),
    )
    assert changed == set(RETURNS_RF_METRICS)


def test_compute_all_applies_rf_with_benchmark(returns, benchmark, daily_periods):
    """Benchmark metrics read rf off their container; returns metrics take it as
    a kwarg. Both routes must work in the same call."""
    changed = _changed(
        compute_all(returns=returns, benchmark=benchmark, periods_per_year=daily_periods, rf=0.0),
        compute_all(returns=returns, benchmark=benchmark, periods_per_year=daily_periods, rf=RF),
    )
    assert set(RETURNS_RF_METRICS) <= changed
    assert {"alpha", "treynor_ratio"} <= changed


def test_prebuilt_returns_input_still_applies_rf(returns, daily_periods):
    """A pre-built container carries no rf of its own, so the kwarg must still
    reach the metric."""
    inp = ReturnsInput(returns, periods_per_year=daily_periods)
    assert compute(inp, "sharpe_ratio", rf=0.0).value != compute(inp, "sharpe_ratio", rf=RF).value


def test_rolling_applies_rf(returns, daily_periods):
    """rolling() dispatches through the same path and inherited the defect."""
    inp = ReturnsInput(returns, periods_per_year=daily_periods)
    assert not np.allclose(
        rolling(inp, "sharpe_ratio", 60).value,
        rolling(inp, "sharpe_ratio", 60, rf=RF).value,
        equal_nan=True,
    )


def test_by_regime_applies_rf(returns, daily_periods):
    """by_regime() dispatches through the same path and inherited the defect."""
    inp = ReturnsInput(returns, periods_per_year=daily_periods)
    regimes = np.where(np.arange(len(returns)) < len(returns) // 2, "first", "second")
    assert not np.allclose(
        by_regime(inp, "sharpe_ratio", regimes).value,
        by_regime(inp, "sharpe_ratio", regimes, rf=RF).value,
        equal_nan=True,
    )


def test_benchmark_tier_applies_rf(returns, benchmark, daily_periods):
    """Benchmark metrics never declare rf; they read it off BenchmarkInput.
    Fixing the returns tier must not have broken that."""
    zero = compute_benchmark(returns, benchmark, periods_per_year=daily_periods, rf=0.0)
    rate = compute_benchmark(returns, benchmark, periods_per_year=daily_periods, rf=RF)
    assert {"alpha", "treynor_ratio"} <= _changed(zero, rate)


def test_compare_tier_applies_rf(returns, benchmark, daily_periods):
    """Compare metrics also read rf off their container rather than declaring
    it. ``sharpe_difference_test`` is the one that depends on it."""
    matrix = np.column_stack([returns, benchmark])
    changed = _changed(
        compute_compare(matrix, periods_per_year=daily_periods, rf=0.0),
        compute_compare(matrix, periods_per_year=daily_periods, rf=RF),
    )
    assert "sharpe_difference_test" in changed


def test_compute_all_applies_rf_on_compare_tier(returns, benchmark, daily_periods):
    """The compare tier is only built when compare data is passed explicitly,
    so it needs its own case."""
    matrix = np.column_stack([returns, benchmark])
    changed = _changed(
        compute_all(compare=matrix, periods_per_year=daily_periods, rf=0.0),
        compute_all(compare=matrix, periods_per_year=daily_periods, rf=RF),
    )
    assert "sharpe_difference_test" in changed


def test_benchmark_input_carries_rf(returns, benchmark, daily_periods):
    inp = BenchmarkInput(returns, benchmark=benchmark, periods_per_year=daily_periods, rf=RF)
    assert inp.rf == RF


def test_compare_input_carries_rf(returns, benchmark, daily_periods):
    matrix = np.column_stack([returns, benchmark])
    inp = CompareInput(matrix, periods_per_year=daily_periods, rf=RF)
    assert inp.rf == RF


@pytest.mark.parametrize(
    "tier_metric",
    [("win_rate", "trades"), ("gross_exposure", "exposure")],
)
def test_tiers_without_an_rf_consumer_accept_it_harmlessly(
    tier_metric, returns, daily_periods, rng
):
    """rf is accepted on every entry point for signature uniformity. Tiers with
    no consumer must ignore it rather than raise, since callers routinely pass
    it alongside periods_per_year."""
    import pandas as pd

    name, tier = tier_metric
    data = (
        pd.DataFrame({"pnl": rng.normal(0.002, 0.02, size=80)})
        if tier == "trades"
        else rng.random((252, 3))
    )
    with_rf = compute(data, name, periods_per_year=daily_periods, rf=RF)
    without = compute(data, name, periods_per_year=daily_periods)
    assert _same(with_rf.value, without.value)


def test_batch_tiers_without_an_rf_consumer_accept_it_harmlessly(returns, daily_periods, rng):
    """Same guarantee as above, but through the batch entry points, which route
    through ``_compute_all`` rather than ``_compute_one``. Neither declares rf
    as a named parameter, so it arrives in **kwargs and is bound by
    ``_compute_all``'s own rf parameter."""
    import pandas as pd

    trades = pd.DataFrame({"pnl": rng.normal(0.002, 0.02, size=80)})
    positions = rng.random((252, 3))

    for name, data, fn in (
        ("trades", trades, compute_trades),
        ("exposure", positions, compute_exposure),
    ):
        with_rf = fn(data, periods_per_year=daily_periods, rf=RF)
        without = fn(data, periods_per_year=daily_periods)
        assert not _changed(without, with_rf), f"{name} tier was affected by rf"
        assert len(with_rf.results) > 0


@pytest.mark.parametrize("entry", ["compute", "rolling", "by_regime"])
def test_misspelled_keyword_still_raises(entry, returns, daily_periods):
    """Dropping rf for metrics that do not declare it must not turn into
    swallowing every unknown keyword. A typo has to stay loud."""
    inp = ReturnsInput(returns, periods_per_year=daily_periods)
    regimes = np.where(np.arange(len(returns)) < len(returns) // 2, "a", "b")
    calls = {
        "compute": lambda: compute(returns, "sharpe_ratio", periods_per_year=daily_periods, rff=RF),
        "rolling": lambda: rolling(inp, "sharpe_ratio", 60, rff=RF),
        "by_regime": lambda: by_regime(inp, "sharpe_ratio", regimes, rff=RF),
    }
    with pytest.raises(TypeError):
        calls[entry]()
