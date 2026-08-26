"""Regression tests: constructor data must reach the container that wants it.

``_build_input`` used to forward only ``periods_per_year``.  Every other Input
constructor parameter fell through to the metric function, which then raised an
error blaming itself for a keyword it was never meant to take.  Passing
``benchmark=`` to a benchmark metric raised ``ValueError: ... Provide benchmark=
to BenchmarkInput``, advice the caller had already followed, and passing
``asset_returns=`` to an exposure metric raised ``TypeError: long_book_return()
got an unexpected keyword argument 'asset_returns'``, which reads as if the
metric does not support asset returns when in fact ``ExposureInput`` does.

The working routes at the time were a pre-built container or, for the benchmark
tier only, a ``(returns, benchmark)`` tuple.  Both still work and are asserted
here, because the keyword form was added alongside them, not in place of them.
"""

import numpy as np
import pytest

from stratstat import (
    BenchmarkInput,
    CompareInput,
    ExposureInput,
    ReturnsInput,
    compute,
)


@pytest.fixture
def returns(rng):
    return rng.normal(0.10 / 252, 0.20 / np.sqrt(252), size=504)


@pytest.fixture
def benchmark():
    return np.random.default_rng(7).normal(0.08 / 252, 0.18 / np.sqrt(252), size=504)


@pytest.fixture
def positions():
    return np.random.default_rng(3).random((504, 3))


@pytest.fixture
def asset_returns():
    return np.random.default_rng(5).normal(0.0004, 0.01, size=(504, 3))


def test_benchmark_reachable_by_keyword(returns, benchmark, daily_periods):
    """The case that raised advice the caller had already taken."""
    result = compute(returns, "alpha", periods_per_year=daily_periods, benchmark=benchmark)
    assert np.isfinite(result.value)


def test_keyword_form_matches_tuple_form(returns, benchmark, daily_periods):
    """The keyword form is a new spelling of an existing route, not a new
    computation. It must agree exactly, not merely approximately."""
    by_kwarg = compute(returns, "alpha", periods_per_year=daily_periods, benchmark=benchmark).value
    by_tuple = compute((returns, benchmark), "alpha", periods_per_year=daily_periods).value
    assert by_kwarg == by_tuple


def test_keyword_form_matches_prebuilt_container(returns, benchmark, daily_periods):
    by_kwarg = compute(returns, "alpha", periods_per_year=daily_periods, benchmark=benchmark).value
    container = BenchmarkInput(returns, benchmark=benchmark, periods_per_year=daily_periods)
    assert by_kwarg == compute(container, "alpha").value


def test_exposure_asset_returns_reachable_by_keyword(positions, asset_returns, daily_periods):
    """ExposureInput.asset_returns means asset level returns. It was
    unreachable, and the error blamed the metric."""
    result = compute(
        positions, "long_book_return", periods_per_year=daily_periods, asset_returns=asset_returns
    )
    assert np.isfinite(result.value)


def test_exposure_keyword_form_matches_prebuilt(positions, asset_returns, daily_periods):
    by_kwarg = compute(
        positions, "long_book_return", periods_per_year=daily_periods, asset_returns=asset_returns
    ).value
    container = ExposureInput(
        positions, asset_returns=asset_returns, periods_per_year=daily_periods
    )
    assert by_kwarg == compute(container, "long_book_return").value


def test_compare_weights_reachable_by_keyword(returns, benchmark, daily_periods):
    matrix = np.column_stack([returns, benchmark])
    weights = np.array([0.6, 0.4])
    by_kwarg = compute(
        matrix, "sharpe_difference_test", periods_per_year=daily_periods, weights=weights
    ).value
    container = CompareInput(matrix, weights=weights, periods_per_year=daily_periods)
    assert np.allclose(
        np.asarray(by_kwarg, dtype=float),
        np.asarray(compute(container, "sharpe_difference_test").value, dtype=float),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("metric", "extra"),
    [
        ("alpha", {"benchmarkk": True}),
        ("alpha", {"nonsense": 1}),
        ("sharpe_ratio", {"rff": 0.04}),
        ("gross_exposure", {"nonsense": 1}),
    ],
)
def test_unrecognised_keyword_still_raises(
    metric, extra, returns, benchmark, positions, daily_periods
):
    """Forwarding recognised parameters must not turn into swallowing every
    keyword. Only what the target container declares is taken; the rest falls
    through and raises, so a typo stays loud.

    The data each metric needs is supplied, so the raise can only come from the
    bad keyword and not from something legitimately missing.
    """
    data, kwargs = returns, dict(extra)
    if metric == "alpha":
        kwargs["benchmark"] = benchmark
    elif metric == "gross_exposure":
        data = positions

    with pytest.raises(TypeError):
        compute(data, metric, periods_per_year=daily_periods, **kwargs)


def test_prebuilt_container_is_not_silently_overridden(returns, benchmark, daily_periods):
    """A pre-built container already carries its data. Forwarding happens only
    when this code constructs the container, so a stray keyword against a
    pre-built one is not quietly swallowed against an object that would ignore
    it."""
    container = BenchmarkInput(returns, benchmark=benchmark, periods_per_year=daily_periods)
    other = np.zeros_like(benchmark)
    with pytest.raises(TypeError):
        compute(container, "alpha", benchmark=other)


def test_returns_container_takes_no_secondary_data(returns, daily_periods):
    """ReturnsInput declares only data and periods_per_year, so nothing extra
    should be absorbed by it. rf in particular must stay in kwargs to reach the
    12 returns tier metrics that declare it themselves."""
    inp = ReturnsInput(returns, periods_per_year=daily_periods)
    assert compute(inp, "sharpe_ratio", rf=0.0).value != compute(inp, "sharpe_ratio", rf=0.04).value
