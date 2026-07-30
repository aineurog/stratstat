"""Shared test fixtures for StratStat.

Provides sample returns data, benchmarks, and known-value references
for metric validation tests.
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_returns(rng):
    """Single-strategy returns: 252 daily periods, ~10% annual return, ~20% annual vol."""
    n = 252
    return rng.normal(0.10 / 252, 0.20 / np.sqrt(252), size=n)


@pytest.fixture
def multi_returns(rng):
    """Three-strategy returns matrix: 252 daily periods, 3 strategies."""
    n = 252
    return rng.normal(0.10 / 252, 0.20 / np.sqrt(252), size=(n, 3))


@pytest.fixture
def flat_returns():
    """All-zero returns (edge case)."""
    return np.zeros(252)


@pytest.fixture
def daily_periods():
    """Standard daily periods_per_year."""
    return 252
