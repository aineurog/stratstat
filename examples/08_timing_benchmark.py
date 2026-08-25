"""Wall-clock timing benchmark: StratStat vs QuantStats (0.0.81).

Measures end-to-end wall-clock time for the *public* APIs on the same
deterministic data:

* per-metric — a representative set of the mutual metrics (apples-to-apples),
  timed through ``stratstat.compute`` (registry + convention resolution) and
  ``quantstats.stats.<fn>`` (direct call) respectively; and

* full report — ``stratstat.compute_all`` (all 176 metrics) vs
  ``quantstats.reports.metrics(mode="full")`` (75 metrics).

Absolute timings are machine-dependent; the informative number is the ratio.
Run: python examples/08_timing_benchmark.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import stratstat as ss
from stratstat.inputs import ReturnsInput

PERIODS_PER_YEAR = 252

try:
    import quantstats as qs
except ImportError:  # pragma: no cover
    raise SystemExit(
        "QuantStats is not installed. Install it with `pip install quantstats` "
        "to run this benchmark (it is an optional dev-time reference only)."
    ) from None


def _time_us(fn, repeats: int) -> float:
    """Mean wall-clock time per call in microseconds (after one warm-up)."""
    fn()  # warm-up (imports, numba jit, cache)
    start = time.perf_counter_ns()
    for _ in range(repeats):
        fn()
    end = time.perf_counter_ns()
    return (end - start) / repeats / 1e3


# ---------------------------------------------------------------------------
# Deterministic data (datetime index so QS drawdown helpers work)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
returns = rng.normal(0.0004, 0.01, size=PERIODS_PER_YEAR * 4)
returns[9::10] = 0.0

idx = pd.bdate_range("2020-01-01", periods=len(returns))
series = pd.Series(returns, index=idx, dtype="float64")
inp = ReturnsInput(returns, periods_per_year=PERIODS_PER_YEAR)

print("=" * 88)
print("STRATSTAT vs QUANTSTATS — wall-clock timing")
print("=" * 88)
print(f"periods_per_year = {PERIODS_PER_YEAR}, n = {len(returns)}")
print("Times are microseconds per call (mean of N repeats after one warm-up).")
print()

# ---------------------------------------------------------------------------
# Per-metric
# ---------------------------------------------------------------------------
# (label, ss callable, qs callable, repeats)
bench = [
    (
        "sharpe_ratio",
        lambda: ss.compute(inp, "sharpe_ratio"),
        lambda: qs.stats.sharpe(series),
        1000,
    ),
    (
        "sortino_ratio",
        lambda: ss.compute(inp, "sortino_ratio"),
        lambda: qs.stats.sortino(series),
        1000,
    ),
    (
        "annualized_volatility",
        lambda: ss.compute(inp, "annualized_volatility"),
        lambda: qs.stats.volatility(series),
        1000,
    ),
    ("cagr", lambda: ss.compute(inp, "cagr"), lambda: qs.stats.cagr(series), 1000),
    ("skewness", lambda: ss.compute(inp, "skewness"), lambda: qs.stats.skew(series), 1000),
    (
        "excess_kurtosis",
        lambda: ss.compute(inp, "excess_kurtosis"),
        lambda: qs.stats.kurtosis(series),
        1000,
    ),
    (
        "max_drawdown",
        lambda: ss.compute(inp, "max_drawdown"),
        lambda: qs.stats.max_drawdown(series),
        500,
    ),
    (
        "ulcer_index",
        lambda: ss.compute(inp, "ulcer_index"),
        lambda: qs.stats.ulcer_index(series),
        500,
    ),
    (
        "consecutive_wins_losses",
        lambda: ss.compute(inp, "consecutive_wins_losses"),
        lambda: qs.stats.consecutive_wins(series),
        500,
    ),
    (
        "psr",
        lambda: ss.compute(inp, "psr"),
        lambda: qs.stats.probabilistic_sharpe_ratio(series),
        500,
    ),
    (
        "monte_carlo sharpe (1000 sims)",
        lambda: ss.compute(inp, "monte_carlo_distribution", target="sharpe", sims=1000, seed=42),
        lambda: qs.stats.montecarlo_sharpe(series, sims=1000, seed=42),
        3,
    ),
]

print("Per-metric (us/call)")
print("-" * 88)
print(f"{'metric':<28s} {'StratStat':>12s} {'QuantStats':>12s} {'QS/SS':>8s}")
print("-" * 88)
for label, ss_fn, qs_fn, repeats in bench:
    ss_us = _time_us(ss_fn, repeats)
    qs_us = _time_us(qs_fn, repeats)
    ratio = qs_us / ss_us if ss_us > 0 else float("inf")
    print(f"{label:<28s} {ss_us:>11.2f} {qs_us:>11.2f} {ratio:>7.2f}x")

print()
print("  (ratio > 1 means StratStat is faster; ratio < 1 means QuantStats is faster.)")
print("  StratStat is timed through the public registry (``ss.compute``), which")
print("  resolves the metric + conventions on each call; QuantStats is a direct call.")
print()

# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------
print("Full report")
print("-" * 88)

ss_full_us = _time_us(lambda: ss.compute_all(inp), 3)
qs_full_us = _time_us(lambda: qs.reports.metrics(series, mode="full", display=False), 3)

ss_count = len(ss.list_metrics())
qs_count = len(qs.reports.metrics(series, mode="full", display=False))

print(f"  StratStat compute_all  (all {ss_count} metrics): {ss_full_us:,.1f} us")
print(f"  QuantStats reports     ({qs_count} metrics):     {qs_full_us:,.1f} us")
print(
    f"  per-metric: StratStat {ss_full_us / ss_count:.2f} us, "
    f"QuantStats {qs_full_us / qs_count:.2f} us"
)
print()
print("=" * 88)
print("Note: absolute times are machine-dependent; compare the ratios, and note")
print("the different metric counts (StratStat exposes 176 metrics to QuantStats'")
print("75 in full-report mode).")
