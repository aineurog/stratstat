#!/usr/bin/env python
"""Benchmark the numba kernels against their numpy fallbacks.

Run from the repository root:

    python benchmarks/benchmark_numba.py

Each row times one kernel and reports:

* ``compile (ms)``: the first call to the numba function, which includes the
  one time JIT compile cost.  This is what the compile gate (``numba_worthwhile``
  in ``core/_utils.py``) is designed to avoid paying for small workloads.
* ``numba (ms)``: the per call time after compilation.
* ``numpy (ms)``: the numpy fallback per call time.
* ``speedup``: ``numpy (ms) / numba (ms)``.  Values above 1.0 mean numba wins.

Requires numpy.  numba is optional; without it the script reports fallback
timing only and skips the numba columns.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make ``stratstat`` importable from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import stratstat.core.compare as compare_mod  # noqa: E402
import stratstat.core.returns.inference as inference_mod  # noqa: E402
import stratstat.core.returns.risk as risk_mod  # noqa: E402
from stratstat.core._utils import is_numba_available  # noqa: E402

HAS_NUMBA = is_numba_available()


def _compile_cost(fn, args) -> float:
    """Seconds for the first call to ``fn`` (for numba this includes JIT)."""
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def _measure(fn, args, min_time: float = 0.2) -> float:
    """Seconds per call, measuring for at least ``min_time`` wall clock.

    The first invocation warms ``fn``; for numba this triggers compilation, so
    call :func:`_compile_cost` first when compile time matters.
    """
    fn(*args)  # warm up
    reps = 1
    while True:
        start = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        elapsed = time.perf_counter() - start
        if elapsed >= min_time:
            return elapsed / reps
        reps *= 2


def _numba_fn(mod, name):
    """Return the numba kernel from ``mod``, or None when numba is absent."""
    return getattr(mod, name, None) if HAS_NUMBA else None


def _cases() -> list[tuple[str, object, object, tuple]]:
    """Build (name, numba_fn, numpy_fn, args) for each accelerated kernel."""
    cases: list[tuple[str, object, object, tuple]] = []

    # 1. Block index assembly.
    n, block_len, n_reps = 252, 3, 5000
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(1)
    block_starts = rng.integers(0, n - block_len + 1, size=(n_reps, n_blocks))
    cases.append((
        f"block indices ({n_reps}x{n})",
        _numba_fn(inference_mod, "_assemble_block_indices_numba"),
        inference_mod._assemble_block_indices,
        (block_starts, n, block_len),
    ))

    # 2. Sharpe bootstrap.  Build indices with the numpy assembler so this
    # setup does not trigger the numba compile that case 1 is meant to measure.
    rng = np.random.default_rng(2)
    r = rng.normal(0.0004, 0.01, size=n)
    idx_starts = np.random.default_rng(3).integers(
        0, n - block_len + 1, size=(n_reps, n_blocks)
    )
    indices = inference_mod._assemble_block_indices(idx_starts, n, block_len)
    cases.append((
        f"sharpe bootstrap ({n_reps}x{n})",
        _numba_fn(inference_mod, "_sharpe_bootstrap_numba"),
        inference_mod._sharpe_bootstrap_fallback,
        (r, indices, 1),
    ))

    # 3. Stationary bootstrap (as used by White's Reality Check).
    rng = np.random.default_rng(4)
    data = rng.normal(0.0003, 0.008, size=(252, 3))
    n_boot = 1000
    n_periods = data.shape[0]
    drng = np.random.default_rng(5)
    starts = drng.integers(0, n_periods, size=(n_boot, n_periods))
    blens = drng.geometric(1.0, size=(n_boot, n_periods))
    cases.append((
        f"stationary bootstrap ({n_boot}x{n_periods}x3)",
        _numba_fn(compare_mod, "_stationary_bootstrap_numba"),
        compare_mod._stationary_bootstrap_fallback,
        (data, starts, blens),
    ))

    # 4. PBO overfit count.
    rng = np.random.default_rng(6)
    pdata = rng.normal(0.0002, 0.01, size=(252, 5))
    split_points = np.arange(80, 130, dtype=np.int64)
    cases.append((
        "pbo overfit (50 splits x 252 x 5)",
        _numba_fn(compare_mod, "_pbo_overfit_numba"),
        compare_mod._pbo_overfit_fallback,
        (pdata, split_points, 3, 0, 0.0),
    ))

    # 5. Drawdown episode walk (the original numba path in risk.py).
    rng = np.random.default_rng(7)
    dr = rng.normal(0.0004, 0.01, size=10000)
    eq = risk_mod._equity_curve(dr.reshape(-1, 1), "simple")
    rm, dd_ser = risk_mod._drawdown_series(eq)
    cases.append((
        "drawdown walk (10000 periods)",
        _numba_fn(risk_mod, "_drawdown_episodes_numba"),
        risk_mod._drawdown_episodes,
        (eq[:, 0], rm[:, 0], dd_ser[:, 0]),
    ))

    return cases


def main() -> None:
    print("StratStat numba vs numpy fallback")
    print("=================================\n")
    if not HAS_NUMBA:
        print("numba not installed: reporting fallback timing only.\n")

    header = (
        f"{'kernel':<36} {'compile (ms)':>13} {'numba (ms)':>12} "
        f"{'numpy (ms)':>12} {'speedup':>9}"
    )
    print(header)
    print("-" * len(header))

    for name, numba_fn, numpy_fn, args in _cases():
        numpy_ms = _measure(numpy_fn, args) * 1e3
        if numba_fn is None:
            print(f"{name:<36} {'-':>13} {'-':>12} {numpy_ms:>12.2f} {'-':>9}")
            continue

        compile_ms = _compile_cost(numba_fn, args) * 1e3
        numba_ms = _measure(numba_fn, args) * 1e3
        speedup = numpy_ms / numba_ms if numba_ms > 0 else float("nan")
        print(
            f"{name:<36} {compile_ms:>13.1f} {numba_ms:>12.2f} "
            f"{numpy_ms:>12.2f} {speedup:>8.1f}x"
        )

    print("\nTimes are per call; speedup is numpy/numba (higher is better).")


if __name__ == "__main__":
    main()
