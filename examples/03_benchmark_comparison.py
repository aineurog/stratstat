"""Benchmark-relative metrics and comparison against a market index.

Covers BenchmarkInput, all benchmark-tier metrics (alpha, beta, tracking
error, information ratio, up/down capture, batting average, Treynor ratio,
information coefficient, directional consistency, etc.).

Run: python examples/03_benchmark_comparison.py
"""

import numpy as np

import stratstat as ss
from stratstat.inputs import BenchmarkInput

# ---------------------------------------------------------------------------
# Generate synthetic strategy and benchmark returns
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 252  # 1 year of daily data

# Benchmark: ~8% annual return, 15% vol
bench_returns = rng.normal(0.0003, 0.0095, size=n)

# Strategy: benchmark + some alpha + tracking error
# Roughly beta=1.1, alpha ~3% annual
true_beta = 1.1
true_alpha = 0.03 / 252  # ~3% annualised
strategy_returns = true_alpha + true_beta * bench_returns + rng.normal(0, 0.005, size=n)

print("=" * 60)
print("BENCHMARK COMPARISON")
print("=" * 60)

# -- Create a BenchmarkInput -------------------------------------------------
bm_inp = BenchmarkInput(returns=strategy_returns, benchmark=bench_returns, periods_per_year=252)

# -- Compute individual benchmark metrics ------------------------------------
alpha = ss.compute(bm_inp, "alpha")
beta = ss.compute(bm_inp, "beta")
r2 = ss.compute(bm_inp, "r_squared")
tracking_err = ss.compute(bm_inp, "tracking_error")
info_ratio = ss.compute(bm_inp, "information_ratio")
up_cap = ss.compute(bm_inp, "up_capture")
down_cap = ss.compute(bm_inp, "down_capture")
corr = ss.compute(bm_inp, "correlation")

print(f"\n{alpha}")
print(f"{beta}")
print(f"{r2}")
print(f"{tracking_err}")
print(f"{info_ratio}")
print(f"{up_cap}")
print(f"{down_cap}")
print(f"{corr}")

# -- All benchmark-tier metrics at once ---------------------------------------
print("\n\nAll benchmark metrics:")
all_bench = ss.compute_all(bm_inp, category="benchmark")
print(all_bench)

# -- Batting average and Treynor ----------------------------------------------
batting = ss.compute(bm_inp, "batting_average")
treynor = ss.compute(bm_inp, "treynor_ratio")
print(f"\n{batting}")
print(f"{treynor}")

# -- Information Coefficient (rank IC) ----------------------------------------
ic = ss.compute(bm_inp, "information_coefficient")
dc = ss.compute(bm_inp, "directional_consistency")
print(f"\n{ic}")
print(f"{dc}")

# -- Active return, outperformance stats --------------------------------------
active_ret = ss.compute(bm_inp, "active_return")
outperf = ss.compute(bm_inp, "outperformance")
print(f"\n{active_ret}")
print(f"{outperf}")

# ---------------------------------------------------------------------------
# Multiple strategies vs the same benchmark (one at a time)
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 60)
print("COMPARING MULTIPLE STRATEGIES")
print("=" * 60)

rng = np.random.default_rng(99)
strat2 = true_alpha * 0.5 + 0.9 * bench_returns + rng.normal(0, 0.007, size=n)
strat3 = true_alpha * 1.5 + 1.2 * bench_returns + rng.normal(0, 0.006, size=n)

for i, strat in enumerate([strategy_returns, strat2, strat3], 1):
    bm = BenchmarkInput(returns=strat, benchmark=bench_returns, periods_per_year=252)
    alpha_val = ss.compute(bm, "alpha")
    beta_val = ss.compute(bm, "beta")
    ir_val = ss.compute(bm, "information_ratio")
    print(
        f"\n  Strategy {i}: {alpha_val.value:.4f} alpha, "
        f"{beta_val.value:.2f} beta, {ir_val.value:.4f} IR"
    )
