"""Monte Carlo comparison between StratStat and QuantStats (0.0.81).

QuantStats' ``montecarlo*`` helpers **shuffle** the return series
(permutation without replacement); StratStat **resamples with replacement**
(Efron's bootstrap).  This is a fundamental methodological difference, so the
distributions are compared side by side rather than asserted equal.

The key consequence is that shuffling leaves the multiset of returns intact,
which makes three of QuantStats' four Monte Carlo statistics degenerate:

* ``montecarlo_sharpe`` — mean and standard deviation are invariant under
  permutation, so every shuffled path has the same Sharpe (std ~ 0).
* ``montecarlo_cagr``  — the product of (1 + r) is invariant under permutation,
  so every path has the same terminal equity (and hence the same CAGR).
* ``montecarlo`` terminal-equity ``.stats`` — same product argument.

Only **maximum drawdown** depends on the *order* of returns, so it is the one
QuantStats Monte Carlo statistic with a genuine spread.  We compare that
against StratStat's bootstrapped max drawdown, and document the permutation-vs
bootstrap difference.

Run: python examples/07_monte_carlo_comparison.py
"""

from __future__ import annotations

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
        "to run this comparison (it is an optional dev-time reference only)."
    ) from None


def _dist(qs_dict: dict) -> list[float]:
    """Re-order a QuantStats distribution dict into StratStat's summary layout."""
    return [
        qs_dict["min"],
        qs_dict["percentile_5"],
        qs_dict["median"],
        qs_dict["mean"],
        qs_dict["percentile_95"],
        qs_dict["max"],
        qs_dict["std"],
    ]


def _show(label: str, ss_arr: np.ndarray, qs_arr: list[float]) -> None:
    idx = ["min", "p05", "median", "mean", "p95", "max", "std"]
    print(f"{label}")
    print(f"  {'':>6s} " + "".join(f"{k:>12s}" for k in idx))
    print(f"  {'SS':>6s} " + "".join(f"{v:12.4f}" for v in ss_arr))
    print(f"  {'QS':>6s} " + "".join(f"{v:12.4f}" for v in qs_arr))


# ---------------------------------------------------------------------------
# Deterministic sample data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
returns = rng.normal(0.0004, 0.01, size=PERIODS_PER_YEAR * 4)
returns[9::10] = 0.0

series = pd.Series(returns, dtype="float64")
inp = ReturnsInput(returns, periods_per_year=PERIODS_PER_YEAR)

SIMS = 1000
SEED = 42
BUST = -0.25
GOAL = 0.0

print("=" * 96)
print("STRATSTAT (bootstrap) vs QUANTSTATS (shuffle) — Monte Carlo")
print("=" * 96)
print(f"periods_per_year = {PERIODS_PER_YEAR}, n = {len(returns)}, sims = {SIMS}")
print(f"bust threshold = {BUST}, goal threshold = {GOAL}")
print()

# ---------------------------------------------------------------------------
# 1. Sharpe — degenerate under shuffle
# ---------------------------------------------------------------------------
print("1. Sharpe ratio distribution")
print("-" * 96)
ss_sharpe = ss.compute(inp, "monte_carlo_distribution", target="sharpe", sims=SIMS, seed=SEED).value
qs_sharpe = qs.stats.montecarlo_sharpe(series, sims=SIMS, seed=SEED)
_show("Sharpe [min, p05, median, mean, p95, max, std]", ss_sharpe, _dist(qs_sharpe))
print("  QuantStats std ~ 0 (a permutation preserves mean & std, so every shuffled")
print("  path's Sharpe is identical). The tiny residual is an artifact: QuantStats")
print("  rebuilds returns via pct_change(), which drops the first observation.")
print("  StratStat's bootstrap yields a genuine spread.")
print()

# ---------------------------------------------------------------------------
# 2. CAGR — degenerate under shuffle
# ---------------------------------------------------------------------------
print("2. CAGR distribution")
print("-" * 96)
ss_cagr = ss.compute(inp, "monte_carlo_distribution", target="cagr", sims=SIMS, seed=SEED).value
qs_cagr = qs.stats.montecarlo_cagr(series, sims=SIMS, seed=SEED)
_show("CAGR [min, p05, median, mean, p95, max, std]", ss_cagr, _dist(qs_cagr))
print("  QuantStats std ~ 0: shuffle preserves the product of (1+r), so every")
print("  path ends at the same terminal equity (and hence the same CAGR).")
print()

# ---------------------------------------------------------------------------
# 3. Terminal equity — degenerate under shuffle
# ---------------------------------------------------------------------------
print("3. Terminal total-return (equity) distribution")
print("-" * 96)
ss_equity = ss.compute(inp, "monte_carlo_distribution", target="equity", sims=SIMS, seed=SEED).value
qs_mc = qs.stats.montecarlo(series, sims=SIMS, bust=BUST, goal=GOAL, seed=SEED)
qs_equity = _dist(qs_mc.stats)
_show("Equity [min, p05, median, mean, p95, max, std]", ss_equity, qs_equity)
print("  QuantStats std ~ 0 for the same product-invariance reason.")
print()

# ---------------------------------------------------------------------------
# 4. Max drawdown — the ONE non-degenerate QuantStats MC statistic
# ---------------------------------------------------------------------------
print("4. Max drawdown distribution (both non-degenerate — order matters)")
print("-" * 96)
ss_mdd = ss.compute(
    inp, "monte_carlo_distribution", target="max_drawdown", sims=SIMS, seed=SEED
).value
qs_mdd = _dist(qs_mc.maxdd)
_show("Max drawdown [min, p05, median, mean, p95, max, std]", ss_mdd, qs_mdd)
print("  Both spread because drawdown depends on return order. The distributions")
print("  differ because StratStat samples with replacement and QuantStats")
print("  permutes without replacement — two different resampling models, neither")
print("  a bug.")
print()

# ---------------------------------------------------------------------------
# 5. Bust / goal probabilities
# ---------------------------------------------------------------------------
print("5. Bust / goal probabilities")
print("-" * 96)
ss_prob = ss.compute(
    inp, "monte_carlo_probabilities", bust=BUST, goal=GOAL, sims=SIMS, seed=SEED
).value
qs_bust = qs_mc.bust_probability
qs_goal = qs_mc.goal_probability
print(f"  bust (MDD <= {BUST}):   StratStat={ss_prob[0]:.4f}  QuantStats={qs_bust:.4f}")
print(f"  goal (equity >= {GOAL}): StratStat={ss_prob[1]:.4f}  QuantStats={qs_goal:.4f}")
print()
print("  QuantStats' goal probability is 0 or 1 (degenerate): every shuffled path")
print("  reaches the same terminal equity, so it either all pass or all fail the")
print("  goal. StratStat bootstraps the terminal return, giving a genuine fraction.")
print()

print("=" * 96)
print("Conclusion: QuantStats' Monte Carlo is a permutation (shuffle) test, which")
print("degenerates for any statistic invariant under reordering (Sharpe, CAGR,")
print("equity) and only produces a spread for order-dependent statistics (max")
print("drawdown). StratStat uses Efron's bootstrap, which yields a genuine spread")
print("for every target.")
