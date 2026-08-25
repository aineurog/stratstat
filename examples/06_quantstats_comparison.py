"""Full parity check between StratStat and QuantStats (0.0.81).

QuantStats is the reference implementation StratStat is benchmarked against.
This example compares **every** metric the two libraries expose on the same
input (a return series, and a benchmark series for the benchmark tier), and
reports whether the values agree.

Run: python examples/06_quantstats_comparison.py

QuantStats is NOT a StratStat dependency — it is used only here, as an
optional dev-time reference.  If it is not installed the script prints a hint
and exits cleanly, so it can never break the test suite.

The comparison is split into six tiers:

1. EXACT MATCH — same formula, same result to floating-point precision.
2. CONVENTION — identical once StratStat's formula-selection option is set
   (e.g. ``rounding="percent_ceil"`` reproduces QuantStats' exposure).
3. SIGN CONVENTION — same magnitude, opposite sign (loss sign conventions).
4. FORMULA — genuinely different definitions; the values are shown for
   documentation, not as failures.
5. BENCHMARK — metrics that need a benchmark series (beta/R² match; alpha,
   information ratio and Treynor use different annualization/numerators).
6. QS-ONLY — QuantStats metrics with no returns-tier StratStat equivalent
   (StratStat has a *trades-tier* counterpart instead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import stratstat as ss
from stratstat.inputs import BenchmarkInput, ReturnsInput

PERIODS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import quantstats as qs
except ImportError:  # pragma: no cover
    raise SystemExit(
        "QuantStats is not installed. Install it with `pip install quantstats` "
        "to run this comparison (it is an optional dev-time reference only)."
    ) from None


def _isclose(a: float, b: float, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    """True when two scalars agree to floating-point tolerance (NaN-aware)."""
    a_nan, b_nan = np.isnan(a), np.isnan(b)
    if a_nan or b_nan:
        return a_nan and b_nan
    return bool(np.isclose(a, b, rtol=rtol, atol=atol))


def _fmt(x: float) -> str:
    if isinstance(x, float) and np.isnan(x):
        return "nan"
    return f"{x:.8f}"


# ---------------------------------------------------------------------------
# Deterministic sample data: 4 years of daily returns with injected flat days
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
returns = rng.normal(0.0004, 0.01, size=PERIODS_PER_YEAR * 4)
returns[9::10] = 0.0  # every 10th day flat -> exposure ~0.90, not a round %

benchmark = rng.normal(0.0003, 0.009, size=returns.shape[0])

# A DatetimeIndex is required by several QuantStats helpers (``resample``,
# the phantom-baseline drawdown logic).  It does not change any daily value.
index = pd.bdate_range("2020-01-01", periods=len(returns))
series = pd.Series(returns, index=index, dtype="float64")
bench_series = pd.Series(benchmark, index=index, dtype="float64")

inp = ReturnsInput(returns, periods_per_year=PERIODS_PER_YEAR)
binp = BenchmarkInput(returns, benchmark=benchmark, periods_per_year=PERIODS_PER_YEAR)

print("=" * 96)
print("STRATSTAT vs QUANTSTATS — full mutual-metric parity check")
print("=" * 96)
print(f"periods_per_year = {PERIODS_PER_YEAR}, n = {len(returns)}")
print(
    f"returns: mean={returns.mean():.6f}, std={returns.std(ddof=1):.6f}, "
    f"non-zero share={float((returns != 0).mean()):.6f}"
)
print(f"benchmark: mean={benchmark.mean():.6f}, std={benchmark.std(ddof=1):.6f}")
print()


# ---------------------------------------------------------------------------
# 1. EXACT MATCH — identical formula, identical result
# ---------------------------------------------------------------------------
def _consec_wins():
    return float(ss.compute(inp, "consecutive_wins_losses").value["max_win_streak"])


def _consec_losses():
    return float(ss.compute(inp, "consecutive_wins_losses").value["max_loss_streak"])


exact = [
    (
        "geometric_mean_return",
        lambda: ss.compute(inp, "geometric_mean_return").value,
        lambda: qs.stats.expected_return(series),
    ),
    (
        "cumulative_return",
        lambda: ss.compute(inp, "cumulative_return").value,
        lambda: qs.stats.comp(series),
    ),
    ("best_period", lambda: ss.compute(inp, "best_period").value, lambda: qs.stats.best(series)),
    ("worst_period", lambda: ss.compute(inp, "worst_period").value, lambda: qs.stats.worst(series)),
    ("consecutive wins (max)", _consec_wins, lambda: qs.stats.consecutive_wins(series)),
    ("consecutive losses (max)", _consec_losses, lambda: qs.stats.consecutive_losses(series)),
    (
        "avg_up_period",
        lambda: ss.compute(inp, "avg_up_period").value,
        lambda: qs.stats.avg_win(series),
    ),
    (
        "avg_down_period",
        lambda: ss.compute(inp, "avg_down_period").value,
        lambda: qs.stats.avg_loss(series),
    ),
    (
        "annualized_volatility",
        lambda: ss.compute(inp, "annualized_volatility").value,
        lambda: qs.stats.volatility(series),
    ),
    ("skewness", lambda: ss.compute(inp, "skewness").value, lambda: qs.stats.skew(series)),
    (
        "excess_kurtosis",
        lambda: ss.compute(inp, "excess_kurtosis").value,
        lambda: qs.stats.kurtosis(series),
    ),
    ("cagr", lambda: ss.compute(inp, "cagr").value, lambda: qs.stats.cagr(series)),
    (
        "sharpe_ratio",
        lambda: ss.compute(inp, "sharpe_ratio", ddof=1).value,
        lambda: qs.stats.sharpe(series),
    ),
    (
        "sortino_ratio",
        lambda: ss.compute(inp, "sortino_ratio").value,
        lambda: qs.stats.sortino(series),
    ),
    ("omega_ratio", lambda: ss.compute(inp, "omega_ratio").value, lambda: qs.stats.omega(series)),
    (
        "calmar_ratio",
        lambda: ss.compute(inp, "calmar_ratio").value,
        lambda: qs.stats.calmar(series),
    ),
    (
        "max_drawdown",
        lambda: ss.compute(inp, "max_drawdown").value,
        lambda: qs.stats.max_drawdown(series),
    ),
    (
        "period_payoff_ratio",
        lambda: ss.compute(inp, "period_payoff_ratio").value,
        lambda: qs.stats.payoff_ratio(series),
    ),
    (
        "period_profit_factor",
        lambda: ss.compute(inp, "period_profit_factor").value,
        lambda: qs.stats.profit_factor(series),
    ),
    (
        "period_kelly_criterion",
        lambda: ss.compute(inp, "period_kelly_criterion").value,
        lambda: qs.stats.kelly_criterion(series),
    ),
    (
        "smart_sharpe",
        lambda: ss.compute(inp, "smart_sharpe").value,
        lambda: qs.stats.smart_sharpe(series),
    ),
    (
        "smart_sortino",
        lambda: ss.compute(inp, "smart_sortino").value,
        lambda: qs.stats.smart_sortino(series),
    ),
    (
        "adjusted_sortino_ratio",
        lambda: ss.compute(inp, "adjusted_sortino_ratio").value,
        lambda: qs.stats.adjusted_sortino(series),
    ),
    (
        "autocorr_penalty",
        lambda: ss.compute(inp, "autocorr_penalty").value,
        lambda: qs.stats.autocorr_penalty(series),
    ),
    (
        "psr  (se_formula=lo)",
        lambda: ss.compute(inp, "psr", se_formula="lo").value,
        lambda: qs.stats.probabilistic_sharpe_ratio(series),
    ),
    (
        "probabilistic_sortino_ratio  (lo)",
        lambda: ss.compute(inp, "probabilistic_sortino_ratio", se_formula="lo").value,
        lambda: qs.stats.probabilistic_sortino_ratio(series),
    ),
    (
        "probabilistic_adjusted_sortino_ratio  (lo)",
        lambda: ss.compute(inp, "probabilistic_adjusted_sortino_ratio", se_formula="lo").value,
        lambda: qs.stats.probabilistic_adjusted_sortino_ratio(series),
    ),
]

print("1. EXACT MATCH (same formula, same value)")
print("-" * 96)
header = f"{'metric':<42s} {'StratStat':>14s} {'QuantStats':>14s}  status"
print(header)
print("-" * 96)
n_exact = 0
exact_fail = []
for label, ssfn, qsfn in exact:
    ss_val = float(ssfn())
    qs_val = float(qsfn())
    ok = _isclose(ss_val, qs_val)
    print(f"{label:<42s} {_fmt(ss_val):>14s} {_fmt(qs_val):>14s}  {'MATCH' if ok else 'MISMATCH'}")
    if ok:
        n_exact += 1
    else:
        exact_fail.append(label)
print(f"  -> {n_exact}/{len(exact)} match exactly")
if exact_fail:
    print(f"     MISMATCHES: {', '.join(exact_fail)}")

print()
print("   Note: ``psr`` and the two probabilistic Sortino ratios need the Lo (2002)")
print("   standard-error formula (``se_formula='lo'``) to reproduce QuantStats, which")
print("   uses that variant.  StratStat's default is Bailey & Lopez de Prado's PSR SE.")
print("   ``max_drawdown``/``calmar_ratio`` match here because the first return is")
print("   positive; QuantStats seeds a phantom baseline that only matters when the")
print("   first return is negative (see the FORMULA section).")
print()

# ---------------------------------------------------------------------------
# 2. CONVENTION — parity restored via a formula-selection option
# ---------------------------------------------------------------------------
convention = [
    (
        "exposure_time",
        "rounding",
        "raw",
        "percent_ceil",
        lambda r: ss.compute(inp, "exposure_time", rounding=r).value,
        lambda: qs.stats.exposure(series),
    ),
    (
        "rar",
        "rounding",
        "raw",
        "percent_ceil",
        lambda r: ss.compute(inp, "rar", rounding=r).value,
        lambda: qs.stats.rar(series),
    ),
]

print("2. CONVENTION (option restores exact parity)")
print("-" * 96)
for label, param, default, chosen, ssfn, qsfn in convention:
    qs_val = float(qsfn())
    default_val = float(ssfn(default))
    chosen_val = float(ssfn(chosen))
    print(f"{label}  ({param}={chosen})")
    print(f"  StratStat default ({param}={default}): {_fmt(default_val)}")
    print(f"  StratStat {param}={chosen:<11s}              {_fmt(chosen_val)}")
    print(f"  QuantStats (rounded):            {_fmt(qs_val)}")
    print(
        f"  -> {'MATCH' if _isclose(chosen_val, qs_val) else 'MISMATCH'} with "
        f"{param}={chosen}; QuantStats rounds exposure up to the nearest 1%."
    )
print()

# ---------------------------------------------------------------------------
# 3. SIGN CONVENTION — same magnitude, opposite sign
# ---------------------------------------------------------------------------
print("3. SIGN CONVENTION (magnitude matches, sign differs)")
print("-" * 96)
ss_var = float(ss.compute(inp, "var", method="parametric").value)
qs_var = float(qs.stats.var(series))
print(f"  var (parametric):  StratStat={_fmt(ss_var)}  QuantStats={_fmt(qs_var)}")
print(
    f"  -> {'MATCH (magnitude)' if _isclose(abs(ss_var), abs(qs_var)) else 'MISMATCH'}: "
    "StratStat reports the loss magnitude (+); QuantStats reports the signed loss (-)."
)
print()

# ---------------------------------------------------------------------------
# 4. FORMULA — genuinely different definitions (documented, not failures)
# ---------------------------------------------------------------------------
formula = [
    (
        "cvar",
        lambda: ss.compute(inp, "cvar", method="parametric").value,
        lambda: qs.stats.cvar(series),
        "SS closed-form normal ES vs QS empirical mean below parametric VaR",
    ),
    (
        "tail_ratio",
        lambda: ss.compute(inp, "tail_ratio").value,
        lambda: qs.stats.tail_ratio(series),
        "SS tail-mean ratio vs QS quantile ratio |q95/q05|",
    ),
    (
        "common_sense_ratio",
        lambda: ss.compute(inp, "common_sense_ratio").value,
        lambda: qs.stats.common_sense_ratio(series),
        "inherits the tail-ratio difference (both scale by profit factor)",
    ),
    (
        "risk_of_ruin",
        lambda: ss.compute(inp, "risk_of_ruin").value,
        lambda: qs.stats.risk_of_ruin(series),
        "SS normal approximation vs QS gambler's-ruin win-rate formula",
    ),
    (
        "gain_to_pain_ratio",
        lambda: ss.compute(inp, "gain_to_pain_ratio").value,
        lambda: qs.stats.gain_to_pain_ratio(series),
        "SS sum(gains)/|sum(losses)| vs QS total-return/|sum(losses)| (= profit_factor - 1)",
    ),
    (
        "ulcer_index",
        lambda: ss.compute(inp, "ulcer_index").value,
        lambda: qs.stats.ulcer_index(series),
        "SS sqrt(mean(dd^2)) divides by n; QS divides by n-1",
    ),
    (
        "upi",
        lambda: ss.compute(inp, "upi").value,
        lambda: qs.stats.upi(series),
        "SS annualized arithmetic mean / UI vs QS total compounded return / UI",
    ),
    (
        "serenity_ratio",
        lambda: ss.compute(inp, "serenity_ratio").value,
        lambda: qs.stats.serenity_index(series),
        "SS annualized excess / (sigma * UI) vs QS (sum(r) - rf) / (UI * pitfall)",
    ),
    (
        "recovery_factor",
        lambda: ss.compute(inp, "recovery_factor").value,
        lambda: qs.stats.recovery_factor(series),
        "SS total compounded return / |MDD| (signed) vs QS |sum(r)| / |MDD|",
    ),
]

print("4. FORMULA DIFFERENCES (both shown for documentation)")
print("-" * 96)
header = f"{'metric':<22s} {'StratStat':>14s} {'QuantStats':>14s}  reason"
print(header)
print("-" * 96)
for label, ssfn, qsfn, reason in formula:
    ss_val = float(ssfn())
    qs_val = float(qsfn())
    print(f"{label:<22s} {_fmt(ss_val):>14s} {_fmt(qs_val):>14s}  {reason}")
print()

# ---------------------------------------------------------------------------
# 5. BENCHMARK — metrics needing a benchmark series
# ---------------------------------------------------------------------------
bench_rows = [
    (
        "beta",
        lambda: ss.compute(binp, "beta").value,
        lambda: qs.stats.greeks(series, bench_series)["beta"],
        True,
        "",
    ),
    (
        "r_squared",
        lambda: ss.compute(binp, "r_squared").value,
        lambda: qs.stats.r_squared(series, bench_series),
        True,
        "",
    ),
    (
        "alpha",
        lambda: ss.compute(binp, "alpha").value,
        lambda: qs.stats.greeks(series, bench_series)["alpha"],
        False,
        "SS Jensen (CAGR-based) vs QS greeks (arithmetic mean * periods)",
    ),
    (
        "information_ratio",
        lambda: ss.compute(binp, "information_ratio").value,
        lambda: qs.stats.information_ratio(series, bench_series),
        False,
        "SS annualizes (x sqrt(P)); QS returns the raw period ratio",
    ),
    (
        "treynor_ratio",
        lambda: ss.compute(binp, "treynor_ratio").value,
        lambda: qs.stats.treynor_ratio(series, bench_series),
        False,
        "SS annualized mean excess / beta vs QS total comp / beta",
    ),
]

print("5. BENCHMARK TIER")
print("-" * 96)
header = f"{'metric':<22s} {'StratStat':>14s} {'QuantStats':>14s}  status"
print(header)
print("-" * 96)
for label, ssfn, qsfn, expect_match, reason in bench_rows:
    ss_val = float(ssfn())
    qs_val = float(qsfn())
    if expect_match:
        status = "MATCH" if _isclose(ss_val, qs_val) else "MISMATCH"
    else:
        status = "DIFFER (documented)"
    print(f"{label:<22s} {_fmt(ss_val):>14s} {_fmt(qs_val):>14s}  {status}")
    if reason:
        print(f"{'':<22s} {'':>14s} {'':>14s}  {reason}")
print()

# ---------------------------------------------------------------------------
# 6. QS-ONLY — no returns-tier StratStat equivalent (trades-tier instead)
# ---------------------------------------------------------------------------
qs_only = [
    (
        "avg_return",
        lambda: qs.stats.avg_return(series),
        "mean of NON-ZERO returns; SS arithmetic_mean_return uses every period",
    ),
    (
        "win_rate",
        lambda: qs.stats.win_rate(series),
        "positives / non-zero; SS positive_period_ratio uses positives / all periods",
    ),
    (
        "profit_ratio",
        lambda: qs.stats.profit_ratio(series),
        "no SS returns-tier equivalent (SS has trades-tier win_loss/payoff)",
    ),
    (
        "cpc_index",
        lambda: qs.stats.cpc_index(series),
        "no SS returns-tier equivalent (SS has trades-tier cpc_ratio)",
    ),
    (
        "outlier_win_ratio",
        lambda: qs.stats.outlier_win_ratio(series),
        "no SS returns-tier equivalent (SS has trades-tier outlier_win_ratio)",
    ),
    (
        "outlier_loss_ratio",
        lambda: qs.stats.outlier_loss_ratio(series),
        "no SS returns-tier equivalent (SS has trades-tier outlier_loss_ratio)",
    ),
]

print("6. QUANTSTATS-ONLY (informational — StratStat covers these at the trades tier)")
print("-" * 96)
for label, qsfn, note in qs_only:
    print(f"  {label:<22s} QuantStats = {_fmt(float(qsfn())):<12s} {note}")
print()

print("=" * 96)
print(
    f"Summary: {n_exact} exact matches, 2 convention matches, 1 sign-convention, "
    f"{len(formula)} formula differences, "
    f"{sum(1 for r in bench_rows if r[3])} benchmark matches, "
    f"{sum(1 for r in bench_rows if not r[3])} benchmark differences, "
    f"{len(qs_only)} QS-only."
)
print("Monte Carlo (bootstrap vs shuffle) and the timing benchmark are covered in")
print("examples/07_monte_carlo_comparison.py and examples/08_timing_benchmark.py.")
