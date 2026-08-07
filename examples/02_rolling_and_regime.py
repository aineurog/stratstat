"""Rolling window metrics and regime-conditional breakdowns.

The wrappers 'rolling()' and 'by_regime()' work with ANY registered metric.
You do not need per-metric rolling implementations — the registry wires
them automatically.

Run: python examples/02_rolling_and_regime.py
"""

import numpy as np

import stratstat as ss
from stratstat.inputs import ReturnsInput

# Trigger metric registration (modules register on import)
import stratstat.core.benchmark  # noqa: F401
import stratstat.core.returns.descriptive  # noqa: F401
import stratstat.core.returns.inference  # noqa: F401
import stratstat.core.returns.risk  # noqa: F401
import stratstat.core.returns.risk_adjusted  # noqa: F401

# ---------------------------------------------------------------------------
# Generate 2 years of daily returns with a volatility spike in year 2
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 504  # ~2 years

# Year 1: calm, Year 2: volatile
returns = np.concatenate([
    rng.normal(0.0006, 0.008, size=252),
    rng.normal(0.0002, 0.022, size=252),
])

print("=" * 60)
print("ROLLING METRICS")
print("=" * 60)

inp = ReturnsInput(returns, periods_per_year=252)

# -- Rolling Sharpe (60-day window) -----------------------------------------
rolling_sharpe = ss.rolling(inp, "sharpe_ratio", window=60)

print(f"\nRolling Sharpe (60-day window):")
print(f"  type:   {type(rolling_sharpe).__name__}")
print(f"  shape:  {rolling_sharpe.value.shape}")
print(f"  mean:   {np.nanmean(rolling_sharpe.value):.3f}")
print(f"  min:    {np.nanmin(rolling_sharpe.value):.3f}")
print(f"  max:    {np.nanmax(rolling_sharpe.value):.3f}")
print(f"  final:  {rolling_sharpe.value[-1]:.3f}")

# The value is an array — first <window> entries are NaN
print(f"  first valid at index {60}: {rolling_sharpe.value[60]:.3f}")

# -- Rolling volatility -----------------------------------------------------
rolling_vol = ss.rolling(inp, "annualized_volatility", window=60)

print(f"\nRolling Volatility (60-day window):")
print(f"  mean:   {np.nanmean(rolling_vol.value):.3f}")
print(f"  min:    {np.nanmin(rolling_vol.value):.3f}")
print(f"  max:    {np.nanmax(rolling_vol.value):.3f}")

# -- Rolling max drawdown ---------------------------------------------------
rolling_dd = ss.rolling(inp, "max_drawdown", window=60)

print(f"\nRolling Max Drawdown (60-day window):")
print(f"  mean:   {np.nanmean(rolling_dd.value):.3f}")
print(f"  worst:  {np.nanmin(rolling_dd.value):.3f}")

# -- Rolling Sortino --------------------------------------------------------
rolling_sortino = ss.rolling(inp, "sortino_ratio", window=60)

print(f"\nRolling Sortino (60-day window):")
print(f"  mean:   {np.nanmean(rolling_sortino.value):.3f}")

# ---------------------------------------------------------------------------
# By regime: compare bull vs bear months
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 60)
print("BY REGIME")
print("=" * 60)

# Label each period based on its own sign (simple regime split)
labels = np.where(returns > 0, "up_months", "down_months")
n_up = np.sum(labels == "up_months")
n_down = np.sum(labels == "down_months")
print(f"\nRegime split: {n_up} up, {n_down} down")

# Compute CAGR separately for each regime
regime_cagr = ss.by_regime(inp, "cagr", labels)
print(f"\nCAGR by regime:")
print(regime_cagr)

# Compute Sharpe separately for each regime
regime_sharpe = ss.by_regime(inp, "sharpe_ratio", labels)
print(f"\nSharpe by regime:")
print(regime_sharpe)

# Compute volatility separately for each regime
regime_vol = ss.by_regime(inp, "annualized_volatility", labels)
print(f"\nVolatility by regime:")
print(regime_vol)

# -- Any metric works with by_regime ----------------------------------------
regime_sortino = ss.by_regime(inp, "sortino_ratio", labels)
print(f"\nSortino by regime:")
print(regime_sortino)

print("\n\nAny registered metric can be passed to rolling() or by_regime().")
print("No per-metric rolling logic is needed — the registry wires it.")
