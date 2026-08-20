"""Returns-tier metrics.

This subpackage holds all metrics that require only a returns series.
Organized by axis-2 classification (statistical nature):
  - descriptive: mean return, volatility, skewness, kurtosis, etc.
  - risk: VaR, CVaR, max drawdown, drawdown duration, etc.
  - risk_adjusted: Sharpe, Sortino, Calmar, Sterling, etc.
  - inference: PSR, DSR, Lo-adjusted SE, bootstrap CIs, min track record, etc.
  - wrappers: rolling(metric_name, window), by_regime(metric_name, labels).
"""

# Re-exports from submodules will be added here as metrics are implemented.
# from stratstat.core.returns.descriptive import mean_return, volatility, ...
# from stratstat.core.returns.risk import var, cvar, max_drawdown, ...
# from stratstat.core.returns.risk_adjusted import sharpe_ratio, sortino_ratio, ...
# from stratstat.core.returns.inference import psr, dsr, bootstrap_ci, ...
