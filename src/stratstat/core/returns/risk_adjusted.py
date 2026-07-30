"""Risk-adjusted return metrics.

Metrics in this category: Sharpe ratio, Sortino ratio, Calmar ratio,
information ratio, Treynor ratio, Omega ratio, and related measures.

All metrics are tagged: category=("risk_adjusted", "returns").
Multiple metrics here have competing real-world definitions (Sharpe annualization/ddof,
Sortino denominator convention) — these must expose an explicit method=/convention=
parameter with a cited default.
"""

from stratstat.registry import register_metric


# Metrics will be registered here during Phase 3.
