"""Inferential statistics for returns.

Metrics in this category: PSR (Probabilistic Sharpe Ratio), DSR (Deflated Sharpe Ratio),
Lo-adjusted Sharpe standard error, bootstrap confidence intervals,
minimum track record length, and related statistical inference measures.

These are StratStat's key differentiator vs empyrical/pyfolio/QuantStats.
Treat with extra rigor and extra test scrutiny.

Most metrics are tagged: category=("inference", "returns"), backend="resampling".
"""

from stratstat.registry import register_metric


# Metrics will be registered here during Phase 4.
