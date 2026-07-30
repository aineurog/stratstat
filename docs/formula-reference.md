# Formula Reference

> ⚠️ **Phase 1 deliverable.** This document must be reviewed and approved before any
> metric implementation begins. It lists every v0.1 metric with its exact formula,
> citation, and default convention (where applicable).

---

## Status

**This document is a placeholder.** The full formula reference will be written in
Phase 1, covering 30+ metrics across the following categories:

- **Descriptive** — mean return, volatility, skewness, kurtosis, etc.
- **Risk** — VaR, CVaR, max drawdown, drawdown duration, downside deviation, etc.
- **Risk-adjusted** — Sharpe, Sortino, Calmar, information ratio, Treynor, Omega, etc.
- **Inference** — PSR, DSR, Lo-adjusted SE, bootstrap CIs, minimum track record, etc.

Each entry will include:

| Metric | Formula | Citation | Default Convention | Notes |
|--------|---------|----------|-------------------|-------|
| ...    | ...     | ...      | ...               | ...   |

---

## Conventions for metrics with competing definitions

The following metrics have more than one legitimate real-world definition and will
expose an explicit `method=`/`convention=` parameter:

| Metric | Parameter | Options | Default | Citation for default |
|--------|-----------|---------|---------|---------------------|
| Sharpe ratio | `ddof` | `0`, `1` | TBD | TBD |
| Sortino ratio | `denominator` | `full_downside`, `downside_only` | TBD | TBD |
| VaR | `method` | `historical`, `parametric`, `cornish_fisher` | TBD | TBD |
| CVaR | `method` | `historical`, `parametric` | TBD | TBD |
| Beta | `variant` | `least_squares`, `robust` | TBD | TBD |
| Max drawdown | `return_type` | `simple`, `log` | TBD | TBD |
| Drawdown duration | `units` | `periods`, `years` | TBD | TBD |

*This list is provisional — the Phase 1 formula reference doc will finalize it.*
