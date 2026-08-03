# StratStat

**Quantitative strategy evaluation statistics** — a comprehensive, high-performance
Python library for evaluating trading strategies.

StratStat is the modern successor to abandoned/stale libraries like `empyrical` and
`pyfolio`, and a more complete, better-engineered alternative to `QuantStats`.

---

## Features

- **138 metrics** across 8 statistical categories — descriptive, risk, risk-adjusted,
  inference, exposure, trades, benchmark-relative, and multi-strategy comparison.
- **Single source of truth** — every formula implemented exactly once, shared across
  the core API, reporting layer, and comparison layer. No copy-paste.
- **Vectorization first** — batch computation across multiple strategies/columns.
  No Python loops where NumPy can do it in one call.
- **Flexible input** — accepts pandas DataFrames/Series, polars DataFrames/Series,
  and numpy arrays. All normalised internally.
- **Every formula cites its source** — paper, book, or well-established reference
  in every docstring and `MetricResult.meta["ref"]`.
- **Standardised output** — every call returns a `MetricResult` (single metric) or
  `MetricSet` (collection) with rich metadata, not a bare float or dict.
- **Registry-based extensibility** — `@register_metric` decorator makes custom
  metrics work identically to built-ins. No central dispatch to edit.
- **Optional numba acceleration** — `pip install stratstat[fast]` for
  resampling-heavy computations (bootstrap CIs, Reality Check, PBO).
- **Optional Plotly reporting** — `pip install stratstat[report]` for tear sheets,
  dashboards, and chart exports.
- **Generic wrappers** — `rolling(metric_name, window)` and `by_regime(metric_name,
  labels)` apply any registered metric over time windows or market regimes.

---

## Installation

```bash
pip install stratstat            # core (numpy, pandas, polars)
pip install stratstat[fast]      # + numba acceleration
pip install stratstat[report]    # + plotly charts, tear sheets, dashboards
pip install stratstat[all]       # everything
pip install stratstat[dev]       # + pytest, ruff, mypy
```

Python 3.10+ required.

---

## Quick start

```python
import numpy as np
import stratstat as ss

returns = np.random.normal(0.001, 0.02, size=252)

# Single metric
result = ss.compute(returns, "sharpe_ratio", periods_per_year=252)
print(result)
# MetricResult(name='sharpe_ratio', value=0.84, category=('risk_adjusted', 'returns'), ...)

# All metrics in a category
risk_stats = ss.compute_all(returns, category="risk", periods_per_year=252)
print(risk_stats.to_frame())

# Rolling metric (time series)
rolling_sharpe = ss.rolling(returns, "sharpe_ratio", window=60, periods_per_year=252)

# By market regime
labels = np.where(returns > 0, "up", "down")
regime_cagr = ss.by_regime(returns, "cagr", labels, periods_per_year=252)

# Block-bootstrap confidence interval for any metric
ci = ss.compute(returns, "block_bootstrap_ci",
                target_metric="sharpe_ratio", n_reps=5000, random_seed=42)

# Discover available metrics
for m in ss.list_metrics():
    print(m["name"], m["category"], m["requires"])

# ---- Reporting (needs stratstat[report]) ----
from stratstat.report import tear_sheet, dashboard

tear_sheet(returns, periods_per_year=252).show()
dashboard(multi_strat_returns, periods_per_year=252).show()
```

---

## Metric inventory

| Category | Count | Tier | Examples |
|---|---|---|---|
| Descriptive | 16 | returns | CAGR, volatility, skewness, kurtosis, autocorrelation |
| Risk | 20 | returns | VaR, CVaR, max drawdown, drawdown duration, ulcer index, tail ratio |
| Risk-adjusted | 9 | returns | Sharpe, Sortino, Calmar, Omega, Kappa-3, Martin ratio |
| Inference | 8 | returns | PSR, DSR, Lo's SE, Sharpe CI (analytic & bootstrap), block-bootstrap CI |
| Exposure | 23 | exposure | gross/net exposure, HHI, turnover, leverage, beta, position coverage |
| Trades | 37 | trades | win rate, profit factor, expectancy, streaks, MFE/MAE, holding period |
| Benchmark | 18 | benchmark | tracking error, information ratio, alpha, beta, M², up/down capture |
| Compare | 7 | compare | Sharpe-difference test, MCR, diversification ratio, PBO, Reality Check |
| **Wrappers** | — | returns | `rolling()` and `by_regime()` — apply any metric over windows or regimes |

---

## Architecture

```
stratstat/
  core/                    # Pure computation — zero heavy dependencies
    _utils.py              #   NaN-aware vectorised helpers
    returns/
      descriptive.py       #   16 metrics — mean, vol, skew, kurtosis, …
      risk.py              #   20 metrics — VaR, CVaR, drawdowns, EVT, …
      risk_adjusted.py     #    9 metrics — Sharpe, Sortino, Calmar, …
      inference.py         #    8 metrics — PSR, DSR, bootstrap CIs, …
      wrappers.py          #    rolling(), by_regime()
    exposure.py            #   23 metrics — gross/net, HHI, turnover, …
    trades.py              #   37 metrics — win rate, profit factor, …
    benchmark.py           #   18 metrics — tracking error, IR, alpha, …
    compare.py             #    7 metrics — Sharpe test, MCR, PBO, …
  report/                  # Plotly visualisation (optional extra)
    _charts.py             #    7 chart functions
    _tearsheet.py          #    Single-strategy tear sheet
    _dashboard.py          #    Multi-strategy dashboard
    _export.py             #    HTML / PNG / SVG / PDF / Markdown / LaTeX
  inputs.py                #   ReturnsInput, TradeInput, ExposureInput, …
  registry.py              #   @register_metric, compute(), compute_all()
  results.py               #   MetricResult, MetricSet
```

**Core ↔ Report boundary**: Core must never import `plotly`, `matplotlib`, or
`stratstat.report`. This is enforced by CI tests.

---

## Registering a custom metric

```python
from stratstat import register_metric
from stratstat.results import MetricResult
from stratstat.inputs import ReturnsInput

@register_metric(
    name="my_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref="My Paper (2024)",
)
def my_ratio(input_data: ReturnsInput) -> MetricResult:
    r = input_data.values[:, 0]
    value = float(np.mean(r) / np.std(r))
    return MetricResult(
        name="my_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": "My Paper (2024)"},
    )

# Now works everywhere — compute(), compute_all(), rolling(), tear_sheet(), …
result = ss.compute(returns, "my_ratio")
```

---

## Development

```bash
git clone https://github.com/aineurog/stratstat.git
cd stratstat
pip install -e ".[dev]"

pytest tests/ -q          # 742 tests
ruff check src/ tests/    # lint
mypy src/stratstat/       # type-check
```

---

## License

MIT — see [LICENSE](LICENSE).
