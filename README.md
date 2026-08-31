# StratStat

[![CI](https://github.com/aineurog/stratstat/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aineurog/stratstat/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Strategy evaluation statistics for Python. One library that replaces
`empyrical`, `pyfolio`, and `QuantStats` with more metrics, a cleaner API, and
faster computation.

StratStat computes 176 statistics across five input tiers. Give it a return
series, a trade log, positions, a benchmark, or several strategies, and it
returns typed results with the formula and its citation attached to every
number.

## Contents

- [Why StratStat](#why-stratstat)
- [Coming from empyrical, pyfolio, or QuantStats?](#coming-from-empyrical-pyfolio-or-quantstats)
- [Install](#install)
- [Getting started](#getting-started)
- [Metrics](#metrics)
- [Input containers](#input-containers)
- [Strategy and Comparison containers](#strategy-and-comparison-containers)
- [Resampling and Monte Carlo](#resampling-and-monte-carlo)
- [Reports](#reports)
- [Reference](#reference)
- [Architecture](#architecture)
- [Development](#development)
- [License](#license)

## Why StratStat

Three libraries (`empyrical`, `pyfolio`, `QuantStats`) split strategy
evaluation across three incompatible APIs. StratStat is one library with more
metrics, one call shape, and the working shown on every number.

- **More coverage.** 176 metrics across five input tiers (returns, benchmark,
  exposure, trades, compare). QuantStats exposes roughly 75.
- **Provenance on every result.** Every `MetricResult` carries its name,
  category, annualization factor, and the citation for its formula, so you can
  reproduce a number and defend it.
- **No engine lock in.** Pass numpy arrays, pandas objects, polars objects, or
  plain dicts. The core runs on numpy only, no scipy, so it works on a headless
  server.
- **Built to extend.** Register a metric with one decorator and it works in
  `compute`, `compute_all`, rolling windows, tear sheets, and reports.

## Coming from empyrical, pyfolio, or QuantStats?

| You use | The StratStat equivalent |
|---|---|
| `empyrical.sharpe_ratio(returns)` | `stratstat.compute(returns, "sharpe_ratio", periods_per_year=252)` |
| `pyfolio.timeseries.perf_stats(returns)` | `stratstat.compute_returns(returns, periods_per_year=252)` |
| `quantstats.reports.metrics(mode="full")` | `stratstat.compute_all(returns, periods_per_year=252)` |

The quickest tour is `python examples/00_quickstart.py`. For a metric by metric
comparison against QuantStats see `examples/06_quantstats_comparison.py`, and
for wall clock timing see `examples/08_timing_benchmark.py`.

## Install

```bash
pip install stratstat                # numpy and pandas
pip install stratstat[polars]        # polars Series and DataFrame input
pip install stratstat[fast]          # numba acceleration for bootstraps
pip install stratstat[report]        # plotly charts, tear sheets, dashboards
pip install stratstat[image]         # static image export (PNG, SVG)
pip install stratstat[pdf]           # PDF report export
pip install stratstat[all]           # everything
```

Python 3.10 or later.

## Getting started

Every tier has one entry point. The snippets below are self contained and
runnable as they are. They use synthetic data, so nothing has to be
downloaded. Start with returns, then add the other tiers as your data allows.

### Returns

```python
import numpy as np
import stratstat as ss

rng = np.random.default_rng(42)
returns = rng.normal(0.0004, 0.01, size=252)  # 252 daily returns

# One metric
result = ss.compute(returns, "sharpe_ratio", periods_per_year=252)
print(result)
# MetricResult(name='sharpe_ratio', value=-0.14, category=('risk_adjusted', 'returns'))

# Every metric in one category
risk = ss.compute_all(returns, category="risk", periods_per_year=252)

# Every returns tier metric
all_returns = ss.compute_returns(returns, periods_per_year=252)

# Any metric over a rolling window, or per market regime
ss.rolling(returns, "sharpe_ratio", window=60, periods_per_year=252)
ss.by_regime(returns, "cagr", np.where(returns > 0, "up", "down"), periods_per_year=252)

# Discover what is available
for m in ss.list_metrics():
    print(m["name"], m["category"], m["requires"])
```

### Benchmark

```python
bench = rng.normal(0.0003, 0.008, size=252)
bm = ss.compute_benchmark(returns, bench, periods_per_year=252)
print(bm["alpha"], bm["beta"], bm["information_ratio"])
```

### Trades

```python
trades = {
    "pnl": [0.02, -0.01, 0.03, 0.015, -0.02, 0.01],
    "side": ["long", "short", "long", "long", "short", "long"],
    "duration": [5, 3, 8, 4, 2, 6],
}
trade_stats = ss.compute_trades(trades, periods_per_year=252)
print(trade_stats["win_rate"], trade_stats["profit_factor"])
```

`pnl` is the only required column. `side` unlocks the long and short
breakdowns; `duration` unlocks the holding period metrics. See the
[data contract](docs/usage-guide.md#data-contract) for the rest.

### Exposure

```python
positions = rng.normal(0.05, 0.30, size=(252, 4))  # 252 periods, 4 assets
exposure = ss.compute_exposure(positions, periods_per_year=252)
print(exposure["long_exposure_pct"], exposure["short_exposure_pct"])
print(exposure["effective_n_positions"], exposure["turnover"])
```

Metrics that need inputs beyond positions (leverage, long and short beta,
active share) are skipped here and named in `exposure.meta["skipped"]`. Pass
`asset_returns`, `benchmark`, and `benchmark_weights` to unlock them.

### Compare

```python
a = rng.normal(0.0004, 0.01, size=252)
b = rng.normal(0.0003, 0.012, size=252)
cmp = ss.compute_compare(np.column_stack([a, b]), periods_per_year=252)
print(cmp["correlation_matrix"], cmp["diversification_ratio"])
```

Compare metrics are tagged with the primary category `relative`, so filter
them with `category="relative"` rather than `"compare"`.

### Everything at once

```python
everything = ss.compute_all(
    returns,
    trades=trades,
    benchmark=bench,
    exposure=positions,
    periods_per_year=252,
)
```

Each tier runs only when its data is present. `everything.meta` records what
was skipped, excluded, and deduplicated, so nothing is dropped silently.

## Metrics

176 metrics across eight categories. Each returns a `MetricResult` with a
`meta["ref"]` citation. The full list with formulas is in the
[formula reference](docs/formula-reference.md).

| Category | Count | What it measures | Example metrics |
|---|---|---|---|
| Descriptive | 27 | Distribution and summary of returns | `cagr`, `annualized_volatility`, `skewness` |
| Risk | 24 | Downside, tail, and drawdown risk | `max_drawdown`, `var`, `cvar` |
| Risk adjusted | 23 | Return per unit of risk | `sharpe_ratio`, `sortino_ratio`, `calmar_ratio` |
| Inference | 14 | Confidence and significance of performance | `psr`, `sharpe_ci_analytic`, `bias_ratio` |
| Benchmark | 20 | Performance relative to a benchmark | `alpha`, `beta`, `information_ratio` |
| Exposure | 24 | Position and book structure | `gross_exposure`, `net_exposure`, `turnover` |
| Trades | 37 | Round trip and trade log statistics | `win_rate`, `profit_factor`, `expectancy` |
| Compare | 7 | Several strategies side by side | `correlation_matrix`, `diversification_ratio`, `pbo` |

`rolling(metric_name, window)` slides any metric over time, and
`by_regime(metric_name, labels)` computes any metric per regime. Filter the
full list with `ss.list_metrics()`, or compute one category with
`ss.compute_all(returns, category="risk")`.

## Input containers

Each tier reads a specific input shape. StratStat normalizes numpy arrays,
pandas objects, polars objects, and plain dicts into typed containers, so you
can pass data directly or wrap it yourself.

| Container | Tier | What it holds |
|---|---|---|
| `ReturnsInput` | returns | One or several return series |
| `BenchmarkInput` | benchmark | Strategy returns plus a benchmark series |
| `ExposureInput` | exposure | Position weights, optional asset returns and benchmark weights |
| `TradeInput` | trades | A trade log with `pnl` and optional columns |
| `CompareInput` | compare | Several strategy return columns |

Both forms work:

```python
ss.compute(returns_array, "sharpe_ratio", periods_per_year=252)
ss.compute(ss.ReturnsInput(returns_array, periods_per_year=252), "sharpe_ratio")
```

Every tier also has its own entry point:

```python
ss.compute_returns(returns, periods_per_year=252)
ss.compute_benchmark(returns, bench, periods_per_year=252)
ss.compute_exposure(positions, asset_returns=asset_returns, periods_per_year=252)
ss.compute_trades(trades, periods_per_year=252)
ss.compute_compare(np.column_stack([a, b]), periods_per_year=252)
```

## Strategy and Comparison containers

A `Strategy` holds one strategy's inputs once and caches the derived quantities
several metrics share, the equity curve, the running maximum, the drawdown
series, and the drawdown episodes, so they are computed once, not per metric.

```python
s = ss.Strategy(
    returns=returns,
    trades=trades,
    benchmark=bench,
    periods_per_year=252,
)
s.compute_all()          # every metric the inputs support
s.compute("risk")        # one category
s.report("strategy.html")
```

When `returns` and `trades` are both present, construction also reconciles the
trade log against the equity curve and warns if the two do not agree. The
figures are kept on `s.reconciliation`.

A `Comparison` stacks several period aligned return series into one matrix and
runs the vectorized engine once, then slices per strategy.

```python
c = ss.Comparison({"a": a, "b": b}, periods_per_year=252)
c.compute_all()
c["a"].compute("risk")
c.labels
```

## Resampling and Monte Carlo

The inference tier ships resampling metrics with a numba fast path and a numpy
fallback that agree within floating point tolerance.

```python
ci = ss.compute(
    returns,
    "block_bootstrap_ci",
    target_metric="sharpe_ratio",
    n_reps=5000,
    random_seed=42,
)
print(ci.value)  # [lower, upper]

dist = ss.compute(returns, "monte_carlo_distribution", target="sharpe", sims=1000)
print(dist.value)  # [min, p05, median, mean, p95, max, std]

probs = ss.compute(returns, "monte_carlo_probabilities", bust=-0.20, goal=0.20)
print(probs.value)  # [p_bust, p_goal]
```

`compute_all` runs the analytic metrics and records the resampling metrics it
leaves out in `meta["excluded_resampling"]`, so you can run them explicitly with
`compute` when you want them. Nothing is dropped silently.

## Reports

The reporting layer is an optional extra that depends on core, never the other
way around. Install it with `pip install stratstat[report]`.

```python
from stratstat.report import tear_sheet, dashboard, generate_report

# A single strategy tear sheet
tear_sheet(returns, benchmark=bench, periods_per_year=252).show()

# A multi strategy dashboard
dashboard(np.column_stack([a, b]), periods_per_year=252).show()

# A standalone HTML report on disk
generate_report(
    returns,
    "report.html",
    benchmark=bench,
    positions=positions,
    trades=trades,
    periods_per_year=252,
    title="My Strategy",
)
```

`generate_report` writes one file with tabs for overview, performance,
exposure, trades, and benchmark. Each tab pulls its numbers from the registry,
so the methodology section lists the citations for exactly the metrics that were
computed. Export helpers cover HTML, image, markdown, LaTeX, and JSON.

## Reference

The details that rarely change are documented separately so the README stays a
quick start.

- [Formula reference](docs/formula-reference.md) — every metric's formula and
  citation.
- [Data contract](docs/usage-guide.md#data-contract) — required and optional
  trade log columns, units, and excursion precedence.
- [Column mapping](docs/usage-guide.md#column-mapping) — mapping your column
  names to the contract with a `Schema`.
- [Conventions](docs/usage-guide.md#conventions) — the parameters that choose
  between legitimate formula variants.
- [Custom metrics](docs/usage-guide.md#custom-metrics) — registering your own
  metric.

The full documentation builds from `docs/` with Sphinx.

## Architecture

```
src/stratstat/
  core/                     # pure computation, numpy only
    _utils.py               #   shared helpers
    returns/
      descriptive.py        #   27 metrics
      risk.py               #   24 metrics
      risk_adjusted.py      #   23 metrics
      inference.py          #   14 metrics
      wrappers.py           #   rolling, by_regime
    benchmark.py            #   20 metrics
    exposure.py             #   24 metrics
    trades.py               #   37 metrics
    compare.py              #    7 metrics
  report/                   # plotly, optional extra
  inputs.py                 #   input containers
  results.py                #   MetricResult, MetricSet
  registry.py               #   register_metric, compute
  schema.py                 #   Schema, column mapping
  container.py              #   Strategy, Comparison
  conventions.py            #   session defaults
```

`core` never imports `plotly`, `matplotlib`, or `stratstat.report`. The boundary
is enforced by a CI test, so you can run StratStat on a headless server with no
graphics libraries. There is no scipy dependency either: the normal distribution
functions are implemented in numpy. Every metric has a single implementation in
`core` that the tear sheet and HTML report call through the registry, so a
formula is fixed once and fixed everywhere.

## Development

```bash
git clone https://github.com/aineurog/stratstat.git
cd stratstat
pip install -e ".[dev]"

pytest tests/ -q
ruff check src/ tests/
mypy src/
```

## License

MIT. See [LICENSE](LICENSE).
