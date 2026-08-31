# StratStat

Strategy evaluation statistics for Python. One library that replaces
`empyrical`, `pyfolio`, and `QuantStats` with more metrics, a cleaner API, and
faster computation.

StratStat computes 176 statistics across five input tiers. Give it a return
series, a trade log, positions, a benchmark, or several strategies, and it
returns typed results with the formula and its citation attached to every
number.

## Why this library exists

The Python tooling for strategy evaluation is fragmented. `empyrical` and
`pyfolio` are no longer maintained. `QuantStats` is hard to extend and mixes
computation with reporting. Each one duplicates formula code across its
codebase.

StratStat is built on three rules.

**One implementation per formula.** The core metric, the tear sheet, and the
HTML report all call the same registered function. Fix a formula once and it is
fixed everywhere.

**A registry, not a switch statement.** A metric is a decorated function.
Register it once and it works in `compute`, `compute_all`, `rolling`,
`by_regime`, tear sheets, dashboards, and reports. There is no central dispatch
to edit.

**Provenance on every result.** Each `MetricResult` carries its name, category,
annualization factor, and the citation for its formula. You always know where a
number came from.

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
# MetricResult(name='sharpe_ratio', value=0.84, category=('risk_adjusted', 'returns'))

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
breakdowns; `duration` unlocks the holding period metrics. See the data
contract below for the rest.

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
`meta["ref"]` citation. The full formulas live in the
[formula reference](docs/formula-reference.md).

### Descriptive (27)

`cagr`, `annualized_volatility`, `cumulative_return`, `arithmetic_mean_return`,
`geometric_mean_return`, `skewness`, `excess_kurtosis`, `best_period`,
`worst_period`, `positive_period_ratio`, `negative_period_ratio`,
`autocorrelation`, `variance`, `return_range`, `percentiles`,
`coefficient_of_variation`, `outlier_iqr`, `stability`, `hurst_exponent`,
`fractal_dimension`, `consecutive_wins_losses`, `exposure_time`,
`avg_up_period`, `avg_down_period`, `period_profit_factor`,
`period_payoff_ratio`, `period_kelly_criterion`

### Risk (24)

`max_drawdown`, `longest_drawdown_duration`, `time_to_recovery`,
`average_drawdown`, `average_drawdown_duration`, `ulcer_index`,
`downside_deviation`, `upside_deviation`, `downside_semivariance`, `var`,
`modified_var`, `cvar`, `tail_ratio`, `common_sense_ratio`, `hill_tail_index`,
`gpd_tail_fit`, `risk_of_ruin`, `drawdown_volatility`,
`drawdown_periods_count`, `current_drawdown`, `current_drawdown_duration`,
`drawdown_total_duration`, `pain_index`, `prospect_ratio`

### Risk adjusted (23)

`sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `omega_ratio`,
`sterling_ratio`, `burke_ratio`, `kappa_3`, `martin_ratio`,
`gain_to_pain_ratio`, `pain_ratio`, `recovery_factor`, `k_ratio`,
`serenity_ratio`, `upi`, `modified_sharpe_ratio`, `upside_potential_ratio`,
`risk_return_ratio`, `roys_safety_first`, `autocorr_penalty`, `smart_sharpe`,
`smart_sortino`, `adjusted_sortino_ratio`, `rar`

### Inference (14)

`jarque_bera`, `psr`, `dsr`, `lo_sharpe_se`, `sharpe_ci_analytic`,
`sharpe_ci_bootstrap`, `min_track_record_length`, `block_bootstrap_ci`,
`bias_ratio`, `skewness_adjusted_sharpe`, `probabilistic_sortino_ratio`,
`probabilistic_adjusted_sortino_ratio`, `monte_carlo_distribution`,
`monte_carlo_probabilities`

### Benchmark (20)

`alpha`, `beta`, `r_squared`, `tracking_error`, `information_ratio`,
`up_capture`, `down_capture`, `up_down_capture`, `correlation`, `active_return`,
`batting_average`, `treynor_ratio`, `outperformance`, `outperformance_ratio`,
`underperforming_periods`, `max_outperformance`, `max_underperformance`,
`benchmark_volatility`, `information_coefficient`, `directional_consistency`

### Exposure (24)

`gross_exposure`, `net_exposure`, `leverage`, `long_exposure_pct`,
`short_exposure_pct`, `long_book_return`, `short_book_return`, `long_beta`,
`short_beta`, `position_concentration`, `effective_n_positions`, `turnover`,
`avg_holding_weight`, `position_coverage`, `long_position_coverage`,
`short_position_coverage`, `exposure_volatility`, `net_exposure_volatility`,
`exposure_cv`, `exposure_utilization`, `exposure_directional_bias`,
`exposure_percentiles`, `period_counts`, `active_share`

### Trades (37)

`total_trades`, `win_rate`, `win_rate_long`, `win_rate_short`, `avg_win`,
`avg_loss`, `win_loss_ratio`, `profit_factor`, `expectancy`,
`avg_holding_period`, `holding_period_distribution`, `max_consecutive_wins`,
`max_consecutive_losses`, `pnl_distribution`, `implementation_shortfall`,
`best_trade`, `worst_trade`, `avg_winning_duration`, `avg_losing_duration`,
`payoff_ratio`, `cpc_ratio`, `sqn`, `trade_duration_std`, `trade_return_std`,
`geometric_mean_return_per_trade`, `outlier_win_ratio`, `outlier_loss_ratio`,
`mfe`, `mae`, `kelly_criterion`, `long_short_trade_count`,
`long_short_trade_pct`, `long_short_winning_losing`, `long_short_avg_duration`,
`long_short_total_pnl`, `long_short_avg_pnl`, `long_short_best_worst`

### Compare (7)

`correlation_matrix`, `diversification_ratio`, `sharpe_difference_test`,
`whites_reality_check`, `pbo`, `marginal_contribution_to_risk`, `component_var`

### Wrappers

`rolling(metric_name, window)` slides any metric over time.
`by_regime(metric_name, labels)` computes any metric per regime.

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

## Data contract

`pnl` is the only required column in a trade log. Every other column is
optional and enables a specific set of metrics.

| Column | Meaning | Enables |
|---|---|---|
| `pnl` | Profit or loss per trade | every trade metric |
| `side` | `"long"` or `"short"`, a sign, or a bool | long and short breakdowns |
| `duration` | Holding period counted in periods | holding period metrics |
| `entry_time` | Entry timestamp | derives `duration`, bar excursions |
| `exit_time` | Exit timestamp | derives `duration`, bar excursions |
| `fill_price` | Price actually obtained | implementation shortfall |
| `decision_price` | Price when the signal fired | implementation shortfall |
| `price_path` | Full price path per trade | MFE and MAE |
| `position_size` | Fraction of account committed to the trade | basis conversion |
| `max_price` | Highest price while the trade was open | MFE |
| `min_price` | Lowest price while the trade was open | MAE |
| `mfe` | Precomputed favorable excursion, as a fraction | MFE |
| `mae` | Precomputed adverse excursion, as a fraction | MAE |

`max_price` and `min_price` are side neutral. StratStat applies the long or
short logic, so you do not invert the columns for shorts.

Two parameters state how `pnl` is measured. They are declared at call time and
recorded in `meta` on every affected result.

| Parameter | Default | Values | Meaning |
|---|---|---|---|
| `pnl_basis` | `"trade"` | `"trade"`, `"account"` | The capital base pnl is measured against |
| `pnl_unit` | `"fraction"` | `"fraction"`, `"currency"` | Whether pnl is a fraction or a currency amount |

`pnl_basis` defaults to `"trade"` because the cited literature defines its
statistics per trade. When `position_size` is absent it defaults to 1.0 and the
two bases coincide. With `position_size` present, StratStat converts between
bases so each metric uses the one it is defined on. Kelly and SQN are defined
per bet and use trade basis. Profit factor, expectancy, and reconciliation use
account basis.

`pnl_unit` defaults to `"fraction"`. The metrics that require a fraction, the
Kelly criterion and the geometric mean return per trade, refuse to run when
`pnl_unit` is `"currency"`.

### Units

Supply each input in the scale listed here. StratStat does not convert between
units.

| Input | Expected |
|---|---|
| returns | fraction, 0.01 is one percent |
| `rf` | annual, deannualized geometrically |
| trade `pnl` | fraction by default, currency when declared |
| benchmark returns | same scale and frequency as strategy returns |
| positions | weights, 0.5 is fifty percent |
| compare `weights` | fractions summing to 1 |
| `duration` | periods |
| `equity` | level series |
| MFE and MAE outputs | fractions |

`rf` is annual and is deannualized geometrically with
`(1 + rf) ** (1 / periods_per_year) - 1`, matching QuantStats. The annual value
and the derived per period value are both recorded in `meta`.

### Excursion precedence

The `mfe` and `mae` metrics derive per trade excursion from the first source
available, in this order.

| Priority | Source | Requires |
|---|---|---|
| 1 | `mfe` and `mae` columns | nothing else |
| 2 | `max_price` and `min_price` columns | nothing else |
| 3 | derived from `prices` bars | `entry_time`, `exit_time`, bars covering the span |
| 4 | `price_path` | one array per trade |

The chosen route is recorded as `meta["excursion_source"]`.

## Column mapping

StratStat expects canonical column names. Real data rarely uses them, so a
`Schema` maps your names to the contract. Three levels.

Supply canonical names directly:

```python
trades = {"pnl": [0.02, -0.01], "side": ["long", "short"], "duration": [5, 3]}
ss.compute_trades(trades)
```

Use the inline `columns=` shorthand for a one off call:

```python
trades = {"profit": [0.02, -0.01], "direction": ["long", "short"], "bars_held": [5, 3]}
ss.compute_trades(
    trades,
    columns={"pnl": "profit", "side": "direction", "duration": "bars_held"},
)
```

Or build a `Schema` once and reuse it, or set a session default:

```python
schema = ss.Schema(trades={"pnl": "profit", "side": "direction"})
ss.compute_trades(trades, schema=schema)

ss.set_schema({"trades": {"pnl": "profit", "side": "direction"}})
ss.compute_trades(trades)
ss.clear_schema()
```

The key is always the canonical name and the value is your column. Matching is
exact, never inferred or case folded. `ss.describe_columns` reports which of
your columns are recognized and which metrics you give up by leaving a column
out.

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

## Conventions

Some metrics have more than one legitimate definition. StratStat exposes the
choice as a parameter on the metric and records the convention actually used in
`meta`. You can also set a session default.

```python
ss.compute(returns, "sharpe_ratio", periods_per_year=252, ddof=0)
ss.compute(returns, "var", confidence=0.99)

ss.set_default("sharpe_ratio", "ddof=0")
ss.set_default("var", "method=parametric")
ss.get_default("sharpe_ratio")
```

Convention parameters include Sharpe ddof, Sortino denominator, VaR and CVaR
estimator and confidence, beta variant, drawdown duration units, tail cutoffs,
and the annualized volatility return type. The full list with defaults and
citations is in the [formula reference](docs/formula-reference.md).

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

## Custom metrics

Adding your own metric takes one function and one decorator. After that it
behaves like a built in everywhere.

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
def my_ratio(input_data: ReturnsInput, rf: float = 0.0) -> MetricResult:
    r = input_data.values
    excess = np.nanmean(r, axis=0) - rf
    sigma = np.nanstd(r, axis=0, ddof=1)
    value = float(excess[0] / sigma[0]) if input_data.is_single else excess / sigma
    return MetricResult(
        name="my_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": "My Paper (2024)", "rf": rf},
    )


ss.compute(returns, "my_ratio")
```

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
functions are implemented in numpy.

## Development

```bash
git clone https://github.com/aineurog/stratstat.git
cd stratstat
pip install -e ".[dev]"

pytest tests/ -q          # 1163 passed, 1 skipped
ruff check src/ tests/
mypy src/stratstat/
```

## License

MIT. See [LICENSE](LICENSE).
