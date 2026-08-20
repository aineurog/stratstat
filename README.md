# StratStat

Quantitative strategy evaluation for Python. A single library that replaces
`empyrical`, `pyfolio`, and `QuantStats` with a cleaner API, more metrics, and
better performance.

If you backtest trading strategies and need to answer questions like "what is
the Sharpe ratio?", "how deep was the worst drawdown?", or "is this strategy
actually better than the benchmark?", StratStat gives you the answer in one
function call.

---

## Why this library exists

The Python ecosystem for strategy evaluation is fragmented. `empyrical`
and `pyfolio` are abandoned. `QuantStats` is monolithic and hard to extend.
All three duplicate formula implementations across their codebases.

StratStat does three things differently:

Every formula lives in exactly one place. The core metric, the tear sheet, and
the HTML report all call the same underlying function. No copy paste.

Every metric is registered through a decorator. Adding a custom metric means
writing one function and decorating it. It then works everywhere: `compute()`,
`compute_all()`, `rolling()`, `by_regime()`, the tear sheet, the dashboard, and
the HTML report. No central dispatch to edit.

Every metric result carries its provenance. Name, value, category tags,
annualization factor, and the academic citation for the formula all travel
together in a `MetricResult` object. You always know where a number came from.

---

## Installation

```bash
pip install stratstat                # core: numpy, pandas
pip install stratstat[polars]        # polars Series/DataFrame input support
pip install stratstat[fast]          # numba acceleration for bootstraps
pip install stratstat[report]        # plotly charts, tear sheets, dashboards
pip install stratstat[all]           # everything
```

Python 3.10 or later.

---

## Quick start

```python
import numpy as np
import stratstat as ss

# Generate 252 days of fake daily returns
rng = np.random.default_rng(42)
returns = rng.normal(0.0004, 0.01, size=252)

# Compute a single metric
result = ss.compute(returns, "sharpe_ratio", periods_per_year=252)
print(result)
# MetricResult(name='sharpe_ratio', value=0.84, category=('risk_adjusted', 'returns'))

# Compute every metric in a category at once
all_risk = ss.compute_all(returns, category="risk", periods_per_year=252)
print(all_risk)
# ═══ Risk ═══
#   max_drawdown              -0.253
#   downside_deviation        0.009
#   var                       0.021
#   ...

# Apply any metric over a rolling window
rolling_sharpe = ss.rolling(returns, "sharpe_ratio", window=60, periods_per_year=252)

# Compute any metric separately for each market regime
labels = np.where(returns > 0, "up_market", "down_market")
regime_stats = ss.by_regime(returns, "cagr", labels, periods_per_year=252)

# Bootstrap a confidence interval for any metric
ci = ss.compute(
    returns, "block_bootstrap_ci",
    target_metric="sharpe_ratio",
    n_reps=5000,
    random_seed=42,
)

# Discover what metrics are available
for m in ss.list_metrics():
    print(m["name"], m["category"], m["requires"])
```

### With benchmark data

```python
from stratstat.inputs import BenchmarkInput

bench_returns = rng.normal(0.0003, 0.008, size=252)
bm_inp = BenchmarkInput(returns=returns, benchmark=bench_returns, periods_per_year=252)

alpha = ss.compute(bm_inp, "alpha")
all_bench = ss.compute_all(bm_inp, category="benchmark")
```

### Reporting (needs `pip install stratstat[report]`)

```python
from stratstat.report import tear_sheet, dashboard, generate_report

# Interactive 4-panel Plotly figure
tear_sheet(returns, benchmark=bench_returns, periods_per_year=252).show()

# Multi-strategy comparison dashboard
dashboard(multi_strat_returns, periods_per_year=252).show()

# Self-contained HTML report saved to disk
generate_report(
    returns,
    "my_report.html",
    benchmark=bench_returns,
    periods_per_year=252,
    title="My Strategy Analysis",
)
```

Open `my_report.html` in any browser. It contains equity curve and drawdown
charts, a monthly returns heatmap, grouped statistics tables for every
applicable metric, and a methodology section with citations.

---

## All 161 metrics

### Descriptive (21 metrics)
`cagr`, `annualized_volatility`, `cumulative_return`, `arithmetic_mean_return`,
`geometric_mean_return`, `skewness`, `excess_kurtosis`, `best_period`,
`worst_period`, `positive_period_ratio`, `negative_period_ratio`,
`autocorrelation`, `variance`, `return_range`, `percentiles`,
`coefficient_of_variation`, `outlier_iqr`, `stability`, `hurst_exponent`,
`fractal_dimension`, `consecutive_wins_losses`

### Risk (24 metrics)
`max_drawdown`, `longest_drawdown_duration`, `time_to_recovery`,
`average_drawdown`, `average_drawdown_duration`, `ulcer_index`,
`downside_deviation`, `downside_semivariance`, `upside_deviation`, `var`,
`modified_var`, `cvar`, `tail_ratio`, `common_sense_ratio`, `hill_tail_index`,
`gpd_tail_fit`, `risk_of_ruin`, `drawdown_volatility`,
`drawdown_periods_count`, `current_drawdown`, `current_drawdown_duration`,
`drawdown_total_duration`, `pain_index`, `prospect_ratio`

### Risk-Adjusted (18 metrics)
`sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `omega_ratio`,
`sterling_ratio`, `burke_ratio`, `kappa_3`, `martin_ratio`,
`gain_to_pain_ratio`, `pain_ratio`, `recovery_factor`, `k_ratio`,
`serenity_ratio`, `upi`, `modified_sharpe_ratio`, `upside_potential_ratio`,
`risk_return_ratio`, `roys_safety_first`

### Inference (10 metrics)
`jarque_bera`, `psr`, `dsr`, `lo_sharpe_se`, `sharpe_ci_analytic`,
`sharpe_ci_bootstrap`, `min_track_record_length`, `block_bootstrap_ci`,
`bias_ratio`, `skewness_adjusted_sharpe`

### Benchmark (20 metrics)
`alpha`, `beta`, `r_squared`, `tracking_error`, `information_ratio`,
`up_capture`, `down_capture`, `up_down_capture`, `correlation`, `active_return`,
`batting_average`, `treynor_ratio`, `outperformance`, `outperformance_ratio`,
`underperforming_periods`, `max_outperformance`, `max_underperformance`,
`benchmark_volatility`, `information_coefficient`, `directional_consistency`

### Exposure (24 metrics)
`gross_exposure`, `net_exposure`, `leverage`, `position_coverage`,
`exposure_cv`, `exposure_directional_bias`, `avg_holding_weight`,
`position_concentration`, `effective_n_positions`, `turnover`,
`exposure_volatility`, `net_exposure_volatility`, `long_beta`, `short_beta`,
`long_book_return`, `short_book_return`, `active_share`, and more

### Trades (37 metrics)
`total_trades`, `win_rate`, `win_rate_long`, `win_rate_short`, `avg_win`,
`avg_loss`, `win_loss_ratio`, `profit_factor`, `expectancy`, `sqn`,
`payoff_ratio`, `cpc_ratio`, `best_trade`, `worst_trade`, `mfe`, `mae`,
`implementation_shortfall`, `kelly_criterion`, and more

### Relative / Comparison (7 metrics)
`correlation_matrix`, `diversification_ratio`, `sharpe_difference_test`,
`whites_reality_check`, `pbo`, `marginal_contribution_to_risk`,
`component_var`

### Wrappers
`rolling(metric_name, window)` slides any metric over time.
`by_regime(metric_name, labels)` computes any metric per regime.

---

## Input types

StratStat uses typed input containers. Each tier of metrics expects its
own input type. The library normalizes raw data automatically, but using
the container directly gives you more control.

| Container | Used by | What it holds |
|-----------|---------|---------------|
| `ReturnsInput(returns, periods_per_year)` | returns tier | 1-D or 2-D array of period returns |
| `BenchmarkInput(returns, benchmark, periods_per_year, rf)` | benchmark tier | Strategy returns plus benchmark series |
| `ExposureInput(positions, returns, ...)` | exposure tier | Position weights, optional returns and benchmark weights |
| `TradeInput(trades)` | trades tier | Dict with pnl, side, duration, and other trade fields |
| `CompareInput(returns, weights, benchmark)` | compare tier | Multiple strategy columns for head to head comparison |

You can pass raw arrays and StratStat wraps them automatically. Both of these
work:

```python
ss.compute(returns_array, "sharpe_ratio", periods_per_year=252)
ss.compute(ReturnsInput(returns_array, periods_per_year=252), "sharpe_ratio")
```

---

## Result types

Every metric returns a `MetricResult`:

```python
@dataclass
class MetricResult:
    name: str                  # "sharpe_ratio"
    value: float | np.ndarray  # 0.84 or array([0.84, 0.92, 0.76])
    category: tuple[str, ...]  # ("risk_adjusted", "returns")
    periods_per_year: int      # 252
    meta: dict                 # {"ref": "Sharpe (1966)...", "rf": 0.0, "ddof": 1}
```

Batch calls return a `MetricSet`, which is a collection of `MetricResult`
objects with serialization methods:

```python
results = ss.compute_all(returns, category="risk")

print(results)              # sectioned terminal display grouped by category
results.to_frame()          # pandas DataFrame with category and meta columns
results.to_csv("out.csv")   # write CSV
results.to_markdown()       # markdown table string
results.to_json()           # JSON string
results.to_clipboard()      # copy to system clipboard
```

In Jupyter, `MetricSet` renders as styled HTML tables automatically.

---

## Architecture

```
stratstat/
  core/                          # Pure computation, no heavy dependencies
    _utils.py                    #   Shared helpers (CAGR, OLS beta, etc.)
    returns/
      descriptive.py             #   21 metrics
      risk.py                    #   24 metrics
      risk_adjusted.py           #   18 metrics
      inference.py               #   10 metrics
      wrappers.py                #   rolling(), by_regime()
    benchmark.py                 #   20 metrics
    exposure.py                  #   24 metrics
    trades.py                    #   37 metrics
    compare.py                   #    7 metrics
  report/                        # Plotly visualization (optional extra)
    _charts.py                   #   7 individual chart functions
    _common.py                   #   Dynamic metric discovery from registry
    _tearsheet.py                #   Single strategy tear sheet
    _dashboard.py                #   Multi strategy dashboard
    _report.py                   #   Self contained HTML report generator
    _export.py                   #   HTML, PNG, SVG, Markdown, LaTeX, JSON export
  inputs.py                      #   ReturnsInput, BenchmarkInput, ExposureInput, etc.
  registry.py                    #   @register_metric, compute(), compute_all()
  results.py                     #   MetricResult, MetricSet
  conventions.py                 #   Session wide metric defaults
```

Core never imports `matplotlib`, `plotly`, or `stratstat.report`. This boundary
is enforced by CI tests. You can use StratStat in a headless environment without
installing any graphics libraries.

---

## Custom metrics

Adding your own metric takes one function and one decorator. After that it
behaves exactly like a built in: `compute()`, `compute_all()`, `rolling()`,
`by_regime()`, tear sheets, dashboards, and HTML reports all pick it up
automatically.

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

result = ss.compute(returns, "my_ratio")
```

---

## Design principles

**Single source of truth.** Every formula exists exactly once. The core API,
reporting layer, reporting helpers, rolling wrappers, and comparison layer all
call the same registered function. When you fix a formula, it is fixed
everywhere.

**Decorator based registry.** `@register_metric` is the only place a metric
declares its name, input requirements, categories, and citation. Adding a new
metric never requires editing a central dispatch block, a switch statement, or
a hardcoded list.

**Standardized output.** Every function returns a `MetricResult` or `MetricSet`.
No bare floats. No dicts with inconsistent keys. The return type carries its own
provenance so you can inspect what convention was used, what the annualization
factor was, and what paper the formula came from.

**Vectorized for multiple strategies.** Every metric works on a single column
or a 2-D array of multiple strategies. The computation happens along axis 0
with no Python loops. `compute_all` runs every metric in a category across all
columns in one pass.

**Strict typing.** All public functions are fully type annotated. Mypy strict
mode passes on the entire codebase. This catches shape mismatches before they
become runtime errors.

**Core separate from visuals.** The computation layer never imports Plotly,
Matplotlib, or any rendering library. You can run StratStat on a server with no
display. The reporting module is an optional extra that depends on core, never
the reverse.

**Citation tracked.** Every metric's docstring and `meta["ref"]` field contain
the academic reference for its formula. When you generate an HTML report, the
methodology section automatically collects and displays all citations for the
metrics that were computed.

---

## Development

```bash
git clone https://github.com/aineurog/stratstat.git
cd stratstat
pip install -e ".[dev]"

pytest tests/ -q           # 893 passed, 1 skipped
ruff check src/ tests/     # lint
mypy src/stratstat/        # type check
```

---

## License

MIT. See [LICENSE](LICENSE).
