# Usage guide

The reference details for day to day use: the trade log data contract, column
mapping, formula conventions, and custom metrics. The README links here so it
can stay a quick start.

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
citations is in the [formula reference](formula-reference.md).

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
