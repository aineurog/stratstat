# StratStat

**Quantitative strategy evaluation statistics** — a comprehensive, high-performance
Python library for evaluating trading strategies.

StratStat is the modern successor to abandoned/stale libraries like `empyrical` and
`pyfolio`, and a more complete, better-engineered alternative to `QuantStats`.

> ⚠️ **Early development.** StratStat is under active development. The v0.1 release
> will include 30+ metrics across descriptive, risk, risk-adjusted, and inference
> categories, with a decorator-based plugin system for extensibility.

## Features (planned v0.1)

- **Single source of truth** — every metric implemented exactly once, shared across
  the core API, reporting layer, and comparison layer.
- **Vectorization first** — batch computation across multiple strategies/columns.
- **Flexible input** — accepts pandas, polars, and numpy inputs.
- **Correctness with citation** — every formula cites its source (paper, book, or
  well-established reference).
- **Optional numba acceleration** — `pip install stratstat[fast]` for path-dependent
  and resampling-heavy computations.
- **Standardized output** — every call returns a `MetricResult` or `MetricSet` with
  rich metadata.
- **Plugin system** — register custom metrics via decorator; they work identically to
  built-ins.

## Installation

```bash
pip install stratstat            # core (numpy only)
pip install stratstat[fast]      # + numba acceleration
pip install stratstat[report]    # + plotly-based reporting
pip install stratstat[all]       # everything
```

## Quick start

```python
import numpy as np
import stratstat as ss

returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01])

result = ss.compute(returns, "sharpe_ratio")
print(result)
# MetricResult(name='sharpe_ratio', value=..., category=('risk_adjusted',), ...)

all_stats = ss.compute_all(returns, category="risk_adjusted")
print(all_stats.to_frame())
```

## License

MIT — see [LICENSE](LICENSE).
