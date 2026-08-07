# Examples

Runnable scripts that demonstrate StratStat's capabilities end to end.
Each script is self-contained and uses synthetic data — no external files
needed.

## Running

```bash
# Compute metrics
python examples/01_computing_metrics.py

# Rolling windows and regime breakdowns
python examples/02_rolling_and_regime.py

# Benchmark comparison
python examples/03_benchmark_comparison.py

# Tear sheet and dashboard charts
python examples/04_visualizations.py

# HTML and PDF reports
python examples/05_generating_reports.py
```

## What each example covers

| # | File | What you will see |
|---|------|-------------------|
| 01 | `computing_metrics` | `compute()`, `compute_all()`, `list_metrics()`, `MetricResult`, `MetricSet`, serialization to DataFrame / CSV / JSON / markdown |
| 02 | `rolling_and_regime` | `rolling()` for any metric, `by_regime()` for bull/bear splits |
| 03 | `benchmark_comparison` | `BenchmarkInput`, alpha, beta, IR, up/down capture, batting average, IC, directional consistency |
| 04 | `visualizations` | `tear_sheet()` and `dashboard()` — interactive Plotly charts |
| 05 | `generating_reports` | `generate_report()` for both HTML and PDF output |

## Dependencies

- Examples 01 through 03 need only `pip install stratstat`
- Example 04 needs `pip install stratstat[report]`
- Example 05 needs `pip install stratstat[report]` for HTML, plus `pip install stratstat[pdf]` for PDF output
