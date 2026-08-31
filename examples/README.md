# Examples

Runnable scripts that demonstrate StratStat end to end. Each script is self
contained and uses synthetic data, so nothing has to be downloaded.

## Running

```bash
python examples/00_quickstart.py
python examples/01_computing_metrics.py
python examples/02_rolling_and_regime.py
python examples/03_benchmark_comparison.py
python examples/04_visualizations.py
python examples/05_generating_reports.py
python examples/06_quantstats_comparison.py
python examples/07_monte_carlo_comparison.py
python examples/08_timing_benchmark.py
```

## What each example covers

| # | File | What you will see |
|---|---|---|
| 00 | `quickstart` | a tour of the five tiers (returns, benchmark, trades, exposure, compare) with one entry point each |
| 01 | `computing_metrics` | `compute`, `compute_all`, `list_metrics`, `MetricResult`, `MetricSet`, and serialization to DataFrame, CSV, JSON, and markdown |
| 02 | `rolling_and_regime` | `rolling` for any metric and `by_regime` for bull and bear splits |
| 03 | `benchmark_comparison` | `BenchmarkInput`, alpha, beta, information ratio, up and down capture, batting average, information coefficient, and directional consistency |
| 04 | `visualizations` | `tear_sheet` and `dashboard` interactive Plotly charts |
| 05 | `generating_reports` | `generate_report` for HTML and PDF output |
| 06 | `quantstats_comparison` | a metric by metric parity check against QuantStats, grouped into exact match, convention, sign, formula, benchmark, and QuantStats-only tiers |
| 07 | `monte_carlo_comparison` | StratStat bootstrap versus QuantStats shuffle, and why shuffle leaves three of four Monte Carlo statistics degenerate |
| 08 | `timing_benchmark` | wall clock timing of the public API against QuantStats, per metric and full report |

## Dependencies

- Examples 00 through 03 need only `pip install stratstat`
- Example 04 needs `pip install stratstat[report]`
- Example 05 needs `pip install stratstat[report]` for HTML and
  `pip install stratstat[pdf]` for PDF output
- Examples 06, 07, and 08 import QuantStats only as an optional reference and
  print a hint and exit cleanly if it is not installed

Output files from the examples are written under `examples/output/`.
