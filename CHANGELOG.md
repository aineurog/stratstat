# Changelog

All notable changes to StratStat are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-20

### Added

- 161 registered metrics across eight modules, each with a formula citation in
  its `meta["ref"]` field:
  - `core.returns.descriptive` (21 metrics): CAGR, volatility, skewness,
    kurtosis, percentiles, Hurst exponent, stability, and more.
  - `core.returns.risk` (24 metrics): max drawdown, drawdown duration, VaR,
    CVaR, tail ratio, EVT tail fits, pain index, and more.
  - `core.returns.risk_adjusted` (18 metrics): Sharpe, Sortino, Calmar, Omega,
    Kappa 3, gain to pain, and more.
  - `core.returns.inference` (10 metrics): PSR, DSR, Lo's Sharpe standard error,
    analytic and bootstrap Sharpe intervals, minimum track record length, and more.
  - `core.benchmark` (20 metrics): alpha, beta, tracking error, information
    ratio, up/down capture, Treynor ratio, and more.
  - `core.exposure` (24 metrics): gross/net exposure, concentration, turnover,
    leverage, long/short beta, active share, and more.
  - `core.trades` (37 metrics): win rate, profit factor, expectancy, streaks,
    SQN, MFE/MAE, Kelly criterion, and more.
  - `core.compare` (7 metrics): correlation matrix, diversification ratio,
    Sharpe difference test, White's Reality Check, PBO, risk contributions.
- Decorator based metric registry: `@register_metric`, `compute()`,
  `compute_all()`, `list_metrics()`, `get_metric()`.
- Typed input containers: `ReturnsInput`, `ExposureInput`, `TradeInput`,
  `BenchmarkInput`, `CompareInput`, accepting numpy, pandas, and polars inputs.
- Standardized result types: `MetricResult` and `MetricSet` with dict, frame,
  JSON, CSV, markdown, and clipboard serialization.
- Session wide convention overrides via `set_default()` / `get_default()` for
  metrics with competing definitions.
- Generic `rolling()` and `by_regime()` wrappers that apply to any registered
  returns tier metric.
- Optional Plotly reporting layer: tear sheet, dashboard, and self contained
  HTML report generator, plus HTML, image, markdown, LaTeX, and JSON export.
- Optional numba acceleration for drawdown walks, block and stationary
  bootstrap, and the probability of backtest overfitting, each with a pure
  numpy fallback that matches within floating point tolerance.
- MIT license.

### Changed

- polars is now an optional extra (`stratstat[polars]`) instead of a hard
  dependency. The core install requires only numpy and pandas.
- Package version is now read from `stratstat.__version__` (single source of
  truth) instead of being duplicated in `pyproject.toml`.

## [0.1.0.dev0] - unreleased

- Project scaffold: package structure, registry skeleton, CI pipeline.
