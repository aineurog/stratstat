Metric Inventory
================

StratStat ships **138 registered metrics** across 8 categories. Every metric returns
:class:`~stratstat.results.MetricResult` with a ``.meta["ref"]`` field citing the
formula's source.

.. list-table::
   :header-rows: 1

   * - Category
     - Tier (``requires``)
     - Count
     - Examples
   * - Descriptive
     - returns
     - 16
     - CAGR, volatility, skewness, kurtosis, autocorrelation, percentiles
   * - Risk
     - returns
     - 20
     - VaR, CVaR, max drawdown, drawdown duration, ulcer index, tail ratio, EVT
   * - Risk-Adjusted
     - returns
     - 9
     - Sharpe, Sortino, Calmar, Omega, Kappa-3, Martin, gain-to-pain
   * - Inference
     - returns
     - 8
     - PSR, DSR, Lo's SE, Sharpe CI (analytic & bootstrap), block-bootstrap CI
   * - Exposure
     - exposure
     - 23
     - gross/net exposure, HHI, turnover, leverage, beta, position coverage
   * - Trades
     - trades
     - 37
     - win rate, profit factor, expectancy, streaks, MFE/MAE, holding periods
   * - Benchmark
     - benchmark
     - 18
     - tracking error, information ratio, alpha, beta, M², up/down capture
   * - Compare
     - compare
     - 7
     - Sharpe-difference test, MCR, diversification ratio, PBO, Reality Check
   * - **Wrappers**
     - returns
     - —
     - ``rolling()`` and ``by_regime()`` — apply any metric over windows/regimes

For the full API reference with signatures and docstrings, see the :doc:`api` page.

Discovering metrics at runtime
------------------------------

.. code-block:: python

   import stratstat as ss

   for m in ss.list_metrics():
       print(m["name"], m["category"], m["requires"])

   # Filter by category
   for m in ss.list_metrics(category="risk"):
       print(m["name"])
