Quick Start
===========

Computing a single metric
-------------------------

.. code-block:: python

   import numpy as np
   import stratstat as ss

   returns = np.random.normal(0.001, 0.02, size=252)

   result = ss.compute(returns, "sharpe_ratio", periods_per_year=252)
   print(result)
   # MetricResult(name='sharpe_ratio', value=0.84, ...)

Computing all metrics in a category
-----------------------------------

.. code-block:: python

   risk_stats = ss.compute_all(returns, category="risk", periods_per_year=252)
   print(risk_stats.to_frame())

Rolling metrics (time series)
-----------------------------

.. code-block:: python

   rolling_sharpe = ss.rolling(returns, "sharpe_ratio", window=60,
                               periods_per_year=252)

Metrics by market regime
------------------------

.. code-block:: python

   labels = np.where(returns > 0, "up", "down")
   regime_cagr = ss.by_regime(returns, "cagr", labels, periods_per_year=252)

Block-bootstrap confidence intervals
-------------------------------------

.. code-block:: python

   ci = ss.compute(returns, "block_bootstrap_ci",
                   target_metric="sharpe_ratio", n_reps=5000, random_seed=42)
   print(ci.value)  # [lower, upper]

Discovering available metrics
-----------------------------

.. code-block:: python

   for m in ss.list_metrics():
       print(m["name"], m["category"], m["requires"])

   # Filter by category
   for m in ss.list_metrics(category="risk"):
       print(m["name"])

Generating reports
------------------

Install ``stratstat[report]`` for visualisation support:

.. code-block:: bash

   pip install stratstat[report]

.. code-block:: python

   from stratstat.report import tear_sheet, dashboard

   # Single-strategy tear sheet
   fig = tear_sheet(returns, periods_per_year=252)
   fig.show()

   # Multi-strategy dashboard
   multi = np.column_stack([strat_a, strat_b, strat_c])
   fig = dashboard(multi, periods_per_year=252)
   fig.show()

   # Export
   from stratstat.report import to_html
   to_html(fig, "report.html")

Registering a custom metric
---------------------------

.. code-block:: python

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

   # Now works everywhere
   ss.compute(returns, "my_ratio")
