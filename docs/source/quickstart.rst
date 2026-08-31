Quick Start
===========

Install StratStat, then feed it data you already hold in memory. It fetches
nothing and computes on whatever you give it.

Computing a single metric
-------------------------

.. code-block:: python

   import numpy as np
   import stratstat as ss

   rng = np.random.default_rng(42)
   returns = rng.normal(0.0004, 0.01, size=252)

   result = ss.compute(returns, "sharpe_ratio", periods_per_year=252)
   print(result)
   # MetricResult(name='sharpe_ratio', value=0.84, category=('risk_adjusted', 'returns'))

Computing every metric on a tier
--------------------------------

.. code-block:: python

   risk = ss.compute_all(returns, category="risk", periods_per_year=252)
   print(risk.to_frame())

   # Every metric on every tier you have data for
   everything = ss.compute_all(
       returns, trades=trades_df, benchmark=bench, periods_per_year=252
   )

   # One tier at a time
   ss.compute_returns(returns, periods_per_year=252)
   ss.compute_benchmark(returns, bench, periods_per_year=252)
   ss.compute_trades(trades, periods_per_year=252)
   ss.compute_exposure(positions, asset_returns=asset_returns, periods_per_year=252)
   ss.compute_compare(np.column_stack([a, b]), periods_per_year=252)

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

Resampling and Monte Carlo
--------------------------

.. code-block:: python

   ci = ss.compute(returns, "block_bootstrap_ci",
                   target_metric="sharpe_ratio", n_reps=5000, random_seed=42)
   print(ci.value)  # [lower, upper]

   dist = ss.compute(returns, "monte_carlo_distribution", target="sharpe", sims=1000)
   print(dist.value)  # [min, p05, median, mean, p95, max, std]

   probs = ss.compute(returns, "monte_carlo_probabilities", bust=-0.20, goal=0.20)
   print(probs.value)  # [p_bust, p_goal]

Strategy and Comparison containers
----------------------------------

A :class:`~stratstat.container.Strategy` holds one strategy's inputs and caches
the derived quantities several metrics share.

.. code-block:: python

   s = ss.Strategy(returns=returns, trades=trades_df, benchmark=bench,
                   periods_per_year=252)
   s.compute_all()
   s.compute("risk")
   s.report("strategy.html")

A :class:`~stratstat.container.Comparison` stacks several period aligned return
series and runs the vectorized engine once, then slices per strategy.

.. code-block:: python

   c = ss.Comparison({"momentum": mom, "carry": carry, "mean_reversion": mr},
                     periods_per_year=252)
   c.compute_all()
   c["momentum"].compute("risk")

Column mapping with a Schema
----------------------------

.. code-block:: python

   trades = {"profit": [0.02, -0.01], "direction": ["long", "short"],
             "bars_held": [5, 3]}

   # Inline shorthand for one call
   ss.compute_trades(trades, columns={"pnl": "profit", "side": "direction",
                                      "duration": "bars_held"})

   # Reusable object, or a session default
   schema = ss.Schema(trades={"pnl": "profit", "side": "direction"})
   ss.compute_trades(trades, schema=schema)

   ss.set_schema({"trades": {"pnl": "profit", "side": "direction"}})
   ss.compute_trades(trades)
   ss.clear_schema()

   # Which columns are recognized, and which metrics you give up
   ss.describe_columns(trades)

Session conventions
-------------------

.. code-block:: python

   ss.compute(returns, "sharpe_ratio", periods_per_year=252, ddof=0)
   ss.compute(returns, "var", confidence=0.99)

   ss.set_default("sharpe_ratio", "ddof=0")
   ss.set_default("var", "method=parametric")
   ss.get_default("sharpe_ratio")

Discovering available metrics
-----------------------------

.. code-block:: python

   for m in ss.list_metrics():
       print(m["name"], m["category"], m["requires"])

   for m in ss.list_metrics(category="risk"):
       print(m["name"])

Generating reports
------------------

Install ``stratstat[report]`` for visualization support:

.. code-block:: bash

   pip install stratstat[report]

.. code-block:: python

   from stratstat.report import tear_sheet, dashboard, generate_report

   # Single strategy tear sheet
   fig = tear_sheet(returns, benchmark=bench, periods_per_year=252)
   fig.show()

   # Multi strategy dashboard
   multi = np.column_stack([strat_a, strat_b, strat_c])
   fig = dashboard(multi, periods_per_year=252)
   fig.show()

   # Standalone HTML report on disk
   generate_report(returns, "report.html", benchmark=bench,
                   positions=positions, trades=trades_df,
                   periods_per_year=252, title="My Strategy")

   # Export helpers
   from stratstat.report import to_html, to_markdown
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
