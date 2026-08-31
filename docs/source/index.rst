StratStat
=========

**Strategy evaluation statistics for Python.** A single library that replaces
``empyrical``, ``pyfolio``, and ``QuantStats`` with more metrics, a cleaner
API, and faster computation.

StratStat computes 176 statistics across five input tiers. Give it a return
series, a trade log, positions, a benchmark, or several strategies, and it
returns typed results with the formula and its citation attached to every
number.

Features at a glance
--------------------

- **176 metrics** across 8 statistical categories
- **Vectorized** batch computation across many strategies in one pass
- **Flexible input** for numpy, pandas, and polars
- **Standardized output** where every call returns ``MetricResult`` or
  ``MetricSet``
- **Registry based** so ``@register_metric`` makes custom metrics first class
- **Generic wrappers** where ``rolling()`` and ``by_regime()`` apply any metric
  over windows or regimes
- **Optional reporting** with tear sheets, dashboards, and chart exports via
  Plotly

Installation
------------

.. code-block:: bash

   pip install stratstat            # core (numpy, pandas)
   pip install stratstat[polars]    # polars Series and DataFrame input
   pip install stratstat[fast]      # numba acceleration
   pip install stratstat[report]    # plotly charts, tear sheets, dashboards
   pip install stratstat[pdf]       # PDF report export
   pip install stratstat[all]       # everything

Python 3.10+ required.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   metrics

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 2
   :caption: Reference

   formula-reference

.. toctree::
   :maxdepth: 1
   :caption: Links

   GitHub <https://github.com/aineurog/stratstat>
   Issues <https://github.com/aineurog/stratstat/issues>
