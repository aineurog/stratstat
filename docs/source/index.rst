StratStat
=========

**Quantitative strategy evaluation statistics** — a comprehensive, high-performance
Python library for evaluating trading strategies.

Features at a glance
--------------------

- **161 metrics** across 8 statistical categories
- **Vectorization first** — batch computation across multiple strategies
- **Flexible input** — pandas, polars, numpy
- **Standardised output** — every call returns ``MetricResult`` or ``MetricSet``
- **Registry-based** — ``@register_metric`` makes custom metrics first-class
- **Generic wrappers** — ``rolling()`` and ``by_regime()`` apply any metric over
  windows or regimes
- **Optional reporting** — tear sheets, dashboards, and chart exports via Plotly

Installation
------------

.. code-block:: bash

   pip install stratstat            # core (numpy, pandas)
   pip install stratstat[polars]    # + polars Series/DataFrame input support
   pip install stratstat[fast]      # + numba acceleration
   pip install stratstat[report]    # + plotly charts, tear sheets, dashboards
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
