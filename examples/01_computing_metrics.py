"""Compute individual metrics, batch metrics, and explore what is available.

This example covers the fundamental compute() and compute_all() APIs,
the MetricResult and MetricSet types, and serialization to DataFrame,
CSV, JSON, and markdown.

Run: python examples/01_computing_metrics.py
"""

import numpy as np

import stratstat as ss
from stratstat.inputs import ReturnsInput

# ---------------------------------------------------------------------------
# Generate synthetic daily returns for a single strategy
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
daily_returns = rng.normal(0.0004, 0.01, size=252)  # ~10% annual return

print("=" * 60)
print("COMPUTING INDIVIDUAL METRICS")
print("=" * 60)

# Wrap in a ReturnsInput so the annualization factor is available to every
# metric.  (Metrics read periods_per_year from the input object, not kwargs.)
inp = ReturnsInput(daily_returns, periods_per_year=252)

# -- Single metric ----------------------------------------------------------
sharp = ss.compute(inp, "sharpe_ratio")
print("\nSingle metric:")
print(f"  {sharp}")

# Every result carries provenance
print(f"\n  name:             {sharp.name}")
print(f"  value:            {sharp.value:.4f}")
print(f"  category:         {sharp.category}")
print(f"  periods_per_year: {sharp.periods_per_year}")
print(f"  citation:         {sharp.meta.get('ref', 'N/A')[:80]}...")

# -- Batch: all metrics in a category ---------------------------------------
print("\n\nAll risk metrics:")
risk_metrics = ss.compute_all(inp, category="risk")
print(risk_metrics)

# -- Batch: all returns-tier metrics ----------------------------------------
print("\n\nAll descriptive metrics:")
descriptive = ss.compute_all(inp, category="descriptive")
print(descriptive)

# ---------------------------------------------------------------------------
# Multi-strategy: 2-D array, one column per strategy
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 60)
print("MULTI-STRATEGY (3 columns)")
print("=" * 60)

rng = np.random.default_rng(99)
multi = rng.normal(0.0005, 0.012, size=(252, 3))  # shape (252, 3)
multi_inp = ReturnsInput(multi, periods_per_year=252)

all_metrics = ss.compute_all(multi_inp, category="risk_adjusted")
print(all_metrics)

# Each metric's value is now an array of length 3
print(f"\nSharpe ratio value: {all_metrics[0].value}")

# ---------------------------------------------------------------------------
# Discovering available metrics
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 60)
print("METRIC DISCOVERY")
print("=" * 60)

all_names = ss.list_metrics()
print(f"\nTotal registered metrics: {len(all_names)}")

# Filter by input tier
returns_metrics = ss.list_metrics(requires="returns")
benchmark_metrics = ss.list_metrics(requires="benchmark")
print(f"  returns-tier:   {len(returns_metrics)}")
print(f"  benchmark-tier: {len(benchmark_metrics)}")

# Filter by category
risk_adj = ss.list_metrics(category="risk_adjusted")
print("\nRisk-adjusted metrics:")
for m in risk_adj:
    print(f"  {m['name']:<30s}  {m.get('ref', '')[:60]}")

# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 60)
print("SERIALIZATION")
print("=" * 60)

ms = ss.compute_all(inp, category="risk")

# To DataFrame
df = ms.to_frame()
print(f"\nDataFrame: {df.shape[0]} rows, columns: {list(df.columns)}")

# To dict
d = ms.to_dict()
print(f"\nDict keys (first 5): {list(d.keys())[:5]}")

# To JSON
json_str = ms.to_json()
print(f"\nJSON (first 200 chars):\n{json_str[:200]}...")

# To markdown
print(f"\nMarkdown:\n{ms.to_markdown()[:300]}...")

# To CSV
ms.to_csv("examples/output/risk_metrics.csv")
print("\nWrote examples/output/risk_metrics.csv")
