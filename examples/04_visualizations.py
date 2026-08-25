"""Tear sheet and dashboard visualizations.

The tear sheet is a 4-panel interactive Plotly figure for a single strategy.
The dashboard compares multiple strategies side by side with rankings.

Both are dynamic — they auto-discover metrics from the registry, so every
new metric you register automatically appears in the stats tables.

Run: python examples/04_visualizations.py
"""

import numpy as np

from stratstat.report import dashboard, tear_sheet

# ---------------------------------------------------------------------------
# Generate synthetic data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 252

# Single strategy with a benchmark
strategy_returns = rng.normal(0.0008, 0.018, size=n)
bench_returns = rng.normal(0.0003, 0.012, size=n)

# Multi-strategy: 3 strategies with different risk/return profiles
strat1 = rng.normal(0.0010, 0.020, size=n)  # higher return, higher vol
strat2 = rng.normal(0.0004, 0.012, size=n)  # lower return, lower vol
strat3 = rng.normal(0.0006, 0.016, size=n)  # middle of the road
multi_returns = np.column_stack([strat1, strat2, strat3])

# ---------------------------------------------------------------------------
# Tear Sheet — single strategy deep dive
# ---------------------------------------------------------------------------
print("Generating tear sheet...")
fig_ts = tear_sheet(
    strategy_returns,
    benchmark=bench_returns,
    periods_per_year=252,
    title="Strategy Analysis — Tear Sheet",
)

# Save as interactive HTML
fig_ts.write_html("examples/output/tearsheet.html", include_plotlyjs="cdn")
print("  Saved: examples/output/tearsheet.html")

# Uncomment to display interactively:
# fig_ts.show()

# ---------------------------------------------------------------------------
# Dashboard — multi-strategy comparison
# ---------------------------------------------------------------------------
print("Generating dashboard...")
fig_db = dashboard(
    multi_returns,
    periods_per_year=252,
    title="Multi-Strategy Comparison — Dashboard",
    rolling_window=60,
)

fig_db.write_html("examples/output/dashboard.html", include_plotlyjs="cdn")
print("  Saved: examples/output/dashboard.html")

# Uncomment to display interactively:
# fig_db.show()

print("\nOpen the HTML files in a browser to explore the interactive charts.")
print("The tear sheet shows: equity curve, drawdown, monthly heatmap, stats.")
print("The dashboard shows: rankings, rolling metrics, correlation, equity.")
