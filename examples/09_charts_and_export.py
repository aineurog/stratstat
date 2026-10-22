"""Individual chart gallery and export helpers.

Examples 04 and 05 show the high-level compositions (``tear_sheet``,
``dashboard``, ``generate_report``). This example goes one level down:

  * Each individual chart function is called directly, so you can see
    what every chart does and how to render just one of them.
  * The export helpers are demonstrated for writing a figure (``to_html``,
    ``to_image``) or a ``MetricSet`` (``to_json``, ``to_markdown``,
    ``to_latex``) to disk.

Run: python examples/09_charts_and_export.py

Requires: pip install stratstat[report]   (and [image] for to_image)
"""

import numpy as np

import stratstat as ss
from stratstat.report import (
    benchmark_overlay_chart,
    cumulative_return_chart,
    drawdown_chart,
    effective_n_chart,
    equity_curve,
    exposure_heatmap,
    exposure_over_time,
    monthly_heatmap,
    returns_distribution,
    rolling_metric_chart,
    to_html,
    to_image,
    to_json,
    to_latex,
    to_markdown,
    trade_duration_histogram,
    trade_markers_chart,
)

OUT = "examples/output/"

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 252  # 1 year of daily data

strategy_returns = rng.normal(0.0008, 0.018, size=n)
bench_returns = rng.normal(0.0003, 0.012, size=n)

# Trade log: pnl, side, and duration (duration is required for the histogram)
n_trades = 50
trade_pnl = rng.normal(0.006, 0.035, size=n_trades)
trade_pnl[:30] = np.abs(trade_pnl[:30])  # ~60% winners
trade_pnl[30:] = -np.abs(trade_pnl[30:])
rng.shuffle(trade_pnl)
trade_side = np.array(["long"] * 28 + ["short"] * 22)
rng.shuffle(trade_side)
trade_duration = np.abs(rng.normal(10, 4, size=n_trades))
trades = {"pnl": trade_pnl, "side": trade_side, "duration": trade_duration}

# Position weights: (n_periods, n_assets); last two assets are short
rng_pos = np.random.default_rng(7)
positions = rng_pos.normal(0.1, 0.12, size=(n, 8))
positions[:, 6:] = -np.abs(positions[:, 6:])

# ---------------------------------------------------------------------------
# Chart gallery — one call per chart
# ---------------------------------------------------------------------------
# Each returns a plotly Figure. The (name, fig) pairs are collected and
# written out with to_html() below, so every chart is saved as a standalone
# interactive file.
charts = [
    ("equity", equity_curve(strategy_returns, title="Equity Curve")),
    (
        "cumulative_return",
        cumulative_return_chart(
            strategy_returns, benchmark=bench_returns, title="Cumulative Return vs Benchmark"
        ),
    ),
    ("drawdown", drawdown_chart(strategy_returns, title="Drawdown (Underwater)")),
    ("monthly_heatmap", monthly_heatmap(strategy_returns, title="Monthly Returns Heatmap")),
    (
        "returns_distribution",
        returns_distribution(strategy_returns, title="Returns Distribution"),
    ),
    (
        "rolling_sharpe",
        rolling_metric_chart(
            strategy_returns, "sharpe_ratio", window=60, periods_per_year=252,
            title="Rolling Sharpe Ratio",
        ),
    ),
    (
        "benchmark_overlay",
        benchmark_overlay_chart(
            strategy_returns, bench_returns, title="Benchmark Comparison"
        ),
    ),
    (
        "trade_markers",
        trade_markers_chart(
            trades, returns=strategy_returns, title="Trade Markers"
        ),
    ),
    (
        "trade_duration",
        trade_duration_histogram(trades, title="Trade Duration Distribution"),
    ),
    (
        "exposure_over_time",
        exposure_over_time(positions, title="Gross vs Net Exposure"),
    ),
    (
        "effective_n",
        effective_n_chart(positions, title="Effective Number of Positions"),
    ),
    (
        "exposure_heatmap",
        exposure_heatmap(positions, title="Asset Exposure Heatmap"),
    ),
]

for name, fig in charts:
    to_html(fig, f"{OUT}chart_{name}.html")
print(f"Saved {len(charts)} charts as interactive HTML under {OUT}")

# ---------------------------------------------------------------------------
# Export helpers — figures
# ---------------------------------------------------------------------------
# to_html already used above; to_image renders a static file and needs kaleido.
fig = equity_curve(strategy_returns, title="Equity Curve (static export)")
try:
    to_image(fig, f"{OUT}chart_equity.png", format="png", width=1200, height=600, scale=2)
    print(f"Saved static PNG: {OUT}chart_equity.png")
except ImportError as e:
    print(f"  Skipped to_image (kaleido not installed): {e}")

# ---------------------------------------------------------------------------
# Export helpers — MetricSet
# ---------------------------------------------------------------------------
# compute_all returns a MetricSet, which the file-writing helpers below accept.
metrics = ss.compute_all(strategy_returns, periods_per_year=252)

# Note the two different JSON/Markdown entry points:
#   metrics.to_json()        -> returns a string (MetricSet method)
#   to_json(metrics, path)   -> writes a file  (report export helper)
to_json(metrics, f"{OUT}metrics.json")
to_markdown(metrics, f"{OUT}metrics.md")
to_latex(metrics, f"{OUT}metrics.tex")
print("Saved metrics as JSON, Markdown, and LaTeX under examples/output/")

print("\nDone. Open the chart_*.html files in a browser to explore each chart,")
print("and the metrics.* files in any editor to see the export formats.")
