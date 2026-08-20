"""Generate standalone HTML and PDF strategy reports.

generate_report() produces a self-contained file with:
  - Summary stat cards (CAGR, Sharpe, Max DD, VaR, ...)
  - Equity curve and drawdown (side by side)
  - Returns distribution and monthly heatmap (side by side)
  - Rolling Sharpe and rolling volatility
  - Tabbed statistics tables grouped by category with citation links
  - Numbered methodology references

HTML files open in any browser. PDF files need weasyprint:
    pip install stratstat[pdf]

Run: python examples/05_generating_reports.py
"""

import numpy as np

import stratstat as ss
from stratstat.report import generate_report

# ---------------------------------------------------------------------------
# Generate synthetic data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 252  # 1 year of daily data

strategy_returns = rng.normal(0.0008, 0.018, size=n)
bench_returns = rng.normal(0.0003, 0.012, size=n)

# ---------------------------------------------------------------------------
# HTML Report — basic (returns only)
# ---------------------------------------------------------------------------
print("Generating HTML report (basic)...")
generate_report(
    strategy_returns,
    "examples/output/report_basic.html",
    periods_per_year=252,
    title="Strategy Analysis — Basic Report",
)
print("  Saved: examples/output/report_basic.html")

# ---------------------------------------------------------------------------
# HTML Report — with benchmark
# ---------------------------------------------------------------------------
print("Generating HTML report (with benchmark)...")
generate_report(
    strategy_returns,
    "examples/output/report_full.html",
    benchmark=bench_returns,
    periods_per_year=252,
    title="Strategy Analysis — Full Report with Benchmark",
)
print("  Saved: examples/output/report_full.html")

# ---------------------------------------------------------------------------
# PDF Report — basic
# ---------------------------------------------------------------------------
print("Generating PDF report (basic)...")
try:
    generate_report(
        strategy_returns,
        "examples/output/report_basic.pdf",
        periods_per_year=252,
        title="Strategy Analysis — Basic PDF Report",
    )
    print("  Saved: examples/output/report_basic.pdf")
except ImportError as e:
    print(f"  Skipped (weasyprint not installed): {e}")

# ---------------------------------------------------------------------------
# PDF Report — with benchmark
# ---------------------------------------------------------------------------
print("Generating PDF report (with benchmark)...")
try:
    generate_report(
        strategy_returns,
        "examples/output/report_full.pdf",
        benchmark=bench_returns,
        periods_per_year=252,
        title="Strategy Analysis — Full PDF Report with Benchmark",
    )
    print("  Saved: examples/output/report_full.pdf")
except ImportError as e:
    print(f"  Skipped (weasyprint not installed): {e}")

# ---------------------------------------------------------------------------
# Short series — edge case
# ---------------------------------------------------------------------------
print("Generating report for short series (6 months)...")
short_returns = np.array([0.01, -0.02, 0.03, -0.01, 0.015, 0.02])
generate_report(
    short_returns,
    "examples/output/report_short.html",
    periods_per_year=12,
    title="Short Series Report (6 months)",
)
print("  Saved: examples/output/report_short.html")

# ---------------------------------------------------------------------------
# With exposure data (position weights)
# ---------------------------------------------------------------------------
print("Generating report with exposure data...")
rng2 = np.random.default_rng(7)
positions = rng2.normal(0.1, 0.12, size=(n, 8))
# Make the last 2 assets short
positions[:, 6:] = -np.abs(positions[:, 6:])

generate_report(
    strategy_returns,
    "examples/output/report_exposure.html",
    positions=positions,
    periods_per_year=252,
    title="Strategy Analysis — With Exposure",
)
print("  Saved: examples/output/report_exposure.html")

# ---------------------------------------------------------------------------
# With trade log
# ---------------------------------------------------------------------------
print("Generating report with trade data...")
rng3 = np.random.default_rng(13)
n_trades = 50
# P&L: some wins, some losses
trade_pnl = rng3.normal(0.006, 0.035, size=n_trades)
# Make ~60% winners
trade_pnl[:30] = np.abs(trade_pnl[:30])
trade_pnl[30:] = -np.abs(trade_pnl[30:])
rng3.shuffle(trade_pnl)

# Basic trade log (P&L only)
generate_report(
    strategy_returns,
    "examples/output/report_trades.html",
    positions=positions,
    trades={"pnl": trade_pnl},
    periods_per_year=252,
    title="Strategy Analysis — With Trades",
)
print("  Saved: examples/output/report_trades.html")

# Trade log with side + duration
side = np.array(["long"] * 28 + ["short"] * 22)
rng3.shuffle(side)
duration = np.abs(rng3.normal(10, 4, size=n_trades))

generate_report(
    strategy_returns,
    "examples/output/report_trades_full.html",
    positions=positions,
    trades={"pnl": trade_pnl, "side": side, "duration": duration},
    periods_per_year=252,
    title="Strategy Analysis — Full Trade Report",
)
print("  Saved: examples/output/report_trades_full.html")

# ---------------------------------------------------------------------------
# With pre-computed metrics (MetricSet)
# ---------------------------------------------------------------------------
print("Generating report with pre-computed metrics...")
# compute_all accepts raw data directly — no need to construct an Input object
metrics = ss.compute_all(strategy_returns, periods_per_year=252)
print(f"  Computed {len(metrics)} metrics ahead of time")

generate_report(
    strategy_returns,
    "examples/output/report_from_metrics.html",
    benchmark=bench_returns,
    metrics=metrics,
    periods_per_year=252,
    title="Strategy Analysis — From Pre-Computed Metrics",
)
print("  Saved: examples/output/report_from_metrics.html")

# compute() also accepts raw data directly
cagr_result = ss.compute(strategy_returns, "cagr", periods_per_year=252)
print(f"  Single metric compute: {cagr_result}")

print("\nDone. Open the .html files in a browser, or the .pdf files in any")
print("PDF viewer. Each file is self-contained — all plotly charts render")
print("offline with no CDN dependency.")
