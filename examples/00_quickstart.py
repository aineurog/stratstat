"""StratStat quick start, one tier at a time.

A self contained tour of the five input tiers: returns, benchmark, trades,
exposure, and compare. Everything is synthetic, so the only dependency is
``pip install stratstat``. Run it with:

    python examples/00_quickstart.py
"""

import numpy as np

import stratstat as ss


def main() -> None:
    rng = np.random.default_rng(42)

    # Returns ----------------------------------------------------------------
    returns = rng.normal(0.0004, 0.01, size=252)
    sharpe = ss.compute(returns, "sharpe_ratio", periods_per_year=252)
    print("returns.sharpe_ratio =", float(sharpe.value))

    risk = ss.compute_all(returns, category="risk", periods_per_year=252)
    print("returns.risk has", len(risk), "metrics, e.g.", risk[0].name)

    rolling = ss.rolling(returns, "sharpe_ratio", window=60, periods_per_year=252)
    print("returns.rolling sharpe shape =", np.asarray(rolling.value).shape)

    # Benchmark --------------------------------------------------------------
    bench = rng.normal(0.0003, 0.008, size=252)
    bm = ss.compute_benchmark(returns, bench, periods_per_year=252)
    print("benchmark.alpha =", round(float(bm["alpha"].value), 4))
    print("benchmark.beta  =", round(float(bm["beta"].value), 4))

    # Trades -----------------------------------------------------------------
    trades = {
        "pnl": [0.02, -0.01, 0.03, 0.015, -0.02, 0.01],
        "side": ["long", "short", "long", "long", "short", "long"],
        "duration": [5, 3, 8, 4, 2, 6],
    }
    trade_stats = ss.compute_trades(trades, periods_per_year=252)
    print("trades.win_rate      =", round(float(trade_stats["win_rate"].value), 4))
    print("trades.profit_factor =", round(float(trade_stats["profit_factor"].value), 4))

    # Exposure ---------------------------------------------------------------
    positions = rng.normal(0.05, 0.30, size=(252, 4))
    exposure = ss.compute_exposure(positions, periods_per_year=252)
    print("exposure.long_pct    =", round(float(exposure["long_exposure_pct"].value), 4))
    print("exposure.short_pct   =", round(float(exposure["short_exposure_pct"].value), 4))
    print("exposure.effective_n =", round(float(exposure["effective_n_positions"].value), 4))
    print("exposure.skipped     =", exposure.meta["skipped"])

    # Compare ----------------------------------------------------------------
    a = rng.normal(0.0004, 0.01, size=252)
    b = rng.normal(0.0003, 0.012, size=252)
    cmp = ss.compute_compare(np.column_stack([a, b]), periods_per_year=252)
    print("compare.correlation   =\n", np.asarray(cmp["correlation_matrix"].value))
    print("compare.diversification =", round(float(cmp["diversification_ratio"].value), 4))


if __name__ == "__main__":
    main()
