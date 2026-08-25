"""Tests for the public compute_all() / domain-function API and MetricSet helpers.

Covers the multi-tier fan-out, resampling exclusion, alias deduplication, and
the name/tier/category accessors added to ``MetricSet``.
"""

import numpy as np
import pytest

from stratstat import (
    compute_all,
    compute_benchmark,
    compute_compare,
    compute_exposure,
    compute_returns,
    compute_trades,
)
from stratstat.registry import list_metrics, register_metric, requires_of
from stratstat.results import MetricResult, MetricSet


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.001, 0.02, size=252)


@pytest.fixture(scope="module")
def benchmark():
    rng = np.random.default_rng(7)
    return rng.normal(0.0005, 0.015, size=252)


@pytest.fixture(scope="module")
def trades():
    rng = np.random.default_rng(11)
    return {
        "pnl": rng.normal(0.002, 0.03, size=60),
        "side": np.where(rng.normal(size=60) > 0, "long", "short"),
    }


# ---------------------------------------------------------------------------
# Domain functions route to the correct tier
# ---------------------------------------------------------------------------


class TestDomainFunctions:
    def test_compute_returns_only_returns_tier(self, returns):
        ms = compute_returns(returns, periods_per_year=252)
        assert set(ms.by_tier()) == {"returns"}
        assert "sharpe_ratio" in ms
        assert "cagr" in ms

    def test_compute_trades_only_trades_tier(self, trades):
        ms = compute_trades(trades)
        assert set(ms.by_tier()) == {"trades"}
        assert "profit_factor" in ms
        assert "avg_win" in ms

    def test_compute_benchmark_only_benchmark_tier(self, returns, benchmark):
        ms = compute_benchmark(returns, benchmark, periods_per_year=252)
        assert set(ms.by_tier()) == {"benchmark"}
        assert "beta" in ms

    def test_compute_exposure_only_exposure_tier(self):
        rng = np.random.default_rng(3)
        positions = rng.normal(0, 1, size=(100, 4))
        ms = compute_exposure(positions, periods_per_year=252)
        assert set(ms.by_tier()) == {"exposure"}
        assert "gross_exposure" in ms

    def test_compute_compare_excludes_resampling(self, benchmark):
        rng = np.random.default_rng(5)
        multi = rng.normal(0.001, 0.02, size=(252, 2))
        ms = compute_compare(multi, benchmark=benchmark, periods_per_year=252)
        assert set(ms.by_tier()) == {"compare"}
        assert set(ms.meta.get("excluded_resampling", [])) == {
            "whites_reality_check", "pbo",
        }


# ---------------------------------------------------------------------------
# compute_all fan-out, dedup, and exclusion
# ---------------------------------------------------------------------------


class TestComputeAll:
    def test_fan_out_across_tiers(self, returns, trades, benchmark):
        ms = compute_all(
            returns=returns, trades=trades, benchmark=benchmark,
            periods_per_year=252,
        )
        assert set(ms.by_tier()) == {"returns", "trades", "benchmark"}

    def test_resampling_metrics_excluded(self, returns):
        ms = compute_all(returns=returns, periods_per_year=252)
        names = {r.name for r in ms}
        for name in (
            "monte_carlo_distribution",
            "monte_carlo_probabilities",
            "sharpe_ci_bootstrap",
            "block_bootstrap_ci",
        ):
            assert name not in names
        # Only the returns tier runs here (compare requires an explicit
        # ``compare``), so exactly its four resampling metrics are excluded.
        assert set(ms.meta.get("excluded_resampling", [])) == {
            "monte_carlo_distribution",
            "monte_carlo_probabilities",
            "sharpe_ci_bootstrap",
            "block_bootstrap_ci",
        }

    def test_dedup_drops_period_twins(self, returns, trades):
        ms = compute_all(returns=returns, trades=trades, periods_per_year=252)
        names = {r.name for r in ms}
        # Trade-level canonical present ...
        assert "avg_win" in names
        assert "avg_loss" in names
        assert "profit_factor" in names
        assert "payoff_ratio" in names
        assert "kelly_criterion" in names
        # ... period-level twins dropped.
        assert "avg_up_period" not in names
        assert "avg_down_period" not in names
        assert "period_profit_factor" not in names
        assert "period_payoff_ratio" not in names
        assert "period_kelly_criterion" not in names
        assert set(ms.meta.get("deduplicated", [])) == {
            "avg_up_period",
            "avg_down_period",
            "period_profit_factor",
            "period_payoff_ratio",
            "period_kelly_criterion",
        }

    def test_deduplicate_false_keeps_both(self, returns, trades):
        ms = compute_all(
            returns=returns, trades=trades, periods_per_year=252,
            deduplicate=False,
        )
        names = {r.name for r in ms}
        assert "avg_win" in names and "avg_up_period" in names

    def test_alias_kept_when_canonical_absent(self, returns):
        """avg_up_period survives when the trades tier is not provided."""
        ms = compute_all(returns=returns, periods_per_year=252)
        names = {r.name for r in ms}
        assert "avg_up_period" in names
        assert "avg_win" not in names

    def test_include_flag_disables_tier(self, returns, trades):
        ms = compute_all(returns=returns, trades=trades, include_trades=False)
        assert set(ms.by_tier()) == {"returns"}
        assert "trades" in ms.meta.get("excluded_tiers", [])

    def test_compare_tier_not_run_without_compare_data(self, returns):
        """Multi-column returns do NOT implicitly trigger the compare tier."""
        rng = np.random.default_rng(4)
        multi = rng.normal(0.001, 0.02, size=(252, 2))
        ms = compute_all(returns=multi, periods_per_year=252)
        assert set(ms.by_tier()) == {"returns"}
        assert "compare" in ms.meta.get("excluded_tiers", [])
        assert "correlation_matrix" not in {r.name for r in ms}

    def test_tiers_restricts_to_listed(self, returns, trades):
        ms = compute_all(returns=returns, trades=trades, tiers=["returns"])
        assert set(ms.by_tier()) == {"returns"}

    def test_category_filter(self, returns):
        ms = compute_all(returns=returns, category="risk", periods_per_year=252)
        assert all(r.category[0] == "risk" for r in ms)
        assert len(ms) > 0


# ---------------------------------------------------------------------------
# MetricSet accessors
# ---------------------------------------------------------------------------


class TestMetricSetAccessors:
    def _set(self):
        return MetricSet(results=[
            MetricResult(name="a", value=1.0, category=("descriptive", "returns")),
            MetricResult(name="b", value=2.0, category=("risk", "returns")),
        ])

    def test_getitem_by_name(self):
        ms = self._set()
        assert ms["a"].value == 1.0

    def test_getitem_missing_name_raises_keyerror(self):
        ms = self._set()
        with pytest.raises(KeyError):
            ms["missing"]

    def test_getitem_by_index(self):
        ms = self._set()
        assert ms[0].name == "a"

    def test_get(self):
        ms = self._set()
        assert ms.get("a").value == 1.0
        assert ms.get("missing") is None
        assert ms.get("missing", "fallback") == "fallback"

    def test_contains(self):
        ms = self._set()
        assert "a" in ms
        assert "missing" not in ms

    def test_by_category(self):
        ms = self._set()
        grouped = ms.by_category()
        assert set(grouped) == {"descriptive", "risk"}
        assert len(grouped["risk"]) == 1


# ---------------------------------------------------------------------------
# Registry: alias_of and requires_of
# ---------------------------------------------------------------------------


class TestRegistryAliasOf:
    def test_list_metrics_includes_alias_of(self):
        for m in list_metrics():
            assert "alias_of" in m
            assert isinstance(m["alias_of"], str)

    def test_requires_of(self):
        assert requires_of("sharpe_ratio") == "returns"
        assert requires_of("profit_factor") == "trades"
        assert requires_of("nonexistent") is None

    def test_register_metric_stores_alias_of(self):
        @register_metric(
            name="__api_test_metric",
            requires="returns",
            category=("descriptive", "returns"),
            backend="vectorized",
            alias_of="__api_test_canonical",
        )
        def __api_test_metric(input_data):  # pragma: no cover - synthetic
            return MetricResult(name="__api_test_metric", value=0.0)

        try:
            assert requires_of("__api_test_metric") == "returns"
            entry = next(
                m for m in list_metrics() if m["name"] == "__api_test_metric"
            )
            assert entry["alias_of"] == "__api_test_canonical"
        finally:
            from stratstat.registry import _registry

            _registry.pop("__api_test_metric", None)


# ---------------------------------------------------------------------------
# to_frame tier column
# ---------------------------------------------------------------------------


class TestToFrameTier:
    def test_to_frame_has_tier_column(self, returns):
        ms = compute_returns(returns, periods_per_year=252)
        df = ms.to_frame()
        assert "tier" in df.columns
        assert set(df["tier"].dropna()) == {"returns"}
