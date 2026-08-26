"""Tests for the Strategy and Comparison containers."""

import warnings

import numpy as np
import pytest

from stratstat import Comparison, MetricSet, Strategy

# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


def test_strategy_builds_returns_input():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 60)
    s = Strategy(r, periods_per_year=252)
    assert s.returns_input is not None
    assert s.returns_input.n_periods == 60
    assert s.returns_input.n_strategies == 1
    assert s.periods_per_year == 252


def test_strategy_compute_all():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 120)
    s = Strategy(r, periods_per_year=252)
    ms = s.compute_all()
    assert isinstance(ms, MetricSet)
    assert "sharpe_ratio" in ms
    assert "cagr" in ms
    assert "max_drawdown" in ms


def test_strategy_compute_category():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, 120)
    s = Strategy(r, periods_per_year=252)
    ms = s.compute("risk")
    assert len(ms) > 0
    assert all(m.category[0] == "risk" for m in ms)


def test_strategy_with_trades_and_benchmark():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 120)
    bench = rng.normal(0, 0.005, 120)
    trades = {"pnl": np.array([0.5, -0.2, 0.3, 0.1])}
    # These trades compound to a very different total than the random returns,
    # so construction legitimately warns about the mismatch.
    with pytest.warns(UserWarning, match="does not reconcile"):
        s = Strategy(r, trades=trades, benchmark=bench, periods_per_year=252)
    ms = s.compute_all()
    tiers = set(ms.by_tier())
    assert {"returns", "trades", "benchmark"} <= tiers


def test_strategy_with_exposure():
    rng = np.random.default_rng(4)
    pos = rng.normal(0.5, 0.1, (30, 3))
    rets = rng.normal(0, 0.01, (30, 3))
    s = Strategy(positions=pos, asset_returns=rets)
    assert s.exposure_input is not None
    assert s.exposure_input.n_assets == 3
    assert "exposure" in s.compute_all().by_tier()


def test_strategy_equity_prefers_strategy_returns():
    """The container routes strategy returns to the exposure tier so leverage
    derives equity from them before falling back to positions + asset returns."""
    rng = np.random.default_rng(11)
    r = rng.normal(0, 0.01, 30)
    pos = rng.normal(0.5, 0.1, (30, 3))
    aret = rng.normal(0, 0.01, (30, 3))

    # route 2: strategy returns win over positions + asset returns
    s = Strategy(returns=r, positions=pos, asset_returns=aret)
    assert s.exposure_input is not None
    assert s.exposure_input.equity_source == "strategy_returns"

    # route 3: no strategy returns, fall back to positions + asset returns
    s2 = Strategy(positions=pos, asset_returns=aret)
    assert s2.exposure_input is not None
    assert s2.exposure_input.equity_source == "positions"


def test_reconciliation_matches_silently():
    """Identical trade pnl and strategy returns reconcile without a warning."""
    r = np.array([0.1, -0.05, 0.2, 0.0])
    trades = {"pnl": r.copy()}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = Strategy(r, trades=trades)
    assert s.reconciliation is not None
    assert np.isclose(s.reconciliation["compounded_account_pnl"], np.prod(1 + r) - 1)
    assert np.isclose(s.reconciliation["equity_total_return"], np.prod(1 + r) - 1)
    assert s.reconciliation["converted_to_account_basis"] is False


def test_reconciliation_warns_on_mismatch():
    """A trade log that compounds to a different total than the returns warns."""
    r = np.array([0.1, 0.1, 0.1])
    trades = {"pnl": np.array([0.1, 0.1])}  # missing a trade
    with pytest.warns(UserWarning, match="does not reconcile"):
        s = Strategy(r, trades=trades)
    assert s.reconciliation is not None
    assert not np.isclose(
        s.reconciliation["compounded_account_pnl"],
        s.reconciliation["equity_total_return"],
    )


def test_reconciliation_applies_position_size():
    """position_size converts trade pnl to account basis before comparing."""
    r = np.array([0.05, -0.02])
    # Trade basis pnl alone compounds to 5.6%, off the 2.9% the returns imply.
    trades = {"pnl": np.array([0.10, -0.04]), "position_size": np.array([0.5, 0.5])}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = Strategy(r, trades=trades)
    assert s.reconciliation is not None
    assert s.reconciliation["converted_to_account_basis"] is True
    assert np.isclose(s.reconciliation["compounded_account_pnl"], np.prod(1 + r) - 1)
    assert np.isclose(s.reconciliation["equity_total_return"], np.prod(1 + r) - 1)


def test_reconciliation_skips_without_trades():
    """No trade log means no reconciliation is recorded."""
    r = np.array([0.1, -0.05, 0.2])
    s = Strategy(r)
    assert s.reconciliation is None


def test_drawdown_cache_computed_once():
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.01, 80)
    s = Strategy(r)
    assert s._drawdowns is None
    eq = s.equity_curve
    assert s._drawdowns is not None
    assert eq.shape == (81, 1)  # prepended 1.0 row
    assert eq[0, 0] == 1.0
    # Repeated access returns the cached object, not a recomputation.
    assert s.equity_curve is eq
    assert s.drawdown_series is s.drawdown_series
    assert s.drawdown_episodes is s.drawdown_episodes


def test_drawdown_requires_returns():
    s = Strategy()
    with pytest.raises(ValueError):
        _ = s.equity_curve


def test_strategy_report(tmp_path):
    pytest.importorskip("plotly")
    rng = np.random.default_rng(6)
    r = rng.normal(0, 0.01, 60)
    s = Strategy(r, periods_per_year=252)
    out = tmp_path / "report.html"
    s.report(str(out))
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def test_comparison_stacks_and_slices():
    rng = np.random.default_rng(7)
    r1 = rng.normal(0, 0.01, 100)
    r2 = rng.normal(0, 0.012, 100)
    c = Comparison({"a": r1, "b": r2})
    assert c.returns.shape == (100, 2)
    assert c.names == ["a", "b"]
    assert c.labels == ["a", "b"]
    assert len(c) == 2

    sa = c["a"]
    assert isinstance(sa, Strategy)
    assert sa.returns_input.values.shape == (100, 1)
    assert set(c.strategies) == {"a", "b"}


def test_comparison_from_array():
    rng = np.random.default_rng(8)
    m = rng.normal(0, 0.01, (50, 3))
    c = Comparison(m)
    assert c.returns.shape == (50, 3)
    assert c.names == ["s0", "s1", "s2"]


def test_comparison_labels_from_pandas():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(9)
    df = pd.DataFrame(rng.normal(0, 0.01, (40, 2)), columns=["x", "y"])
    c = Comparison(df)
    assert c.names == ["x", "y"]


def test_comparison_compute_all_runs_compare():
    rng = np.random.default_rng(10)
    r1 = rng.normal(0, 0.01, 100)
    r2 = rng.normal(0, 0.012, 100)
    c = Comparison({"a": r1, "b": r2})
    ms = c.compute_all()
    tiers = set(ms.by_tier())
    assert {"returns", "compare"} <= tiers


def test_comparison_rejects_misaligned():
    with pytest.raises(ValueError):
        Comparison({"a": np.zeros(10), "b": np.zeros(12)})


def test_comparison_empty_rejected():
    with pytest.raises(ValueError):
        Comparison({})
