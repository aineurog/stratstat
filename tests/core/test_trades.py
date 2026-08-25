"""Tests for trade-tier metrics.

Covers all 37 registered trade metrics, edge cases, input types,
and registry integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Import triggers @register_metric decorators
import stratstat.core.trades  # noqa: F401
from stratstat.inputs import TradeInput

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_returns():
    """Portfolio-level returns for TradeInput context."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 0.01, size=252).astype(np.float64)


@pytest.fixture
def simple_trades_dict():
    """10-trade log with pnl, side, and duration.

    Mix of wins, losses, longs, and shorts.
    """
    return {
        "pnl": [0.02, -0.01, 0.03, -0.015, 0.01, -0.02, 0.04, -0.005, 0.015, -0.01],
        "side": [
            "long", "short", "long", "short", "long",
            "short", "long", "long", "short", "short",
        ],
        "duration": [5, 3, 8, 2, 4, 6, 10, 3, 7, 4],
    }


@pytest.fixture
def win_only_trades():
    """Trades that are all winners."""
    return {
        "pnl": [0.02, 0.03, 0.01, 0.04, 0.015],
        "side": ["long", "long", "short", "long", "short"],
        "duration": [5, 8, 4, 10, 7],
    }


@pytest.fixture
def loss_only_trades():
    """Trades that are all losers."""
    return {
        "pnl": [-0.02, -0.03, -0.01, -0.04, -0.015],
        "side": ["long", "long", "short", "long", "short"],
        "duration": [5, 8, 4, 10, 7],
    }


@pytest.fixture
def trades_with_prices():
    """Trade log with fill/decision prices for implementation shortfall."""
    return {
        "pnl": [0.02, -0.01, 0.03, -0.015, 0.01],
        "side": ["long", "short", "long", "short", "long"],
        "fill_price": [101.0, 99.0, 102.5, 98.0, 100.5],
        "decision_price": [100.0, 100.0, 101.0, 99.0, 100.0],
    }


@pytest.fixture
def trades_with_intratrade():
    """Trade log with intra-trade price paths for MFE/MAE."""
    return {
        "pnl": [0.02, -0.01, 0.03, -0.015, 0.01],
        "side": ["long", "short", "long", "short", "long"],
        "intratrade_prices": [
            [100.0, 101.0, 102.5, 102.0],   # long: MFE=102.5-100=2.5
            [100.0, 99.0, 98.0, 99.5],       # short: MFE=100-98=2.0
            [100.0, 101.0, 103.0, 102.5],    # long: MFE=103-100=3.0
            [100.0, 99.5, 98.5, 99.0],       # short: MFE=100-98.5=1.5
            [100.0, 101.5, 100.5, 101.0],    # long: MFE=101.5-100=1.5
        ],
    }


@pytest.fixture
def inp_basic(simple_trades_dict, simple_returns):
    """TradeInput with returns + trades (pnl, side, duration)."""
    return TradeInput(returns=simple_returns, trades=simple_trades_dict)


@pytest.fixture
def inp_no_side(simple_trades_dict, simple_returns):
    """TradeInput with no side information."""
    trades = {"pnl": simple_trades_dict["pnl"]}
    return TradeInput(returns=simple_returns, trades=trades)


@pytest.fixture
def inp_no_duration(simple_trades_dict, simple_returns):
    """TradeInput with side but no duration."""
    trades = {"pnl": simple_trades_dict["pnl"], "side": simple_trades_dict["side"]}
    return TradeInput(returns=simple_returns, trades=trades)


@pytest.fixture
def inp_empty():
    """TradeInput with no trades."""
    return TradeInput(trades={"pnl": []})


# ===================================================================
# §7.1  Total Trades
# ===================================================================


class TestTotalTrades:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "total_trades")
        assert result.value == 10
        assert isinstance(result.value, float)

    def test_empty(self, inp_empty):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_empty, "total_trades")
        assert result.value == 0

    def test_pandas_input(self, simple_trades_dict):
        df = pd.DataFrame(simple_trades_dict)
        inp = TradeInput(trades=df)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "total_trades")
        assert result.value == 10

    def test_polars_input(self, simple_trades_dict):
        pl = pytest.importorskip("polars")
        df = pl.from_dict(simple_trades_dict)
        inp = TradeInput(trades=df)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "total_trades")
        assert result.value == 10

    def test_registry_auto_wrap(self, simple_trades_dict):
        from stratstat import compute

        result = compute(simple_trades_dict, "total_trades")
        assert result.value == 10


# ===================================================================
# §7.2  Win Rate (Overall)
# ===================================================================


class TestWinRate:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "win_rate")
        pnl = np.array(inp_basic.pnl)
        expected = np.sum(pnl > 0) / len(pnl)
        assert result.value == pytest.approx(expected)

    def test_all_wins(self, win_only_trades):
        inp = TradeInput(trades=win_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "win_rate")
        assert result.value == 1.0

    def test_all_losses(self, loss_only_trades):
        inp = TradeInput(trades=loss_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "win_rate")
        assert result.value == 0.0

    def test_empty(self, inp_empty):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_empty, "win_rate")
        assert np.isnan(result.value)

    def test_nan_in_pnl(self):
        inp = TradeInput(trades={"pnl": [0.02, np.nan, -0.01, 0.03]})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "win_rate")
        # NaN is neither win nor loss; 2 wins / 4 total = 0.5
        assert result.value == pytest.approx(0.5)


# ===================================================================
# §7.3  Win Rate (Long-Only)
# ===================================================================


class TestWinRateLong:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "win_rate_long")
        pnl = np.array(inp_basic.pnl)
        is_long = inp_basic.is_long
        assert is_long is not None
        long_pnl = pnl[is_long]
        expected = np.sum(long_pnl > 0) / len(long_pnl)
        assert result.value == pytest.approx(expected)

    def test_requires_side(self, inp_no_side):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="requires side"):
            _compute_one(inp_no_side, "win_rate_long")

    def test_no_long_trades_returns_nan(self):
        inp = TradeInput(
            trades={
                "pnl": [0.02, -0.01, 0.03],
                "side": ["short", "short", "short"],
            }
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "win_rate_long")
        assert np.isnan(result.value)

    def test_side_as_ints(self):
        inp = TradeInput(
            trades={
                "pnl": [0.02, -0.01, 0.03],
                "side": [1, -1, 1],
            }
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "win_rate_long")
        # long trades: idx 0 (win=0.02), idx 2 (win=0.03) → 2/2
        assert result.value == 1.0


# ===================================================================
# §7.4  Win Rate (Short-Only)
# ===================================================================


class TestWinRateShort:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "win_rate_short")
        pnl = np.array(inp_basic.pnl)
        is_long = inp_basic.is_long
        assert is_long is not None
        short_pnl = pnl[~is_long]
        expected = np.sum(short_pnl > 0) / len(short_pnl)
        assert result.value == pytest.approx(expected)

    def test_no_short_trades_returns_nan(self):
        inp = TradeInput(
            trades={
                "pnl": [0.02, -0.01, 0.03],
                "side": ["long", "long", "long"],
            }
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "win_rate_short")
        assert np.isnan(result.value)


# ===================================================================
# §7.5  Average Win
# ===================================================================


class TestAvgWin:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "avg_win")
        pnl = np.array(inp_basic.pnl)
        wins = pnl[pnl > 0]
        expected = np.mean(wins)
        assert result.value == pytest.approx(expected)

    def test_no_wins_returns_nan(self, loss_only_trades):
        inp = TradeInput(trades=loss_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "avg_win")
        assert np.isnan(result.value)

    def test_empty(self, inp_empty):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_empty, "avg_win")
        assert np.isnan(result.value)


# ===================================================================
# §7.6  Average Loss
# ===================================================================


class TestAvgLoss:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "avg_loss")
        pnl = np.array(inp_basic.pnl)
        losses = pnl[pnl < 0]
        expected = np.mean(losses)
        assert result.value == pytest.approx(expected)
        assert result.value < 0  # sign preserved

    def test_no_losses_returns_nan(self, win_only_trades):
        inp = TradeInput(trades=win_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "avg_loss")
        assert np.isnan(result.value)

    def test_meta_sign_preserved(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "avg_loss")
        assert result.meta["sign"] == "preserved (negative)"


# ===================================================================
# §7.7  Win/Loss Ratio
# ===================================================================


class TestWinLossRatio:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "win_loss_ratio")
        pnl = np.array(inp_basic.pnl)
        n_win = int(np.sum(pnl > 0))
        n_loss = int(np.sum(pnl < 0))
        assert result.value == pytest.approx(n_win / n_loss)

    def test_all_wins_returns_inf(self, win_only_trades):
        inp = TradeInput(trades=win_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "win_loss_ratio")
        assert result.value == np.inf

    def test_all_losses(self, loss_only_trades):
        inp = TradeInput(trades=loss_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "win_loss_ratio")
        assert result.value == 0.0

    def test_empty(self, inp_empty):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_empty, "win_loss_ratio")
        assert np.isnan(result.value)


# ===================================================================
# §7.8  Profit Factor
# ===================================================================


class TestProfitFactor:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "profit_factor")
        pnl = np.array(inp_basic.pnl)
        gp = np.sum(np.maximum(pnl, 0))
        gl = np.abs(np.sum(np.minimum(pnl, 0)))
        assert result.value == pytest.approx(gp / gl)

    def test_all_wins_returns_inf(self, win_only_trades):
        inp = TradeInput(trades=win_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "profit_factor")
        assert result.value == np.inf

    def test_all_losses(self, loss_only_trades):
        inp = TradeInput(trades=loss_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "profit_factor")
        assert result.value == 0.0

    def test_empty(self, inp_empty):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_empty, "profit_factor")
        assert np.isnan(result.value)


# ===================================================================
# §7.9  Expectancy per Trade
# ===================================================================


class TestExpectancy:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "expectancy")
        pnl = np.array(inp_basic.pnl)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        n = len(pnl)
        wr = len(wins) / n
        avg_w = np.mean(wins)
        avg_l = np.mean(losses)  # negative
        expected = wr * avg_w + (1 - wr) * avg_l
        assert result.value == pytest.approx(expected)

    def test_all_wins(self, win_only_trades):
        inp = TradeInput(trades=win_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "expectancy")
        pnl = np.array(win_only_trades["pnl"])
        assert result.value == pytest.approx(np.mean(pnl))

    def test_empty(self, inp_empty):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_empty, "expectancy")
        assert np.isnan(result.value)


# ===================================================================
# §7.10  Average Holding Period
# ===================================================================


class TestAvgHoldingPeriod:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "avg_holding_period")
        dur = np.array([5, 3, 8, 2, 4, 6, 10, 3, 7, 4], dtype=np.float64)
        assert result.value == pytest.approx(np.mean(dur))

    def test_requires_duration(self, inp_no_duration):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="requires duration"):
            _compute_one(inp_no_duration, "avg_holding_period")

    def test_empty(self, inp_empty):
        inp = TradeInput(trades={"pnl": [], "duration": []})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "avg_holding_period")
        assert np.isnan(result.value)

    def test_entry_exit_time_computes_duration(self):
        inp = TradeInput(
            trades={
                "pnl": [0.02, -0.01],
                "entry_time": [10.0, 20.0],
                "exit_time": [15.0, 23.0],
            }
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "avg_holding_period")
        assert result.value == pytest.approx(np.mean([5.0, 3.0]))


# ===================================================================
# §7.11  Holding Period Distribution
# ===================================================================


class TestHoldingPeriodDistribution:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "holding_period_distribution")
        dur = np.array([5, 3, 8, 2, 4, 6, 10, 3, 7, 4], dtype=np.float64)
        expected = np.array(
            [np.min(dur), np.percentile(dur, 25), np.percentile(dur, 50),
             np.percentile(dur, 75), np.max(dur)]
        )
        np.testing.assert_array_equal(result.value, expected)
        assert result.meta["output_index"] == ["min", "p25", "p50", "p75", "max"]


# ===================================================================
# §7.12  Max Consecutive Wins
# ===================================================================


class TestMaxConsecutiveWins:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "max_consecutive_wins")
        # pnl: [0.02, -0.01, 0.03, -0.015, 0.01, -0.02, 0.04, -0.005, 0.015, -0.01]
        # wins at: 0, 2, 4, 6, 8  → max consecutive = 1
        assert result.value == 1

    def test_longer_run(self):
        inp = TradeInput(
            trades={"pnl": [0.02, 0.03, 0.01, -0.01, 0.04, 0.05, 0.02]}
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "max_consecutive_wins")
        assert result.value == 3  # indices 4,5,6

    def test_all_wins(self, win_only_trades):
        inp = TradeInput(trades=win_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "max_consecutive_wins")
        assert result.value == 5

    def test_all_losses(self, loss_only_trades):
        inp = TradeInput(trades=loss_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "max_consecutive_wins")
        assert result.value == 0

    def test_empty(self, inp_empty):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_empty, "max_consecutive_wins")
        assert result.value == 0

    def test_ties_not_counted(self):
        # PnL == 0 is neither win nor loss
        inp = TradeInput(trades={"pnl": [0.02, 0.0, 0.03, 0.0, 0.01]})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "max_consecutive_wins")
        # The zeros break the streak, so each win is isolated: max = 1
        assert result.value == 1


# ===================================================================
# §7.13  Max Consecutive Losses
# ===================================================================


class TestMaxConsecutiveLosses:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "max_consecutive_losses")
        # losses at: 1, 3, 5, 7, 9 → max consecutive = 1
        assert result.value == 1

    def test_longer_run(self):
        inp = TradeInput(
            trades={"pnl": [0.02, -0.03, -0.01, 0.01, -0.04, -0.05, -0.02]}
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "max_consecutive_losses")
        assert result.value == 3  # indices 4,5,6

    def test_all_losses(self, loss_only_trades):
        inp = TradeInput(trades=loss_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "max_consecutive_losses")
        assert result.value == 5


# ===================================================================
# §7.14  Round-Trip P&L Distribution
# ===================================================================


class TestPnlDistribution:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "pnl_distribution")
        pnl = np.array(inp_basic.pnl, dtype=np.float64)
        # Check the first three entries exactly
        assert result.value[0] == pytest.approx(np.mean(pnl))  # mean
        assert result.value[1] == pytest.approx(np.median(pnl))  # median
        assert result.value[2] == pytest.approx(np.std(pnl, ddof=1))  # std
        assert result.meta["output_index"] == [
            "mean", "median", "std", "skewness", "p5", "p95",
        ]

    def test_empty(self, inp_empty):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_empty, "pnl_distribution")
        assert np.all(np.isnan(result.value))

    def test_single_trade(self):
        inp = TradeInput(trades={"pnl": [0.05]})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "pnl_distribution")
        assert result.value[0] == 0.05  # mean
        assert result.value[1] == 0.05  # median
        assert np.isnan(result.value[2])  # std (ddof=1, single obs)
        assert np.isnan(result.value[3])  # skewness (< 3 obs)


# ===================================================================
# §7.15  Implementation Shortfall
# ===================================================================


class TestImplementationShortfall:
    def test_known_value(self, trades_with_prices):
        inp = TradeInput(trades=trades_with_prices)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "implementation_shortfall")
        # Hand-computed:
        # trade 0: long, +1 * (101-100)/100 = 0.01
        # trade 1: short, -1 * (99-100)/100 = 0.01
        # trade 2: long, +1 * (102.5-101)/101 = 0.014851...
        # trade 3: short, -1 * (98-99)/99 = 0.010101...
        # trade 4: long, +1 * (100.5-100)/100 = 0.005
        expected = np.array([0.01, 0.01, 1.5 / 101, 1.0 / 99, 0.005])
        assert result.value[0] == pytest.approx(np.mean(expected))  # mean

    def test_requires_prices(self, inp_basic):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="requires fill_price"):
            _compute_one(inp_basic, "implementation_shortfall")

    def test_requires_side(self):
        inp = TradeInput(
            trades={
                "pnl": [0.02, -0.01],
                "fill_price": [101.0, 99.0],
                "decision_price": [100.0, 100.0],
            }
        )
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="requires side"):
            _compute_one(inp, "implementation_shortfall")

    def test_output_index(self, trades_with_prices):
        inp = TradeInput(trades=trades_with_prices)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "implementation_shortfall")
        assert result.meta["output_index"] == ["mean", "std", "min", "max"]


# ===================================================================
# §7.16  Best Trade
# ===================================================================


class TestBestTrade:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "best_trade")
        assert result.value == 0.04

    def test_empty(self, inp_empty):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_empty, "best_trade")
        assert np.isnan(result.value)

    def test_all_negative(self, loss_only_trades):
        inp = TradeInput(trades=loss_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "best_trade")
        assert result.value == -0.01


# ===================================================================
# §7.17  Worst Trade
# ===================================================================


class TestWorstTrade:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "worst_trade")
        assert result.value == -0.02

    def test_empty(self, inp_empty):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_empty, "worst_trade")
        assert np.isnan(result.value)


# ===================================================================
# §7.18  Avg Winning Trade Duration
# ===================================================================


class TestAvgWinningDuration:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "avg_winning_duration")
        pnl = np.array(inp_basic.pnl)
        dur = np.array(inp_basic.duration)
        assert dur is not None
        win_dur = dur[pnl > 0]
        assert result.value == pytest.approx(np.mean(win_dur))

    def test_no_wins_returns_nan(self, loss_only_trades):
        inp = TradeInput(trades=loss_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "avg_winning_duration")
        assert np.isnan(result.value)


# ===================================================================
# §7.19  Avg Losing Trade Duration
# ===================================================================


class TestAvgLosingDuration:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "avg_losing_duration")
        pnl = np.array(inp_basic.pnl)
        dur = np.array(inp_basic.duration)
        assert dur is not None
        loss_dur = dur[pnl < 0]
        assert result.value == pytest.approx(np.mean(loss_dur))

    def test_no_losses_returns_nan(self, win_only_trades):
        inp = TradeInput(trades=win_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "avg_losing_duration")
        assert np.isnan(result.value)


# ===================================================================
# §7.20  Payoff Ratio
# ===================================================================


class TestPayoffRatio:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "payoff_ratio")
        pnl = np.array(inp_basic.pnl)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        expected = np.mean(wins) / np.abs(np.mean(losses))
        assert result.value == pytest.approx(expected)

    def test_all_wins_returns_inf(self, win_only_trades):
        inp = TradeInput(trades=win_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "payoff_ratio")
        assert result.value == np.inf

    def test_all_losses(self, loss_only_trades):
        inp = TradeInput(trades=loss_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "payoff_ratio")
        assert np.isnan(result.value)


# ===================================================================
# §7.21  CPC Ratio
# ===================================================================


class TestCpcRatio:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "cpc_ratio")
        # PF = sum(wins)/|sum(losses)| = 0.115/0.06 = 23/12
        # Payoff = avg_win/|avg_loss| = 0.023/0.012 = 23/12
        # WR = 5/10 = 0.5
        # CPC = (23/12) * (23/12) * 0.5 = 529/288
        expected = (23.0 / 12.0) * (23.0 / 12.0) * 0.5
        assert result.value == pytest.approx(expected)

    def test_empty(self, inp_empty):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_empty, "cpc_ratio")
        assert np.isnan(result.value)


# ===================================================================
# §7.22  SQN (System Quality Number)
# ===================================================================


class TestSqn:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "sqn")
        pnl = np.array(inp_basic.pnl)
        mu = np.mean(pnl)
        sigma = np.std(pnl, ddof=1)
        n = len(pnl)
        expected = (mu / sigma) * np.sqrt(n)
        assert result.value == pytest.approx(expected)

    def test_single_trade(self):
        inp = TradeInput(trades={"pnl": [0.05]})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "sqn")
        assert np.isnan(result.value)

    def test_constant_positive(self):
        inp = TradeInput(trades={"pnl": [0.02, 0.02, 0.02]})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "sqn")
        assert result.value == np.inf  # positive mean, zero std

    def test_constant_negative(self):
        inp = TradeInput(trades={"pnl": [-0.02, -0.02, -0.02]})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "sqn")
        assert result.value == -np.inf


# ===================================================================
# §7.23  Trade Duration Std
# ===================================================================


class TestTradeDurationStd:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "trade_duration_std")
        dur = np.array([5, 3, 8, 2, 4, 6, 10, 3, 7, 4], dtype=np.float64)
        assert result.value == pytest.approx(np.std(dur, ddof=1))

    def test_single_trade(self):
        inp = TradeInput(trades={"pnl": [0.05], "duration": [5]})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "trade_duration_std")
        assert np.isnan(result.value)


# ===================================================================
# §7.24  Trade Return Std
# ===================================================================


class TestTradeReturnStd:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "trade_return_std")
        pnl = np.array(inp_basic.pnl)
        assert result.value == pytest.approx(np.std(pnl, ddof=1))

    def test_single_trade(self):
        inp = TradeInput(trades={"pnl": [0.05]})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "trade_return_std")
        assert np.isnan(result.value)


# ===================================================================
# §7.25  Geometric Mean Return (per Trade)
# ===================================================================


class TestGeometricMeanReturnPerTrade:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "geometric_mean_return_per_trade")
        pnl = np.array(inp_basic.pnl)
        log_vals = np.log(1.0 + pnl)
        expected = np.exp(np.mean(log_vals)) - 1.0
        assert result.value == pytest.approx(expected)

    def test_large_loss_ignored(self):
        # PnL <= -1.0 makes log(1+PnL) undefined — these are filtered out
        inp = TradeInput(trades={"pnl": [0.02, -1.5, 0.03, -2.0]})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "geometric_mean_return_per_trade")
        # Only 0.02 and 0.03 are valid
        valid = np.array([0.02, 0.03])
        log_vals = np.log(1.0 + valid)
        expected = np.exp(np.mean(log_vals)) - 1.0
        assert result.value == pytest.approx(expected)

    def test_all_100pct_losses(self):
        inp = TradeInput(trades={"pnl": [-1.5, -2.0]})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "geometric_mean_return_per_trade")
        assert np.isnan(result.value)


# ===================================================================
# §7.26  Outlier Win Ratio
# ===================================================================


class TestOutlierWinRatio:
    def test_no_outliers(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "outlier_win_ratio")
        assert isinstance(result.value, float)
        # All wins are fairly close — no outliers expected
        assert result.value == 0.0

    def test_clear_outlier(self):
        trades = {"pnl": [0.01, 0.015, 0.02, 0.01, 0.5]}
        inp = TradeInput(trades=trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "outlier_win_ratio")
        assert result.value == 0.2  # 1 out of 5

    def test_few_wins(self):
        inp = TradeInput(trades={"pnl": [0.02, -0.01, -0.02]})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "outlier_win_ratio")
        assert np.isnan(result.value)


# ===================================================================
# §7.27  Outlier Loss Ratio
# ===================================================================


class TestOutlierLossRatio:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "outlier_loss_ratio")
        assert isinstance(result.value, float)

    def test_clear_outlier(self):
        trades = {"pnl": [-0.01, -0.015, -0.02, -0.01, -0.5]}
        inp = TradeInput(trades=trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "outlier_loss_ratio")
        assert result.value == 0.2  # 1 out of 5

    def test_few_losses(self):
        inp = TradeInput(trades={"pnl": [-0.01, 0.02, 0.03]})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "outlier_loss_ratio")
        assert np.isnan(result.value)


# ===================================================================
# §7.28  MFE (Maximum Favorable Excursion)
# ===================================================================


class TestMfe:
    def test_known_value(self, trades_with_intratrade):
        inp = TradeInput(trades=trades_with_intratrade)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "mfe")
        # Hand-computed dollar excursions:
        # Trade 0 (long): entry=100, max=102.5 → 102.5-100 = 2.5
        # Trade 1 (short): entry=100, min=98 → 100-98 = 2.0
        # Trade 2 (long): entry=100, max=103 → 103-100 = 3.0
        # Trade 3 (short): entry=100, min=98.5 → 100-98.5 = 1.5
        # Trade 4 (long): entry=100, max=101.5 → 101.5-100 = 1.5
        expected_mfes = np.array([2.5, 2.0, 3.0, 1.5, 1.5])
        assert result.value[0] == pytest.approx(np.mean(expected_mfes))  # mean
        assert result.value[1] == pytest.approx(np.max(expected_mfes))  # max
        assert result.value[2] == pytest.approx(np.min(expected_mfes))  # min

    def test_requires_intratrade(self, inp_basic):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="requires intratrade_prices"):
            _compute_one(inp_basic, "mfe")

    def test_empty(self, inp_empty):
        inp = TradeInput(
            trades={"pnl": [], "side": [], "intratrade_prices": []}
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "mfe")
        assert np.all(np.isnan(result.value))

    def test_output_index(self, trades_with_intratrade):
        inp = TradeInput(trades=trades_with_intratrade)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "mfe")
        assert result.meta["output_index"] == ["mean", "max", "min"]


# ===================================================================
# §7.29  MAE (Maximum Adverse Excursion)
# ===================================================================


class TestMae:
    def test_known_value(self, trades_with_intratrade):
        inp = TradeInput(trades=trades_with_intratrade)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "mae")
        # All paths start at entry and only go favorably:
        # Trade 0 (long): min=100 → MAE=0.0
        # Trade 1 (short): max=100 → MAE=0.0
        # Trade 2 (long): min=100 → MAE=0.0
        # Trade 3 (short): max=100 → MAE=0.0
        # Trade 4 (long): min=100 → MAE=0.0
        assert result.value[0] == 0.0  # mean
        assert result.value[1] == 0.0  # max
        assert result.value[2] == 0.0  # min

    def test_with_adverse_move(self):
        # Long trade with drawdown
        inp = TradeInput(
            trades={
                "pnl": [0.02],
                "side": ["long"],
                "intratrade_prices": [[100.0, 99.0, 95.0, 102.0]],
            }
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "mae")
        # Long: MAE = entry - min = 100 - 95 = 5.0
        assert result.value[0] == pytest.approx(5.0)

    def test_requires_intratrade(self, inp_basic):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="requires intratrade_prices"):
            _compute_one(inp_basic, "mae")


# ===================================================================
# §7.30  Kelly Criterion
# ===================================================================


class TestKellyCriterion:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "kelly_criterion")
        pnl = np.array(inp_basic.pnl)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        w = len(wins) / len(pnl)
        avg_w = np.mean(wins)
        avg_l_abs = np.abs(np.mean(losses))
        payoff = avg_w / avg_l_abs
        expected = w - (1 - w) / payoff
        assert result.value == pytest.approx(expected)

    def test_all_wins(self, win_only_trades):
        inp = TradeInput(trades=win_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "kelly_criterion")
        assert result.value == 1.0  # full Kelly

    def test_all_losses(self, loss_only_trades):
        inp = TradeInput(trades=loss_only_trades)
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "kelly_criterion")
        assert result.value == 0.0  # don't bet

    def test_empty(self, inp_empty):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_empty, "kelly_criterion")
        assert np.isnan(result.value)


# ===================================================================
# §7.31  Long/Short Trade Count
# ===================================================================


class TestLongShortTradeCount:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "long_short_trade_count")
        is_long = inp_basic.is_long
        assert is_long is not None
        n_long = int(np.sum(is_long))
        n_short = int(np.sum(~is_long))
        np.testing.assert_array_equal(result.value, [n_long, n_short])

    def test_output_index(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "long_short_trade_count")
        assert result.meta["output_index"] == ["long", "short"]


# ===================================================================
# §7.32  Long/Short Trade %
# ===================================================================


class TestLongShortTradePct:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "long_short_trade_pct")
        is_long = inp_basic.is_long
        assert is_long is not None
        expected_long = np.mean(is_long)
        expected_short = np.mean(~is_long)
        np.testing.assert_array_almost_equal(
            result.value, [expected_long, expected_short]
        )

    def test_empty(self):
        inp = TradeInput(trades={"pnl": [], "side": []})
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "long_short_trade_pct")
        assert np.all(np.isnan(result.value))


# ===================================================================
# §7.33  Long/Short Winning/Losing Trades
# ===================================================================


class TestLongShortWinningLosing:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "long_short_winning_losing")
        pnl = np.array(inp_basic.pnl)
        is_long = inp_basic.is_long
        assert is_long is not None
        n_long_win = int(np.sum(is_long & (pnl > 0)))
        n_long_loss = int(np.sum(is_long & (pnl < 0)))
        n_short_win = int(np.sum(~is_long & (pnl > 0)))
        n_short_loss = int(np.sum(~is_long & (pnl < 0)))
        np.testing.assert_array_equal(
            result.value, [n_long_win, n_long_loss, n_short_win, n_short_loss]
        )

    def test_output_index(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "long_short_winning_losing")
        assert result.meta["output_index"] == [
            "long_win", "long_loss", "short_win", "short_loss",
        ]


# ===================================================================
# §7.34  Long/Short Avg Duration
# ===================================================================


class TestLongShortAvgDuration:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "long_short_avg_duration")
        dur = np.array(inp_basic.duration)
        is_long = inp_basic.is_long
        assert dur is not None
        assert is_long is not None
        long_dur = dur[is_long]
        short_dur = dur[~is_long]
        expected = [np.mean(long_dur), np.mean(short_dur)]
        np.testing.assert_array_almost_equal(result.value, expected)

    def test_requires_duration(self, inp_no_duration):
        from stratstat.registry import _compute_one

        with pytest.raises(ValueError, match="requires duration"):
            _compute_one(inp_no_duration, "long_short_avg_duration")


# ===================================================================
# §7.35  Long/Short Total PnL %
# ===================================================================


class TestLongShortTotalPnl:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "long_short_total_pnl")
        pnl = np.array(inp_basic.pnl)
        is_long = inp_basic.is_long
        assert is_long is not None
        long_total = np.sum(pnl[is_long])
        short_total = np.sum(pnl[~is_long])
        np.testing.assert_array_almost_equal(
            result.value, [long_total, short_total]
        )


# ===================================================================
# §7.36  Long/Short Avg PnL %
# ===================================================================


class TestLongShortAvgPnl:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "long_short_avg_pnl")
        pnl = np.array(inp_basic.pnl)
        is_long = inp_basic.is_long
        assert is_long is not None
        avg_long = np.mean(pnl[is_long])
        avg_short = np.mean(pnl[~is_long])
        np.testing.assert_array_almost_equal(
            result.value, [avg_long, avg_short]
        )

    def test_no_long_returns_nan(self):
        inp = TradeInput(
            trades={
                "pnl": [0.02, -0.01, 0.03],
                "side": ["short", "short", "short"],
            }
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "long_short_avg_pnl")
        assert np.isnan(result.value[0])  # long avg
        assert not np.isnan(result.value[1])  # short avg


# ===================================================================
# §7.37  Long/Short Best/Worst Trade %
# ===================================================================


class TestLongShortBestWorst:
    def test_known_value(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "long_short_best_worst")
        pnl = np.array(inp_basic.pnl)
        is_long = inp_basic.is_long
        assert is_long is not None
        long_pnl = pnl[is_long]
        short_pnl = pnl[~is_long]
        expected = [
            np.max(long_pnl), np.min(long_pnl),
            np.max(short_pnl), np.min(short_pnl),
        ]
        np.testing.assert_array_equal(result.value, expected)

    def test_output_index(self, inp_basic):
        from stratstat.registry import _compute_one

        result = _compute_one(inp_basic, "long_short_best_worst")
        assert result.meta["output_index"] == [
            "best_long", "worst_long", "best_short", "worst_short",
        ]

    def test_no_long_returns_nan(self):
        inp = TradeInput(
            trades={
                "pnl": [0.02, -0.01, 0.03],
                "side": ["short", "short", "short"],
            }
        )
        from stratstat.registry import _compute_one

        result = _compute_one(inp, "long_short_best_worst")
        assert np.isnan(result.value[0])  # best_long
        assert np.isnan(result.value[1])  # worst_long
        assert not np.isnan(result.value[2])  # best_short


# ===================================================================
# TradeInput feature tests
# ===================================================================


class TestTradeInputFeatures:
    def test_auto_creates_from_trades_only(self, simple_trades_dict):
        inp = TradeInput(trades=simple_trades_dict)
        assert inp.n_trades == 10
        assert inp.returns is None
        assert inp.n_periods == 0

    def test_has_side_true(self, inp_basic):
        assert inp_basic.has_side is True

    def test_has_side_false(self, inp_no_side):
        assert inp_no_side.has_side is False

    def test_has_duration_true(self, inp_basic):
        assert inp_basic.has_duration is True

    def test_has_duration_false(self, inp_no_duration):
        assert inp_no_duration.has_duration is False

    def test_has_prices(self, trades_with_prices):
        inp = TradeInput(trades=trades_with_prices)
        assert inp.has_prices is True

    def test_has_intratrade(self, trades_with_intratrade):
        inp = TradeInput(trades=trades_with_intratrade)
        assert inp.has_intratrade is True

    def test_intratrade_2d_rows_per_trade(self):
        # A 2D array should be split so each row is one trade's price path.
        itp_2d = np.array(
            [[100.0, 101.0, 102.0], [100.0, 99.0, 98.0], [100.0, 101.0, 103.0]]
        )
        inp = TradeInput(
            trades={"pnl": [0.01, -0.01, 0.02], "intratrade_prices": itp_2d}
        )
        paths = inp.intratrade_prices
        assert paths is not None
        assert len(paths) == 3
        np.testing.assert_array_equal(paths[1], [100.0, 99.0, 98.0])

    def test_intratrade_1d_single_path(self):
        # A 1D array should become a single price path.
        itp_1d = np.array([100.0, 101.0, 102.0, 101.5])
        inp = TradeInput(
            trades={"pnl": [0.01], "intratrade_prices": itp_1d}
        )
        paths = inp.intratrade_prices
        assert paths is not None
        assert len(paths) == 1
        np.testing.assert_array_equal(paths[0], itp_1d)

    def test_repr(self, inp_basic):
        r = repr(inp_basic)
        assert "TradeInput" in r
        assert "n_trades=10" in r

    def test_side_normalize_strings(self, simple_trades_dict):
        inp = TradeInput(trades=simple_trades_dict)
        is_long = inp.is_long
        assert is_long is not None
        expected = np.array(
            [True, False, True, False, True, False, True, True, False, False]
        )
        np.testing.assert_array_equal(is_long, expected)

    def test_side_normalize_ints(self):
        inp = TradeInput(
            trades={"pnl": [0.02, -0.01, 0.03], "side": [+1, -1, +1]}
        )
        is_long = inp.is_long
        assert is_long is not None
        np.testing.assert_array_equal(is_long, [True, False, True])

    def test_side_normalize_bools(self):
        inp = TradeInput(
            trades={"pnl": [0.02, -0.01, 0.03], "side": [True, False, True]}
        )
        is_long = inp.is_long
        assert is_long is not None
        np.testing.assert_array_equal(is_long, [True, False, True])

    def test_side_normalize_case_insensitive(self):
        inp = TradeInput(
            trades={
                "pnl": [0.02, -0.01, 0.03],
                "side": ["LONG", "Short", "long"],
            }
        )
        is_long = inp.is_long
        assert is_long is not None
        np.testing.assert_array_equal(is_long, [True, False, True])

    def test_missing_pnl_raises(self):
        with pytest.raises(ValueError, match="must contain a 'pnl' column"):
            TradeInput(trades={"side": ["long", "short"]})

    def test_bad_trades_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported trades type"):
            TradeInput(trades=[1, 2, 3])

    def test_pandas_input_with_side(self):
        df = pd.DataFrame(
            {"pnl": [0.02, -0.01, 0.03], "side": ["long", "short", "long"]}
        )
        inp = TradeInput(trades=df)
        assert inp.n_trades == 3
        assert inp.has_side is True

    def test_polars_input_with_side(self):
        pl = pytest.importorskip("polars")
        df = pl.from_dict(
            {"pnl": [0.02, -0.01, 0.03], "side": ["long", "short", "long"]}
        )
        inp = TradeInput(trades=df)
        assert inp.n_trades == 3
        assert inp.has_side is True

    def test_returns_optional_with_none(self):
        inp = TradeInput(trades={"pnl": [0.02, -0.01]})
        assert inp.returns is None

    def test_returns_stored_when_provided(self, simple_returns, simple_trades_dict):
        inp = TradeInput(returns=simple_returns, trades=simple_trades_dict)
        assert inp.returns is not None
        assert inp.n_periods == 252

    def test_positions_stored(self):
        positions = np.random.default_rng(0).normal(size=(10, 5))
        inp = TradeInput(
            trades={"pnl": [0.02] * 10},
            positions=positions,
        )
        assert inp.positions is not None
        assert inp.positions.shape == (10, 5)


# ===================================================================
# Registry integration tests
# ===================================================================


class TestRegistryIntegration:
    def test_all_trades_metrics_registered(self):
        from stratstat.registry import list_metrics

        metrics = list_metrics(requires="trades")
        names = {m["name"] for m in metrics}
        assert len(names) == 37

    def test_compute_single_auto_wrap(self, simple_trades_dict):
        from stratstat import compute

        result = compute(simple_trades_dict, "total_trades")
        assert result.name == "total_trades"
        assert result.value == 10

    def test_compute_all_pnl_category(self, simple_trades_dict):
        from stratstat import compute_all

        results = compute_all(trades=simple_trades_dict)
        names = {r.name for r in results}
        assert "avg_win" in names
        assert "avg_loss" in names
        assert "profit_factor" in names
        assert "expectancy" in names

    def test_compute_all_breakdown_category(self, simple_trades_dict):
        """compute_all(trades=...) includes long/short breakdown metrics."""
        from stratstat import compute_all

        results = compute_all(trades=simple_trades_dict)
        names = {r.name for r in results}
        assert "long_short_trade_count" in names
        assert "long_short_trade_pct" in names
        assert "long_short_winning_losing" in names
        assert "long_short_best_worst" in names

    def test_unknown_metric_raises(self, simple_trades_dict):
        from stratstat import compute
        from stratstat.exceptions import UnknownMetricError

        with pytest.raises(UnknownMetricError):
            compute(simple_trades_dict, "nonexistent_metric")

    def test_compute_with_tradeinput_instance(self, inp_basic):
        from stratstat import compute

        result = compute(inp_basic, "win_rate")
        assert result.name == "win_rate"
