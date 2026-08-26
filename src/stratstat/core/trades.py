"""Trade-tier metrics.

Metrics requiring returns + trade/transaction log: win rate, profit factor,
expectancy, average win/loss, max consecutive wins/losses, MFE/MAE, Kelly
criterion, and long/short breakdowns.

All tagged: category varies, requires="trades".
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from stratstat.core._utils import sample_skewness
from stratstat.exceptions import MetricNotApplicableError
from stratstat.inputs import TradeInput
from stratstat.registry import register_metric
from stratstat.results import MetricResult

# ---------------------------------------------------------------------------
# Citation strings
# ---------------------------------------------------------------------------

_REF_SCHWAGER = "Schwager (1995), Schwager on Futures: Technical Analysis, Wiley, Ch. 38."
_REF_THARP = "Tharp (1998), Trade Your Way to Financial Freedom, McGraw-Hill, Ch. 5."
_REF_THARP_CH7 = "Tharp (1998), Trade Your Way to Financial Freedom, McGraw-Hill, Ch. 7."
_REF_HYNDMAN = "Hyndman & Fan (1996); standard order statistics."
_REF_FISHER = "Fisher (1925); standard statistic."
_REF_TUKEY = "Tukey (1977); outlier criterion based on IQR."
_REF_CAMPBELL = (
    "Campbell, Lo & MacKinlay (1997), The Econometrics of Financial Markets, "
    "Princeton University Press, §1.4."
)
_REF_PEROLD = 'Perold (1988), "The Implementation Shortfall: Paper versus Reality," JPM, 14(3).'
_REF_SWEENEY = 'Sweeney (1988), "The Maximum Favorable Excursion Methodology."'
_REF_KELLY = (
    'Kelly (1956), "A New Interpretation of Information Rate," '
    "Bell System Technical Journal, 35(4). "
    'Thorp (1997), "The Kelly Criterion in Blackjack, Sports Betting, '
    'and the Stock Market."'
)
_REF_YOUNG = "Young (1991), CPC Index; see also Schwager (1995, Ch. 38)."


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


# Mapping from field names to (has_attr, display_name) for validation.
_FIELD_CHECKS: dict[str, tuple[str, str]] = {
    "side": ("has_side", "side"),
    "duration": ("has_duration", "duration (or entry_time/exit_time)"),
    "fill_price": ("has_prices", "fill_price, decision_price"),
    "price_path": ("has_price_path", "price_path (or intratrade_prices)"),
}


def _require_field(inp: TradeInput, field: str, metric_name: str) -> None:
    """Check that *inp* has *field*; raise ``ValueError`` if not.

    Used by metrics that need optional trade-log columns (``side``,
    ``duration``, ``fill_price``, ``price_path``).
    """
    if field not in _FIELD_CHECKS:
        raise ValueError(
            f"Unknown field {field!r} requested by {metric_name}. "
            f"Known fields: {list(_FIELD_CHECKS)}."
        )
    attr, display = _FIELD_CHECKS[field]
    if not getattr(inp, attr):
        raise MetricNotApplicableError(
            f"{metric_name} requires {display}. "
            f"Provide the required field(s) in the trade log "
            f"passed to TradeInput."
        )


def _win_mask(pnl: NDArray[np.floating]) -> NDArray[np.bool_]:
    """Boolean mask for winning trades (PnL > 0)."""
    return pnl > 0.0


def _loss_mask(pnl: NDArray[np.floating]) -> NDArray[np.bool_]:
    """Boolean mask for losing trades (PnL < 0)."""
    return pnl < 0.0


def _max_consecutive(mask: NDArray[np.bool_]) -> int:
    """Longest run of consecutive ``True`` values in *mask*.

    Returns 0 if no ``True`` values are present.
    """
    if len(mask) == 0:
        return 0
    # Pad with False at both ends to detect runs at boundaries.
    padded = np.concatenate([np.array([False]), mask, np.array([False])])
    diffs: NDArray[np.intp] = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))


# ===================================================================
# §7.1  Total Trades
# ===================================================================


@register_metric(
    name="total_trades",
    requires="trades",
    category=("trades",),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def total_trades(inp: TradeInput) -> MetricResult:
    r"""Total number of round-trip trades.

    .. math::
        N = \text{number of round-trip trades}
    """
    return MetricResult(
        name="total_trades",
        value=float(inp.n_trades),
        category=("trades",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER},
    )


# ===================================================================
# §7.2  Win Rate (Overall)
# ===================================================================


@register_metric(
    name="win_rate",
    requires="trades",
    category=("trades",),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def win_rate(inp: TradeInput) -> MetricResult:
    r"""Fraction of trades with positive P&L.

    .. math::
        \text{WR} = \frac{N_{\text{win}}}{N}
    """
    pnl = inp.pnl
    n = len(pnl)
    if n == 0:
        value: float = np.nan
    else:
        value = float(np.sum(_win_mask(pnl)) / n)
    return MetricResult(
        name="win_rate",
        value=value,
        category=("trades",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER},
    )


# ===================================================================
# §7.3  Win Rate (Long-Only)
# ===================================================================


@register_metric(
    name="win_rate_long",
    requires="trades",
    category=("trades",),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def win_rate_long(inp: TradeInput) -> MetricResult:
    r"""Win rate restricted to long trades.

    .. math::
        \text{WR}_{\text{long}} = \frac{N_{\text{long, win}}}
        {N_{\text{long}}}
    """
    _require_field(inp, "side", "win_rate_long")
    pnl = inp.pnl
    is_long = inp.is_long
    assert is_long is not None
    mask = is_long
    n_long = int(np.sum(mask))
    if n_long == 0:
        value: float = np.nan
    else:
        value = float(np.sum(_win_mask(pnl) & mask) / n_long)
    return MetricResult(
        name="win_rate_long",
        value=value,
        category=("trades",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER},
    )


# ===================================================================
# §7.4  Win Rate (Short-Only)
# ===================================================================


@register_metric(
    name="win_rate_short",
    requires="trades",
    category=("trades",),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def win_rate_short(inp: TradeInput) -> MetricResult:
    r"""Win rate restricted to short trades.

    .. math::
        \text{WR}_{\text{short}} = \frac{N_{\text{short, win}}}
        {N_{\text{short}}}
    """
    _require_field(inp, "side", "win_rate_short")
    pnl = inp.pnl
    is_long = inp.is_long
    assert is_long is not None
    mask = ~is_long
    n_short = int(np.sum(mask))
    if n_short == 0:
        value: float = np.nan
    else:
        value = float(np.sum(_win_mask(pnl) & mask) / n_short)
    return MetricResult(
        name="win_rate_short",
        value=value,
        category=("trades",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER},
    )


# ===================================================================
# §7.5  Average Win
# ===================================================================


@register_metric(
    name="avg_win",
    requires="trades",
    category=("trades", "pnl"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def avg_win(inp: TradeInput) -> MetricResult:
    r"""Average P&L of winning trades.

    .. math::
        \bar{W} = \frac{1}{N_{\text{win}}}
        \sum_{j:\text{win}} \text{PnL}_j
    """
    pnl = inp.pnl
    wins = pnl[_win_mask(pnl)]
    if len(wins) == 0:
        value: float = np.nan
    else:
        value = float(np.mean(wins))
    return MetricResult(
        name="avg_win",
        value=value,
        category=("trades", "pnl"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER},
    )


# ===================================================================
# §7.6  Average Loss
# ===================================================================


@register_metric(
    name="avg_loss",
    requires="trades",
    category=("trades", "pnl"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def avg_loss(inp: TradeInput) -> MetricResult:
    r"""Average P&L of losing trades (sign preserved; negative value).

    .. math::
        \bar{L} = \frac{1}{N_{\text{loss}}}
        \sum_{j:\text{loss}} \text{PnL}_j
    """
    pnl = inp.pnl
    losses = pnl[_loss_mask(pnl)]
    if len(losses) == 0:
        value: float = np.nan
    else:
        value = float(np.mean(losses))
    return MetricResult(
        name="avg_loss",
        value=value,
        category=("trades", "pnl"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER, "sign": "preserved (negative)"},
    )


# ===================================================================
# §7.7  Win/Loss Ratio
# ===================================================================


@register_metric(
    name="win_loss_ratio",
    requires="trades",
    category=("trades",),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def win_loss_ratio(inp: TradeInput) -> MetricResult:
    r"""Ratio of winning trades to losing trades.

    .. math::
        \text{WLR} = \frac{N_{\text{win}}}{N_{\text{loss}}}
    """
    pnl = inp.pnl
    n_win = int(np.sum(_win_mask(pnl)))
    n_loss = int(np.sum(_loss_mask(pnl)))
    if n_loss == 0:
        value: float = np.inf if n_win > 0 else np.nan
    else:
        value = float(n_win / n_loss)
    return MetricResult(
        name="win_loss_ratio",
        value=value,
        category=("trades",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER},
    )


# ===================================================================
# §7.8  Profit Factor
# ===================================================================


@register_metric(
    name="profit_factor",
    requires="trades",
    category=("trades", "pnl"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def profit_factor(inp: TradeInput) -> MetricResult:
    r"""Gross profit divided by gross loss (absolute value).

    .. math::
        \text{PF} = \frac{\sum_{j}\max(\text{PnL}_j, 0)}
        {|\sum_{j}\min(\text{PnL}_j, 0)|}

    Defined on account basis: when ``pnl_basis`` is ``"trade"`` and a
    ``position_size`` column is present, each trade's pnl is converted to
    account basis first, and ``meta["converted"]`` records that the
    conversion ran.
    """
    pnl, converted = inp.pnl_account_basis()
    gross_profit = np.nansum(np.maximum(pnl, 0.0))
    gross_loss = np.abs(np.nansum(np.minimum(pnl, 0.0)))
    if gross_loss == 0.0:
        value: float = np.inf if gross_profit > 0.0 else np.nan
    else:
        value = float(gross_profit / gross_loss)
    return MetricResult(
        name="profit_factor",
        value=value,
        category=("trades", "pnl"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER, "pnl_basis": inp.pnl_basis, "converted": converted},
    )


# ===================================================================
# §7.9  Expectancy per Trade
# ===================================================================


@register_metric(
    name="expectancy",
    requires="trades",
    category=("trades", "pnl"),
    backend="vectorized",
    ref=_REF_THARP,
)
def expectancy(inp: TradeInput) -> MetricResult:
    r"""Expected P&L per trade (expectancy).

    .. math::
        \mathbb{E}[\text{PnL}] = \text{WR} \cdot \bar{W} +
        (1 - \text{WR}) \cdot \bar{L}

    with :math:`\bar{L}` as a negative number.  Defined on account basis:
    when ``pnl_basis`` is ``"trade"`` and a ``position_size`` column is
    present, each trade's pnl is converted to account basis first, and
    ``meta["converted"]`` records that the conversion ran.
    """
    pnl, converted = inp.pnl_account_basis()
    wins = pnl[_win_mask(pnl)]
    losses = pnl[_loss_mask(pnl)]
    n = len(pnl)
    if n == 0:
        value: float = np.nan
    else:
        wr = len(wins) / n
        avg_w = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_l = float(np.mean(losses)) if len(losses) > 0 else 0.0
        value = wr * avg_w + (1.0 - wr) * avg_l
    return MetricResult(
        name="expectancy",
        value=value,
        category=("trades", "pnl"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_THARP, "pnl_basis": inp.pnl_basis, "converted": converted},
    )


# ===================================================================
# §7.10  Average Holding Period
# ===================================================================


@register_metric(
    name="avg_holding_period",
    requires="trades",
    category=("trades", "duration"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def avg_holding_period(inp: TradeInput) -> MetricResult:
    r"""Average trade holding-period duration.

    .. math::
        \bar{H} = \frac{1}{N}\sum_{j=1}^{N}
        (t_{j,\text{exit}} - t_{j,\text{entry}})
    """
    _require_field(inp, "duration", "avg_holding_period")
    dur = inp.duration
    assert dur is not None
    if len(dur) == 0:
        value: float = np.nan
    else:
        value = float(np.nanmean(dur))
    return MetricResult(
        name="avg_holding_period",
        value=value,
        category=("trades", "duration"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER, "duration_unit": "periods"},
    )


# ===================================================================
# §7.11  Holding Period Distribution
# ===================================================================


@register_metric(
    name="holding_period_distribution",
    requires="trades",
    category=("trades", "duration"),
    backend="vectorized",
    ref=_REF_HYNDMAN,
)
def holding_period_distribution(inp: TradeInput) -> MetricResult:
    r"""Distribution summary of trade holding-period durations.

    Returns ``[min, p25, p50, p75, max]``.
    """
    _require_field(inp, "duration", "holding_period_distribution")
    dur = inp.duration
    assert dur is not None
    valid = dur[np.isfinite(dur)]
    if len(valid) == 0:
        arr: NDArray[np.floating] = np.full(5, np.nan)
    else:
        arr = np.array(
            [
                np.min(valid),
                np.percentile(valid, 25),
                np.percentile(valid, 50),
                np.percentile(valid, 75),
                np.max(valid),
            ]
        )
    return MetricResult(
        name="holding_period_distribution",
        value=arr,
        category=("trades", "duration"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_HYNDMAN,
            "duration_unit": "periods",
            "output_index": ["min", "p25", "p50", "p75", "max"],
        },
    )


# ===================================================================
# §7.12  Max Consecutive Wins
# ===================================================================


@register_metric(
    name="max_consecutive_wins",
    requires="trades",
    category=("trades",),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def max_consecutive_wins(inp: TradeInput) -> MetricResult:
    r"""Longest run of consecutive winning trades.

    Ties (PnL == 0) are treated as neither win nor loss.
    """
    pnl = inp.pnl
    value: float = float(_max_consecutive(_win_mask(pnl)))
    return MetricResult(
        name="max_consecutive_wins",
        value=value,
        category=("trades",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER},
    )


# ===================================================================
# §7.13  Max Consecutive Losses
# ===================================================================


@register_metric(
    name="max_consecutive_losses",
    requires="trades",
    category=("trades",),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def max_consecutive_losses(inp: TradeInput) -> MetricResult:
    r"""Longest run of consecutive losing trades.

    Ties (PnL == 0) are treated as neither win nor loss.
    """
    pnl = inp.pnl
    value: float = float(_max_consecutive(_loss_mask(pnl)))
    return MetricResult(
        name="max_consecutive_losses",
        value=value,
        category=("trades",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER},
    )


# ===================================================================
# §7.14  Round-Trip P&L Distribution
# ===================================================================


@register_metric(
    name="pnl_distribution",
    requires="trades",
    category=("trades", "pnl"),
    backend="vectorized",
    ref=_REF_THARP_CH7,
)
def pnl_distribution(inp: TradeInput) -> MetricResult:
    r"""Summary statistics of per-trade P&L.

    Returns ``[mean, median, std, skewness, p5, p95]``.
    """
    pnl = inp.pnl
    valid = pnl[np.isfinite(pnl)]
    if len(valid) == 0:
        arr: NDArray[np.floating] = np.full(6, np.nan)
    else:
        arr = np.array(
            [
                np.mean(valid),
                np.median(valid),
                np.std(valid, ddof=1),
                sample_skewness(valid),
                np.percentile(valid, 5),
                np.percentile(valid, 95),
            ]
        )
    return MetricResult(
        name="pnl_distribution",
        value=arr,
        category=("trades", "pnl"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_THARP_CH7,
            "output_index": ["mean", "median", "std", "skewness", "p5", "p95"],
        },
    )


# ===================================================================
# §7.15  Implementation Shortfall
# ===================================================================


@register_metric(
    name="implementation_shortfall",
    requires="trades",
    category=("trades", "execution"),
    backend="vectorized",
    ref=_REF_PEROLD,
)
def implementation_shortfall(inp: TradeInput) -> MetricResult:
    r"""Per-trade implementation shortfall (average across all trades).

    .. math::
        \text{IS}_j = \text{side}_j \cdot
        \frac{P_{\text{fill},j} - P_{\text{decision},j}}
        {P_{\text{decision},j}}

    where :math:`\text{side}_j` is +1 for buys (long entries) and −1
    for sells (short entries).

    Live trading note: ``fill_price`` is the price actually obtained and
    ``decision_price`` the price when the signal fired.  In a backtest the
    fill equals the decision by construction, so the shortfall is trivially
    zero; a non-zero value is only meaningful when the two genuinely differ.

    Returns ``[mean, std, min, max]`` of the per-trade shortfall series.
    """
    _require_field(inp, "fill_price", "implementation_shortfall")
    _require_field(inp, "side", "implementation_shortfall")
    fill = inp.fill_price
    decision = inp.decision_price
    is_long = inp.is_long
    assert fill is not None
    assert decision is not None
    assert is_long is not None

    # side_sign: +1 for long (buy), -1 for short (sell)
    side_sign = np.where(is_long, 1.0, -1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shortfall = side_sign * (fill - decision) / decision
    valid = shortfall[np.isfinite(shortfall)]
    if len(valid) == 0:
        arr: NDArray[np.floating] = np.full(4, np.nan)
    else:
        arr = np.array(
            [
                np.mean(valid),
                np.std(valid, ddof=1),
                np.min(valid),
                np.max(valid),
            ]
        )
    return MetricResult(
        name="implementation_shortfall",
        value=arr,
        category=("trades", "execution"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_PEROLD,
            "output_index": ["mean", "std", "min", "max"],
        },
    )


# ===================================================================
# §7.16  Best Trade
# ===================================================================


@register_metric(
    name="best_trade",
    requires="trades",
    category=("trades",),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def best_trade(inp: TradeInput) -> MetricResult:
    r"""Maximum trade P&L.

    .. math::
        \max_j \text{PnL}_j
    """
    pnl = inp.pnl
    if len(pnl) == 0:
        value: float = np.nan
    else:
        value = float(np.nanmax(pnl))
    return MetricResult(
        name="best_trade",
        value=value,
        category=("trades",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER},
    )


# ===================================================================
# §7.17  Worst Trade
# ===================================================================


@register_metric(
    name="worst_trade",
    requires="trades",
    category=("trades",),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def worst_trade(inp: TradeInput) -> MetricResult:
    r"""Minimum trade P&L (largest loss).

    .. math::
        \min_j \text{PnL}_j
    """
    pnl = inp.pnl
    if len(pnl) == 0:
        value: float = np.nan
    else:
        value = float(np.nanmin(pnl))
    return MetricResult(
        name="worst_trade",
        value=value,
        category=("trades",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER},
    )


# ===================================================================
# §7.18  Avg Winning Trade Duration
# ===================================================================


@register_metric(
    name="avg_winning_duration",
    requires="trades",
    category=("trades", "duration"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def avg_winning_duration(inp: TradeInput) -> MetricResult:
    r"""Average holding-period duration of winning trades.

    .. math::
        \bar{H}_{\text{win}} = \frac{1}{N_{\text{win}}}
        \sum_{j:\text{win}} H_j
    """
    _require_field(inp, "duration", "avg_winning_duration")
    dur = inp.duration
    assert dur is not None
    pnl = inp.pnl
    win_dur = dur[_win_mask(pnl)]
    win_dur = win_dur[np.isfinite(win_dur)]
    if len(win_dur) == 0:
        value: float = np.nan
    else:
        value = float(np.mean(win_dur))
    return MetricResult(
        name="avg_winning_duration",
        value=value,
        category=("trades", "duration"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER, "duration_unit": "periods"},
    )


# ===================================================================
# §7.19  Avg Losing Trade Duration
# ===================================================================


@register_metric(
    name="avg_losing_duration",
    requires="trades",
    category=("trades", "duration"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def avg_losing_duration(inp: TradeInput) -> MetricResult:
    r"""Average holding-period duration of losing trades.

    .. math::
        \bar{H}_{\text{loss}} = \frac{1}{N_{\text{loss}}}
        \sum_{j:\text{loss}} H_j
    """
    _require_field(inp, "duration", "avg_losing_duration")
    dur = inp.duration
    assert dur is not None
    pnl = inp.pnl
    loss_dur = dur[_loss_mask(pnl)]
    loss_dur = loss_dur[np.isfinite(loss_dur)]
    if len(loss_dur) == 0:
        value: float = np.nan
    else:
        value = float(np.mean(loss_dur))
    return MetricResult(
        name="avg_losing_duration",
        value=value,
        category=("trades", "duration"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER, "duration_unit": "periods"},
    )


# ===================================================================
# §7.20  Payoff Ratio
# ===================================================================


@register_metric(
    name="payoff_ratio",
    requires="trades",
    category=("trades", "pnl"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def payoff_ratio(inp: TradeInput) -> MetricResult:
    r"""Average win divided by absolute average loss.

    .. math::
        \text{Payoff} = \frac{\bar{W}}{|\bar{L}|}
    """
    pnl = inp.pnl
    wins = pnl[_win_mask(pnl)]
    losses = pnl[_loss_mask(pnl)]
    avg_w = float(np.mean(wins)) if len(wins) > 0 else np.nan
    avg_l_abs = float(np.abs(np.mean(losses))) if len(losses) > 0 else np.nan
    if np.isnan(avg_w) or np.isnan(avg_l_abs) or avg_l_abs == 0.0:
        value: float = np.inf if avg_w > 0.0 else np.nan
    else:
        value = float(avg_w / avg_l_abs)
    return MetricResult(
        name="payoff_ratio",
        value=value,
        category=("trades", "pnl"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SCHWAGER},
    )


# ===================================================================
# §7.21  CPC Ratio
# ===================================================================


@register_metric(
    name="cpc_ratio",
    requires="trades",
    category=("trades", "pnl"),
    backend="vectorized",
    ref=_REF_YOUNG,
)
def cpc_ratio(inp: TradeInput) -> MetricResult:
    r"""Young's CPC Index: profit factor × payoff ratio × win rate.

    .. math::
        \text{CPC} = \text{PF} \times \text{Payoff} \times \text{WR}

    Delegates to the registered ``profit_factor``, ``payoff_ratio``,
    and ``win_rate`` metrics rather than recomputing their formulas.
    """
    from stratstat.registry import _compute_one

    if inp.n_trades == 0:
        return MetricResult(
            name="cpc_ratio",
            value=np.nan,
            category=("trades", "pnl"),
            periods_per_year=inp.periods_per_year,
            meta={"ref": _REF_YOUNG},
        )

    pf = cast(float, _compute_one(inp, "profit_factor").value)
    payoff = cast(float, _compute_one(inp, "payoff_ratio").value)
    wr = cast(float, _compute_one(inp, "win_rate").value)

    # Propagate inf/nan correctly: inf * any_finite = inf; nan * anything = nan
    value: float = pf * payoff * wr
    return MetricResult(
        name="cpc_ratio",
        value=value,
        category=("trades", "pnl"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_YOUNG},
    )


# ===================================================================
# §7.22  SQN (System Quality Number)
# ===================================================================


@register_metric(
    name="sqn",
    requires="trades",
    category=("trades", "pnl"),
    backend="vectorized",
    ref=_REF_THARP,
)
def sqn(inp: TradeInput) -> MetricResult:
    r"""System Quality Number (SQN).

    .. math::
        \text{SQN} = \frac{\bar{r}_{\text{trade}}}
        {\sigma_{\text{trade}}} \cdot \sqrt{N}

    Defined per bet on trade basis: when ``pnl_basis`` is ``"account"`` and a
    ``position_size`` column is present, each trade's pnl is converted to trade
    basis first.
    """
    pnl, converted = inp.pnl_trade_basis()
    valid = pnl[np.isfinite(pnl)]
    n = len(valid)
    if n < 2:
        value: float = np.nan
    else:
        mean_val = np.mean(valid)
        std_val = np.std(valid, ddof=1)
        if std_val == 0.0:
            value = np.inf if mean_val > 0.0 else (-np.inf if mean_val < 0.0 else np.nan)
        else:
            value = float((mean_val / std_val) * np.sqrt(n))
    return MetricResult(
        name="sqn",
        value=value,
        category=("trades", "pnl"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_THARP, "pnl_basis": inp.pnl_basis, "converted": converted},
    )


# ===================================================================
# §7.23  Trade Duration Std
# ===================================================================


@register_metric(
    name="trade_duration_std",
    requires="trades",
    category=("trades", "duration"),
    backend="vectorized",
    ref=_REF_FISHER,
)
def trade_duration_std(inp: TradeInput) -> MetricResult:
    r"""Standard deviation of trade holding-period durations.

    .. math::
        \sigma_H = \text{std}(H_1, \dots, H_N)
    """
    _require_field(inp, "duration", "trade_duration_std")
    dur = inp.duration
    assert dur is not None
    valid = dur[np.isfinite(dur)]
    if len(valid) < 2:
        value: float = np.nan
    else:
        value = float(np.std(valid, ddof=1))
    return MetricResult(
        name="trade_duration_std",
        value=value,
        category=("trades", "duration"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_FISHER, "ddof": 1, "duration_unit": "periods"},
    )


# ===================================================================
# §7.24  Trade Return Std
# ===================================================================


@register_metric(
    name="trade_return_std",
    requires="trades",
    category=("trades", "pnl"),
    backend="vectorized",
    ref=_REF_FISHER,
)
def trade_return_std(inp: TradeInput) -> MetricResult:
    r"""Standard deviation of per-trade returns (P&L).

    .. math::
        \sigma_{\text{trade}} = \text{std}(\text{PnL}_1, \dots,
        \text{PnL}_N)
    """
    pnl = inp.pnl
    valid = pnl[np.isfinite(pnl)]
    if len(valid) < 2:
        value: float = np.nan
    else:
        value = float(np.std(valid, ddof=1))
    return MetricResult(
        name="trade_return_std",
        value=value,
        category=("trades", "pnl"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_FISHER, "ddof": 1},
    )


# ===================================================================
# §7.25  Geometric Mean Return (per Trade)
# ===================================================================


@register_metric(
    name="geometric_mean_return_per_trade",
    requires="trades",
    category=("trades", "pnl"),
    backend="vectorized",
    ref=_REF_CAMPBELL,
)
def geometric_mean_return_per_trade(inp: TradeInput) -> MetricResult:
    r"""Geometric mean return per trade.

    .. math::
        \bar{r}_{g,\text{trade}} = \exp\!\left(
        \frac{1}{N}\sum_{j=1}^{N}\ln(1 + \text{PnL}_j)\right) - 1

    Requires ``pnl`` as a return fraction; ``log(1 + pnl)`` is meaningless for
    a currency-valued pnl, so ``pnl_unit="currency"`` raises.
    """
    if inp.pnl_unit == "currency":
        raise MetricNotApplicableError(
            "geometric_mean_return_per_trade requires pnl as a fraction "
            "(pnl_unit='fraction'); got pnl_unit='currency'."
        )
    pnl = inp.pnl
    valid = pnl[np.isfinite(pnl)]
    # PnL is a return fraction; 1 + PnL must be > 0 for log
    one_plus = 1.0 + valid
    positive = one_plus > 0.0
    log_vals = np.log(one_plus[positive])
    if len(log_vals) == 0:
        value: float = np.nan
    else:
        value = float(np.exp(np.mean(log_vals)) - 1.0)
    return MetricResult(
        name="geometric_mean_return_per_trade",
        value=value,
        category=("trades", "pnl"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_CAMPBELL, "pnl_unit": inp.pnl_unit},
    )


# ===================================================================
# §7.26  Outlier Win Ratio
# ===================================================================


@register_metric(
    name="outlier_win_ratio",
    requires="trades",
    category=("trades",),
    backend="vectorized",
    ref=_REF_TUKEY,
)
def outlier_win_ratio(inp: TradeInput) -> MetricResult:
    r"""Fraction of winning trades with P&L exceeding the upper Tukey fence.

    Outlier criterion: :math:`\text{PnL} > Q_3 + 1.5 \times \text{IQR}`
    of the winning-trade P&L distribution.
    """
    pnl = inp.pnl
    win_pnl = pnl[_win_mask(pnl)]
    win_pnl = win_pnl[np.isfinite(win_pnl)]
    n_win = len(win_pnl)
    if n_win < 4:
        value: float = np.nan
    else:
        q1 = np.percentile(win_pnl, 25)
        q3 = np.percentile(win_pnl, 75)
        iqr = q3 - q1
        if iqr == 0.0:
            value = 0.0
        else:
            upper = q3 + 1.5 * iqr
            value = float(np.sum(win_pnl > upper) / n_win)
    return MetricResult(
        name="outlier_win_ratio",
        value=value,
        category=("trades",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_TUKEY},
    )


# ===================================================================
# §7.27  Outlier Loss Ratio
# ===================================================================


@register_metric(
    name="outlier_loss_ratio",
    requires="trades",
    category=("trades",),
    backend="vectorized",
    ref=_REF_TUKEY,
)
def outlier_loss_ratio(inp: TradeInput) -> MetricResult:
    r"""Fraction of losing trades with P&L below the lower Tukey fence.

    Outlier criterion: :math:`\text{PnL} < Q_1 - 1.5 \times \text{IQR}`
    of the losing-trade P&L distribution.
    """
    pnl = inp.pnl
    loss_pnl = pnl[_loss_mask(pnl)]
    loss_pnl = loss_pnl[np.isfinite(loss_pnl)]
    n_loss = len(loss_pnl)
    if n_loss < 4:
        value: float = np.nan
    else:
        q1 = np.percentile(loss_pnl, 25)
        q3 = np.percentile(loss_pnl, 75)
        iqr = q3 - q1
        if iqr == 0.0:
            value = 0.0
        else:
            lower = q1 - 1.5 * iqr
            value = float(np.sum(loss_pnl < lower) / n_loss)
    return MetricResult(
        name="outlier_loss_ratio",
        value=value,
        category=("trades",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_TUKEY},
    )


# ===================================================================
# §7.28  MFE (Maximum Favorable Excursion)
# ===================================================================


@register_metric(
    name="mfe",
    requires="trades",
    category=("trades", "excursion"),
    backend="vectorized",
    ref=_REF_SWEENEY,
)
def mfe(inp: TradeInput) -> MetricResult:
    r"""Maximum Favorable Excursion (summary across all trades).

    For each trade, MFE is the largest favorable move observed while the
    trade was open, expressed as a fraction of the entry price.  For long
    trades this is :math:`(\max P_{\text{path}} - P_{\text{entry}}) /
    P_{\text{entry}}`; for short trades it is
    :math:`(P_{\text{entry}} - \min P_{\text{path}}) / P_{\text{entry}}`.

    The per-trade excursion source follows the precedence in section 3.3 and
    is recorded in ``meta["excursion_source"]``.

    Returns ``[mean, max, min]`` across all trades.
    """
    mfe_vals, _, source = inp.excursions()
    valid = mfe_vals[np.isfinite(mfe_vals)]
    if len(valid) == 0:
        arr: NDArray[np.floating] = np.full(3, np.nan)
    else:
        arr = np.array([np.mean(valid), np.max(valid), np.min(valid)])
    return MetricResult(
        name="mfe",
        value=arr,
        category=("trades", "excursion"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_SWEENEY,
            "output_index": ["mean", "max", "min"],
            "excursion_source": source,
        },
    )


# ===================================================================
# §7.29  MAE (Maximum Adverse Excursion)
# ===================================================================


@register_metric(
    name="mae",
    requires="trades",
    category=("trades", "excursion"),
    backend="vectorized",
    ref=_REF_SWEENEY,
)
def mae(inp: TradeInput) -> MetricResult:
    r"""Maximum Adverse Excursion (summary across all trades).

    For each trade, MAE is the largest adverse move observed while the
    trade was open, expressed as a fraction of the entry price (reported
    as a positive number).  For long trades this is
    :math:`(P_{\text{entry}} - \min P_{\text{path}}) / P_{\text{entry}}`;
    for short trades it is
    :math:`(\max P_{\text{path}} - P_{\text{entry}}) / P_{\text{entry}}`.

    The per-trade excursion source follows the precedence in section 3.3 and
    is recorded in ``meta["excursion_source"]``.

    Returns ``[mean, max, min]`` across all trades.
    """
    _, mae_vals, source = inp.excursions()
    valid = mae_vals[np.isfinite(mae_vals)]
    if len(valid) == 0:
        arr: NDArray[np.floating] = np.full(3, np.nan)
    else:
        arr = np.array([np.mean(valid), np.max(valid), np.min(valid)])
    return MetricResult(
        name="mae",
        value=arr,
        category=("trades", "excursion"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_SWEENEY,
            "output_index": ["mean", "max", "min"],
            "excursion_source": source,
        },
    )


# ===================================================================
# §7.30  Kelly Criterion
# ===================================================================


@register_metric(
    name="kelly_criterion",
    requires="trades",
    category=("trades", "pnl"),
    backend="vectorized",
    ref=_REF_KELLY,
)
def kelly_criterion(inp: TradeInput) -> MetricResult:
    r"""Estimated optimal Kelly bet fraction.

    .. math::
        f^* = W - \frac{1 - W}{\bar{W} / |\bar{L}|}

    Assumes independent, identically distributed trade returns.  Defined per
    bet on trade basis, and requires ``pnl`` as a fraction; a currency-valued
    pnl raises.
    """
    if inp.pnl_unit == "currency":
        raise MetricNotApplicableError(
            "kelly_criterion requires pnl as a fraction (pnl_unit='fraction'); "
            "got pnl_unit='currency'."
        )
    pnl, converted = inp.pnl_trade_basis()
    wins = pnl[_win_mask(pnl)]
    losses = pnl[_loss_mask(pnl)]
    n = len(pnl)

    if n == 0:
        value: float = np.nan
    elif len(wins) == 0:
        value = 0.0  # no wins → don't bet
    elif len(losses) == 0:
        value = 1.0  # no losses → full Kelly (unlikely in practice)
    else:
        w = len(wins) / n
        avg_w = float(np.mean(wins))
        avg_l_abs = float(np.abs(np.mean(losses)))
        if avg_l_abs == 0.0:
            value = 1.0
        else:
            payoff = avg_w / avg_l_abs
            value = 0.0 if payoff == 0.0 else w - (1.0 - w) / payoff

    return MetricResult(
        name="kelly_criterion",
        value=value,
        category=("trades", "pnl"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_KELLY, "pnl_unit": inp.pnl_unit, "converted": converted},
    )


# ===================================================================
# §7.31  Long/Short Trade Count
# ===================================================================


@register_metric(
    name="long_short_trade_count",
    requires="trades",
    category=("trades", "breakdown"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def long_short_trade_count(inp: TradeInput) -> MetricResult:
    r"""Number of long and short trades.

    Returns ``[N_long, N_short]``.
    """
    _require_field(inp, "side", "long_short_trade_count")
    is_long = inp.is_long
    assert is_long is not None
    n_long = int(np.sum(is_long))
    n_short = int(np.sum(~is_long))
    arr: NDArray[np.floating] = np.array([n_long, n_short], dtype=np.float64)
    return MetricResult(
        name="long_short_trade_count",
        value=arr,
        category=("trades", "breakdown"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_SCHWAGER,
            "output_index": ["long", "short"],
        },
    )


# ===================================================================
# §7.32  Long/Short Trade %
# ===================================================================


@register_metric(
    name="long_short_trade_pct",
    requires="trades",
    category=("trades", "breakdown"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def long_short_trade_pct(inp: TradeInput) -> MetricResult:
    r"""Fraction of all trades that are long vs. short.

    Returns ``[p_long, p_short]``.
    """
    _require_field(inp, "side", "long_short_trade_pct")
    is_long = inp.is_long
    assert is_long is not None
    n = len(is_long)
    if n == 0:
        arr: NDArray[np.floating] = np.full(2, np.nan)
    else:
        arr = np.array([np.mean(is_long), np.mean(~is_long)], dtype=np.float64)
    return MetricResult(
        name="long_short_trade_pct",
        value=arr,
        category=("trades", "breakdown"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_SCHWAGER,
            "output_index": ["long_pct", "short_pct"],
        },
    )


# ===================================================================
# §7.33  Long/Short Winning/Losing Trades
# ===================================================================


@register_metric(
    name="long_short_winning_losing",
    requires="trades",
    category=("trades", "breakdown"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def long_short_winning_losing(inp: TradeInput) -> MetricResult:
    r"""Winning and losing trade counts split by side.

    Returns ``[N_long_win, N_long_loss, N_short_win, N_short_loss]``.
    """
    _require_field(inp, "side", "long_short_winning_losing")
    pnl = inp.pnl
    is_long = inp.is_long
    assert is_long is not None
    wins = _win_mask(pnl)
    losses = _loss_mask(pnl)

    n_long_win = int(np.sum(is_long & wins))
    n_long_loss = int(np.sum(is_long & losses))
    n_short_win = int(np.sum(~is_long & wins))
    n_short_loss = int(np.sum(~is_long & losses))

    arr: NDArray[np.floating] = np.array(
        [n_long_win, n_long_loss, n_short_win, n_short_loss],
        dtype=np.float64,
    )
    return MetricResult(
        name="long_short_winning_losing",
        value=arr,
        category=("trades", "breakdown"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_SCHWAGER,
            "output_index": [
                "long_win",
                "long_loss",
                "short_win",
                "short_loss",
            ],
        },
    )


# ===================================================================
# §7.34  Long/Short Avg Duration
# ===================================================================


@register_metric(
    name="long_short_avg_duration",
    requires="trades",
    category=("trades", "breakdown"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def long_short_avg_duration(inp: TradeInput) -> MetricResult:
    r"""Average holding-period duration for long and short trades.

    Returns ``[H_bar_long, H_bar_short]``.
    """
    _require_field(inp, "side", "long_short_avg_duration")
    _require_field(inp, "duration", "long_short_avg_duration")
    dur = inp.duration
    is_long = inp.is_long
    assert dur is not None
    assert is_long is not None

    long_dur = dur[is_long]
    short_dur = dur[~is_long]
    long_dur = long_dur[np.isfinite(long_dur)]
    short_dur = short_dur[np.isfinite(short_dur)]

    h_long = float(np.mean(long_dur)) if len(long_dur) > 0 else np.nan
    h_short = float(np.mean(short_dur)) if len(short_dur) > 0 else np.nan

    arr: NDArray[np.floating] = np.array([h_long, h_short])
    return MetricResult(
        name="long_short_avg_duration",
        value=arr,
        category=("trades", "breakdown"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_SCHWAGER,
            "duration_unit": "periods",
            "output_index": ["long_avg_duration", "short_avg_duration"],
        },
    )


# ===================================================================
# §7.35  Long/Short Total PnL %
# ===================================================================


@register_metric(
    name="long_short_total_pnl",
    requires="trades",
    category=("trades", "breakdown"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def long_short_total_pnl(inp: TradeInput) -> MetricResult:
    r"""Total P&L from long and short trades.

    Returns ``[PnL_long_total, PnL_short_total]``.
    """
    _require_field(inp, "side", "long_short_total_pnl")
    pnl = inp.pnl
    is_long = inp.is_long
    assert is_long is not None

    long_total = float(np.nansum(pnl[is_long]))
    short_total = float(np.nansum(pnl[~is_long]))

    arr: NDArray[np.floating] = np.array([long_total, short_total])
    return MetricResult(
        name="long_short_total_pnl",
        value=arr,
        category=("trades", "breakdown"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_SCHWAGER,
            "output_index": ["long_total_pnl", "short_total_pnl"],
        },
    )


# ===================================================================
# §7.36  Long/Short Avg PnL %
# ===================================================================


@register_metric(
    name="long_short_avg_pnl",
    requires="trades",
    category=("trades", "breakdown"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def long_short_avg_pnl(inp: TradeInput) -> MetricResult:
    r"""Average P&L per long and short trade.

    Returns ``[PnL_bar_long, PnL_bar_short]``.
    """
    _require_field(inp, "side", "long_short_avg_pnl")
    pnl = inp.pnl
    is_long = inp.is_long
    assert is_long is not None

    long_pnl = pnl[is_long]
    short_pnl = pnl[~is_long]
    long_pnl = long_pnl[np.isfinite(long_pnl)]
    short_pnl = short_pnl[np.isfinite(short_pnl)]

    avg_long = float(np.mean(long_pnl)) if len(long_pnl) > 0 else np.nan
    avg_short = float(np.mean(short_pnl)) if len(short_pnl) > 0 else np.nan

    arr: NDArray[np.floating] = np.array([avg_long, avg_short])
    return MetricResult(
        name="long_short_avg_pnl",
        value=arr,
        category=("trades", "breakdown"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_SCHWAGER,
            "output_index": ["long_avg_pnl", "short_avg_pnl"],
        },
    )


# ===================================================================
# §7.37  Long/Short Best/Worst Trade %
# ===================================================================


@register_metric(
    name="long_short_best_worst",
    requires="trades",
    category=("trades", "breakdown"),
    backend="vectorized",
    ref=_REF_SCHWAGER,
)
def long_short_best_worst(inp: TradeInput) -> MetricResult:
    r"""Best and worst trade P&L split by side.

    Returns ``[best_long, worst_long, best_short, worst_short]``.
    """
    _require_field(inp, "side", "long_short_best_worst")
    pnl = inp.pnl
    is_long = inp.is_long
    assert is_long is not None

    long_pnl = pnl[is_long]
    short_pnl = pnl[~is_long]

    best_long = float(np.nanmax(long_pnl)) if len(long_pnl) > 0 else np.nan
    worst_long = float(np.nanmin(long_pnl)) if len(long_pnl) > 0 else np.nan
    best_short = float(np.nanmax(short_pnl)) if len(short_pnl) > 0 else np.nan
    worst_short = float(np.nanmin(short_pnl)) if len(short_pnl) > 0 else np.nan

    arr: NDArray[np.floating] = np.array([best_long, worst_long, best_short, worst_short])
    return MetricResult(
        name="long_short_best_worst",
        value=arr,
        category=("trades", "breakdown"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_SCHWAGER,
            "output_index": [
                "best_long",
                "worst_long",
                "best_short",
                "worst_short",
            ],
        },
    )
