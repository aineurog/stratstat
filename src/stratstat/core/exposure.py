"""Exposure-tier metrics.

Metrics requiring positions/weights: gross exposure, net exposure,
leverage, long/short exposure percentages, long/short book contribution,
long/short beta, position concentration (HHI), effective N positions,
turnover, average holding weight, position coverage, exposure volatility,
exposure CV, exposure utilization, exposure directional bias, exposure
percentiles, period counts, active share.

All tagged: category varies, requires="exposure".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from stratstat.core._utils import ols_beta
from stratstat.inputs import ExposureInput
from stratstat.registry import register_metric
from stratstat.results import MetricResult

# ---------------------------------------------------------------------------
# Citation strings
# ---------------------------------------------------------------------------

_REF_ANG_CH2 = (
    "Ang (2014), Asset Management: A Systematic Approach to Factor "
    "Investing, Oxford University Press, Ch. 2."
)
_BACON_BASE = (
    "Bacon (2008), Practical Portfolio Performance Measurement and "
    "Attribution, 2nd ed."
)
_REF_BACON_112 = f"{_BACON_BASE}, §11.2."
_REF_BACON_113 = f"{_BACON_BASE}, §11.3."
_REF_BACON_115 = f"{_BACON_BASE}, §11.5."
_REF_BACON_116 = f"{_BACON_BASE}, §11.6."
_REF_GIPS = "CFA Institute, GIPS Standards (2020)."
_REF_AFP2014 = (
    "Asness, Frazzini & Pedersen (2014), 'Low-Risk Investing Without "
    "Industry Bets,' Financial Analysts Journal, 70(4)."
)
_REF_HHI_HIRSCHMAN = (
    "Hirschman (1964), 'The Paternity of an Index,' American Economic "
    "Review, 54(5)."
)
_REF_HHI_ADELMAN = (
    "Adelman (1969), 'Comment on the \"H\" Concentration Measure as a "
    "Numbers-Equivalent,' Review of Economics and Statistics, 51(1)."
)
_REF_TURNOVER = (
    "Morningstar (2020), Morningstar Portfolio Turnover Methodology."
)
_REF_PEARSON = (
    "Pearson (1896); coefficient of variation. Standard descriptive "
    "statistic."
)
_REF_HYNDMAN = (
    "Hyndman & Fan (1996); standard order statistics for percentiles."
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _gross_exposure_series(
    positions: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Gross exposure at each period: sum of absolute weights.

    Args:
        positions: (n_periods, n_assets) weight array.

    Returns:
        (n_periods,) array of gross exposure values.
    """
    arr: NDArray[np.floating] = np.nansum(np.abs(positions), axis=1)
    return arr


def _net_exposure_series(
    positions: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Net exposure at each period: sum of signed weights.

    Args:
        positions: (n_periods, n_assets) weight array.

    Returns:
        (n_periods,) array of net exposure values.
    """
    arr: NDArray[np.floating] = np.nansum(positions, axis=1)
    return arr


def _long_exposure_series(
    positions: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Long exposure at each period: sum of positive weights.

    Args:
        positions: (n_periods, n_assets) weight array.

    Returns:
        (n_periods,) array of long exposure values.
    """
    pos = np.where(positions > 0.0, positions, 0.0)
    arr: NDArray[np.floating] = np.nansum(pos, axis=1)
    return arr


def _short_exposure_series(
    positions: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Short exposure at each period: sum of absolute negative weights.

    Args:
        positions: (n_periods, n_assets) weight array.

    Returns:
        (n_periods,) array of short exposure values (positive).
    """
    neg = np.where(positions < 0.0, np.abs(positions), 0.0)
    arr: NDArray[np.floating] = np.nansum(neg, axis=1)
    return arr


# ---------------------------------------------------------------------------
# §6.1  Gross Exposure
# ---------------------------------------------------------------------------

_REF_GROSS = _REF_ANG_CH2


@register_metric(
    name="gross_exposure",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_GROSS,
)
def gross_exposure(inp: ExposureInput) -> MetricResult:
    r"""Gross exposure: sum of absolute position weights.

    .. math::
        \text{GE}_t = \sum_{i=1}^{N} |w_{i,t}|

    Returns ``[current, max, mean]`` across all periods.
    """
    ge = _gross_exposure_series(inp.positions)
    current = float(ge[-1]) if len(ge) > 0 else np.nan
    ge_max = float(np.nanmax(ge)) if len(ge) > 0 else np.nan
    ge_mean = float(np.nanmean(ge)) if len(ge) > 0 else np.nan
    arr: NDArray[np.floating] = np.array([current, ge_max, ge_mean])
    return MetricResult(
        name="gross_exposure",
        value=arr,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_GROSS,
            "output_index": ["current", "max", "mean"],
        },
    )


# ---------------------------------------------------------------------------
# §6.2  Net Exposure
# ---------------------------------------------------------------------------

_REF_NET = _REF_ANG_CH2


@register_metric(
    name="net_exposure",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_NET,
)
def net_exposure(inp: ExposureInput) -> MetricResult:
    r"""Net exposure: sum of signed position weights.

    .. math::
        \text{NE}_t = \sum_{i=1}^{N} w_{i,t}

    Returns ``[current, max, min, mean, range]`` across all periods.
    """
    ne = _net_exposure_series(inp.positions)
    current = float(ne[-1]) if len(ne) > 0 else np.nan
    ne_max = float(np.nanmax(ne)) if len(ne) > 0 else np.nan
    ne_min = float(np.nanmin(ne)) if len(ne) > 0 else np.nan
    ne_mean = float(np.nanmean(ne)) if len(ne) > 0 else np.nan
    ne_range = ne_max - ne_min if len(ne) > 0 else np.nan
    arr: NDArray[np.floating] = np.array(
        [current, ne_max, ne_min, ne_mean, ne_range]
    )
    return MetricResult(
        name="net_exposure",
        value=arr,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_NET,
            "output_index": ["current", "max", "min", "mean", "range"],
        },
    )


# ---------------------------------------------------------------------------
# §6.3  Leverage
# ---------------------------------------------------------------------------

_REF_LEVERAGE = _REF_ANG_CH2


@register_metric(
    name="leverage",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_LEVERAGE,
)
def leverage(inp: ExposureInput) -> MetricResult:
    r"""Gross-exposure-to-equity leverage (mean over all periods).

    This is the ratio of gross exposure to portfolio equity,
    not notional leverage or margin leverage.

    .. math::
        \text{Leverage}_t = \frac{\text{GE}_t}{\text{equity}_t}

    Requires portfolio equity (provided directly or computed from
    positions + returns).
    """
    if not inp.has_equity:
        raise ValueError(
            "leverage requires equity. Provide equity= to ExposureInput, "
            "or provide asset-level returns= so equity can be computed."
        )
    eq = inp.equity
    assert eq is not None  # narrow type for mypy
    ge = _gross_exposure_series(inp.positions)
    with np.errstate(divide="ignore", invalid="ignore"):
        lev = np.where(eq > 0.0, ge / eq, np.nan)
    value: float = float(np.nanmean(lev))
    return MetricResult(
        name="leverage",
        value=value,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_LEVERAGE},
    )


# ---------------------------------------------------------------------------
# §6.4  Long Exposure %
# ---------------------------------------------------------------------------

_REF_LONG_EXP = _REF_GIPS + " Also see " + _REF_BACON_113


@register_metric(
    name="long_exposure_pct",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_LONG_EXP,
)
def long_exposure_pct(inp: ExposureInput) -> MetricResult:
    r"""Average long exposure as a fraction of portfolio value.

    .. math::
        \text{LE\%}_t = \sum_{i} w_{i,t} \cdot
        \mathbf{1}_{[w_{i,t} > 0]}
    """
    le = _long_exposure_series(inp.positions)
    value: float = float(np.nanmean(le))
    return MetricResult(
        name="long_exposure_pct",
        value=value,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_LONG_EXP},
    )


# ---------------------------------------------------------------------------
# §6.5  Short Exposure %
# ---------------------------------------------------------------------------

_REF_SHORT_EXP = _REF_LONG_EXP  # same citation


@register_metric(
    name="short_exposure_pct",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_SHORT_EXP,
)
def short_exposure_pct(inp: ExposureInput) -> MetricResult:
    r"""Average short exposure as a fraction of portfolio value.

    .. math::
        \text{SE\%}_t = \sum_{i} |w_{i,t}| \cdot
        \mathbf{1}_{[w_{i,t} < 0]}
    """
    se = _short_exposure_series(inp.positions)
    value: float = float(np.nanmean(se))
    return MetricResult(
        name="short_exposure_pct",
        value=value,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SHORT_EXP},
    )


# ---------------------------------------------------------------------------
# §6.6  Long-Book Contribution to Return
# ---------------------------------------------------------------------------

_REF_LONG_BOOK = _REF_BACON_115


@register_metric(
    name="long_book_return",
    requires="exposure",
    category=("exposure", "attribution"),
    backend="vectorized",
    ref=_REF_LONG_BOOK,
)
def long_book_return(inp: ExposureInput) -> MetricResult:
    r"""Average contribution of long positions to portfolio return.

    .. math::
        R_t^{\text{long}} = \sum_{i} w_{i,t-1} \cdot r_{i,t} \cdot
        \mathbf{1}_{[w_{i,t-1} > 0]}
    """
    if not inp.has_returns:
        raise ValueError(
            "long_book_return requires asset-level returns. "
            "Provide returns= to ExposureInput."
        )
    ret = inp.returns
    assert ret is not None
    positions = inp.positions
    w_lag = np.roll(positions, shift=1, axis=0)
    w_lag[0, :] = np.nan
    long_only = np.where(w_lag > 0.0, w_lag, 0.0)
    contrib = np.nansum(long_only * ret, axis=1)
    value: float = float(np.nanmean(contrib))
    return MetricResult(
        name="long_book_return",
        value=value,
        category=("exposure", "attribution"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_LONG_BOOK},
    )


# ---------------------------------------------------------------------------
# §6.7  Short-Book Contribution to Return
# ---------------------------------------------------------------------------

_REF_SHORT_BOOK = _REF_BACON_115


@register_metric(
    name="short_book_return",
    requires="exposure",
    category=("exposure", "attribution"),
    backend="vectorized",
    ref=_REF_SHORT_BOOK,
)
def short_book_return(inp: ExposureInput) -> MetricResult:
    r"""Average contribution of short positions to portfolio return.

    .. math::
        R_t^{\text{short}} = \sum_{i} w_{i,t-1} \cdot r_{i,t} \cdot
        \mathbf{1}_{[w_{i,t-1} < 0]}
    """
    if not inp.has_returns:
        raise ValueError(
            "short_book_return requires asset-level returns. "
            "Provide returns= to ExposureInput."
        )
    ret = inp.returns
    assert ret is not None
    positions = inp.positions
    w_lag = np.roll(positions, shift=1, axis=0)
    w_lag[0, :] = np.nan
    short_only = np.where(w_lag < 0.0, w_lag, 0.0)
    contrib = np.nansum(short_only * ret, axis=1)
    value: float = float(np.nanmean(contrib))
    return MetricResult(
        name="short_book_return",
        value=value,
        category=("exposure", "attribution"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SHORT_BOOK},
    )


# ---------------------------------------------------------------------------
# §6.8  Long Beta
# ---------------------------------------------------------------------------

_REF_LONG_BETA = _REF_AFP2014


@register_metric(
    name="long_beta",
    requires="exposure",
    category=("exposure", "risk"),
    backend="vectorized",
    ref=_REF_LONG_BETA,
)
def longols_beta(inp: ExposureInput) -> MetricResult:
    r"""Beta of long-book returns vs. benchmark.

    .. math::
        \beta_{\text{long}} = \frac{\text{Cov}(R^{\text{long}}, R_m)}
        {\text{Var}(R_m)}
    """
    if not inp.has_returns:
        raise ValueError(
            "long_beta requires asset-level returns. "
            "Provide returns= to ExposureInput."
        )
    if not inp.has_benchmark:
        raise ValueError(
            "long_beta requires benchmark returns. "
            "Provide benchmark= to ExposureInput."
        )
    ret = inp.returns
    bench = inp.benchmark
    assert ret is not None
    assert bench is not None
    positions = inp.positions
    w_lag = np.roll(positions, shift=1, axis=0)
    w_lag[0, :] = np.nan
    long_only = np.where(w_lag > 0.0, w_lag, 0.0)
    r_long = np.nansum(long_only * ret, axis=1)
    value: float = ols_beta(r_long, bench)
    return MetricResult(
        name="long_beta",
        value=value,
        category=("exposure", "risk"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_LONG_BETA},
    )


# ---------------------------------------------------------------------------
# §6.9  Short Beta
# ---------------------------------------------------------------------------

_REF_SHORT_BETA = _REF_AFP2014


@register_metric(
    name="short_beta",
    requires="exposure",
    category=("exposure", "risk"),
    backend="vectorized",
    ref=_REF_SHORT_BETA,
)
def shortols_beta(inp: ExposureInput) -> MetricResult:
    r"""Beta of short-book returns vs. benchmark.

    .. math::
        \beta_{\text{short}} = \frac{\text{Cov}(R^{\text{short}}, R_m)}
        {\text{Var}(R_m)}
    """
    if not inp.has_returns:
        raise ValueError(
            "short_beta requires asset-level returns. "
            "Provide returns= to ExposureInput."
        )
    if not inp.has_benchmark:
        raise ValueError(
            "short_beta requires benchmark returns. "
            "Provide benchmark= to ExposureInput."
        )
    ret = inp.returns
    bench = inp.benchmark
    assert ret is not None
    assert bench is not None
    positions = inp.positions
    w_lag = np.roll(positions, shift=1, axis=0)
    w_lag[0, :] = np.nan
    short_only = np.where(w_lag < 0.0, w_lag, 0.0)
    r_short = np.nansum(short_only * ret, axis=1)
    value: float = ols_beta(r_short, bench)
    return MetricResult(
        name="short_beta",
        value=value,
        category=("exposure", "risk"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SHORT_BETA},
    )


# ---------------------------------------------------------------------------
# §6.10  Position Concentration (HHI)
# ---------------------------------------------------------------------------

_REF_HHI = _REF_HHI_HIRSCHMAN + " " + _REF_HHI_ADELMAN


@register_metric(
    name="position_concentration",
    requires="exposure",
    category=("exposure", "concentration"),
    backend="vectorized",
    ref=_REF_HHI,
)
def position_concentration(inp: ExposureInput) -> MetricResult:
    r"""Average Herfindahl-Hirschman Index on normalized position weights.

    .. math::
        \text{HHI}_t = \sum_{i=1}^{N}
        \left(\frac{w_{i,t}}{\sum_j |w_{j,t}|}\right)^{\!2}
    """
    positions = inp.positions
    ge = _gross_exposure_series(positions)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_w = np.where(
            ge[:, np.newaxis] > 0.0,
            positions / ge[:, np.newaxis],
            0.0,
        )
    hhi_t = np.nansum(norm_w**2, axis=1)
    # Filter out periods with zero gross exposure (flat).
    hhi_t = np.where(ge > 0.0, hhi_t, np.nan)
    value: float = float(np.nanmean(hhi_t))
    return MetricResult(
        name="position_concentration",
        value=value,
        category=("exposure", "concentration"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_HHI},
    )


# ---------------------------------------------------------------------------
# §6.11  Effective N Positions
# ---------------------------------------------------------------------------

_REF_EFF_N = _REF_HHI_ADELMAN


@register_metric(
    name="effective_n_positions",
    requires="exposure",
    category=("exposure", "concentration"),
    backend="vectorized",
    ref=_REF_EFF_N,
)
def effective_n_positions(inp: ExposureInput) -> MetricResult:
    r"""Average effective number of positions (reciprocal HHI).

    .. math::
        N_{\text{eff},t} = \frac{1}{\text{HHI}_t}
    """
    positions = inp.positions
    ge = _gross_exposure_series(positions)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_w = np.where(
            ge[:, np.newaxis] > 0.0,
            positions / ge[:, np.newaxis],
            0.0,
        )
    hhi_t = np.nansum(norm_w**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff_t = np.where(hhi_t > 0.0, 1.0 / hhi_t, np.nan)
    # Zero HHI where GE is zero leads to NaN; exclude.
    n_eff_t = np.where(ge > 0.0, n_eff_t, np.nan)
    value: float = float(np.nanmean(n_eff_t))
    return MetricResult(
        name="effective_n_positions",
        value=value,
        category=("exposure", "concentration"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_EFF_N},
    )


# ---------------------------------------------------------------------------
# §6.12  Turnover
# ---------------------------------------------------------------------------

_REF_TO = _REF_TURNOVER + " Also see " + _REF_BACON_116


@register_metric(
    name="turnover",
    requires="exposure",
    category=("exposure", "turnover"),
    backend="vectorized",
    ref=_REF_TO,
)
def turnover(inp: ExposureInput) -> MetricResult:
    r"""Annualized mean portfolio turnover.

    .. math::
        \text{TO}_t = \frac{1}{2}\sum_{i=1}^{N}
        |\Delta w_{i,t}|

    Annualized by multiplying the mean period turnover by
    ``periods_per_year``.
    """
    if inp.periods_per_year is None:
        raise ValueError(
            "turnover requires periods_per_year to annualize. "
            "Provide periods_per_year= to ExposureInput."
        )
    positions = inp.positions
    if inp.n_periods < 2:
        value: float = np.nan
        return MetricResult(
            name="turnover",
            value=value,
            category=("exposure", "turnover"),
            periods_per_year=inp.periods_per_year,
            meta={"ref": _REF_TO, "annualized": True},
        )
    delta = np.diff(positions, axis=0)  # (n_periods-1, n_assets)
    to_t = 0.5 * np.nansum(np.abs(delta), axis=1)
    mean_to = float(np.nanmean(to_t))
    ann_to: float = mean_to * float(inp.periods_per_year)
    return MetricResult(
        name="turnover",
        value=ann_to,
        category=("exposure", "turnover"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_TO, "annualized": True},
    )


# ---------------------------------------------------------------------------
# §6.13  Average Holding Weight per Position
# ---------------------------------------------------------------------------

_REF_AVG_W = _REF_BACON_112


@register_metric(
    name="avg_holding_weight",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_AVG_W,
)
def avg_holding_weight(inp: ExposureInput) -> MetricResult:
    r"""Average absolute weight per active position, averaged over time.

    .. math::
        \bar{w}_t = \frac{1}{N_t}\sum_{i=1}^{N_t} |w_{i,t}|

    where :math:`N_t` counts assets with non-zero weight at period *t*.
    """
    positions = inp.positions
    abs_pos = np.abs(positions)
    n_active = np.sum(~np.isclose(abs_pos, 0.0), axis=1)  # (n_periods,)
    sum_abs = np.nansum(abs_pos, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_w_t = np.where(n_active > 0, sum_abs / n_active, np.nan)
    value: float = float(np.nanmean(avg_w_t))
    return MetricResult(
        name="avg_holding_weight",
        value=value,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_AVG_W},
    )


# ---------------------------------------------------------------------------
# §6.14  Position Coverage %
# ---------------------------------------------------------------------------

_REF_COVERAGE = _REF_BACON_112


@register_metric(
    name="position_coverage",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_COVERAGE,
)
def position_coverage(inp: ExposureInput) -> MetricResult:
    r"""Fraction of periods with at least one non-zero position.

    .. math::
        \text{Coverage} = \frac{1}{n}\sum_{t=1}^{n}
        \mathbf{1}_{[\exists i: w_{i,t} \neq 0]}
    """
    positions = inp.positions
    n_periods = inp.n_periods
    if n_periods == 0:
        value: float = np.nan
    else:
        has_position = np.any(~np.isclose(np.abs(positions), 0.0), axis=1)
        value = float(np.sum(has_position) / n_periods)
    return MetricResult(
        name="position_coverage",
        value=value,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_COVERAGE},
    )


# ---------------------------------------------------------------------------
# §6.15  Long/Short Position Coverage %
# ---------------------------------------------------------------------------

_REF_LS_COVERAGE = _REF_COVERAGE


@register_metric(
    name="long_position_coverage",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_LS_COVERAGE,
)
def long_position_coverage(inp: ExposureInput) -> MetricResult:
    r"""Fraction of periods with at least one long position.

    A period is "long" if at least one weight is strictly positive.
    """
    positions = inp.positions
    n_periods = inp.n_periods
    if n_periods == 0:
        value: float = np.nan
    else:
        has_long = np.any(positions > 0.0, axis=1)
        value = float(np.sum(has_long) / n_periods)
    return MetricResult(
        name="long_position_coverage",
        value=value,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_LS_COVERAGE},
    )


@register_metric(
    name="short_position_coverage",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_LS_COVERAGE,
)
def short_position_coverage(inp: ExposureInput) -> MetricResult:
    r"""Fraction of periods with at least one short position.

    A period is "short" if at least one weight is strictly negative.
    """
    positions = inp.positions
    n_periods = inp.n_periods
    if n_periods == 0:
        value: float = np.nan
    else:
        has_short = np.any(positions < 0.0, axis=1)
        value = float(np.sum(has_short) / n_periods)
    return MetricResult(
        name="short_position_coverage",
        value=value,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_LS_COVERAGE},
    )


# ---------------------------------------------------------------------------
# §6.16  Exposure Volatility
# ---------------------------------------------------------------------------

_REF_EXP_VOL = _REF_BACON_113


@register_metric(
    name="exposure_volatility",
    requires="exposure",
    category=("exposure", "risk"),
    backend="vectorized",
    ref=_REF_EXP_VOL,
)
def exposure_volatility(inp: ExposureInput) -> MetricResult:
    r"""Standard deviation of the gross exposure time series.

    .. math::
        \sigma_{\text{GE}} = \text{std}(\text{GE}_1, \dots, \text{GE}_n)
    """
    ge = _gross_exposure_series(inp.positions)
    value: float = float(np.nanstd(ge, ddof=1))
    return MetricResult(
        name="exposure_volatility",
        value=value,
        category=("exposure", "risk"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_EXP_VOL, "ddof": 1},
    )


# ---------------------------------------------------------------------------
# §6.17  Net Exposure Volatility
# ---------------------------------------------------------------------------

_REF_NET_VOL = _REF_EXP_VOL


@register_metric(
    name="net_exposure_volatility",
    requires="exposure",
    category=("exposure", "risk"),
    backend="vectorized",
    ref=_REF_NET_VOL,
)
def net_exposure_volatility(inp: ExposureInput) -> MetricResult:
    r"""Standard deviation of the net exposure time series.

    .. math::
        \sigma_{\text{NE}} = \text{std}(\text{NE}_1, \dots, \text{NE}_n)
    """
    ne = _net_exposure_series(inp.positions)
    value: float = float(np.nanstd(ne, ddof=1))
    return MetricResult(
        name="net_exposure_volatility",
        value=value,
        category=("exposure", "risk"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_NET_VOL, "ddof": 1},
    )


# ---------------------------------------------------------------------------
# §6.18  Exposure Coefficient of Variation
# ---------------------------------------------------------------------------

_REF_EXP_CV = _REF_PEARSON


@register_metric(
    name="exposure_cv",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_EXP_CV,
)
def exposure_cv(inp: ExposureInput) -> MetricResult:
    r"""Coefficient of variation of gross exposure.

    .. math::
        \text{CV}_{\text{exp}} =
        \frac{\sigma_{\text{GE}}}{|\bar{\text{GE}}|}
    """
    ge = _gross_exposure_series(inp.positions)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv: float = float(np.nanstd(ge, ddof=1) / np.abs(np.nanmean(ge)))
    return MetricResult(
        name="exposure_cv",
        value=cv,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_EXP_CV},
    )


# ---------------------------------------------------------------------------
# §6.19  Avg Exposure Utilization
# ---------------------------------------------------------------------------

_REF_UTIL = _REF_BACON_113


@register_metric(
    name="exposure_utilization",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_UTIL,
)
def exposure_utilization(inp: ExposureInput) -> MetricResult:
    r"""Mean gross exposure as a fraction of maximum gross exposure.

    .. math::
        \text{Utilization} = \frac{\bar{\text{GE}}}{\max_t \text{GE}_t}
    """
    ge = _gross_exposure_series(inp.positions)
    ge_mean = np.nanmean(ge)
    ge_max = np.nanmax(ge)
    with np.errstate(divide="ignore", invalid="ignore"):
        util: float = float(ge_mean / ge_max)
    return MetricResult(
        name="exposure_utilization",
        value=util,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_UTIL},
    )


# ---------------------------------------------------------------------------
# §6.20  Exposure Directional Bias
# ---------------------------------------------------------------------------

_REF_BIAS = _REF_BACON_113 + " " + _REF_ANG_CH2


@register_metric(
    name="exposure_directional_bias",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_BIAS,
)
def exposure_directional_bias(inp: ExposureInput) -> MetricResult:
    r"""Absolute mean net exposure relative to mean gross exposure.

    .. math::
        \text{Bias} = \frac{|\bar{\text{NE}}|}{\bar{\text{GE}}}
    """
    ne = _net_exposure_series(inp.positions)
    ge = _gross_exposure_series(inp.positions)
    with np.errstate(divide="ignore", invalid="ignore"):
        bias: float = float(np.abs(np.nanmean(ne)) / np.nanmean(ge))
    return MetricResult(
        name="exposure_directional_bias",
        value=bias,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_BIAS},
    )


# ---------------------------------------------------------------------------
# §6.21  Exposure Percentiles
# ---------------------------------------------------------------------------

_REF_PCTL = _REF_HYNDMAN


@register_metric(
    name="exposure_percentiles",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_PCTL,
)
def exposure_percentiles(inp: ExposureInput) -> MetricResult:
    r"""Percentiles {25, 50, 75, 90, 95} of the gross exposure time series.

    Uses numpy's linear-interpolation percentile method (default).
    """
    ge = _gross_exposure_series(inp.positions)
    qs = np.array([25, 50, 75, 90, 95])
    ge_valid = ge[np.isfinite(ge)]
    values: NDArray[np.floating] = (
        np.percentile(ge_valid, qs)
        if len(ge_valid) > 0
        else np.full(5, np.nan)
    )
    return MetricResult(
        name="exposure_percentiles",
        value=values,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_PCTL,
            "output_index": ["p25", "p50", "p75", "p90", "p95"],
        },
    )


# ---------------------------------------------------------------------------
# §6.22  Period Counts
# ---------------------------------------------------------------------------

_REF_COUNTS = _REF_BACON_112


@register_metric(
    name="period_counts",
    requires="exposure",
    category=("exposure",),
    backend="vectorized",
    ref=_REF_COUNTS,
)
def period_counts(inp: ExposureInput) -> MetricResult:
    r"""Period-count breakdown of the exposure time series.

    Returns ``[total, active, long_only, short_only, long_short, idle]``:

    * **active**: at least one non-zero weight
    * **long_only**: only positive weights (no negatives, >=1 positive)
    * **short_only**: only negative weights (no positives, >=1 negative)
    * **long_short**: both positive and negative weights present
    * **idle** (flat): all weights zero
    """
    positions = inp.positions
    total = inp.n_periods
    abs_pos = np.abs(positions)
    has_position = np.any(~np.isclose(abs_pos, 0.0), axis=1)
    has_long = np.any(positions > 0.0, axis=1)
    has_short = np.any(positions < 0.0, axis=1)

    active = int(np.sum(has_position))
    long_only = int(np.sum(has_long & ~has_short))
    short_only = int(np.sum(~has_long & has_short))
    long_short = int(np.sum(has_long & has_short))
    idle = total - int(np.sum(has_position))

    arr: NDArray[np.floating] = np.array(
        [total, active, long_only, short_only, long_short, idle],
        dtype=np.float64,
    )
    return MetricResult(
        name="period_counts",
        value=arr,
        category=("exposure",),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_COUNTS,
            "output_index": [
                "total",
                "active",
                "long_only",
                "short_only",
                "long_short",
                "idle",
            ],
        },
    )


# ---------------------------------------------------------------------------
# 6.23 Active Share
# Reference: Cremers & Petajisto (2009)
# ---------------------------------------------------------------------------

_ACTIVE_SHARE_REF = (
    "Cremers & Petajisto (2009), "
    '"How Active Is Your Fund Manager? A New Measure That Predicts Performance," '
    "Review of Financial Studies, 22(9)"
)


@register_metric(
    name="active_share",
    requires="exposure",
    category=("relative", "exposure"),
    backend="vectorized",
    ref=_ACTIVE_SHARE_REF,
)
def active_share(inp: ExposureInput) -> MetricResult:
    """Active Share — fraction of the portfolio that differs from the benchmark.

    Formula (per period):
        AS_t = (1/2) * sum_i |w_{i,t} - w_{b,i,t}|

    where w_{i,t} are portfolio position weights and w_{b,i,t} are benchmark
    constituent weights for asset i at time t. Active Share ranges from 0
    (perfectly matching the benchmark) to 1 (zero overlap).

    Returns the time-series mean as the primary value and stores the full
    per-period series in ``meta["series"]``.

    Requires ``benchmark_weights`` on the ``ExposureInput``.

    Args:
        inp: An ``ExposureInput`` with ``benchmark_weights`` set.

    Returns:
        MetricResult with mean Active Share (float) and per-period series
        in ``meta["series"]``.

    Raises:
        ValueError: If ``benchmark_weights`` is not set on the input.
    """
    if inp.benchmark_weights is None:
        raise ValueError(
            "Active Share requires benchmark_weights on the ExposureInput. "
            "Pass benchmark_weights=<array> to ExposureInput."
        )

    positions = inp.positions  # (n_periods, n_assets)
    bw = inp.benchmark_weights  # (n_assets,) or (n_periods, n_assets)

    if bw.ndim == 1:
        # Static benchmark weights — broadcast across all periods
        bw_2d = bw[np.newaxis, :]  # (1, n_assets)
        diff = np.abs(positions - bw_2d)
    else:
        diff = np.abs(positions - bw)

    # Per-period Active Share
    active_share_series: NDArray[np.floating] = 0.5 * np.nansum(diff, axis=1)

    mean_as = float(np.nanmean(active_share_series))

    return MetricResult(
        name="active_share",
        value=mean_as,
        category=("relative", "exposure"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _ACTIVE_SHARE_REF,
            "series": active_share_series,
        },
    )
