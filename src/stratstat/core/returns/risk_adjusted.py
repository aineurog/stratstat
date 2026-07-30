"""Risk-adjusted return metrics.

Metrics: Sharpe ratio, Sortino ratio, Calmar ratio, Omega ratio, Sterling ratio,
Burke ratio, Kappa-3, Martin ratio, Gain-to-Pain ratio.

All tagged: category=("risk_adjusted", "returns"), backend="vectorized".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from stratstat.core._utils import compute_cagr
from stratstat.inputs import ReturnsInput
from stratstat.registry import register_metric
from stratstat.results import MetricResult

from .risk import _analyse_drawdowns, _drawdown_series, _equity_curve

# ---------------------------------------------------------------------------
# 3.1 Sharpe Ratio
# Reference: Sharpe (1966); Sharpe (1994)
# ---------------------------------------------------------------------------

_SHARPE_REF = (
    "Sharpe (1966), 'Mutual Fund Performance,' J. Business, 39(1); "
    "Sharpe (1994), 'The Sharpe Ratio,' JPM, 21(1)"
)


@register_metric(
    name="sharpe_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_SHARPE_REF,
)
def sharpe_ratio(
    input_data: ReturnsInput,
    rf: float = 0.0,
    ddof: int = 1,
) -> MetricResult:
    """Sharpe ratio — annualized excess return per unit of volatility.

    Formula:
        SR = ((mean(r) - rf) / std(r, ddof)) * sqrt(P)

    where P is ``periods_per_year``. Annualization is applied pre-division,
    equivalent to (r̄_excess * sqrt(P)) / sigma.

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.
        rf: Risk-free rate per period (default 0.0).
        ddof: Delta degrees of freedom for standard deviation — 1 for sample
            (default), 0 for population.

    Returns:
        MetricResult with Sharpe ratio (float or array).
    """
    if input_data.periods_per_year is None:
        raise ValueError(
            "Sharpe ratio requires periods_per_year on the ReturnsInput"
        )

    r = input_data.values  # (n_periods, n_strategies)
    p = float(input_data.periods_per_year)

    excess = np.nanmean(r, axis=0) - rf  # shape: (n_strategies,)
    sigma = np.nanstd(r, axis=0, ddof=ddof)  # shape: (n_strategies,)

    # Guard against zero (or near-zero) volatility.
    sigma_safe = np.where(sigma < 1e-15, np.nan, sigma)

    arr = (excess / sigma_safe) * np.sqrt(p)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="sharpe_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _SHARPE_REF,
            "rf": rf,
            "ddof": ddof,
        },
    )


# ---------------------------------------------------------------------------
# 3.2 Sortino Ratio
# Reference: Sortino & Price (1994)
# ---------------------------------------------------------------------------

_SORTINO_REF = (
    "Sortino & Price (1994), 'Performance Measurement in a Downside Risk Framework,' "
    "J. Investing, 3(3)"
)


@register_metric(
    name="sortino_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_SORTINO_REF,
)
def sortino_ratio(
    input_data: ReturnsInput,
    rf: float = 0.0,
    mar: float = 0.0,
    denominator: str = "full_downside",
) -> MetricResult:
    """Sortino ratio — excess return per unit of downside deviation.

    Formula:
        Sortino = ((mean(r) - rf) * P) / (DD * sqrt(P))

    which simplifies to (mean(r) - rf) * sqrt(P) / DD, where DD is the
    downside deviation below MAR.

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.
        rf: Risk-free rate per period (default 0.0).
        mar: Minimum acceptable return for downside deviation (default 0.0).
        denominator: ``"full_downside"`` (default) divides by sqrt of mean
            squared downside over *all* periods; ``"downside_only"`` divides by
            sqrt of mean squared downside over only downside periods.

    Returns:
        MetricResult with Sortino ratio (float or array).
    """
    if denominator not in ("full_downside", "downside_only"):
        raise ValueError(
            f"denominator must be 'full_downside' or 'downside_only', "
            f"got {denominator!r}"
        )

    if input_data.periods_per_year is None:
        raise ValueError(
            "Sortino ratio requires periods_per_year on the ReturnsInput"
        )

    r = input_data.values  # (n_periods, n_strategies)
    p = float(input_data.periods_per_year)

    excess_mean = np.nanmean(r, axis=0) - rf  # shape: (n_strategies,)

    # Downside deviation (vectorized, per column).
    # np.minimum(r - mar, 0.0) returns NaN for NaN inputs, and NaN < 0.0
    # evaluates to False in numpy → NaN is implicitly excluded from both
    # the count and sum below without explicit masking.
    below = np.minimum(r - mar, 0.0)  # shape: (n_periods, n_strategies)

    if denominator == "full_downside":
        dd = np.sqrt(np.nanmean(below**2, axis=0))
    else:
        # Count only strictly below-zero non-NaN periods.
        n_down = np.sum(~np.isnan(below) & (below < 0.0), axis=0).astype(np.float64)
        sum_sq_down = np.nansum(below**2, axis=0)
        dd = np.where(n_down > 0, np.sqrt(sum_sq_down / n_down), np.nan)

    dd_safe = np.where(dd < 1e-15, np.nan, dd)

    arr = excess_mean * np.sqrt(p) / dd_safe

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="sortino_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _SORTINO_REF,
            "rf": rf,
            "mar": mar,
            "denominator": denominator,
        },
    )


# ---------------------------------------------------------------------------
# 3.3 Calmar Ratio
# Reference: Young (1991)
# ---------------------------------------------------------------------------

_CALMAR_REF = "Young (1991), 'Calmar Ratio: A Smoother Tool,' Futures, 20(10)"


@register_metric(
    name="calmar_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_CALMAR_REF,
)
def calmar_ratio(input_data: ReturnsInput) -> MetricResult:
    """Calmar ratio — CAGR divided by the absolute value of max drawdown.

    Formula:
        Calmar = CAGR / |MDD|

    Requires ``periods_per_year`` on the input for CAGR annualization.

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.

    Returns:
        MetricResult with Calmar ratio (float or array). NaN is returned when
        max drawdown is zero (no drawdowns); +inf when MDD is non-negative and
        CAGR is positive.

    Raises:
        ValueError: If ``periods_per_year`` is None.
    """
    if input_data.periods_per_year is None:
        raise ValueError(
            "Calmar ratio requires periods_per_year on the ReturnsInput"
        )

    r = input_data.values
    p = float(input_data.periods_per_year)

    cagr_arr = compute_cagr(r, p)
    equity = _equity_curve(r, "simple")
    _, dd = _drawdown_series(equity)
    mdd = np.nanmin(dd, axis=0)  # most negative drawdown per strategy

    # Guard against zero drawdown.
    mdd_safe = np.where(np.abs(mdd) < 1e-15, np.nan, mdd)

    arr = cagr_arr / np.abs(mdd_safe)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="calmar_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _CALMAR_REF},
    )


# ---------------------------------------------------------------------------
# 3.4 Omega Ratio
# Reference: Keating & Shadwick (2002)
# ---------------------------------------------------------------------------

_OMEGA_REF = "Keating & Shadwick (2002), 'A Universal Performance Measure,' JPM, 6(3)"


@register_metric(
    name="omega_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_OMEGA_REF,
)
def omega_ratio(
    input_data: ReturnsInput, threshold: float = 0.0
) -> MetricResult:
    """Omega ratio — probability-weighted ratio of gains to losses.

    Formula:
        Omega(tau) = sum(max(r_t - tau, 0)) / |sum(min(r_t - tau, 0))|

    where tau is the threshold (default 0.0).

    Args:
        input_data: A ``ReturnsInput``.
        threshold: Return threshold (default 0.0).

    Returns:
        MetricResult with Omega ratio (float or array). +inf when there are
        no returns below the threshold (all upside); NaN when there are no
        returns at all.
    """
    r = input_data.values
    excess = r - threshold

    gains = np.nansum(np.maximum(excess, 0.0), axis=0)  # shape: (n_strategies,)
    losses = np.abs(np.nansum(np.minimum(excess, 0.0), axis=0))

    # Suppress divide-by-zero warning: np.where computes both branches.
    with np.errstate(divide="ignore", invalid="ignore"):
        arr = np.where(losses < 1e-15, np.inf, gains / losses)

    # If BOTH gains and losses are near-zero (empty or all-NaN input),
    # return NaN rather than inf — there is no data to evaluate.
    both_zero = (gains < 1e-15) & (losses < 1e-15)
    arr = np.where(both_zero, np.nan, arr)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="omega_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _OMEGA_REF,
            "threshold": threshold,
        },
    )


# ---------------------------------------------------------------------------
# 3.5 Sterling Ratio
# Reference: Bacon (2008, Sec. 8.3)
# ---------------------------------------------------------------------------

_STERLING_REF = "Bacon (2008, Practical Portfolio Performance Measurement, 2nd ed., Sec. 8.3)"


@register_metric(
    name="sterling_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_STERLING_REF,
)
def sterling_ratio(
    input_data: ReturnsInput, floor: float = 0.10
) -> MetricResult:
    """Sterling ratio — CAGR divided by average drawdown depth plus a floor.

    Formula:
        Sterling = CAGR / (|ADD| + k)

    where ADD is the average drawdown depth (negative) and k is a floor
    constant (default 0.10 = 10%) that prevents division by zero for
    strategies with very shallow drawdowns.

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.
        floor: Non-negative constant added to |ADD| (default 0.10).

    Returns:
        MetricResult with Sterling ratio (float or array).

    Raises:
        ValueError: If ``periods_per_year`` is None or ``floor`` is negative.
    """
    if input_data.periods_per_year is None:
        raise ValueError(
            "Sterling ratio requires periods_per_year on the ReturnsInput"
        )
    if floor < 0.0:
        raise ValueError(f"floor must be >= 0, got {floor}")

    r = input_data.values
    p = float(input_data.periods_per_year)

    cagr_arr = compute_cagr(r, p)

    # Average drawdown depth.
    _, _, _, all_episodes = _analyse_drawdowns(r)
    add_arr = np.zeros(r.shape[1], dtype=np.float64)
    for col in range(r.shape[1]):
        episodes = all_episodes[col]
        if episodes:
            depths = [ep["depth"] for ep in episodes]
            add_arr[col] = float(np.mean(depths))
        else:
            add_arr[col] = 0.0

    # ADD is negative; |ADD| + floor.
    denom = np.abs(add_arr) + floor
    arr = cagr_arr / denom

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="sterling_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _STERLING_REF,
            "floor": floor,
        },
    )


# ---------------------------------------------------------------------------
# 3.6 Burke Ratio
# Reference: See formula reference note on drawdown-based definition
# ---------------------------------------------------------------------------

_BURKE_REF = (
    "Industry convention (drawdown-based); differs from Bacon (2008, Sec. 8.5). "
    "See formula reference for details."
)


@register_metric(
    name="burke_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_BURKE_REF,
)
def burke_ratio(input_data: ReturnsInput) -> MetricResult:
    """Burke ratio — CAGR divided by the root of sum of squared drawdowns.

    Formula:
        Burke = CAGR / sqrt(sum(d_t^2))

    where d_t is the per-period percentage drawdown.

    Note: This drawdown-based definition differs from Bacon (2008, Sec. 8.5),
    which defines it as excess return over sqrt of mean squared negative
    returns. The drawdown-based form matches the project specification.

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.

    Returns:
        MetricResult with Burke ratio (float or array). NaN when there are
        no drawdowns (zero denominator).

    Raises:
        ValueError: If ``periods_per_year`` is None.
    """
    if input_data.periods_per_year is None:
        raise ValueError(
            "Burke ratio requires periods_per_year on the ReturnsInput"
        )

    r = input_data.values
    p = float(input_data.periods_per_year)

    cagr_arr = compute_cagr(r, p)

    equity = _equity_curve(r, "simple")
    _, dd = _drawdown_series(equity)  # dd values are <= 0

    denom = np.sqrt(np.nansum(dd**2, axis=0))
    denom_safe = np.where(denom < 1e-15, np.nan, denom)

    arr = cagr_arr / denom_safe

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="burke_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _BURKE_REF},
    )


# ---------------------------------------------------------------------------
# 3.7 Kappa-3
# Reference: Kaplan & Knowles (2004)
# ---------------------------------------------------------------------------

_KAPPA3_REF = (
    "Kaplan & Knowles (2004), 'Kappa: A Generalized Downside Risk-Adjusted "
    "Performance Measure,' JPM"
)


@register_metric(
    name="kappa_3",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_KAPPA3_REF,
)
def kappa_3(
    input_data: ReturnsInput, mar: float = 0.0
) -> MetricResult:
    """Kappa-3 — excess return over the cube root of the third lower partial moment.

    Formula:
        Kappa_3 = (mean(r) - mar) / (LPM_3)^(1/3)

    where LPM_3 = (1/n) * sum(max(mar - r_t, 0)^3) is the third lower
    partial moment.

    Args:
        input_data: A ``ReturnsInput``.
        mar: Minimum acceptable return (default 0.0).

    Returns:
        MetricResult with Kappa-3 (float or array). NaN when LPM_3 is zero
        (no downside risk).
    """
    r = input_data.values

    excess_mean = np.nanmean(r, axis=0) - mar  # shape: (n_strategies,)

    below = np.maximum(mar - r, 0.0)  # shape: (n_periods, n_strategies)
    lpm3 = np.nanmean(below**3, axis=0)  # third lower partial moment

    # Guard: if LPM_3 is zero, Kappa is undefined (NaN).
    lpm3_safe = np.where(lpm3 < 1e-15, np.nan, lpm3)

    arr = excess_mean / np.cbrt(lpm3_safe)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="kappa_3",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _KAPPA3_REF,
            "mar": mar,
        },
    )


# ---------------------------------------------------------------------------
# 3.8 Martin Ratio (CAGR / Ulcer Index)
# Reference: Martin & McCann (1989); Bacon (2008, Sec. 8.6)
# ---------------------------------------------------------------------------

_MARTIN_REF = (
    "Martin & McCann (1989), 'The Investor's Guide to Fidelity Funds'; "
    "Bacon (2008, Sec. 8.6)"
)


@register_metric(
    name="martin_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_MARTIN_REF,
)
def martin_ratio(input_data: ReturnsInput) -> MetricResult:
    """Martin ratio — CAGR divided by the Ulcer Index.

    Formula:
        Martin = CAGR / UI

    where UI is the Ulcer Index (root-mean-square of percentage drawdowns).

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.

    Returns:
        MetricResult with Martin ratio (float or array). NaN when Ulcer
        Index is zero (no drawdowns).

    Raises:
        ValueError: If ``periods_per_year`` is None.
    """
    if input_data.periods_per_year is None:
        raise ValueError(
            "Martin ratio requires periods_per_year on the ReturnsInput"
        )

    r = input_data.values
    p = float(input_data.periods_per_year)

    cagr_arr = compute_cagr(r, p)

    equity = _equity_curve(r, "simple")
    _, dd = _drawdown_series(equity)
    ui = np.sqrt(np.nanmean(dd**2, axis=0))  # Ulcer Index

    ui_safe = np.where(ui < 1e-15, np.nan, ui)

    arr = cagr_arr / ui_safe

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="martin_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _MARTIN_REF},
    )


# ---------------------------------------------------------------------------
# 3.9 Gain-to-Pain Ratio
# Reference: Bacon (2008, Sec. 8.4)
# ---------------------------------------------------------------------------

_GPR_REF = "Bacon (2008, Practical Portfolio Performance Measurement, 2nd ed., Sec. 8.4)"


@register_metric(
    name="gain_to_pain_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_GPR_REF,
)
def gain_to_pain_ratio(input_data: ReturnsInput) -> MetricResult:
    """Gain-to-Pain ratio — sum of gains divided by absolute sum of losses.

    Formula:
        GPR = sum(max(r_t, 0)) / |sum(min(r_t, 0))|

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with Gain-to-Pain ratio (float or array). +inf when
        there are no losses (all positive returns); NaN when there are no
        returns at all.
    """
    r = input_data.values

    gains = np.nansum(np.maximum(r, 0.0), axis=0)  # shape: (n_strategies,)
    losses = np.abs(np.nansum(np.minimum(r, 0.0), axis=0))

    # Suppress divide-by-zero warning: np.where computes both branches.
    with np.errstate(divide="ignore", invalid="ignore"):
        arr = np.where(losses < 1e-15, np.inf, gains / losses)

    # If BOTH gains and losses are near-zero (empty or all-NaN input),
    # return NaN rather than inf — there is no data to evaluate.
    both_zero = (gains < 1e-15) & (losses < 1e-15)
    arr = np.where(both_zero, np.nan, arr)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="gain_to_pain_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _GPR_REF},
    )
