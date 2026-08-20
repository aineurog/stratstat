"""Risk-adjusted return metrics.

Metrics: Sharpe ratio, Sortino ratio, Calmar ratio, Omega ratio, Sterling ratio,
Burke ratio, Kappa-3, Martin ratio, Gain-to-Pain ratio, pain ratio, recovery
factor, K-ratio, serenity ratio, UPI (Ulcer Performance Index), modified Sharpe
ratio, upside potential ratio, risk return ratio.

All tagged: category=("risk_adjusted", "returns"), backend="vectorized".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from stratstat.conventions import resolve_convention
from stratstat.core._utils import compute_cagr
from stratstat.exceptions import MetricNotApplicableError
from stratstat.inputs import ReturnsInput
from stratstat.registry import register_metric
from stratstat.results import MetricResult

from .risk import _analyse_drawdowns, _drawdown_series, _equity_curve, _var_cornish_fisher

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
    ddof: int | None = None,
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
            (default), 0 for population. Overridden by a session default set
            via ``stratstat.set_default("sharpe_ratio", "ddof=...")``.

    Returns:
        MetricResult with Sharpe ratio (float or array).
    """
    ddof = resolve_convention(ddof, "sharpe_ratio", "ddof", 1)

    if input_data.periods_per_year is None:
        raise MetricNotApplicableError(
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
    denominator: str | None = None,
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
            Overridden by a session default set via
            ``stratstat.set_default("sortino_ratio", "denominator=...")``.

    Returns:
        MetricResult with Sortino ratio (float or array).
    """
    denominator = resolve_convention(
        denominator, "sortino_ratio", "denominator", "full_downside"
    )

    if denominator not in ("full_downside", "downside_only"):
        raise ValueError(
            f"denominator must be 'full_downside' or 'downside_only', "
            f"got {denominator!r}"
        )

    if input_data.periods_per_year is None:
        raise MetricNotApplicableError(
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
        raise MetricNotApplicableError(
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
        raise MetricNotApplicableError(
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
        raise MetricNotApplicableError(
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
        raise MetricNotApplicableError(
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


# ---------------------------------------------------------------------------
# 3.10 Pain Ratio
# Reference: Zephyr Associates
# ---------------------------------------------------------------------------

_PAIN_RATIO_REF = "Zephyr Associates"


@register_metric(
    name="pain_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_PAIN_RATIO_REF,
)
def pain_ratio(input_data: ReturnsInput) -> MetricResult:
    """Pain Ratio — CAGR divided by the absolute value of the Pain Index.

    Formula:
        Pain Ratio = CAGR / |PI|

    where PI = mean(d_t) over all periods (including zero-drawdown periods).
    Higher values indicate a better return per unit of drawdown pain.

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.

    Returns:
        MetricResult with Pain Ratio (float or array). +inf when Pain Index
        is zero (no drawdowns at all). NaN when CAGR is undefined.

    Raises:
        ValueError: If ``periods_per_year`` is None.
    """
    if input_data.periods_per_year is None:
        raise MetricNotApplicableError(
            "Pain ratio requires periods_per_year on the ReturnsInput"
        )

    r = input_data.values
    p = float(input_data.periods_per_year)

    cagr_arr = compute_cagr(r, p)

    equity = _equity_curve(r, "simple")
    _, dd = _drawdown_series(equity)
    pain_idx = np.nanmean(dd, axis=0)  # Pain Index = mean of all drawdowns

    with np.errstate(divide="ignore", invalid="ignore"):
        arr = np.where(np.abs(pain_idx) < 1e-15, np.inf, cagr_arr / np.abs(pain_idx))

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="pain_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _PAIN_RATIO_REF},
    )


# ---------------------------------------------------------------------------
# 3.11 Recovery Factor
# Reference: Industry convention
# ---------------------------------------------------------------------------

_RECOVERY_FACTOR_REF = "Industry convention"


@register_metric(
    name="recovery_factor",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_RECOVERY_FACTOR_REF,
)
def recovery_factor(input_data: ReturnsInput) -> MetricResult:
    """Recovery Factor — total cumulative return divided by absolute max drawdown.

    Formula:
        RF = total_cumulative_return / |MDD|

    Where total_cumulative_return = prod(1+r) - 1 and MDD is the maximum
    drawdown (a negative number). Uses TOTAL return, not CAGR (which is
    the Calmar ratio).

    A 100k→150k strategy with a -20% max drawdown:
        RF = 0.5 / 0.2 = 2.5

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with Recovery Factor (float or array). +inf when
        max drawdown is zero; NaN when cumulative return is undefined.
    """
    r = input_data.values

    total_return = np.nanprod(1.0 + r, axis=0) - 1.0

    equity = _equity_curve(r, "simple")
    _, dd = _drawdown_series(equity)
    mdd = np.nanmin(dd, axis=0)  # most negative drawdown

    with np.errstate(divide="ignore", invalid="ignore"):
        arr = np.where(np.abs(mdd) < 1e-15, np.inf, total_return / np.abs(mdd))

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="recovery_factor",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _RECOVERY_FACTOR_REF},
    )


# ---------------------------------------------------------------------------
# 3.12 K-Ratio
# Reference: Kestner (1996), revised 2003
# ---------------------------------------------------------------------------

_K_RATIO_REF = "Kestner (1996), revised 2003"


@register_metric(
    name="k_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_K_RATIO_REF,
)
def k_ratio(input_data: ReturnsInput) -> MetricResult:
    """K-Ratio — slope of the log(VAMI) regression line divided by its standard error.

    Measures the consistency of equity curve growth. A higher K-Ratio
    indicates a smoother, more linear equity curve. Penalises erratic
    or inconsistent growth patterns.

    Formula:
        y_t = sum_{tau=1}^{t} ln(1 + r_tau)   (log VAMI)
        y_t = alpha + beta * t + epsilon_t
        K = beta / SE(beta)

    where SE(beta) = sqrt( MSE / sum((t - t_bar)^2) ).

    Requires at least 3 periods.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with K-Ratio (float or array). NaN for fewer than 3
        observations or when the equity curve is perfectly flat.
    """
    r = input_data.values  # (n_periods, n_strategies)
    n = input_data.n_periods
    n_strat = r.shape[1]

    if n < 3:
        nan_arr = np.full(n_strat, np.nan, dtype=np.float64)
        nan_value: float | NDArray[np.floating] = (
            float(nan_arr[0]) if input_data.is_single else nan_arr
        )
        return MetricResult(
            name="k_ratio",
            value=nan_value,
            category=("risk_adjusted", "returns"),
            periods_per_year=input_data.periods_per_year,
            meta={
                "ref": _K_RATIO_REF,
                "note": "Requires at least 3 observations.",
            },
        )

    # Time index 1..n
    x = np.arange(1, n + 1, dtype=np.float64)
    x_bar = np.mean(x)

    arr = np.zeros(n_strat, dtype=np.float64)
    for col in range(n_strat):
        col_data = r[:, col]
        valid = ~np.isnan(col_data)
        n_valid = int(np.sum(valid))
        if n_valid < 3:
            arr[col] = np.nan
            continue

        # Log VAMI = cumulative sum of log(1+r)
        log_ret = np.where(valid, np.log(1.0 + col_data), 0.0)
        y = np.cumsum(log_ret)  # (n,)

        y_valid = y[valid]
        x_valid = x[valid]
        y_bar = np.mean(y_valid)

        # OLS slope
        ss_xy = float(np.sum((x_valid - x_bar) * (y_valid - y_bar)))
        ss_xx_valid = float(np.sum((x_valid - x_bar) ** 2))

        if ss_xx_valid < 1e-30:
            arr[col] = np.nan
            continue

        beta = ss_xy / ss_xx_valid
        alpha = y_bar - beta * x_bar

        y_pred = alpha + beta * x_valid
        residuals = y_valid - y_pred
        mse = np.sum(residuals**2) / (n_valid - 2) if n_valid > 2 else np.inf

        if mse < 1e-30:
            # Perfect fit — infinite K-Ratio (or cap?)
            arr[col] = np.inf
        else:
            se_beta = np.sqrt(mse / ss_xx_valid)
            arr[col] = beta / se_beta

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="k_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _K_RATIO_REF},
    )


# ---------------------------------------------------------------------------
# 3.13 Serenity Ratio
# Reference: Industry metric; used by PortfolioMetrics
# ---------------------------------------------------------------------------

_SERENITY_REF = "Industry metric; used by PortfolioMetrics"


@register_metric(
    name="serenity_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_SERENITY_REF,
)
def serenity_ratio(
    input_data: ReturnsInput, rf: float = 0.0
) -> MetricResult:
    """Serenity Ratio — excess return divided by the product of volatility and Ulcer Index.

    Formula:
        SR = (Rp - Rf) / (sigma_p * UI)

    where Rp is the annualized arithmetic mean return, Rf is the annualized
    risk-free rate, sigma_p is the annualized volatility, and UI is the
    Ulcer Index. Both volatility risk and drawdown risk must be low for a
    high score.

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.
        rf: Risk-free rate per period (default 0.0).

    Returns:
        MetricResult with Serenity Ratio (float or array). NaN when
        volatility or Ulcer Index is zero.

    Raises:
        ValueError: If ``periods_per_year`` is None.
    """
    if input_data.periods_per_year is None:
        raise MetricNotApplicableError(
            "Serenity ratio requires periods_per_year on the ReturnsInput"
        )

    r = input_data.values
    p = float(input_data.periods_per_year)

    mean_ret = np.nanmean(r, axis=0)  # per-period mean
    ann_ret = mean_ret * p  # annualized
    rf_ann = rf * p  # annualized risk-free
    excess = ann_ret - rf_ann

    sigma = np.nanstd(r, axis=0, ddof=1) * np.sqrt(p)  # annualized vol

    equity = _equity_curve(r, "simple")
    _, dd = _drawdown_series(equity)
    ui = np.sqrt(np.nanmean(dd**2, axis=0))  # Ulcer Index

    denom = sigma * ui
    with np.errstate(divide="ignore", invalid="ignore"):
        arr = np.where(denom < 1e-15, np.nan, excess / denom)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="serenity_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _SERENITY_REF,
            "rf": rf,
        },
    )


# ---------------------------------------------------------------------------
# 3.14 UPI (Ulcer Performance Index)
# Reference: Martin & McCann (1989)
# ---------------------------------------------------------------------------

_UPI_REF = "Martin & McCann (1989, 'The Investor's Guide to Fidelity Funds')"


@register_metric(
    name="upi",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_UPI_REF,
)
def upi(
    input_data: ReturnsInput, rf: float = 0.0
) -> MetricResult:
    """Ulcer Performance Index — excess return divided by the Ulcer Index.

    Formula:
        UPI = (Rp - Rf) / UI

    where Rp is the annualized arithmetic mean return, Rf is the
    annualized risk-free rate, and UI is the Ulcer Index.

    Differs from the Martin ratio (which uses CAGR in the numerator):
        Martin = CAGR / UI
        UPI    = (Rp - Rf) / UI

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.
        rf: Risk-free rate per period (default 0.0).

    Returns:
        MetricResult with UPI (float or array). NaN when Ulcer Index
        is zero.

    Raises:
        ValueError: If ``periods_per_year`` is None.
    """
    if input_data.periods_per_year is None:
        raise MetricNotApplicableError(
            "UPI requires periods_per_year on the ReturnsInput"
        )

    r = input_data.values
    p = float(input_data.periods_per_year)

    mean_ret = np.nanmean(r, axis=0)
    ann_ret = mean_ret * p
    rf_ann = rf * p
    excess = ann_ret - rf_ann

    equity = _equity_curve(r, "simple")
    _, dd = _drawdown_series(equity)
    ui = np.sqrt(np.nanmean(dd**2, axis=0))

    ui_safe = np.where(ui < 1e-15, np.nan, ui)
    with np.errstate(invalid="ignore"):
        arr = excess / ui_safe

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="upi",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _UPI_REF,
            "rf": rf,
        },
    )


# ---------------------------------------------------------------------------
# 3.15 Modified Sharpe Ratio
# Reference: Gregoriou & Gueyie (2003)
# ---------------------------------------------------------------------------

_MOD_SHARPE_REF = "Gregoriou & Gueyie (2003)"


@register_metric(
    name="modified_sharpe_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_MOD_SHARPE_REF,
)
def modified_sharpe_ratio(
    input_data: ReturnsInput,
    rf: float = 0.0,
    confidence: float = 0.95,
) -> MetricResult:
    """Modified Sharpe Ratio — excess return divided by Modified VaR.

    Formula:
        MSR = (Rp - Rf) / Modified_VaR

    where Modified VaR uses the Cornish-Fisher expansion to adjust for
    skewness and excess kurtosis (rather than assuming normality).

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.
        rf: Risk-free rate per period (default 0.0).
        confidence: Confidence level for Modified VaR (default 0.95).

    Returns:
        MetricResult with Modified Sharpe Ratio (float or array). NaN
        when Modified VaR is zero or undefined.

    Raises:
        ValueError: If ``periods_per_year`` is None.
    """
    if input_data.periods_per_year is None:
        raise MetricNotApplicableError(
            "Modified Sharpe ratio requires periods_per_year on the ReturnsInput"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence must be in (0, 1), got {confidence}"
        )

    r = input_data.values
    p = float(input_data.periods_per_year)

    mean_ret = np.nanmean(r, axis=0)
    ann_ret = mean_ret * p
    rf_ann = rf * p
    excess = ann_ret - rf_ann

    # Modified VaR using Cornish-Fisher (annualized)
    mod_var = _var_cornish_fisher(r, confidence) * np.sqrt(p)

    mod_var_safe = np.where(np.abs(mod_var) < 1e-15, np.nan, mod_var)
    with np.errstate(invalid="ignore"):
        arr = excess / mod_var_safe

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="modified_sharpe_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _MOD_SHARPE_REF,
            "rf": rf,
            "confidence": confidence,
        },
    )


# ---------------------------------------------------------------------------
# 3.16 Upside Potential Ratio
# Reference: Sortino, van der Meer & Plantinga (1999)
# ---------------------------------------------------------------------------

_UPR_REF = "Sortino, van der Meer & Plantinga (1999)"


@register_metric(
    name="upside_potential_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_UPR_REF,
)
def upside_potential_ratio(
    input_data: ReturnsInput, mar: float = 0.0
) -> MetricResult:
    """Upside Potential Ratio — upside potential divided by downside deviation.

    Formula:
        UPR = upside_potential / DD

    where:
        upside_potential = (1/n) * sum(max(r_t - mar, 0))
        DD = sqrt((1/n) * sum(min(r_t - mar, 0)^2))

    This is a Sortino variant that replaces the mean excess return in the
    numerator with upside potential (average of only positive excess returns).

    Args:
        input_data: A ``ReturnsInput``.
        mar: Minimum acceptable return (default 0.0).

    Returns:
        MetricResult with Upside Potential Ratio (float or array). +inf
        when downside deviation is zero (no downside); NaN when there is
        no upside potential either.
    """
    r = input_data.values

    excess = r - mar
    upside = np.maximum(excess, 0.0)  # positive excess returns
    downside = np.minimum(excess, 0.0)  # negative excess returns

    up_pot = np.nanmean(upside, axis=0)  # upside potential
    dd = np.sqrt(np.nanmean(downside**2, axis=0))  # downside deviation

    with np.errstate(divide="ignore", invalid="ignore"):
        arr = np.where(dd < 1e-15, np.inf, up_pot / dd)

    both_zero = (up_pot < 1e-15) & (dd < 1e-15)
    arr = np.where(both_zero, np.nan, arr)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="upside_potential_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _UPR_REF,
            "mar": mar,
        },
    )


# ---------------------------------------------------------------------------
# 3.17 Risk Return Ratio
# Reference: Industry convention
# ---------------------------------------------------------------------------

_RISK_RETURN_REF = "Industry convention"


@register_metric(
    name="risk_return_ratio",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_RISK_RETURN_REF,
)
def risk_return_ratio(input_data: ReturnsInput) -> MetricResult:
    """Risk Return Ratio — annualized return divided by absolute max drawdown.

    Formula:
        RRR = annualized_return / |MDD|

    where annualized_return = mean(r) * periods_per_year and MDD is the
    maximum drawdown. Simpler than the Calmar ratio (which uses CAGR).
    Both use MDD as the risk denominator.

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.

    Returns:
        MetricResult with Risk Return Ratio (float or array). +inf when
        max drawdown is zero.

    Raises:
        ValueError: If ``periods_per_year`` is None.
    """
    if input_data.periods_per_year is None:
        raise MetricNotApplicableError(
            "Risk return ratio requires periods_per_year on the ReturnsInput"
        )

    r = input_data.values
    p = float(input_data.periods_per_year)

    ann_ret = np.nanmean(r, axis=0) * p  # annualized arithmetic return

    equity = _equity_curve(r, "simple")
    _, dd = _drawdown_series(equity)
    mdd = np.nanmin(dd, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        arr = np.where(np.abs(mdd) < 1e-15, np.inf, ann_ret / np.abs(mdd))

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="risk_return_ratio",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _RISK_RETURN_REF},
    )


# ===================================================================
# §3.18  Roy's Safety-First Ratio
# ===================================================================

_ROYS_SAFETY_FIRST_REF = (
    'Roy (1952), "Safety First and the Holding of Assets," Econometrica, 20(3).'
)


@register_metric(
    name="roys_safety_first",
    requires="returns",
    category=("risk_adjusted", "returns"),
    backend="vectorized",
    ref=_ROYS_SAFETY_FIRST_REF,
)
def roys_safety_first(
    input_data: ReturnsInput,
    mar: float = 0.0,
    ddof: int = 1,
) -> MetricResult:
    """Roy's Safety-First Ratio — excess return over MAR per unit of total risk.

    Formula:
        RSF = (R̄_p − MAR) / σ_p

    where R̄_p is the annualized mean return, MAR is the minimum acceptable
    return (annualized), and σ_p is the annualized standard deviation.

    Unlike Sharpe, which uses excess over the risk-free rate, Roy uses a
    minimum acceptable return (MAR) set by the investor.

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.
        mar: Minimum acceptable return as a *period* rate (default 0.0).
            Annualized internally for the ratio.
        ddof: Delta degrees of freedom for standard deviation (default 1).

    Returns:
        MetricResult with Roy's Safety-First ratio.
    """
    if input_data.periods_per_year is None:
        raise MetricNotApplicableError(
            "Roy's Safety-First ratio requires periods_per_year on the "
            "ReturnsInput"
        )

    r = input_data.values
    p = float(input_data.periods_per_year)

    mean_p = np.nanmean(r, axis=0)
    sigma_p = np.nanstd(r, axis=0, ddof=ddof)

    # Annualize: R̄_ann = mean_p * p, σ_ann = sigma_p * sqrt(p)
    ann_mean = mean_p * p
    ann_sigma = sigma_p * np.sqrt(p)
    ann_mar = mar * p

    sigma_safe = np.where(ann_sigma < 1e-15, np.nan, ann_sigma)
    arr = (ann_mean - ann_mar) / sigma_safe

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="roys_safety_first",
        value=value,
        category=("risk_adjusted", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _ROYS_SAFETY_FIRST_REF,
            "mar": mar,
            "ddof": ddof,
        },
    )
