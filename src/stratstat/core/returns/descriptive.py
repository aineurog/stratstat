"""Descriptive statistics for returns.

Metrics: CAGR, annualized volatility, cumulative return, arithmetic mean return,
geometric mean return, skewness, excess kurtosis, best/worst period, positive-period
ratio, autocorrelation (lag-1), variance, return range, percentiles, coefficient of
variation, outlier count & percentage (IQR method).

All metrics are tagged: category=("descriptive", "returns"), backend="vectorized".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from stratstat.core._utils import compute_cagr as _compute_cagr
from stratstat.inputs import ReturnsInput
from stratstat.registry import register_metric
from stratstat.results import MetricResult


# ---------------------------------------------------------------------------
# 1.1 CAGR — Compound Annual Growth Rate
# ---------------------------------------------------------------------------
@register_metric(
    name="cagr",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Damodaran (2012, Investment Valuation, 3rd ed., Ch. 3)",
)
def cagr(input_data: ReturnsInput) -> MetricResult:
    """Compound Annual Growth Rate.

    Formula:
        CAGR = exp(P * mean(ln(1 + r))) - 1

    where P is ``periods_per_year`` and the mean is taken over all periods.
    Equivalent to (V_f / V_i)^(1/T) - 1 where T is the series length in years.

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.

    Returns:
        MetricResult with annualized CAGR as a float (single strategy) or
        1-D array (multi-strategy).

    Raises:
        ValueError: If ``periods_per_year`` is None.
    """
    if input_data.periods_per_year is None:
        raise ValueError(
            "CAGR requires periods_per_year to be set on the ReturnsInput. "
            "Pass periods_per_year=252 (daily) or periods_per_year=12 (monthly)."
        )

    r = input_data.values  # (n_periods, n_strategies)
    p = float(input_data.periods_per_year)

    arr = _compute_cagr(r, p)  # shape: (n_strategies,)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="cagr",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": "Damodaran (2012, Investment Valuation, 3rd ed., Ch. 3)"},
    )


# ---------------------------------------------------------------------------
# 1.2 Annualized Volatility
# ---------------------------------------------------------------------------
@register_metric(
    name="annualized_volatility",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="CFA Institute, Quantitative Methods (CFA Program Curriculum, Level I, Vol. 1)",
)
def annualized_volatility(input_data: ReturnsInput) -> MetricResult:
    """Annualized volatility (standard deviation of returns).

    Formula:
        sigma_ann = sigma * sqrt(P)

    where sigma is the sample standard deviation (ddof=1) of period returns
    and P is ``periods_per_year``.

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.

    Returns:
        MetricResult with annualized volatility.

    Raises:
        ValueError: If ``periods_per_year`` is None.
    """
    if input_data.periods_per_year is None:
        raise ValueError(
            "Annualized volatility requires periods_per_year to be set on the ReturnsInput."
        )

    r = input_data.values
    p = float(input_data.periods_per_year)

    sigma = np.nanstd(r, axis=0, ddof=1)  # shape: (n_strategies,)
    arr = sigma * np.sqrt(p)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="annualized_volatility",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": (
                "CFA Institute, Quantitative Methods "
                "(CFA Program Curriculum, Level I, Vol. 1)"
            ),
            "ddof": 1,
        },
    )


# ---------------------------------------------------------------------------
# 1.3 Cumulative Return
# ---------------------------------------------------------------------------
@register_metric(
    name="cumulative_return",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Bacon (2008, Practical Portfolio Performance Measurement and Attribution,"
    " 2nd ed., Sec. 2.1)",
)
def cumulative_return(input_data: ReturnsInput) -> MetricResult:
    """Cumulative (total) return over the full period.

    Formula:
        R_cum = prod(1 + r_t) - 1

    NaN entries are treated as a factor of 1 (np.nanprod), which is
    equivalent to omitting those periods from the compound return.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with cumulative return.
    """
    r = input_data.values  # (n_periods, n_strategies)

    # np.nanprod treats NaN as 1, matching "omit NaN periods" semantics.
    arr = np.nanprod(1.0 + r, axis=0) - 1.0

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="cumulative_return",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": (
                "Bacon (2008, Practical Portfolio Performance Measurement "
                "and Attribution, 2nd ed., Sec. 2.1)"
            ),
        },
    )


# ---------------------------------------------------------------------------
# 1.4 Arithmetic Mean Return
# ---------------------------------------------------------------------------
@register_metric(
    name="arithmetic_mean_return",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Casella & Berger (2002, Statistical Inference, Sec. 5.2)",
)
def arithmetic_mean_return(input_data: ReturnsInput) -> MetricResult:
    """Arithmetic mean of period returns (native frequency).

    Formula:
        r_bar = (1/n) * sum(r_t)

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with mean return at the data's native frequency.
    """
    r = input_data.values
    arr = np.nanmean(r, axis=0)  # shape: (n_strategies,)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="arithmetic_mean_return",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": "Casella & Berger (2002, Statistical Inference, Sec. 5.2)"},
    )


# ---------------------------------------------------------------------------
# 1.5 Geometric Mean Return
# ---------------------------------------------------------------------------
@register_metric(
    name="geometric_mean_return",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Campbell, Lo & MacKinlay (1997, The Econometrics of Financial Markets, Sec. 1.4)",
)
def geometric_mean_return(input_data: ReturnsInput) -> MetricResult:
    """Geometric mean of period returns (native frequency, per-period).

    Formula:
        r_bar_g = exp(mean(ln(1 + r))) - 1

    This is the per-period geometric mean, NOT annualized. For the annualized
    version, see ``cagr``.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with geometric mean return at native frequency.
    """
    r = input_data.values

    log_returns = np.log(1.0 + r)
    mean_log = np.nanmean(log_returns, axis=0)  # shape: (n_strategies,)
    arr = np.exp(mean_log) - 1.0

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="geometric_mean_return",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": (
                "Campbell, Lo & MacKinlay (1997, The Econometrics of "
                "Financial Markets, Sec. 1.4)"
            ),
        },
    )


# ---------------------------------------------------------------------------
# 1.6 Skewness (bias-corrected sample skewness)
# ---------------------------------------------------------------------------
@register_metric(
    name="skewness",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Fisher (1930); Cramer (1946, Mathematical Methods of Statistics, Sec. 27.4)",
)
def skewness(input_data: ReturnsInput) -> MetricResult:
    """Bias-corrected (adjusted) sample skewness.

    Formula:
        gamma_1 = n / ((n-1)(n-2)) * sum(((r_t - r_bar) / sigma)^3)

    where sigma uses the sample standard deviation (ddof=1).

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with sample skewness.
    """
    r = input_data.values  # (n_periods, n_strategies)
    n = input_data.n_periods

    # Degenerate case: fewer than 3 observations
    if n < 3:
        nan_arr = np.full(r.shape[1], np.nan, dtype=np.float64)
        nan_value: float | NDArray[np.floating] = (
            float(nan_arr[0]) if input_data.is_single else nan_arr
        )
        return MetricResult(
            name="skewness",
            value=nan_value,
            category=("descriptive", "returns"),
            periods_per_year=input_data.periods_per_year,
            meta={
                "ref": (
                    "Fisher (1930); Cramer (1946, Mathematical Methods "
                    "of Statistics, Sec. 27.4)"
                ),
                "note": "Undefined for fewer than 3 observations.",
            },
        )

    mean = np.nanmean(r, axis=0, keepdims=True)  # (1, n_strategies)
    std = np.nanstd(r, axis=0, ddof=1, keepdims=True)  # (1, n_strategies)

    # Flag columns where std is effectively zero (constant series)
    zero_std = (std < 1e-15).squeeze(axis=0)  # (n_strategies,)

    # Avoid division by zero for constant series
    std_safe = np.where(std < 1e-15, np.nan, std)

    # NaN z-scores (from NaN returns in the original data, or from zero-std
    # columns via std_safe) are zeroed. This is equivalent to omitting those
    # observations from the moment sum — they contribute zero to m3.
    z = (r - mean) / std_safe  # (n_periods, n_strategies)
    z = np.where(np.isnan(z), 0.0, z)

    m3 = np.nansum(z**3, axis=0)  # (n_strategies,)
    # Count non-NaN observations per strategy column
    n_eff = np.sum(~np.isnan(r), axis=0).astype(np.float64)  # (n_strategies,)

    with np.errstate(invalid="ignore"):
        factor = n_eff / ((n_eff - 1.0) * (n_eff - 2.0))
        factor = np.where(n_eff < 3, np.nan, factor)
        arr = factor * m3
        # Constant returns (zero variance) -> undefined skewness
        arr = np.where(zero_std, np.nan, arr)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="skewness",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": (
                "Fisher (1930); Cramer (1946, Mathematical Methods "
                "of Statistics, Sec. 27.4)"
            ),
            "bias_corrected": True,
        },
    )


# ---------------------------------------------------------------------------
# 1.7 Excess Kurtosis (bias-corrected, fisher=True)
# ---------------------------------------------------------------------------
@register_metric(
    name="excess_kurtosis",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Fisher (1930); Cramer (1946, Mathematical Methods of Statistics, Sec. 27.4)",
)
def excess_kurtosis(input_data: ReturnsInput) -> MetricResult:
    """Bias-corrected sample excess kurtosis (fisher=True).

    Returns 0 for a normal distribution.

    Formula:
        gamma_2 = [n(n+1) / ((n-1)(n-2)(n-3))] * sum(((r - r_bar)/sigma)^4)
                  - 3(n-1)^2 / ((n-2)(n-3))

    where sigma uses the sample standard deviation (ddof=1).

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with sample excess kurtosis.
    """
    r = input_data.values
    n = input_data.n_periods

    # Degenerate case: fewer than 4 observations
    if n < 4:
        nan_arr = np.full(r.shape[1], np.nan, dtype=np.float64)
        nan_value: float | NDArray[np.floating] = (
            float(nan_arr[0]) if input_data.is_single else nan_arr
        )
        return MetricResult(
            name="excess_kurtosis",
            value=nan_value,
            category=("descriptive", "returns"),
            periods_per_year=input_data.periods_per_year,
            meta={
                "ref": (
                    "Fisher (1930); Cramer (1946, Mathematical Methods "
                    "of Statistics, Sec. 27.4)"
                ),
                "note": "Undefined for fewer than 4 observations.",
            },
        )

    mean = np.nanmean(r, axis=0, keepdims=True)
    std = np.nanstd(r, axis=0, ddof=1, keepdims=True)
    zero_std = (std < 1e-15).squeeze(axis=0)  # (n_strategies,)
    std_safe = np.where(std < 1e-15, np.nan, std)

    # NaN z-scores (from NaN returns or zero-std columns) are zeroed.
    # This is equivalent to omitting those observations — they contribute zero to m4.
    z = (r - mean) / std_safe
    z = np.where(np.isnan(z), 0.0, z)

    m4 = np.nansum(z**4, axis=0)
    n_eff = np.sum(~np.isnan(r), axis=0).astype(np.float64)

    with np.errstate(invalid="ignore"):
        term1 = n_eff * (n_eff + 1.0) / (
            (n_eff - 1.0) * (n_eff - 2.0) * (n_eff - 3.0)
        )
        term2 = 3.0 * (n_eff - 1.0) ** 2 / ((n_eff - 2.0) * (n_eff - 3.0))
        arr = term1 * m4 - term2
        arr = np.where(n_eff < 4, np.nan, arr)
        # Constant returns (zero variance) -> undefined kurtosis
        arr = np.where(zero_std, np.nan, arr)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="excess_kurtosis",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": (
                "Fisher (1930); Cramer (1946, Mathematical Methods "
                "of Statistics, Sec. 27.4)"
            ),
            "bias_corrected": True,
            "fisher": True,
        },
    )


# ---------------------------------------------------------------------------
# 1.8 Best Period
# ---------------------------------------------------------------------------
@register_metric(
    name="best_period",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Bacon (2008, Practical Portfolio Performance Measurement and Attribution,"
    " 2nd ed., Sec. 3.10)",
)
def best_period(input_data: ReturnsInput) -> MetricResult:
    """Maximum single-period return at the data's native frequency.

    Formula:
        r_best = max_t r_t

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with the best (maximum) single-period return.
    """
    r = input_data.values
    arr = np.nanmax(r, axis=0)  # shape: (n_strategies,)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="best_period",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": (
                "Bacon (2008, Practical Portfolio Performance Measurement "
                "and Attribution, 2nd ed., Sec. 3.10)"
            ),
        },
    )


# ---------------------------------------------------------------------------
# 1.9 Worst Period
# ---------------------------------------------------------------------------
@register_metric(
    name="worst_period",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Bacon (2008, Practical Portfolio Performance Measurement and Attribution,"
    " 2nd ed., Sec. 3.10)",
)
def worst_period(input_data: ReturnsInput) -> MetricResult:
    """Minimum single-period return at the data's native frequency.

    Formula:
        r_worst = min_t r_t

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with the worst (minimum) single-period return.
    """
    r = input_data.values
    arr = np.nanmin(r, axis=0)  # shape: (n_strategies,)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="worst_period",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": (
                "Bacon (2008, Practical Portfolio Performance Measurement "
                "and Attribution, 2nd ed., Sec. 3.10)"
            ),
        },
    )


# ---------------------------------------------------------------------------
# 1.10 Positive-Period Ratio
# ---------------------------------------------------------------------------
@register_metric(
    name="positive_period_ratio",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Bacon (2008, Practical Portfolio Performance Measurement and Attribution,"
    " 2nd ed., Sec. 3.11)",
)
def positive_period_ratio(input_data: ReturnsInput) -> MetricResult:
    """Fraction of periods with strictly positive return (> 0).

    Formula:
        PPR = (1/n) * sum(1_{r_t > 0})

    Zero is treated as non-positive.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with positive-period ratio.
    """
    r = input_data.values
    valid = ~np.isnan(r)  # boolean mask of non-NaN entries
    positive = r > 0  # NaN > 0 is False, but we'll mask it anyway
    n_valid: NDArray[np.floating] = np.sum(valid, axis=0).astype(np.float64)
    n_positive: NDArray[np.floating] = np.sum(positive & valid, axis=0).astype(np.float64)

    with np.errstate(invalid="ignore"):
        arr = n_positive / np.maximum(n_valid, 1.0)
        arr = np.where(n_valid == 0, np.nan, arr)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="positive_period_ratio",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": (
                "Bacon (2008, Practical Portfolio Performance Measurement "
                "and Attribution, 2nd ed., Sec. 3.11)"
            ),
        },
    )


# ---------------------------------------------------------------------------
# 1.11 Autocorrelation (Lag-1)
# ---------------------------------------------------------------------------
@register_metric(
    name="autocorrelation",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Campbell, Lo & MacKinlay (1997, The Econometrics of Financial Markets, Sec. 2.4)",
)
def autocorrelation(input_data: ReturnsInput) -> MetricResult:
    """Lag-1 autocorrelation (Pearson correlation of r[t] with r[t+1]).

    Formula:
        rho_1 = sum((r_t - r_bar)(r_{t-1} - r_bar)) / sum((r_t - r_bar)^2)

    where the sums run over t=2..n for the numerator and t=1..n for the
    denominator (Campbell, Lo & MacKinlay convention).

    Args:
        input_data: A ``ReturnsInput``. Must have at least 2 periods.

    Returns:
        MetricResult with lag-1 autocorrelation.
    """
    r = input_data.values  # (n_periods, n_strategies)
    n = input_data.n_periods

    arr = np.zeros(r.shape[1], dtype=np.float64)

    if n < 2:
        arr[:] = np.nan
    else:
        for col in range(r.shape[1]):
            col_data = r[:, col]
            valid = ~np.isnan(col_data)
            n_valid = np.sum(valid)
            if n_valid < 2:
                arr[col] = np.nan
                continue

            # Full-sample mean over all non-NaN observations.
            mean_val = np.mean(col_data[valid])

            # Centred series with NaN positions zeroed (contribute nothing to sums).
            centered = np.where(valid, col_data - mean_val, 0.0)

            # Denominator: sum of squared deviations over all non-NaN periods.
            denom = np.sum(centered[valid] ** 2)

            # Numerator: only pairs that are truly consecutive in the original
            # series (i.e. both r[t] and r[t+1] are non-NaN).
            valid_pair = valid[:-1] & valid[1:]
            num = np.sum(centered[:-1][valid_pair] * centered[1:][valid_pair])

            if denom < 1e-30:
                arr[col] = np.nan
            else:
                arr[col] = num / denom

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="autocorrelation",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": (
                "Campbell, Lo & MacKinlay (1997, The Econometrics of "
                "Financial Markets, Sec. 2.4)"
            ),
        },
    )


# ---------------------------------------------------------------------------
# 1.12 Variance
# ---------------------------------------------------------------------------
@register_metric(
    name="variance",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Fisher (1925, Statistical Methods for Research Workers)",
)
def variance(input_data: ReturnsInput) -> MetricResult:
    """Sample variance of returns (ddof=1).

    Formula:
        s^2 = (1/(n-1)) * sum((r_t - r_bar)^2)

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with sample variance.
    """
    r = input_data.values
    arr = np.nanvar(r, axis=0, ddof=1)  # shape: (n_strategies,)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="variance",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": "Fisher (1925, Statistical Methods for Research Workers)",
            "ddof": 1,
        },
    )


# ---------------------------------------------------------------------------
# 1.13 Return Range
# ---------------------------------------------------------------------------
@register_metric(
    name="return_range",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Bacon (2008, Practical Portfolio Performance Measurement and Attribution,"
    " 2nd ed., Sec. 3.10)",
)
def return_range(input_data: ReturnsInput) -> MetricResult:
    """Range of period returns (max minus min).

    Formula:
        R_range = max_t r_t - min_t r_t

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with return range.
    """
    r = input_data.values
    max_vals = np.nanmax(r, axis=0)
    min_vals = np.nanmin(r, axis=0)
    arr = max_vals - min_vals

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="return_range",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": (
                "Bacon (2008, Practical Portfolio Performance Measurement "
                "and Attribution, 2nd ed., Sec. 3.10)"
            ),
        },
    )


# ---------------------------------------------------------------------------
# 1.14 Percentiles
# ---------------------------------------------------------------------------
_PERCENTILE_LEVELS = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=np.float64)

_PERCENTILE_REF = (
    'Hyndman & Fan (1996), "Sample Quantiles in Statistical Packages,"'
    " The American Statistician, 50(4)"
)


@register_metric(
    name="percentiles",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref=_PERCENTILE_REF,
)
def percentiles(input_data: ReturnsInput) -> MetricResult:
    """Percentiles of the empirical return distribution.

    Computes percentiles at levels [1, 5, 10, 25, 50, 75, 90, 95, 99]
    using linear interpolation (Hyndman & Fan 1996, type 7, the default
    for ``numpy.percentile``).

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult whose value is a 1-D array of shape (9,) for a single
        strategy, or a 2-D array of shape (9, n_strategies) for multiple
        strategies. The ``meta`` field records the percentile levels.
    """
    r = input_data.values  # (n_periods, n_strategies)

    # np.nanpercentile along axis 0 returns (len(levels), n_strategies)
    arr: NDArray[np.floating] = np.nanpercentile(
        r, _PERCENTILE_LEVELS, axis=0, method="linear"
    )

    # For a single strategy, squeeze to 1-D
    if input_data.is_single:
        value: float | NDArray[np.floating] = arr.squeeze(axis=1)  # (9,)
    else:
        value = arr

    return MetricResult(
        name="percentiles",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _PERCENTILE_REF,
            "levels": _PERCENTILE_LEVELS.tolist(),
            "method": "linear (type 7)",
        },
    )


# ---------------------------------------------------------------------------
# 1.15 Coefficient of Variation
# ---------------------------------------------------------------------------
@register_metric(
    name="coefficient_of_variation",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Pearson (1896); Everitt & Skrondal (2010, The Cambridge Dictionary of Statistics)",
)
def coefficient_of_variation(input_data: ReturnsInput) -> MetricResult:
    """Coefficient of variation (relative standard deviation).

    Formula:
        CV = sigma / |r_bar|

    where sigma is the sample standard deviation (ddof=1) and r_bar is the
    arithmetic mean. Returns NaN when the mean is zero.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with coefficient of variation.
    """
    r = input_data.values

    mean = np.nanmean(r, axis=0)  # (n_strategies,)
    std = np.nanstd(r, axis=0, ddof=1)  # (n_strategies,)

    with np.errstate(invalid="ignore"):
        arr = std / np.abs(mean)
        # Where mean is effectively zero, result is undefined
        arr = np.where(np.abs(mean) < 1e-15, np.nan, arr)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="coefficient_of_variation",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": (
                "Pearson (1896); Everitt & Skrondal (2010, "
                "The Cambridge Dictionary of Statistics)"
            ),
        },
    )


# ---------------------------------------------------------------------------
# 1.16 Outlier Count & % (IQR Method)
# ---------------------------------------------------------------------------
@register_metric(
    name="outlier_iqr",
    requires="returns",
    category=("descriptive", "returns"),
    backend="vectorized",
    ref="Tukey (1977, Exploratory Data Analysis)",
)
def outlier_iqr(input_data: ReturnsInput) -> MetricResult:
    """Outlier count and percentage using the IQR (interquartile range) method.

    A return is an outlier if:
        r_t < Q1 - 1.5 * IQR   or   r_t > Q3 + 1.5 * IQR

    where IQR = Q3 - Q1 and Q1, Q3 are the 25th and 75th percentiles.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult whose value is a 1-D array ``[count, percentage]`` for a
        single strategy, or a 2-D array of shape ``(2, n_strategies)`` for
        multiple strategies (row 0 = outlier counts, row 1 = outlier percentages).
        The ``meta`` field documents the indices via ``output_index``.
    """
    r = input_data.values  # (n_periods, n_strategies)

    q1 = np.nanpercentile(r, 25, axis=0)  # (n_strategies,)
    q3 = np.nanpercentile(r, 75, axis=0)  # (n_strategies,)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # Comparisons with NaN→False per numpy semantics, so NaN positions are
    # implicitly excluded from the outlier mask without an explicit skip.
    outlier_mask = (r < lower) | (r > upper)  # (n_periods, n_strategies)
    count_arr = np.nansum(
        outlier_mask.astype(np.float64), axis=0
    ).astype(np.float64)  # (n_strategies,)
    n_eff = np.sum(~np.isnan(r), axis=0).astype(np.float64)  # (n_strategies,)
    with np.errstate(invalid="ignore"):
        pct_arr = count_arr / n_eff * 100.0
        pct_arr = np.where(n_eff == 0, np.nan, pct_arr)

    # Stack as (2, n_strategies): row 0 = count, row 1 = percentage
    arr: NDArray[np.floating] = np.stack([count_arr, pct_arr], axis=0)

    if input_data.is_single:
        value: NDArray[np.floating] | float = arr.squeeze(axis=1)  # shape (2,)
    else:
        value = arr

    return MetricResult(
        name="outlier_iqr",
        value=value,
        category=("descriptive", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": "Tukey (1977, Exploratory Data Analysis)",
            "method": "IQR (1.5 * IQR beyond Q1/Q3)",
            "output_index": ["count", "percentage"],
        },
    )
