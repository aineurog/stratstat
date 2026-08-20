"""Inference statistics for returns.

Metrics: Jarque-Bera statistic, Probabilistic Sharpe Ratio (PSR),
Deflated Sharpe Ratio (DSR), Lo's autocorrelation-adjusted Sharpe SE,
Sharpe ratio CI (analytic), Sharpe ratio CI (bootstrap),
minimum track record length, generic block-bootstrap CI,
bias ratio, skewness-adjusted Sharpe (ASR).

All tagged: category=("inference", "returns").  Backend varies: analytic
metrics are vectorized; bootstrap confidence intervals use resampling.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from stratstat.exceptions import MetricNotApplicableError
from stratstat.inputs import ReturnsInput
from stratstat.registry import _compute_one, register_metric
from stratstat.results import MetricResult

from .risk import _norm_cdf, _norm_ppf

# ---------------------------------------------------------------------------
# Helpers: period statistics (return raw ndarrays, not MetricResult)
# ---------------------------------------------------------------------------


def _period_sharpe(r: NDArray[np.floating], ddof: int = 1) -> NDArray[np.floating]:
    """Period (non-annualized) Sharpe ratio per strategy column.

    Formula:
        SR = mean(r) / std(r, ddof)

    Args:
        r: Returns array of shape (n_periods, n_strategies).
        ddof: Delta degrees of freedom (default 1 for sample std).

    Returns:
        Array of shape (n_strategies,). NaN where std ≈ 0.
    """
    mu = np.nanmean(r, axis=0)
    sigma = np.nanstd(r, axis=0, ddof=ddof)
    sigma_safe = np.where(sigma < 1e-15, np.nan, sigma)
    arr: NDArray[np.floating] = mu / sigma_safe
    return arr


def _sample_skewness(r: NDArray[np.floating]) -> NDArray[np.floating]:
    """Bias-corrected sample skewness per strategy column.

    Formula:
        γ₁ = n_eff / ((n_eff-1)(n_eff-2)) * Σ z³
    where z = (r - mean) / std (ddof=1).

    Returns array of shape (n_strategies,). NaN for fewer than 3 obs or
    constant returns.
    """
    n_eff = np.sum(~np.isnan(r), axis=0).astype(np.float64)

    mean = np.nanmean(r, axis=0, keepdims=True)
    std = np.nanstd(r, axis=0, ddof=1, keepdims=True)
    zero_std = (std < 1e-15).squeeze(axis=0)

    std_safe = np.where(std < 1e-15, np.nan, std)
    z = (r - mean) / std_safe
    z = np.where(np.isnan(z), 0.0, z)

    m3 = np.nansum(z**3, axis=0)

    with np.errstate(invalid="ignore"):
        factor = n_eff / ((n_eff - 1.0) * (n_eff - 2.0))
        factor = np.where(n_eff < 3, np.nan, factor)
        arr: NDArray[np.floating] = factor * m3
        arr = np.where(zero_std, np.nan, arr)

    return arr


def _sample_excess_kurtosis(r: NDArray[np.floating]) -> NDArray[np.floating]:
    """Bias-corrected sample excess kurtosis per strategy column.

    Formula:
        γ₂ = n(n+1)/((n-1)(n-2)(n-3)) * Σ z⁴ - 3(n-1)²/((n-2)(n-3))

    Returns 0 for a normal distribution. Array shape (n_strategies,).
    NaN for fewer than 4 obs or constant returns.
    """
    n_eff = np.sum(~np.isnan(r), axis=0).astype(np.float64)

    mean = np.nanmean(r, axis=0, keepdims=True)
    std = np.nanstd(r, axis=0, ddof=1, keepdims=True)
    zero_std = (std < 1e-15).squeeze(axis=0)

    std_safe = np.where(std < 1e-15, np.nan, std)
    z = (r - mean) / std_safe
    z = np.where(np.isnan(z), 0.0, z)

    m4 = np.nansum(z**4, axis=0)

    with np.errstate(invalid="ignore"):
        factor_a = n_eff * (n_eff + 1.0) / ((n_eff - 1.0) * (n_eff - 2.0) * (n_eff - 3.0))
        factor_b = 3.0 * (n_eff - 1.0) ** 2 / ((n_eff - 2.0) * (n_eff - 3.0))
        arr: NDArray[np.floating] = np.where(
            n_eff < 4, np.nan, factor_a * m4 - factor_b
        )
        arr = np.where(zero_std, np.nan, arr)

    return arr


def _autocorr_lag1(r: NDArray[np.floating]) -> NDArray[np.floating]:
    """Lag-1 autocorrelation per strategy column.

    Returns array of shape (n_strategies,).
    """
    centered = r - np.nanmean(r, axis=0, keepdims=True)
    valid = ~np.isnan(r)
    valid_pair = valid[:-1] & valid[1:]

    num = np.sum(
        centered[:-1] * centered[1:] * valid_pair.astype(np.float64), axis=0
    )
    den = np.sum(centered**2 * valid.astype(np.float64), axis=0)

    with np.errstate(invalid="ignore"):
        arr: NDArray[np.floating] = np.where(den < 1e-15, np.nan, num / den)
    return arr


def _psr_z(
    sr: NDArray[np.floating],
    sr_benchmark: float,
    skew: NDArray[np.floating],
    excess_kurt: NDArray[np.floating],
    n: int,
) -> NDArray[np.floating]:
    """Compute the PSR z-score.

    Formula (Bailey & Lopez de Prado 2012):
        z = (SR - SR*) * sqrt(n-1) / sqrt(1 - γ₃·SR + (γ₄-1)/4 · SR²)

    where γ₃ is skewness and γ₄ is raw kurtosis (= excess_kurt + 3).
    """
    kurt = excess_kurt + 3.0  # raw kurtosis
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    denom_safe = np.where(denom < 1e-15, np.nan, denom)
    z: NDArray[np.floating] = (sr - sr_benchmark) * np.sqrt(float(n) - 1.0) / denom_safe
    return z


# ---------------------------------------------------------------------------
# 4.1 Jarque-Bera Statistic
# Reference: Jarque & Bera (1987)
# ---------------------------------------------------------------------------

_JB_REF = (
    "Jarque & Bera (1987), 'A Test for Normality of Observations and "
    "Regression Residuals,' International Statistical Review, 55(2)"
)


@register_metric(
    name="jarque_bera",
    requires="returns",
    category=("inference", "returns"),
    backend="vectorized",
    ref=_JB_REF,
)
def jarque_bera(input_data: ReturnsInput) -> MetricResult:
    """Jarque-Bera test statistic for normality.

    Formula:
        JB = n/6 * (γ₁² + γ₂²/4)

    where γ₁ is sample skewness and γ₂ is sample excess kurtosis.
    Under the null of normality, JB ~ χ²(2). The p-value is
    p = exp(-JB/2) (survival function of χ²(2)).

    Requires at least 4 observations for both skewness and kurtosis.
    Returns NaN for fewer observations or constant returns.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with the JB statistic as a float (or array).
        The p-value is stored in ``meta["p_value"]``.
    """
    r = input_data.values

    skew = _sample_skewness(r)
    kurt = _sample_excess_kurtosis(r)

    # JB = n/6 * (skew² + kurt²/4)
    # Use effective n per strategy for robustness.
    n_eff = np.sum(~np.isnan(r), axis=0).astype(np.float64)
    n_eff = np.where(n_eff < 4, np.nan, n_eff)

    arr: NDArray[np.floating] = n_eff / 6.0 * (skew**2 + kurt**2 / 4.0)

    # p-value: χ²(2) survival function = exp(-x/2)
    p_values: NDArray[np.floating] = np.exp(-arr / 2.0)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="jarque_bera",
        value=value,
        category=("inference", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _JB_REF,
            "p_value": float(p_values[0]) if input_data.is_single else p_values,
            "null_distribution": "chi-squared(2)",
        },
    )


# ---------------------------------------------------------------------------
# 4.2 Probabilistic Sharpe Ratio (PSR)
# Reference: Bailey & Lopez de Prado (2012)
# ---------------------------------------------------------------------------

_PSR_REF = (
    "Bailey & Lopez de Prado (2012), 'The Sharpe Ratio Efficient Frontier,' "
    "Journal of Risk, 15(2); de Prado (2018, Advances in Financial Machine "
    "Learning, Ch. 14)"
)


@register_metric(
    name="psr",
    requires="returns",
    category=("inference", "returns"),
    backend="vectorized",
    ref=_PSR_REF,
)
def psr(
    input_data: ReturnsInput,
    sr_benchmark: float = 0.0,
    ddof: int = 1,
) -> MetricResult:
    """Probabilistic Sharpe Ratio — probability that true SR exceeds benchmark.

    Formula:
        PSR(SR*) = Φ( (SR - SR*) * sqrt(n-1) /
                      sqrt(1 - γ₃·SR + (γ₄-1)/4 · SR²) )

    where SR is the period (non-annualized) Sharpe ratio, SR* is the
    benchmark, γ₃ is sample skewness, γ₄ is raw kurtosis (= excess + 3),
    and Φ is the standard normal CDF.

    Important: SR and sr_benchmark are at the input data's native
    frequency, NOT annualized. To use annualized values, divide by
    sqrt(periods_per_year) first.

    Args:
        input_data: A ``ReturnsInput``.
        sr_benchmark: Benchmark Sharpe ratio at period frequency (default 0.0).
        ddof: Delta degrees of freedom for Sharpe std computation (default 1).

    Returns:
        MetricResult with PSR value in [0, 1] (float or array).
        NaN when std ≈ 0 or fewer than 4 observations.
    """
    r = input_data.values
    n = r.shape[0]

    sr = _period_sharpe(r, ddof=ddof)
    skew = _sample_skewness(r)
    excess_kurt = _sample_excess_kurtosis(r)

    z = _psr_z(sr, sr_benchmark, skew, excess_kurt, n)

    # Φ(z) per strategy
    arr = np.array([_norm_cdf(float(zi)) for zi in np.atleast_1d(z)], dtype=np.float64)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="psr",
        value=value,
        category=("inference", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _PSR_REF,
            "sr_benchmark": sr_benchmark,
            "ddof": ddof,
            "sr_period": float(sr[0]) if input_data.is_single else sr,
        },
    )


# ---------------------------------------------------------------------------
# 4.3 Deflated Sharpe Ratio (DSR)
# Reference: Bailey & Lopez de Prado (2014)
# ---------------------------------------------------------------------------

_DSR_REF = (
    "Bailey & Lopez de Prado (2014), 'The Deflated Sharpe Ratio: Correcting "
    "for Selection Bias, Backtest Overfitting, and Non-Normality,' JPM, 40(5)"
)


@register_metric(
    name="dsr",
    requires="returns",
    category=("inference", "returns"),
    backend="vectorized",
    ref=_DSR_REF,
)
def dsr(
    input_data: ReturnsInput,
    sr_benchmark: float = 0.0,
    sr_trials: NDArray[np.floating] | None = None,
    ddof: int = 1,
) -> MetricResult:
    """Deflated Sharpe Ratio — PSR corrected for multiple testing.

    Formula:
        DSR = Φ( z * (1 - 1/M * Σ 1[SR_m < SR_hat]) )

    where z is the PSR z-score, M is the number of trials, and SR_m
    are the maximum Sharpe ratios from M independent trials. The
    deflation term corrects for selection bias: when many trials can
    produce a higher SR by chance, the DSR is substantially lower
    than the PSR.

    If ``sr_trials`` is None or empty, DSR degrades to PSR
    (no multiple-testing correction).

    Args:
        input_data: A ``ReturnsInput``.
        sr_benchmark: Benchmark Sharpe at period frequency (default 0.0).
        sr_trials: Array of maximum SRs from M trials, all at period
            frequency. If None, no deflation is applied.
        ddof: Delta degrees of freedom for Sharpe std (default 1).

    Returns:
        MetricResult with DSR value in [0, 1] (float or array).
    """
    r = input_data.values
    n = r.shape[0]

    sr = _period_sharpe(r, ddof=ddof)
    skew = _sample_skewness(r)
    excess_kurt = _sample_excess_kurtosis(r)

    z = _psr_z(sr, sr_benchmark, skew, excess_kurt, n)

    # Deflation factor
    if sr_trials is not None and len(sr_trials) > 0:
        # For each strategy, count how many trial max SRs are below the
        # observed SR. proportion = count / M, deflation = 1 - proportion.
        m = len(sr_trials)
        deflation = np.array(
            [
                1.0 - float(np.sum(sr_trials < float(s))) / float(m)
                for s in np.atleast_1d(sr)
            ],
            dtype=np.float64,
        )
    else:
        deflation = np.ones_like(sr, dtype=np.float64)

    z_dsr: NDArray[np.floating] = z * deflation
    arr = np.array(
        [_norm_cdf(float(zi)) for zi in np.atleast_1d(z_dsr)], dtype=np.float64
    )

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    m_trials = len(sr_trials) if sr_trials is not None else 0

    return MetricResult(
        name="dsr",
        value=value,
        category=("inference", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _DSR_REF,
            "sr_benchmark": sr_benchmark,
            "ddof": ddof,
            "sr_period": float(sr[0]) if input_data.is_single else sr,
            "m_trials": m_trials,
            "note": (
                "DSR = PSR without deflation"
                if m_trials == 0
                else f"DSR deflated by {m_trials} trials"
            ),
        },
    )


# ---------------------------------------------------------------------------
# 4.4 Lo's Autocorrelation-Adjusted Sharpe SE
# Reference: Lo (2002)
# ---------------------------------------------------------------------------

_LO_SE_REF = "Lo (2002), 'The Statistics of Sharpe Ratios,' Financial Analysts Journal, 58(4)"


@register_metric(
    name="lo_sharpe_se",
    requires="returns",
    category=("inference", "returns"),
    backend="vectorized",
    ref=_LO_SE_REF,
)
def lo_sharpe_se(
    input_data: ReturnsInput,
    adjust: bool = True,
    ddof: int = 1,
) -> MetricResult:
    """Lo's standard error of the Sharpe ratio.

    Formula:
        SE_IID = sqrt( (1/T) * (1 + SR²/2 - γ₃·SR + (γ₄-3)/4 · SR²) )
        SE_adj = SE_IID * sqrt((1 + ρ₁)/(1 - ρ₁))

    where ρ₁ is the lag-1 autocorrelation of returns. Set
    ``adjust=False`` to return the IID SE without autocorrelation
    adjustment.

    Args:
        input_data: A ``ReturnsInput``.
        adjust: If True (default), apply autocorrelation adjustment.
        ddof: Delta degrees of freedom for Sharpe std (default 1).

    Returns:
        MetricResult with the standard error (float or array).
    """
    r = input_data.values
    n = r.shape[0]

    sr = _period_sharpe(r, ddof=ddof)
    skew = _sample_skewness(r)
    excess_kurt = _sample_excess_kurtosis(r)

    # IID SE² = 1/T * (1 + SR²/2 - γ₃·SR + (γ₄-3)/4 · SR²)
    # (γ₄ - 3) = excess_kurtosis, so this simplifies to:
    se_iid_sq: NDArray[np.floating] = (1.0 / np.where(n < 1, np.nan, float(n))) * (
        1.0 + sr**2 / 2.0 - skew * sr + excess_kurt / 4.0 * sr**2
    )
    se_iid: NDArray[np.floating] = np.sqrt(np.maximum(se_iid_sq, 0.0))

    if adjust:
        rho = _autocorr_lag1(r)
        # Clip rho to (-1, 1) to avoid invalid sqrt.
        rho_clipped = np.clip(rho, -0.9999, 0.9999)
        adj_factor: NDArray[np.floating] = np.sqrt(
            (1.0 + rho_clipped) / (1.0 - rho_clipped)
        )
        # If rho is NaN (insufficient data), adj_factor is NaN → SE becomes NaN.
        arr = se_iid * adj_factor
    else:
        arr = se_iid

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="lo_sharpe_se",
        value=value,
        category=("inference", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _LO_SE_REF,
            "adjust": adjust,
            "ddof": ddof,
        },
    )


# ---------------------------------------------------------------------------
# 4.5 Sharpe Ratio CI — Analytic
# Reference: Lo (2002)
# ---------------------------------------------------------------------------

_SHARPE_CI_REF = "Lo (2002), 'The Statistics of Sharpe Ratios,' Financial Analysts Journal, 58(4)"


@register_metric(
    name="sharpe_ci_analytic",
    requires="returns",
    category=("inference", "returns"),
    backend="vectorized",
    ref=_SHARPE_CI_REF,
)
def sharpe_ci_analytic(
    input_data: ReturnsInput,
    confidence: float = 0.95,
    adjust: bool = True,
    ddof: int = 1,
) -> MetricResult:
    """Analytic confidence interval for the Sharpe ratio.

    Formula:
        CI = SR ± z_{1-α/2} · SE(SR)

    using Lo's standard error (optionally autocorrelation-adjusted).

    Args:
        input_data: A ``ReturnsInput``.
        confidence: Confidence level (default 0.95 for 95% CI).
        adjust: If True (default), use autocorrelation-adjusted SE.
        ddof: Delta degrees of freedom for Sharpe std (default 1).

    Returns:
        MetricResult with ``ndarray([lower, upper])`` and
        ``meta["output_index"] = ["lower", "upper"]``.
    """
    r = input_data.values
    n = r.shape[0]

    sr = _period_sharpe(r, ddof=ddof)
    skew = _sample_skewness(r)
    excess_kurt = _sample_excess_kurtosis(r)

    # Lo SE
    se_sq = (1.0 / np.where(n < 1, np.nan, float(n))) * (
        1.0 + sr**2 / 2.0 - skew * sr + excess_kurt / 4.0 * sr**2
    )
    se = np.sqrt(np.maximum(se_sq, 0.0))

    if adjust:
        rho = _autocorr_lag1(r)
        rho_clipped = np.clip(rho, -0.9999, 0.9999)
        adj_factor = np.sqrt((1.0 + rho_clipped) / (1.0 - rho_clipped))
        se = se * adj_factor

    alpha = 1.0 - confidence
    z_crit = _norm_ppf(1.0 - alpha / 2.0)

    lower = sr - z_crit * se
    upper = sr + z_crit * se

    # Build output: shape (2,) for single, (2, n_strategies) for multi.
    # Matches the project convention: (k_outputs, n_strategies).
    n_strat = r.shape[1]
    if n_strat == 1 and input_data.is_single:
        arr = np.array(
            [float(lower.flat[0]), float(upper.flat[0])], dtype=np.float64
        )
        value: float | NDArray[np.floating] = arr
    else:
        value = np.stack([lower, upper], axis=0)  # shape (2, n_strategies)

    return MetricResult(
        name="sharpe_ci_analytic",
        value=value,
        category=("inference", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _SHARPE_CI_REF,
            "confidence": confidence,
            "adjust": adjust,
            "ddof": ddof,
            "output_index": ["lower", "upper"],
        },
    )


# ---------------------------------------------------------------------------
# 4.6 Sharpe Ratio CI — Bootstrap
# Reference: Efron & Tibshirani (1994); Ledoit & Wolf (2008)
# ---------------------------------------------------------------------------

_SHARPE_CI_BOOT_REF = (
    "Efron & Tibshirani (1994, An Introduction to the Bootstrap); "
    "Ledoit & Wolf (2008), 'Robust Performance Hypothesis Testing with "
    "the Sharpe Ratio,' J. Empirical Finance, 15(5)"
)


def _block_bootstrap_indices(
    n: int, block_len: int, n_reps: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Generate block bootstrap index arrays.

    Returns an array of shape (n_reps, n) with indices for each
    bootstrap replicate.
    """
    n_blocks = int(np.ceil(n / block_len))
    indices = np.empty((n_reps, n), dtype=np.int64)

    for b in range(n_reps):
        # Sample block starting positions with replacement
        block_starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        rep = np.concatenate([
            np.arange(start, start + block_len) for start in block_starts
        ])
        indices[b] = rep[:n]

    return indices


@register_metric(
    name="sharpe_ci_bootstrap",
    requires="returns",
    category=("inference", "returns"),
    backend="resampling",
    ref=_SHARPE_CI_BOOT_REF,
)
def sharpe_ci_bootstrap(
    input_data: ReturnsInput,
    confidence: float = 0.95,
    n_reps: int = 5000,
    block_len: int | None = None,
    random_seed: int | None = None,
    ddof: int = 1,
) -> MetricResult:
    """Bootstrap confidence interval for the Sharpe ratio.

    Uses the moving-block bootstrap (Kunsch 1989) with equal-tailed
    percentile intervals. Block length defaults to n^(1/3) (Politis &
    White 2004 heuristic).

    Args:
        input_data: A ``ReturnsInput`` (single-strategy only).
        confidence: Confidence level (default 0.95 for 95% CI).
        n_reps: Number of bootstrap replicates (default 5000).
        block_len: Block length. If None, defaults to n^(1/3).
        random_seed: Seed for reproducibility.
        ddof: Delta degrees of freedom for Sharpe std (default 1).

    Returns:
        MetricResult with ``ndarray([lower, upper])``.

    Raises:
        ValueError: If multi-strategy input is provided (bootstrap
            operates on a single strategy at a time).
    """
    if not input_data.is_single:
        raise MetricNotApplicableError(
            "sharpe_ci_bootstrap requires single-strategy input. "
            "Use compute() per strategy or wrap in a loop."
        )

    rng = np.random.default_rng(random_seed)
    r = input_data.values[:, 0]  # 1-D array of period returns
    n = len(r)

    # Block length selection
    if block_len is None:
        block_len = max(1, int(np.ceil(n ** (1.0 / 3.0))))
    block_len = min(block_len, n)

    # Generate bootstrap indices and compute SR for each replicate
    indices = _block_bootstrap_indices(n, block_len, n_reps, rng)
    boot_sr = np.empty(n_reps, dtype=np.float64)

    for b in range(n_reps):
        sample = r[indices[b]]
        mu = np.nanmean(sample)
        sigma = np.nanstd(sample, ddof=ddof)
        boot_sr[b] = mu / sigma if sigma > 1e-15 else np.nan

    # Remove NaN replicates
    boot_sr = boot_sr[~np.isnan(boot_sr)]

    if len(boot_sr) == 0:
        value: float | NDArray[np.floating] = np.array([np.nan, np.nan], dtype=np.float64)
        return MetricResult(
            name="sharpe_ci_bootstrap",
            value=value,
            category=("inference", "returns"),
            periods_per_year=input_data.periods_per_year,
            meta={
                "ref": _SHARPE_CI_BOOT_REF,
                "confidence": confidence,
                "n_reps": n_reps,
                "block_len": block_len,
                "output_index": ["lower", "upper"],
                "note": "All bootstrap replicates produced NaN.",
            },
        )

    alpha = 1.0 - confidence
    lower = np.percentile(boot_sr, 100.0 * alpha / 2.0)
    upper = np.percentile(boot_sr, 100.0 * (1.0 - alpha / 2.0))

    arr = np.array([lower, upper], dtype=np.float64)
    value = arr

    return MetricResult(
        name="sharpe_ci_bootstrap",
        value=value,
        category=("inference", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _SHARPE_CI_BOOT_REF,
            "confidence": confidence,
            "n_reps": n_reps,
            "n_valid_reps": len(boot_sr),
            "block_len": block_len,
            "output_index": ["lower", "upper"],
        },
    )


# ---------------------------------------------------------------------------
# 4.7 Minimum Track Record Length
# Reference: Bailey & Lopez de Prado (2012)
# ---------------------------------------------------------------------------

_MIN_TR_REF = (
    "Bailey & Lopez de Prado (2012), 'The Sharpe Ratio Efficient Frontier,' "
    "Journal of Risk, 15(2); de Prado (2018, Advances in Financial Machine "
    "Learning, Ch. 14)"
)


@register_metric(
    name="min_track_record_length",
    requires="returns",
    category=("inference", "returns"),
    backend="vectorized",
    ref=_MIN_TR_REF,
)
def min_track_record_length(
    input_data: ReturnsInput,
    sr_benchmark: float = 0.0,
    alpha: float = 0.05,
    ddof: int = 1,
) -> MetricResult:
    """Minimum track record length for statistically significant Sharpe.

    Formula:
        T_min = 1 + z_α² · (1 - γ₃·SR + (γ₄-1)/4 · SR²) / (SR - SR*)²

    where z_α = Φ⁻¹(1 - α) is the critical value, SR is the period
    Sharpe ratio, SR* is the benchmark, γ₃ is skewness, and γ₄ is
    raw kurtosis.

    This is the number of observations needed to have (1-α) confidence
    that the true Sharpe ratio exceeds the benchmark.

    Args:
        input_data: A ``ReturnsInput``.
        sr_benchmark: Benchmark Sharpe at period frequency (default 0.0).
        alpha: Significance level (default 0.05 for 95% confidence).
        ddof: Delta degrees of freedom for Sharpe std (default 1).

    Returns:
        MetricResult with minimum number of periods (float or array).
        Returns ``inf`` when SR ≈ sr_benchmark (division by zero).
    """
    r = input_data.values

    sr = _period_sharpe(r, ddof=ddof)
    skew = _sample_skewness(r)
    excess_kurt = _sample_excess_kurtosis(r)

    kurt = excess_kurt + 3.0  # raw kurtosis
    z_crit = _norm_ppf(1.0 - alpha)

    # T_min = 1 + z² * (1 - skew*SR + (kurt-1)/4 * SR²) / (SR - SR*)^2
    num = z_crit**2 * (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    denom = (sr - sr_benchmark) ** 2

    with np.errstate(invalid="ignore"):
        arr: NDArray[np.floating] = np.where(
            denom < 1e-15, np.inf, 1.0 + num / denom
        )

    # If numerator is negative (which shouldn't happen for valid data),
    # return NaN.
    arr = np.where(num < 0.0, np.nan, arr)
    # Truncate fractional periods: T_min must be an integer >= 1
    arr_ceil = np.ceil(np.maximum(arr, 1.0))

    value: float | NDArray[np.floating]
    value = float(arr_ceil[0]) if input_data.is_single else arr_ceil

    return MetricResult(
        name="min_track_record_length",
        value=value,
        category=("inference", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _MIN_TR_REF,
            "sr_benchmark": sr_benchmark,
            "alpha": alpha,
            "ddof": ddof,
            "note": "Minimum number of periods (ceiling of computed value).",
        },
    )


# ---------------------------------------------------------------------------
# 4.8 Generic Block-Bootstrap CI Wrapper
# Reference: Efron & Tibshirani (1994); Kunsch (1989)
# ---------------------------------------------------------------------------

_BOOT_CI_REF = (
    "Efron & Tibshirani (1994, An Introduction to the Bootstrap); "
    "Kunsch (1989), 'The Jackknife and the Bootstrap for General "
    "Stationary Observations,' Annals of Statistics, 17(3)"
)


@register_metric(
    name="block_bootstrap_ci",
    requires="returns",
    category=("inference", "returns"),
    backend="resampling",
    ref=_BOOT_CI_REF,
)
def block_bootstrap_ci(
    input_data: ReturnsInput,
    target_metric: str,
    confidence: float = 0.95,
    n_reps: int = 5000,
    block_len: int | None = None,
    random_seed: int | None = None,
    **metric_kwargs: Any,
) -> MetricResult:
    """Bootstrap confidence interval for any registered returns-tier metric.

    Uses the moving-block bootstrap with equal-tailed percentile intervals.

    Args:
        input_data: A ``ReturnsInput`` (single-strategy only).
        target_metric: Name of a registered returns-tier metric.
        confidence: Confidence level (default 0.95).
        n_reps: Bootstrap replicates (default 5000).
        block_len: Block length. If None, defaults to n^(1/3).
        random_seed: Seed for reproducibility.
        **metric_kwargs: Passed through to the metric function.

    Returns:
        MetricResult with ``ndarray([lower, upper])`` and
        ``meta["output_index"] = ["lower", "upper"]``.

    Raises:
        ValueError: If multi-strategy input or unknown metric.
    """
    if not input_data.is_single:
        raise MetricNotApplicableError(
            "block_bootstrap_ci requires single-strategy input."
        )

    rng = np.random.default_rng(random_seed)
    r = input_data.values[:, 0]  # 1-D
    n = len(r)

    if block_len is None:
        block_len = max(1, int(np.ceil(n ** (1.0 / 3.0))))
    block_len = min(block_len, n)

    indices = _block_bootstrap_indices(n, block_len, n_reps, rng)
    boot_vals = np.empty(n_reps, dtype=np.float64)

    for b in range(n_reps):
        sample = r[indices[b]]
        # Create a new ReturnsInput for the bootstrap sample
        boot_inp = ReturnsInput(sample, periods_per_year=input_data.periods_per_year)
        result = _compute_one(boot_inp, target_metric, **metric_kwargs)
        val = result.value
        if isinstance(val, np.ndarray):
            val = val.flat[0]
        boot_vals[b] = cast(float, val)

    boot_vals = boot_vals[~np.isnan(boot_vals)]

    if len(boot_vals) == 0:
        arr: NDArray[np.floating] = np.array([np.nan, np.nan], dtype=np.float64)
        return MetricResult(
            name=f"{target_metric}_bootstrap_ci",
            value=arr,
            category=("inference", "returns"),
            periods_per_year=input_data.periods_per_year,
            meta={
                "ref": _BOOT_CI_REF,
                "metric": target_metric,
                "confidence": confidence,
                "n_reps": n_reps,
                "block_len": block_len,
                "output_index": ["lower", "upper"],
                "note": "All bootstrap replicates produced NaN.",
            },
        )

    alpha = 1.0 - confidence
    lower = np.percentile(boot_vals, 100.0 * alpha / 2.0)
    upper = np.percentile(boot_vals, 100.0 * (1.0 - alpha / 2.0))

    arr = np.array([lower, upper], dtype=np.float64)

    return MetricResult(
        name=f"{target_metric}_bootstrap_ci",
        value=arr,
        category=("inference", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _BOOT_CI_REF,
            "metric": target_metric,
            "confidence": confidence,
            "n_reps": n_reps,
            "n_valid_reps": len(boot_vals),
            "block_len": block_len,
            "output_index": ["lower", "upper"],
        },
    )


# ---------------------------------------------------------------------------
# 4.9 Bias Ratio
# Reference: Abdulali (2006)
# ---------------------------------------------------------------------------

_BIAS_RATIO_REF = 'Abdulali (2006), "Detecting Smoothed Returns"'


@register_metric(
    name="bias_ratio",
    requires="returns",
    category=("inference", "returns"),
    backend="vectorized",
    ref=_BIAS_RATIO_REF,
)
def bias_ratio(
    input_data: ReturnsInput, bandwidth: float = 1.0
) -> MetricResult:
    """Bias Ratio — detects smoothed or manipulated returns.

    Measures the concentration of returns in a narrow band around zero
    relative to returns outside that band. A high Bias Ratio suggests
    return smoothing or artificial price manipulation.

    Formula:
        bias_ratio = count(|r| < bandwidth * sigma) /
                     max(count(|r| >= bandwidth * sigma), 1)

    where sigma is the sample standard deviation (ddof=1). The default
    bandwidth of 1.0 uses ±1 sigma as the narrow band (Abdulali 2006).

    The denominator is floored at 1 to avoid division by zero when all
    returns fall within the narrow band.

    Args:
        input_data: A ``ReturnsInput``. Must have at least 2 periods.
        bandwidth: Number of standard deviations defining the narrow band
            around zero (default 1.0).

    Returns:
        MetricResult with Bias Ratio (non-negative float or array).
        Higher values indicate more clustering around zero.
    """
    if bandwidth <= 0.0:
        raise ValueError(
            f"bandwidth must be positive, got {bandwidth}"
        )

    r = input_data.values  # (n_periods, n_strategies)
    n_strat = r.shape[1]

    bias_arr = np.zeros(n_strat, dtype=np.float64)
    for col in range(n_strat):
        col_data = r[:, col]
        valid_data = col_data[~np.isnan(col_data)]
        if len(valid_data) < 2:
            bias_arr[col] = np.nan
            continue

        sigma = float(np.std(valid_data, ddof=1))
        if sigma < 1e-15:
            bias_arr[col] = np.nan
            continue

        threshold = bandwidth * sigma
        in_band = np.sum(np.abs(valid_data) < threshold)
        outside = max(len(valid_data) - in_band, 1)  # floor at 1

        bias_arr[col] = float(in_band) / float(outside)

    value: float | NDArray[np.floating]
    value = float(bias_arr[0]) if input_data.is_single else bias_arr

    return MetricResult(
        name="bias_ratio",
        value=value,
        category=("inference", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _BIAS_RATIO_REF,
            "bandwidth": bandwidth,
        },
    )


# ---------------------------------------------------------------------------
# 4.10 Skewness-Adjusted Sharpe Ratio (ASR)
# Reference: Pezier & White (2008)
# ---------------------------------------------------------------------------

_ASR_REF = "Pezier & White (2008)"


@register_metric(
    name="skewness_adjusted_sharpe",
    requires="returns",
    category=("inference", "returns"),
    backend="vectorized",
    ref=_ASR_REF,
)
def skewness_adjusted_sharpe(
    input_data: ReturnsInput,
    rf: float = 0.0,
    ddof: int = 1,
) -> MetricResult:
    """Skewness-Adjusted Sharpe Ratio (ASR).

    Adjusts the standard Sharpe ratio for the skewness and kurtosis of
    the return distribution using the Pezier & White (2008) expansion:

    Formula:
        ASR = SR * (1 + (gamma_3 / 6) * SR - (gamma_4 / 24) * SR^2)

    where SR is the (annualized) Sharpe ratio, gamma_3 is the sample
    skewness, and gamma_4 is the sample excess kurtosis.

    For normally distributed returns (gamma_3 ≈ 0, gamma_4 ≈ 0),
    ASR ≈ SR. Positive skewness increases ASR; excess kurtosis
    decreases it (penalising fat tails).

    Requires ``periods_per_year`` on the input and at least 4 observations.

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.
        rf: Risk-free rate per period (default 0.0).
        ddof: Delta degrees of freedom for standard deviation — 1 for
            sample (default), 0 for population.

    Returns:
        MetricResult with ASR (float or array). NaN when fewer than 4
        observations or zero volatility.
    """
    if input_data.periods_per_year is None:
        raise MetricNotApplicableError(
            "ASR requires periods_per_year on the ReturnsInput"
        )

    r = input_data.values  # (n_periods, n_strategies)
    p = float(input_data.periods_per_year)

    # Period Sharpe ratio per column
    excess = np.nanmean(r, axis=0) - rf
    sigma = np.nanstd(r, axis=0, ddof=ddof)
    sigma_safe = np.where(sigma < 1e-15, np.nan, sigma)
    sr_period = excess / sigma_safe
    sr = sr_period * np.sqrt(p)  # annualized Sharpe

    skew = _sample_skewness(r)  # (n_strat,)
    excess_kurt = _sample_excess_kurtosis(r)  # (n_strat,)

    # ASR expansion
    term1 = (skew / 6.0) * sr
    term2 = (excess_kurt / 24.0) * sr**2

    with np.errstate(invalid="ignore"):
        arr = sr * (1.0 + term1 - term2)

    # If Sharpe is NaN (zero vol, too few obs), ASR should also be NaN
    arr = np.where(np.isnan(sr), np.nan, arr)
    # If skew/kurt are NaN (too few obs), ASR is NaN
    arr = np.where(np.isnan(skew) | np.isnan(excess_kurt), np.nan, arr)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="skewness_adjusted_sharpe",
        value=value,
        category=("inference", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _ASR_REF,
            "rf": rf,
            "ddof": ddof,
        },
    )
