"""Benchmark-tier metrics.

Metrics comparing a single strategy to a benchmark: alpha (Jensen's),
beta, R², tracking error, information ratio, up/down capture, correlation,
active return, batting average, Treynor ratio, outperformance, and related
measures.

All tagged: category varies, requires="benchmark".
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from stratstat.core._utils import compute_cagr, ols_beta
from stratstat.exceptions import MetricNotApplicableError
from stratstat.inputs import BenchmarkInput
from stratstat.registry import register_metric
from stratstat.results import MetricResult

# ---------------------------------------------------------------------------
# Citation strings
# ---------------------------------------------------------------------------

_REF_JENSEN = (
    'Jensen (1968), "The Performance of Mutual Funds in the '
    'Period 1945-1964," Journal of Finance, 23(2).'
)
_REF_SHARPE = (
    'Sharpe (1964), "Capital Asset Prices: A Theory of Market '
    'Equilibrium Under Conditions of Risk," Journal of Finance, 19(3).'
)
_REF_GREENE = "Greene (2018, Econometric Analysis, 8th ed., §3.5)."
_REF_ROLL = (
    'Roll (1992), "A Mean/Variance Analysis of Tracking Error," '
    "JPM, 18(4)."
)
_REF_GOODWIN = (
    'Goodwin (1998), "The Information Ratio," Financial Analysts '
    "Journal, 54(4)."
)
_REF_CAPTURE = (
    "Morningstar (2020), Morningstar Performance Reporting Methodology; "
    "Bacon (2008, §9.4)."
)
_REF_BACON_94 = "Bacon (2008), §9.4."
_REF_PEARSON = "Pearson (1895); standard statistic."
_BACON_BASE = (
    "Bacon (2008), Practical Portfolio Performance Measurement and "
    "Attribution, 2nd ed."
)
_REF_BACON_92 = f"{_BACON_BASE}, §9.2."
_REF_BACON_95 = f"{_BACON_BASE}, §9.5."
_REF_CFA = "CFA Institute, Performance Attribution."
_REF_TREYNOR = (
    'Treynor (1965), "How to Rate Management of Investment Funds," '
    "Harvard Business Review, 43(1)."
)
_REF_BACON_93 = f"{_BACON_BASE}, §9.3."
_REF_CFA_QM = "CFA Institute, Quantitative Methods."


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_periods_per_year(inp: BenchmarkInput, metric_name: str) -> None:
    """Raise ``MetricNotApplicableError`` if ``periods_per_year`` is not set."""
    if inp.periods_per_year is None:
        raise MetricNotApplicableError(
            f"{metric_name} requires periods_per_year for annualization. "
            f"Provide periods_per_year= to BenchmarkInput."
        )


def _require_single_strategy(inp: BenchmarkInput, metric_name: str) -> None:
    """Raise ``MetricNotApplicableError`` if input contains more than one strategy."""
    if inp.n_strategies > 1:
        raise MetricNotApplicableError(
            f"{metric_name} requires a single strategy. "
            f"Got {inp.n_strategies} strategy columns."
        )


def _active_return_series(
    returns: NDArray[np.floating],
    benchmark: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Period-by-period active return: r_t - r_{m,t}.

    *returns* is ``(n_periods,)`` for a single strategy column;
    *benchmark* is ``(n_periods,)``.
    """
    return returns - benchmark


# ===================================================================
# §8.1  Alpha (Jensen's)
# ===================================================================


@register_metric(
    name="alpha",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_JENSEN,
)
def alpha(inp: BenchmarkInput) -> MetricResult:
    r"""Jensen's alpha (annualized).

    .. math::
        \alpha_{\text{ann}} = \text{CAGR}
        - (r_f + \beta \cdot (\text{CAGR}_m - r_f))

    Returns NaN if fewer than 3 valid overlapping observations.
    """
    _require_single_strategy(inp, "alpha")
    _require_periods_per_year(inp, "alpha")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    p = float(inp.periods_per_year)  # type: ignore[arg-type]

    mask = np.isfinite(r) & np.isfinite(bench)
    if mask.sum() < 3:
        value: float = np.nan
        return MetricResult(
            name="alpha",
            value=value,
            category=("benchmark",),
            periods_per_year=inp.periods_per_year,
            meta={"ref": _REF_JENSEN, "annualized": True},
        )

    rc = r[mask]
    bc = bench[mask]
    cagr_r = compute_cagr(rc.reshape(-1, 1), p)[0]
    cagr_m = compute_cagr(bc.reshape(-1, 1), p)[0]

    beta_val = ols_beta(rc, bc)
    value = cagr_r - (inp.rf + beta_val * (cagr_m - inp.rf))

    return MetricResult(
        name="alpha",
        value=value,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_JENSEN, "annualized": True},
    )


# ===================================================================
# §8.2  Beta
# ===================================================================


@register_metric(
    name="beta",
    requires="benchmark",
    category=("benchmark", "risk"),
    backend="vectorized",
    ref=_REF_SHARPE,
)
def beta(inp: BenchmarkInput) -> MetricResult:
    r"""OLS beta vs. benchmark.

    .. math::
        \beta = \frac{\text{Cov}(r, r_m)}{\text{Var}(r_m)}

    Convention: ``variant="least_squares"`` (default, OLS).
    """
    _require_single_strategy(inp, "beta")
    r = inp.returns[:, 0]
    value: float = ols_beta(r, inp.benchmark)
    return MetricResult(
        name="beta",
        value=value,
        category=("benchmark", "risk"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_SHARPE, "variant": "least_squares"},
    )


# ===================================================================
# §8.3  R²
# ===================================================================


@register_metric(
    name="r_squared",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_GREENE,
)
def r_squared(inp: BenchmarkInput) -> MetricResult:
    r"""Coefficient of determination vs. benchmark.

    .. math::
        R^2 = 1 - \frac{\text{Var}(r - \hat{r})}
        {\text{Var}(r)}

    where :math:`\hat{r} = \alpha + \beta r_m`.
    """
    _require_single_strategy(inp, "r_squared")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    mask = np.isfinite(r) & np.isfinite(bench)
    if mask.sum() < 3:
        value: float = np.nan
        return MetricResult(
            name="r_squared",
            value=value,
            category=("benchmark",),
            periods_per_year=inp.periods_per_year,
            meta={"ref": _REF_GREENE},
        )

    rc = r[mask]
    bc = bench[mask]
    beta_val = ols_beta(rc, bc)
    # alpha = mean(r) - beta * mean(bench)
    alpha_val = np.mean(rc) - beta_val * np.mean(bc)
    r_hat = alpha_val + beta_val * bc
    resid = rc - r_hat
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((rc - np.mean(rc)) ** 2)
    value = np.nan if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    return MetricResult(
        name="r_squared",
        value=value,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_GREENE},
    )


# ===================================================================
# §8.4  Tracking Error
# ===================================================================


@register_metric(
    name="tracking_error",
    requires="benchmark",
    category=("benchmark", "risk"),
    backend="vectorized",
    ref=_REF_ROLL,
)
def tracking_error(inp: BenchmarkInput) -> MetricResult:
    r"""Annualized tracking error.

    .. math::
        \text{TE}_{\text{ann}} = \sigma(r - r_m) \cdot \sqrt{P}
    """
    _require_single_strategy(inp, "tracking_error")
    _require_periods_per_year(inp, "tracking_error")
    r = inp.returns[:, 0]
    active = _active_return_series(r, inp.benchmark)
    te_period = float(np.nanstd(active, ddof=1))
    p = float(inp.periods_per_year)  # type: ignore[arg-type]
    value: float = te_period * np.sqrt(p)
    return MetricResult(
        name="tracking_error",
        value=value,
        category=("benchmark", "risk"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_ROLL, "annualized": True},
    )


# ===================================================================
# §8.5  Information Ratio
# ===================================================================


@register_metric(
    name="information_ratio",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_GOODWIN,
)
def information_ratio(inp: BenchmarkInput) -> MetricResult:
    r"""Annualized information ratio.

    .. math::
        \text{IR}_{\text{ann}} =
        \frac{(\bar{r} - \bar{r}_m) \cdot P}
        {\sigma(r - r_m) \cdot \sqrt{P}}
    """
    _require_single_strategy(inp, "information_ratio")
    _require_periods_per_year(inp, "information_ratio")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    active = _active_return_series(r, bench)
    te_period = float(np.nanstd(active, ddof=1))
    if te_period == 0.0:
        value: float = np.nan
    else:
        p = float(inp.periods_per_year)  # type: ignore[arg-type]
        mean_active = float(np.nanmean(active))
        value = (mean_active * p) / (te_period * np.sqrt(p))
    return MetricResult(
        name="information_ratio",
        value=value,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_GOODWIN, "annualized": True},
    )


# ===================================================================
# §8.6  Up-Capture Ratio
# ===================================================================


@register_metric(
    name="up_capture",
    requires="benchmark",
    category=("benchmark", "capture"),
    backend="vectorized",
    ref=_REF_CAPTURE,
)
def up_capture(inp: BenchmarkInput) -> MetricResult:
    r"""Up-capture ratio.

    .. math::
        \text{UC} = \frac{
        \text{mean}_{t: r_{m,t} > 0}(r_t)}
        {\text{mean}_{t: r_{m,t} > 0}(r_{m,t})}
    """
    _require_single_strategy(inp, "up_capture")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    up_mask = bench > 0.0
    if not np.any(up_mask):
        value: float = np.nan
    else:
        num = np.nanmean(np.where(up_mask, r, np.nan))
        denom = np.nanmean(np.where(up_mask, bench, np.nan))
        value = float(num / denom) if denom != 0.0 else np.nan
    return MetricResult(
        name="up_capture",
        value=value,
        category=("benchmark", "capture"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_CAPTURE},
    )


# ===================================================================
# §8.7  Down-Capture Ratio
# ===================================================================


@register_metric(
    name="down_capture",
    requires="benchmark",
    category=("benchmark", "capture"),
    backend="vectorized",
    ref=_REF_CAPTURE,
)
def down_capture(inp: BenchmarkInput) -> MetricResult:
    r"""Down-capture ratio.

    .. math::
        \text{DC} = \frac{
        \text{mean}_{t: r_{m,t} < 0}(r_t)}
        {\text{mean}_{t: r_{m,t} < 0}(r_{m,t})}
    """
    _require_single_strategy(inp, "down_capture")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    down_mask = bench < 0.0
    if not np.any(down_mask):
        value: float = np.nan
    else:
        num = np.nanmean(np.where(down_mask, r, np.nan))
        denom = np.nanmean(np.where(down_mask, bench, np.nan))
        value = float(num / denom) if denom != 0.0 else np.nan
    return MetricResult(
        name="down_capture",
        value=value,
        category=("benchmark", "capture"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_CAPTURE},
    )


# ===================================================================
# §8.8  Up/Down Capture Ratio
# ===================================================================


@register_metric(
    name="up_down_capture",
    requires="benchmark",
    category=("benchmark", "capture"),
    backend="vectorized",
    ref=_REF_BACON_94,
)
def up_down_capture(inp: BenchmarkInput) -> MetricResult:
    r"""Up/down capture ratio.

    .. math::
        \text{UDR} = \frac{\text{UC}}{|\text{DC}|}

    Delegates to the registered ``up_capture`` and ``down_capture``
    metrics.
    """
    _require_single_strategy(inp, "up_down_capture")
    from stratstat.registry import _compute_one

    uc = cast(float, _compute_one(inp, "up_capture").value)
    dc = cast(float, _compute_one(inp, "down_capture").value)
    if np.isnan(uc) or np.isnan(dc) or dc == 0.0:
        value: float = np.nan
    else:
        value = uc / abs(dc)
    return MetricResult(
        name="up_down_capture",
        value=value,
        category=("benchmark", "capture"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_BACON_94},
    )


# ===================================================================
# §8.9  Correlation
# ===================================================================


@register_metric(
    name="correlation",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_PEARSON,
)
def correlation(inp: BenchmarkInput) -> MetricResult:
    r"""Pearson correlation with benchmark.

    .. math::
        \rho = \frac{\text{Cov}(r, r_m)}
        {\sigma(r) \cdot \sigma(r_m)}
    """
    _require_single_strategy(inp, "correlation")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    mask = np.isfinite(r) & np.isfinite(bench)
    if mask.sum() < 3:
        value: float = np.nan
    else:
        value = float(np.corrcoef(r[mask], bench[mask])[0, 1])
    return MetricResult(
        name="correlation",
        value=value,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_PEARSON},
    )


# ===================================================================
# §8.10  Active Return
# ===================================================================


@register_metric(
    name="active_return",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_BACON_92 + " " + _REF_CFA,
)
def active_return(inp: BenchmarkInput) -> MetricResult:
    r"""Annualized active return (mean excess return vs. benchmark).

    .. math::
        \bar{r}_{\text{active, ann}} = (\bar{r} - \bar{r}_m) \cdot P
    """
    _require_single_strategy(inp, "active_return")
    _require_periods_per_year(inp, "active_return")
    r = inp.returns[:, 0]
    active = _active_return_series(r, inp.benchmark)
    p = float(inp.periods_per_year)  # type: ignore[arg-type]
    value: float = float(np.nanmean(active) * p)
    return MetricResult(
        name="active_return",
        value=value,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_BACON_92 + " " + _REF_CFA,
            "annualized": True,
        },
    )


# ===================================================================
# §8.11  Batting Average vs Benchmark
# ===================================================================


@register_metric(
    name="batting_average",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_BACON_95,
)
def batting_average(inp: BenchmarkInput) -> MetricResult:
    r"""Fraction of periods where strategy return exceeds benchmark.

    .. math::
        \text{BA} = \frac{1}{n}\sum_{t=1}^{n}
        \mathbf{1}_{[r_t > r_{m,t}]}
    """
    _require_single_strategy(inp, "batting_average")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    valid = np.isfinite(r) & np.isfinite(bench)
    if valid.sum() == 0:
        value: float = np.nan
    else:
        value = float(np.sum(r[valid] > bench[valid]) / valid.sum())
    return MetricResult(
        name="batting_average",
        value=value,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_BACON_95},
    )


# ===================================================================
# §8.12  Treynor Ratio
# ===================================================================


@register_metric(
    name="treynor_ratio",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_TREYNOR,
)
def treynor_ratio(inp: BenchmarkInput) -> MetricResult:
    r"""Treynor ratio (annualized excess return per unit beta).

    .. math::
        \text{Treynor} = \frac{\bar{r}_{\text{excess}} \cdot P}{\beta}
    """
    _require_single_strategy(inp, "treynor_ratio")
    _require_periods_per_year(inp, "treynor_ratio")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    p = float(inp.periods_per_year)  # type: ignore[arg-type]
    excess = r - inp.rf
    mean_excess_ann = float(np.nanmean(excess) * p)
    beta_val = ols_beta(r, bench)
    if beta_val == 0.0 or np.isnan(beta_val):
        value: float = np.inf if mean_excess_ann > 0.0 else (
            -np.inf if mean_excess_ann < 0.0 else np.nan
        )
    else:
        value = mean_excess_ann / beta_val
    return MetricResult(
        name="treynor_ratio",
        value=value,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_TREYNOR, "annualized": True},
    )


# ===================================================================
# §8.13  Outperformance
# ===================================================================


@register_metric(
    name="outperformance",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_BACON_92,
)
def outperformance(inp: BenchmarkInput) -> MetricResult:
    r"""Total cumulative return difference vs. benchmark.

    .. math::
        R_{\text{out}} = R_{\text{cum}} - R_{m,\text{cum}}

    where :math:`R_{\text{cum}} = \prod_t (1+r_t) - 1`.
    """
    _require_single_strategy(inp, "outperformance")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    r_cum = float(np.nanprod(1.0 + r) - 1.0)
    m_cum = float(np.nanprod(1.0 + bench) - 1.0)
    value: float = r_cum - m_cum
    return MetricResult(
        name="outperformance",
        value=value,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_BACON_92},
    )


# ===================================================================
# §8.14  Outperformance Ratio
# ===================================================================


@register_metric(
    name="outperformance_ratio",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_BACON_92,
)
def outperformance_ratio(inp: BenchmarkInput) -> MetricResult:
    r"""Ratio of total cumulative returns vs. benchmark.

    .. math::
        \text{OR} = \frac{1 + R_{\text{cum}}}{1 + R_{m,\text{cum}}}
    """
    _require_single_strategy(inp, "outperformance_ratio")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    r_cum = np.nanprod(1.0 + r)
    m_cum = np.nanprod(1.0 + bench)
    if m_cum <= 0.0:
        value: float = np.inf if r_cum > 0.0 else np.nan
    else:
        value = float(r_cum / m_cum)
    return MetricResult(
        name="outperformance_ratio",
        value=value,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_BACON_92},
    )


# ===================================================================
# §8.15  Underperforming Periods / %
# ===================================================================


@register_metric(
    name="underperforming_periods",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_BACON_95,
)
def underperforming_periods(inp: BenchmarkInput) -> MetricResult:
    r"""Count and fraction of periods where strategy underperforms benchmark.

    Returns ``[count, pct]``.
    """
    _require_single_strategy(inp, "underperforming_periods")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    valid = np.isfinite(r) & np.isfinite(bench)
    n_valid = int(valid.sum())
    if n_valid == 0:
        arr: NDArray[np.floating] = np.full(2, np.nan)
    else:
        n_under = int(np.sum(r[valid] < bench[valid]))
        arr = np.array([n_under, n_under / n_valid], dtype=np.float64)
    return MetricResult(
        name="underperforming_periods",
        value=arr,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_BACON_95,
            "output_index": ["count", "pct"],
        },
    )


# ===================================================================
# §8.16  Max Outperformance
# ===================================================================


@register_metric(
    name="max_outperformance",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_BACON_93,
)
def max_outperformance(inp: BenchmarkInput) -> MetricResult:
    r"""Maximum value of the cumulative active return series.

    .. math::
        \max_t \sum_{\tau=1}^{t} (r_\tau - r_{m,\tau})
    """
    _require_single_strategy(inp, "max_outperformance")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    active = _active_return_series(r, bench)
    # Cumsum ignoring NaN at the start
    active_filled = np.where(np.isfinite(active), active, 0.0)
    cum_active = np.cumsum(active_filled)
    value: float = float(np.max(cum_active))
    return MetricResult(
        name="max_outperformance",
        value=value,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_BACON_93},
    )


# ===================================================================
# §8.17  Max Underperformance
# ===================================================================


@register_metric(
    name="max_underperformance",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_BACON_93,
)
def max_underperformance(inp: BenchmarkInput) -> MetricResult:
    r"""Deepest cumulative underperformance vs. benchmark.

    Maximum absolute value of cumulative active return below zero.

    .. math::
        \text{MaxUnder} = |\min(0,
        \min_t \sum_{\tau=1}^{t} (r_\tau - r_{m,\tau}))|
    """
    _require_single_strategy(inp, "max_underperformance")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    active = _active_return_series(r, bench)
    active_filled = np.where(np.isfinite(active), active, 0.0)
    cum_active = np.cumsum(active_filled)
    min_cum = np.min(cum_active)
    value: float = float(np.abs(min(0.0, min_cum)))
    return MetricResult(
        name="max_underperformance",
        value=value,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_BACON_93},
    )


# ===================================================================
# §8.18  Benchmark Volatility
# ===================================================================


@register_metric(
    name="benchmark_volatility",
    requires="benchmark",
    category=("benchmark", "risk"),
    backend="vectorized",
    ref=_REF_CFA_QM,
)
def benchmark_volatility(inp: BenchmarkInput) -> MetricResult:
    r"""Annualized benchmark volatility.

    .. math::
        \sigma_{m,\text{ann}} = \sigma(r_m) \cdot \sqrt{P}
    """
    _require_single_strategy(inp, "benchmark_volatility")
    _require_periods_per_year(inp, "benchmark_volatility")
    bench = inp.benchmark
    p = float(inp.periods_per_year)  # type: ignore[arg-type]
    value: float = float(np.nanstd(bench, ddof=1) * np.sqrt(p))
    return MetricResult(
        name="benchmark_volatility",
        value=value,
        category=("benchmark", "risk"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_CFA_QM, "annualized": True},
    )


# ===================================================================
# §8.19  Information Coefficient (Rank IC)
# ===================================================================

_REF_RANK_IC = (
    "Spearman (1904); standard rank-based information coefficient."
)


@register_metric(
    name="information_coefficient",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_RANK_IC,
)
def information_coefficient(inp: BenchmarkInput) -> MetricResult:
    r"""Spearman rank correlation between strategy and benchmark returns.

    Also known as the *rank information coefficient* (rank IC). Unlike
    Pearson correlation, rank IC is robust to outliers and captures
    monotonic (not just linear) association.

    .. math::
        \rho_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}

    where :math:`d_i` is the difference between rank pairs.

    Returns NaN for fewer than 3 valid overlapping observations.
    """
    _require_single_strategy(inp, "information_coefficient")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    mask = np.isfinite(r) & np.isfinite(bench)
    n = int(mask.sum())
    if n < 3:
        value: float = np.nan
        return MetricResult(
            name="information_coefficient",
            value=value,
            category=("benchmark",),
            periods_per_year=inp.periods_per_year,
            meta={"ref": _REF_RANK_IC},
        )

    rc = r[mask]
    bc = bench[mask]

    # Compute ranks manually (no scipy dependency).
    # Use average-rank tie handling (standard Spearman).
    def _avg_rank(x: NDArray[np.floating]) -> NDArray[np.float64]:
        order = np.argsort(x)
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            while j < n and x[order[j]] == x[order[i]]:
                j += 1
            mean_rank = (i + j + 2) / 2.0  # 1-based averaging
            for k in range(i, j):
                ranks[order[k]] = mean_rank
            i = j
        return ranks

    rank_r = _avg_rank(rc)
    rank_b = _avg_rank(bc)

    d = rank_r - rank_b
    rho = 1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1.0))
    value = float(rho)

    return MetricResult(
        name="information_coefficient",
        value=value,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_RANK_IC},
    )


# ===================================================================
# §8.20  Directional Consistency
# ===================================================================

_REF_DIR_CONSISTENCY = (
    "Standard statistic; fraction of periods where strategy and benchmark "
    "returns share the same sign."
)


@register_metric(
    name="directional_consistency",
    requires="benchmark",
    category=("benchmark",),
    backend="vectorized",
    ref=_REF_DIR_CONSISTENCY,
)
def directional_consistency(inp: BenchmarkInput) -> MetricResult:
    r"""Fraction of periods where strategy and benchmark agree on sign.

    .. math::
        \text{DC} = \frac{1}{n} \sum_{t=1}^{n}
        \mathbf{1}_{[\operatorname{sign}(r_t) = \operatorname{sign}(r_{m,t})]}

    Periods where either return is exactly zero (sign = 0) are counted as
    agreement only if both are exactly zero. NaN periods are excluded.

    Returns NaN if no valid overlapping observations.
    """
    _require_single_strategy(inp, "directional_consistency")
    r = inp.returns[:, 0]
    bench = inp.benchmark
    valid = np.isfinite(r) & np.isfinite(bench)
    n_valid = int(valid.sum())
    if n_valid == 0:
        value: float = np.nan
        return MetricResult(
            name="directional_consistency",
            value=value,
            category=("benchmark",),
            periods_per_year=inp.periods_per_year,
            meta={"ref": _REF_DIR_CONSISTENCY},
        )

    signs_r = np.sign(r[valid])
    signs_b = np.sign(bench[valid])
    n_agree = int(np.sum(signs_r == signs_b))
    value = n_agree / n_valid

    return MetricResult(
        name="directional_consistency",
        value=value,
        category=("benchmark",),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_DIR_CONSISTENCY},
    )
