"""Compare-tier metrics.

Metrics comparing two or more strategies: correlation matrix,
diversification ratio, pairwise Sharpe-difference test (Jobson-Korkie /
Memmel), White's Reality Check / SPA, PBO (combinatorial purged CV),
marginal contribution to portfolio risk, and component VaR.

All tagged: category=("relative", "compare"), requires="compare".
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from stratstat.core._utils import numba_worthwhile, sample_skewness
from stratstat.exceptions import MetricNotApplicableError
from stratstat.inputs import CompareInput
from stratstat.registry import register_metric
from stratstat.results import MetricResult

# ---------------------------------------------------------------------------
# Optional numba acceleration for the resampling kernels.
#
# Numba does not support the new-style ``np.random.Generator``, so all random
# draws are generated in Python and passed into the JIT kernels as precomputed
# integer arrays.  Every kernel below has a pure-numpy fallback defined next to
# the function that uses it; the two paths must agree within floating point
# tolerance (see tests/core/test_compare.py::TestNumbaAgreement).
# ---------------------------------------------------------------------------

try:
    import numba  # noqa: F401
    from numba import njit
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False
else:
    _HAS_NUMBA = True

    @njit(cache=False)
    def _stationary_bootstrap_numba(
        data: NDArray[np.floating],
        starts: NDArray[np.int64],
        blens: NDArray[np.int64],
    ) -> NDArray[np.floating]:
        """NaN-aware stationary bootstrap over precomputed starts and lengths."""
        n_boot = starts.shape[0]
        n_periods = data.shape[0]
        n_strat = data.shape[1]
        means = np.empty((n_boot, n_strat), dtype=np.float64)
        for b in range(n_boot):
            idx = np.empty(n_periods, dtype=np.int64)
            t = 0
            k = 0
            while t < n_periods:
                start = starts[b, k]
                blen = blens[b, k]
                k += 1
                for j in range(blen):
                    if t >= n_periods:
                        break
                    idx[t] = (start + j) % n_periods
                    t += 1
            for col in range(n_strat):
                s = 0.0
                cnt = 0
                for i in range(n_periods):
                    v = data[idx[i], col]
                    if not np.isnan(v):
                        s += v
                        cnt += 1
                means[b, col] = s / cnt if cnt > 0 else np.nan
        return means

    @njit(cache=False)
    def _sharpe_slice_numba(r: NDArray[np.floating], lo: int, hi: int, k: int, rf: float) -> float:
        """NaN-aware per-period Sharpe ratio of ``r[lo:hi, k]`` against ``rf``."""
        s = 0.0
        cnt = 0
        for i in range(lo, hi):
            v = r[i, k]
            if not np.isnan(v):
                s += v - rf
                cnt += 1
        if cnt == 0:
            return np.nan
        mu = s / cnt
        ss = 0.0
        for i in range(lo, hi):
            v = r[i, k]
            if not np.isnan(v):
                d = (v - rf) - mu
                ss += d * d
        denom = cnt - 1
        if denom <= 0:
            return np.nan
        sigma = np.sqrt(ss / denom)
        if sigma < 1e-15:
            return np.nan
        return float(mu / sigma)

    @njit(cache=False)
    def _pbo_overfit_numba(
        r: NDArray[np.floating],
        split_points: NDArray[np.int64],
        purge: int,
        embargo: int,
        rf: float,
    ) -> int:
        """Count splits where the best in-sample strategy ranks below OOS median."""
        n_strat = r.shape[1]
        n_obs = r.shape[0]
        n_splits = split_points.shape[0]
        overfit = 0
        for s in range(n_splits):
            split = split_points[s]
            test_start = split + purge + embargo
            is_sr = np.empty(n_strat, dtype=np.float64)
            best_val = -1e300
            best_is = 0
            ok = True
            for k in range(n_strat):
                sr = _sharpe_slice_numba(r, 0, split, k, rf)
                is_sr[k] = sr
                if np.isnan(sr):
                    ok = False
                    break
                if sr > best_val:
                    best_val = sr
                    best_is = k
            if not ok:
                continue
            oos_sr = np.empty(n_strat, dtype=np.float64)
            for k in range(n_strat):
                sr = _sharpe_slice_numba(r, test_start, n_obs, k, rf)
                oos_sr[k] = sr
                if np.isnan(sr):
                    ok = False
                    break
            if not ok:
                continue
            oos_rank = n_strat
            for k in range(n_strat):
                if oos_sr[k] < oos_sr[best_is]:
                    oos_rank -= 1
            if oos_rank > n_strat / 2.0:
                overfit += 1
        return overfit


# ---------------------------------------------------------------------------
# Citation strings
# ---------------------------------------------------------------------------

_REF_PEARSON = "Pearson (1895); Johnson & Wichern (2007, §2.5)."
_REF_CHOUEIFATY = 'Choueifaty & Coignard (2008), "Toward Maximum Diversification," JPM, 35(1).'
_REF_JK = (
    'Jobson & Korkie (1981), "Performance Hypothesis Testing with the '
    'Sharpe and Treynor Measures," J. Finance, 36(4); '
    'Memmel (2003), "Performance Hypothesis Testing with the Sharpe '
    'Ratio," Finance Letters, 1(1).'
)
_REF_WHITE = (
    'White (2000), "A Reality Check for Data Snooping," Econometrica, '
    '68(5); Hansen (2005), "A Test for Superior Predictive Ability," '
    "JBES, 23(4)."
)
_REF_PBO = (
    'Bailey & López de Prado (2014), "Pseudo-Mathematics and Financial '
    'Charlatanism," Notices of the AMS, 61(5); '
    "de Prado (2018, Advances in Financial Machine Learning, Ch. 11-13)."
)
_REF_MCR = (
    'Litterman (1996), "Hot Spots and Hedges," JPM Special Issue; '
    'Qian (2005), "Risk Parity and Diversification," J. of Investing, 14(3).'
)
_REF_CVAR = "Jorion (2006, Value at Risk, 3rd ed., Ch. 7); Litterman (1996)."


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_min_strategies(inp: CompareInput, n: int, metric_name: str) -> None:
    """Raise ``MetricNotApplicableError`` if fewer than *n* strategy columns."""
    if inp.n_strategies < n:
        raise MetricNotApplicableError(
            f"{metric_name} requires at least {n} strategies. Got {inp.n_strategies}."
        )


def _require_exact_strategies(inp: CompareInput, n: int, metric_name: str) -> None:
    """Raise ``MetricNotApplicableError`` if not exactly *n* strategy columns."""
    if inp.n_strategies != n:
        raise MetricNotApplicableError(
            f"{metric_name} requires exactly {n} strategies. Got {inp.n_strategies}."
        )


def _require_benchmark(inp: CompareInput, metric_name: str) -> None:
    """Raise ``MetricNotApplicableError`` if benchmark returns are not provided."""
    if inp.benchmark is None:
        raise MetricNotApplicableError(
            f"{metric_name} requires benchmark returns. Provide benchmark= to CompareInput."
        )


def _require_periods_per_year(inp: CompareInput, metric_name: str) -> None:
    """Raise ``MetricNotApplicableError`` if ``periods_per_year`` is not set."""
    if inp.periods_per_year is None:
        raise MetricNotApplicableError(
            f"{metric_name} requires periods_per_year for "
            f"annualization. Provide periods_per_year= to CompareInput."
        )


def _output_labels(inp: CompareInput, n: int) -> list[str] | None:
    """Strategy labels keyed to a ``(n,)`` output, or None when unnamed.

    ``CompareInput.labels`` carries the column names from a pandas/polars
    DataFrame; a numpy matrix has no names, so per-strategy outputs stay
    anonymous.  Returns None when the labels do not line up with the output
    length, so a mismatched input can never attach the wrong name to a value.
    """
    if inp.labels is not None and len(inp.labels) == n:
        return list(inp.labels)
    return None


def _per_period_sharpe(r: NDArray[np.floating], rf: float) -> float:
    """Per-period (non-annualised) Sharpe ratio for a single series."""
    excess = r - rf
    mu = np.nanmean(excess)
    sigma = np.nanstd(excess, ddof=1)
    if sigma < 1e-15:
        return np.nan
    return float(mu / sigma)


def _cov_matrix(
    r: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Covariance matrix of strategy returns (n_periods, n_strategies).

    Returns ``(n_strategies, n_strategies)`` with NaN for
    insufficient data.
    """
    n_strat = r.shape[1]
    cov: NDArray[np.floating] = np.full((n_strat, n_strat), np.nan)
    for i in range(n_strat):
        for j in range(i, n_strat):
            mask = np.isfinite(r[:, i]) & np.isfinite(r[:, j])
            if mask.sum() < 3:
                cov[i, j] = np.nan
                cov[j, i] = np.nan
            else:
                c = np.cov(r[mask, i], r[mask, j], ddof=1)
                cov[i, j] = c[0, 1]
                cov[j, i] = c[0, 1]
    return cov


# ===================================================================
# §9.1  Correlation Matrix
# ===================================================================


@register_metric(
    name="correlation_matrix",
    requires="compare",
    category=("relative", "compare"),
    backend="vectorized",
    ref=_REF_PEARSON,
)
def correlation_matrix(inp: CompareInput) -> MetricResult:
    r"""Pearson correlation matrix of strategy returns.

    .. math::
        \Sigma_{ij} = \text{Corr}(r^{(i)}, r^{(j)})
        \quad\text{for } i,j = 1,\dots,K

    Returns a ``(K, K)`` symmetric array.  Diagonal entries are 1.0
    for valid series; NaN where fewer than 3 overlapping observations.
    """
    _require_min_strategies(inp, 2, "correlation_matrix")
    r = inp.returns
    n_strat = r.shape[1]
    corr: NDArray[np.floating] = np.full((n_strat, n_strat), np.nan)
    for i in range(n_strat):
        for j in range(i, n_strat):
            mask = np.isfinite(r[:, i]) & np.isfinite(r[:, j])
            if mask.sum() < 3:
                corr[i, j] = np.nan
                corr[j, i] = np.nan
            else:
                c = np.corrcoef(r[mask, i], r[mask, j])[0, 1]
                corr[i, j] = c
                corr[j, i] = c
    meta: dict[str, Any] = {"ref": _REF_PEARSON, "shape": (n_strat, n_strat)}
    labels = _output_labels(inp, n_strat)
    if labels is not None:
        meta["labels"] = labels  # row and column axis, shared (symmetric matrix)
    return MetricResult(
        name="correlation_matrix",
        value=corr,
        category=("relative", "compare"),
        periods_per_year=inp.periods_per_year,
        meta=meta,
    )


# ===================================================================
# §9.2  Diversification Ratio
# ===================================================================


@register_metric(
    name="diversification_ratio",
    requires="compare",
    category=("relative", "compare"),
    backend="vectorized",
    ref=_REF_CHOUEIFATY,
)
def diversification_ratio(inp: CompareInput) -> MetricResult:
    r"""Diversification ratio.

    .. math::
        \text{DR} = \frac{\sum_{i=1}^{K} w_i \sigma_i}{\sigma_p}

    where :math:`\sigma_i` is the per-strategy volatility,
    :math:`\sigma_p = \sqrt{w^\top \Sigma w}` is the portfolio
    volatility, and :math:`w` defaults to equal weight.

    Returns NaN when fewer than 3 valid overlapping observations
    across all strategies.
    """
    _require_min_strategies(inp, 2, "diversification_ratio")
    r = inp.returns
    w = inp.get_weights()
    n_strat = r.shape[1]

    # Per-strategy volatilities
    sigmas = np.array(
        [float(np.nanstd(r[:, i], ddof=1)) for i in range(n_strat)],
        dtype=np.float64,
    )

    # Portfolio volatility
    cov = _cov_matrix(r)
    # Use only rows/cols without NaN on the diagonal
    valid = ~np.isnan(np.diag(cov))
    if valid.sum() < 2:
        value: float = np.nan
        return MetricResult(
            name="diversification_ratio",
            value=value,
            category=("relative", "compare"),
            periods_per_year=inp.periods_per_year,
            meta={"ref": _REF_CHOUEIFATY, "weights": w.tolist()},
        )

    # Subset to strategies with valid pairwise covariances
    sub_cov = cov[valid][:, valid]
    # Guard against NaN in off-diagonals (insufficient overlap between
    # two strategies that individually have enough data)
    if np.any(np.isnan(sub_cov)):
        value = np.nan
        return MetricResult(
            name="diversification_ratio",
            value=value,
            category=("relative", "compare"),
            periods_per_year=inp.periods_per_year,
            meta={"ref": _REF_CHOUEIFATY, "weights": w.tolist()},
        )

    port_var = float(w[valid] @ sub_cov @ w[valid])
    if port_var <= 0.0:
        value = np.nan
    else:
        port_vol = np.sqrt(port_var)
        weighted_sum = float(np.dot(w[valid], sigmas[valid]))
        value = weighted_sum / port_vol

    return MetricResult(
        name="diversification_ratio",
        value=value,
        category=("relative", "compare"),
        periods_per_year=inp.periods_per_year,
        meta={"ref": _REF_CHOUEIFATY, "weights": w.tolist()},
    )


# ===================================================================
# §9.3  Pairwise Sharpe-Difference Test (Jobson-Korkie, Memmel)
# ===================================================================


@register_metric(
    name="sharpe_difference_test",
    requires="compare",
    category=("relative", "compare"),
    backend="vectorized",
    ref=_REF_JK,
)
def sharpe_difference_test(inp: CompareInput) -> MetricResult:
    r"""Pairwise Sharpe-difference z-test (Jobson-Korkie / Memmel).

    .. math::
        z &= \frac{\hat{SR}_1 - \hat{SR}_2}
            {\sqrt{\hat{\sigma}^2}} \\[4pt]
        \hat{\sigma}^2 &= \frac{1}{T}\Big[
            2 - 2\rho_{12}
            + \tfrac{1}{2}(\hat{SR}_1^2 + \hat{SR}_2^2)
            - \rho_{12}^2 \hat{SR}_1 \hat{SR}_2
            - (\gamma_{3,1}\hat{SR}_1
               - \gamma_{3,2}\hat{SR}_2) \cdot 2\rho_{12}
        \Big]

    where :math:`\gamma_{3,k}` is the sample skewness of strategy *k*.

    Requires exactly 2 strategies.  Returns ``[z, p_value, sr_diff]``.

    The null hypothesis is :math:`SR_1 = SR_2`; *p* is two-sided.
    """
    _require_exact_strategies(inp, 2, "sharpe_difference_test")
    r = inp.returns
    r1 = r[:, 0]
    r2 = r[:, 1]
    rf = inp.rf_period

    # Per-period Sharpe ratios
    sr1 = _per_period_sharpe(r1, rf)
    sr2 = _per_period_sharpe(r2, rf)

    # Correlation
    mask = np.isfinite(r1) & np.isfinite(r2)
    n_valid = int(mask.sum())
    if n_valid < 3:
        arr: NDArray[np.floating] = np.full(3, np.nan)
        return MetricResult(
            name="sharpe_difference_test",
            value=arr,
            category=("relative", "compare"),
            periods_per_year=inp.periods_per_year,
            meta={
                "ref": _REF_JK,
                "output_index": ["z", "p_value", "sr_diff"],
                "rf": inp.rf,
                "rf_period": inp.rf_period,
            },
        )

    r1c = r1[mask]
    r2c = r2[mask]
    rho = float(np.corrcoef(r1c, r2c)[0, 1])

    # Skewness of each strategy
    skew1 = sample_skewness(r1c)
    skew2 = sample_skewness(r2c)

    sr_diff = sr1 - sr2

    # Asymptotic variance (Memmel 2003)
    var_jk = (
        2.0
        - 2.0 * rho
        + 0.5 * (sr1**2 + sr2**2)
        - rho**2 * sr1 * sr2
        - (skew1 * sr1 - skew2 * sr2) * 2.0 * rho
    ) / n_valid

    if var_jk <= 0.0:
        z = np.nan
        p_value = np.nan
    else:
        se = np.sqrt(var_jk)
        z = float(sr_diff / se)
        # Two-sided p-value via normal approximation
        from math import erfc

        p_value = float(erfc(abs(z) / np.sqrt(2.0)))

    arr = np.array([z, p_value, sr_diff], dtype=np.float64)

    return MetricResult(
        name="sharpe_difference_test",
        value=arr,
        category=("relative", "compare"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_JK,
            "output_index": ["z", "p_value", "sr_diff"],
            "sr1": sr1,
            "sr2": sr2,
            "rho": rho,
            "n_valid": n_valid,
            "rf": inp.rf,
            "rf_period": inp.rf_period,
        },
    )


# ===================================================================
# §9.4  White's Reality Check / SPA Test
# ===================================================================


def _stationary_bootstrap_fallback(
    data: NDArray[np.floating],
    starts: NDArray[np.int64],
    blens: NDArray[np.int64],
) -> NDArray[np.floating]:
    """Pure-numpy stationary bootstrap over precomputed starts and lengths.

    Reference implementation for ``_stationary_bootstrap_numba``; the two must
    agree within floating point tolerance.
    """
    n_boot = starts.shape[0]
    n_periods = data.shape[0]
    n_strat = data.shape[1]
    means = np.empty((n_boot, n_strat), dtype=np.float64)

    for b in range(n_boot):
        idx = np.zeros(n_periods, dtype=np.intp)
        t = 0
        k = 0
        while t < n_periods:
            start = starts[b, k]
            blen = blens[b, k]
            k += 1
            for j in range(blen):
                if t >= n_periods:
                    break
                idx[t] = (start + j) % n_periods
                t += 1
        means[b] = np.nanmean(data[idx], axis=0)

    return means


def _stationary_bootstrap(
    data: NDArray[np.floating],
    n_boot: int,
    block_size: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Stationary (circular block) bootstrap of Politis & Romano (1994).

    Random block starts and geometric block lengths are drawn once in Python
    (numba does not support ``np.random.Generator``) so the numba and
    pure-numpy paths consume identical draws.  Returns ``(n_boot, n_strategies)``
    array of bootstrap sample means.
    """
    n_periods = data.shape[0]
    n_strat = data.shape[1]
    # Worst case: every block has length 1, so at most n_periods blocks are
    # ever consumed per replicate.
    starts = rng.integers(0, n_periods, size=(n_boot, n_periods))
    blens = rng.geometric(1.0 / block_size, size=(n_boot, n_periods))

    if _HAS_NUMBA and numba_worthwhile(n_boot * n_periods * (n_strat + 1)):
        return _stationary_bootstrap_numba(data, starts, blens)
    return _stationary_bootstrap_fallback(data, starts, blens)


@register_metric(
    name="whites_reality_check",
    requires="compare",
    category=("relative", "compare"),
    backend="resampling",
    ref=_REF_WHITE,
)
def whites_reality_check(
    inp: CompareInput,
    n_boot: int = 1000,
    block_size: int = 1,
    seed: int | None = None,
) -> MetricResult:
    r"""White's Reality Check (White 2000).

    Tests the null hypothesis that the best strategy does not
    outperform the benchmark:

    .. math::
        H_0: \max_k E[f_{k,t}] \le 0

    where :math:`f_{k,t} = r_{k,t} - r_{m,t}` is the period-*t*
    excess return of strategy *k* over the benchmark.

    The test statistic is
    :math:`\bar{V} = \max_k \sqrt{T} \, \bar{f}_k`.

    The null distribution is obtained via stationary bootstrap
    (Politis & Romano 1994), recentering each strategy's excess
    returns to satisfy the least-favourable configuration.

    Parameters
    ----------
    n_boot:
        Number of bootstrap resamples (default 1000).
    block_size:
        Expected block length for stationary bootstrap (default 1 = i.i.d.).
    seed:
        Optional seed for reproducible bootstrap draws.

    Returns
    -------
    ``[statistic, p_value]``.
    """
    _require_min_strategies(inp, 1, "whites_reality_check")
    _require_benchmark(inp, "whites_reality_check")
    r = inp.returns
    # Guarded by _require_benchmark above
    bench = inp.benchmark
    assert bench is not None
    n_periods = r.shape[0]

    # Excess returns over benchmark
    f_kt = r - bench.reshape(-1, 1)  # (n_periods, n_strategies)
    f_bar = np.nanmean(f_kt, axis=0)  # (n_strategies,)

    # Test statistic
    v_obs = float(np.sqrt(n_periods) * np.nanmax(f_bar))

    # Recenter for least-favourable configuration
    # f_tilde = f - max(0, f_bar)  per strategy
    adj = np.maximum(0.0, f_bar)  # (n_strategies,)
    f_tilde = f_kt - adj[np.newaxis, :]

    # Fill NaN with 0 (periods with missing data don't contribute)
    f_tilde = np.where(np.isfinite(f_tilde), f_tilde, 0.0)

    # Bootstrap
    rng = np.random.default_rng(seed)
    boot_means = _stationary_bootstrap(f_tilde, n_boot, float(block_size), rng)
    # Bootstrap max statistic
    v_star = np.sqrt(n_periods) * np.max(boot_means, axis=1)  # (n_boot,)

    p_value = float(np.mean(v_star >= v_obs))

    arr = np.array([v_obs, p_value], dtype=np.float64)

    return MetricResult(
        name="whites_reality_check",
        value=arr,
        category=("relative", "compare"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_WHITE,
            "output_index": ["statistic", "p_value"],
            "n_boot": n_boot,
            "block_size": block_size,
        },
    )


# ===================================================================
# §9.5  PBO (Probability of Backtest Overfitting)
# ===================================================================


def _comb_purged_split_points(
    n_obs: int,
    n_splits: int,
    purge_pct: float,
    embargo_pct: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], int, int]:
    """Generate valid combinatorial purged train/test split points.

    Each split point ``s`` defines an in-sample window ``[0, s)`` and an
    out-of-sample window ``[s + purge + embargo, n_obs)``.  Returns the valid
    split points plus the purge and embargo widths (in periods).
    """
    purge = max(1, int(n_obs * purge_pct))
    embargo = max(0, int(n_obs * embargo_pct))
    # Minimum test size
    min_test = max(1, n_obs // 4)

    points: list[int] = []
    attempts = 0
    max_attempts = n_splits * 20

    while len(points) < n_splits and attempts < max_attempts:
        attempts += 1
        # Random split point (near the middle, with variance)
        split = int(rng.integers(n_obs // 3, 2 * n_obs // 3))
        # With purge + embargo gap
        test_start = split + purge + embargo
        if test_start >= n_obs - min_test:
            continue
        points.append(split)

    return np.asarray(points, dtype=np.int64), purge, embargo


def _pbo_overfit_fallback(
    r: NDArray[np.floating],
    split_points: NDArray[np.int64],
    purge: int,
    embargo: int,
    rf: float,
) -> int:
    """Pure-numpy PBO overfit count over precomputed split points.

    Reference implementation for ``_pbo_overfit_numba``; the two must agree
    within floating point tolerance.
    """
    n_strat = r.shape[1]
    overfit = 0

    for split in split_points:
        split = int(split)
        test_start = split + purge + embargo
        is_sr = np.array(
            [_per_period_sharpe(r[0:split, k], rf) for k in range(n_strat)],
            dtype=np.float64,
        )
        oos_sr = np.array(
            [_per_period_sharpe(r[test_start:, k], rf) for k in range(n_strat)],
            dtype=np.float64,
        )
        if np.any(np.isnan(is_sr)) or np.any(np.isnan(oos_sr)):
            continue

        best_is = int(np.nanargmax(is_sr))
        oos_rank = n_strat - int(np.sum(oos_sr < oos_sr[best_is]))
        if oos_rank > n_strat / 2:
            overfit += 1

    return overfit


@register_metric(
    name="pbo",
    requires="compare",
    category=("relative", "compare"),
    backend="resampling",
    ref=_REF_PBO,
)
def pbo(
    inp: CompareInput,
    n_splits: int = 100,
    purge_pct: float = 0.01,
    embargo_pct: float = 0.0,
    seed: int | None = None,
) -> MetricResult:
    r"""Probability of Backtest Overfitting (Bailey & López de Prado 2014).

    Generates combinatorially-paired train/test splits with purge and
    embargo periods.  For each split, strategies are ranked by
    in-sample Sharpe ratio.  PBO is the probability that the best
    in-sample strategy ranks below the median out-of-sample:

    .. math::
        \text{PBO} = \frac{1}{S} \sum_{s=1}^{S}
        \mathbf{1}_{[\text{rank}_{\text{OOS}}(k_s^*) > K/2]}

    where :math:`k_s^*` is the best IS strategy in split *s*.

    Parameters
    ----------
    n_splits:
        Number of train/test splits (default 100).
    purge_pct:
        Fraction of observations to purge between train and test
        (default 0.01).
    embargo_pct:
        Fraction of observations to embargo after purge (default 0.0).
    seed:
        Optional seed for reproducible split generation.

    Returns
    -------
    ``[pbo, n_splits_used]``.
    """
    _require_min_strategies(inp, 2, "pbo")
    r = inp.returns
    n_periods = r.shape[0]
    rf = inp.rf_period

    rng = np.random.default_rng(seed)
    split_points, purge, embargo = _comb_purged_split_points(
        n_periods, n_splits, purge_pct, embargo_pct, rng
    )

    if len(split_points) == 0:
        arr: NDArray[np.floating] = np.array([np.nan, 0], dtype=np.float64)
        return MetricResult(
            name="pbo",
            value=arr,
            category=("relative", "compare"),
            periods_per_year=inp.periods_per_year,
            meta={
                "ref": _REF_PBO,
                "output_index": ["pbo", "n_splits"],
                "n_splits_requested": n_splits,
                "rf": inp.rf,
                "rf_period": inp.rf_period,
            },
        )

    work = 2 * len(split_points) * n_periods * r.shape[1]
    if _HAS_NUMBA and numba_worthwhile(work):
        overfit_count = _pbo_overfit_numba(r, split_points, purge, embargo, rf)
    else:
        overfit_count = _pbo_overfit_fallback(r, split_points, purge, embargo, rf)

    n_used = len(split_points)
    pbo_val = float(overfit_count / n_used)

    arr = np.array([pbo_val, n_used], dtype=np.float64)

    return MetricResult(
        name="pbo",
        value=arr,
        category=("relative", "compare"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_PBO,
            "output_index": ["pbo", "n_splits"],
            "n_splits_requested": n_splits,
            "purge_pct": purge_pct,
            "embargo_pct": embargo_pct,
            "rf": inp.rf,
            "rf_period": inp.rf_period,
        },
    )


# ===================================================================
# §9.6  Marginal Contribution to Portfolio Risk
# ===================================================================


@register_metric(
    name="marginal_contribution_to_risk",
    requires="compare",
    category=("relative", "compare"),
    backend="vectorized",
    ref=_REF_MCR,
)
def marginal_contribution_to_risk(inp: CompareInput) -> MetricResult:
    r"""Marginal contribution of each strategy to portfolio risk.

    .. math::
        \text{MCR}_i = w_i \cdot \frac{(\Sigma w)_i}{\sigma_p}

    where :math:`\Sigma` is the covariance matrix of strategy returns,
    :math:`w` the weight vector, and
    :math:`\sigma_p = \sqrt{w^\top \Sigma w}`.

    Returns a ``(K,)`` array.  Weights default to equal weight.
    """
    _require_min_strategies(inp, 2, "marginal_contribution_to_risk")
    r = inp.returns
    w = inp.get_weights()
    cov = _cov_matrix(r)
    n_strat = r.shape[1]
    label_meta: dict[str, Any] = {}
    _labels = _output_labels(inp, n_strat)
    if _labels is not None:
        label_meta["output_index"] = _labels

    # Check for NaN in the covariance matrix — _cov_matrix returns NaN
    # for pairs with <3 overlapping observations.  We subset to
    # strategies with complete pairwise data, mirroring the approach
    # in diversification_ratio.
    if np.any(np.isnan(cov)):
        valid = ~np.isnan(np.diag(cov))
        if valid.sum() < 2:
            mcr: NDArray[np.floating] = np.full(n_strat, np.nan)
            return MetricResult(
                name="marginal_contribution_to_risk",
                value=mcr,
                category=("relative", "compare"),
                periods_per_year=inp.periods_per_year,
                meta={
                    "ref": _REF_MCR,
                    "weights": w.tolist(),
                    "portfolio_vol": np.nan,
                    **label_meta,
                },
            )
        sub_cov = cov[valid][:, valid]
        if np.any(np.isnan(sub_cov)):
            mcr = np.full(n_strat, np.nan)
            return MetricResult(
                name="marginal_contribution_to_risk",
                value=mcr,
                category=("relative", "compare"),
                periods_per_year=inp.periods_per_year,
                meta={
                    "ref": _REF_MCR,
                    "weights": w.tolist(),
                    "portfolio_vol": np.nan,
                    **label_meta,
                },
            )
        # Compute MCR on the valid subset, return NaN for excluded strategies
        port_var = float(w[valid] @ sub_cov @ w[valid])
        if port_var <= 0.0:
            mcr = np.full(n_strat, np.nan)
        else:
            port_vol = np.sqrt(port_var)
            marginal = sub_cov @ w[valid]  # (n_valid,)
            mcr_valid = w[valid] * marginal / port_vol
            mcr = np.full(n_strat, np.nan)
            mcr[valid] = mcr_valid
        return MetricResult(
            name="marginal_contribution_to_risk",
            value=mcr,
            category=("relative", "compare"),
            periods_per_year=inp.periods_per_year,
            meta={
                "ref": _REF_MCR,
                "weights": w.tolist(),
                "portfolio_vol": (float(np.sqrt(port_var)) if port_var > 0.0 else np.nan),
                **label_meta,
            },
        )

    port_var = float(w @ cov @ w)
    if port_var <= 0.0:
        mcr = np.full(n_strat, np.nan)
    else:
        port_vol = np.sqrt(port_var)
        marginal = cov @ w  # (K,)
        mcr = w * marginal / port_vol

    return MetricResult(
        name="marginal_contribution_to_risk",
        value=mcr,
        category=("relative", "compare"),
        periods_per_year=inp.periods_per_year,
        meta={
            "ref": _REF_MCR,
            "weights": w.tolist(),
            "portfolio_vol": (float(np.sqrt(port_var)) if port_var > 0.0 else np.nan),
            **label_meta,
        },
    )


# ===================================================================
# §9.7  Component VaR
# ===================================================================


@register_metric(
    name="component_var",
    requires="compare",
    category=("relative", "compare"),
    backend="vectorized",
    ref=_REF_CVAR,
)
def component_var(
    inp: CompareInput,
    confidence: float = 0.95,
) -> MetricResult:
    r"""Component VaR via marginal revaluation.

    For each strategy *i*, component VaR is the difference between
    total portfolio VaR and portfolio VaR with strategy *i* removed:

    .. math::
        \text{CVaR}_i = \text{VaR}_p(w) - \text{VaR}_p(w_{-i})

    where :math:`w_{-i}` sets :math:`w_i = 0` and re-normalises the
    remaining weights to sum to 1.

    .. note::

        Rows where any strategy has a NaN return produce a NaN
        portfolio return and are implicitly excluded by
        ``np.nanquantile`` (pairwise deletion).  This matches the
        standard convention for VaR under incomplete data.

    Parameters
    ----------
    confidence: VaR confidence level (default 0.95).

    Returns
    -------
    ``(K,)`` array of per-strategy component VaR values.
    """
    _require_min_strategies(inp, 2, "component_var")
    r = inp.returns
    w = inp.get_weights()
    n_strat = r.shape[1]
    labels = _output_labels(inp, n_strat)

    # Portfolio return series
    port_ret = r @ w  # (n_periods,)

    # Total portfolio VaR
    total_var = float(-np.nanquantile(port_ret, 1.0 - confidence))

    cvar = np.empty(n_strat, dtype=np.float64)
    for i in range(n_strat):
        # Remove strategy i and re-normalise weights
        w_minus = np.delete(w, i)
        w_minus = w_minus / np.sum(w_minus)
        r_minus = np.delete(r, i, axis=1)
        port_minus = r_minus @ w_minus
        var_minus = float(-np.nanquantile(port_minus, 1.0 - confidence))
        cvar[i] = total_var - var_minus

    meta: dict[str, Any] = {
        "ref": _REF_CVAR,
        "confidence": confidence,
        "total_var": total_var,
        "weights": w.tolist(),
    }
    if labels is not None:
        meta["output_index"] = labels
    return MetricResult(
        name="component_var",
        value=cvar,
        category=("relative", "compare"),
        periods_per_year=inp.periods_per_year,
        meta=meta,
    )
