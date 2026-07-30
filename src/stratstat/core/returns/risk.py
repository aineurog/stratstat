"""Risk metrics for returns.

Metrics: max drawdown, longest drawdown duration, time to recovery,
average drawdown, average drawdown duration, ulcer index, downside deviation,
upside deviation, VaR, CVaR, tail ratio, common-sense ratio, Hill tail index,
GPD tail fit, risk of ruin, drawdown volatility, drawdown periods count,
current drawdown, current drawdown duration, drawdown total duration.

All tagged: category=("risk", "returns"). Backend varies: mostly "vectorized",
drawdown walks are "sequential".
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from stratstat.inputs import ReturnsInput
from stratstat.registry import register_metric
from stratstat.results import MetricResult


class _DrawdownEpisode(TypedDict):
    """A single drawdown episode."""

    start: int
    end: int
    trough_idx: int
    depth: float
    duration: int
    recovered: bool


# ---------------------------------------------------------------------------
# Helper: drawdown analysis (used by many risk metrics)
# ---------------------------------------------------------------------------


def _equity_curve(
    r: NDArray[np.floating], return_type: str = "simple"
) -> NDArray[np.floating]:
    """Build the cumulative return index (equity curve) from period returns.

    Args:
        r: (n_periods, n_strategies) return array.
        return_type: ``"simple"`` (default) — returns are simple returns,
            equity = cumprod(1+r). ``"log"`` — returns are log returns,
            equity = exp(cumsum(r)).

    Returns:
        Equity curve array of shape (n_periods, n_strategies), starting at 1.0
        (prepended row) for drawdown computation convenience.
    """
    if return_type == "log":
        # Log returns: equity = exp(cumsum(r)), prepend 1.0
        cum_log = np.nancumsum(r, axis=0)
        curve = np.exp(cum_log)
    else:
        # Simple returns: equity = cumprod(1+r)
        # np.nancumprod treats NaN as 1, which is correct (NaN period = flat).
        curve = np.nancumprod(1.0 + r, axis=0)

    # Prepend a row of 1.0 for drawdown-from-start computation
    init = np.ones((1, r.shape[1]), dtype=np.float64)
    return np.concatenate([init, curve], axis=0)


def _drawdown_series(
    equity: NDArray[np.floating],
) -> tuple[
    NDArray[np.floating],  # running_max
    NDArray[np.floating],  # drawdown series (0 or negative)
]:
    """Compute the running maximum and percentage drawdown series.

    Args:
        equity: (n_periods, n_strategies) equity curve (with prepended 1.0).

    Returns:
        (running_max, drawdown) where:
        - running_max: shape (n_periods, n_strategies)
        - drawdown: shape (n_periods, n_strategies), values ≤ 0.
          drawdown[t] = (equity[t] - running_max[t]) / running_max[t]
    """
    running_max = np.maximum.accumulate(equity, axis=0)
    dd: NDArray[np.floating] = (equity - running_max) / running_max
    return running_max, dd


def _drawdown_episodes(
    equity: NDArray[np.floating],
    running_max: NDArray[np.floating],
    dd: NDArray[np.floating],
) -> list[_DrawdownEpisode]:
    """Identify distinct drawdown episodes for a single-strategy column.

    An episode begins when equity falls below the running maximum and ends
    the first period equity returns to (or exceeds) the running maximum.

    Args:
        equity: 1-D equity curve with prepended 1.0.
        running_max: 1-D running maximum series.
        dd: 1-D drawdown series.

    Returns:
        List of dicts, each with keys:
        - start: index of first underwater period
        - end: index where equity recovers to running max (or len-1 if ongoing)
        - trough_idx: index of maximum drawdown depth within episode
        - depth: peak-to-trough decline (negative number)
        - duration: number of periods from start to end (inclusive of end)
        - recovered: True if equity returned to peak, False if still underwater
    """
    n = len(equity)
    underwater = equity < running_max

    episodes: list[_DrawdownEpisode] = []
    i = 0
    while i < n:
        if underwater[i]:
            start = i
            # Find trough (minimum drawdown within this episode)
            trough_idx = start
            min_dd = dd[start]
            while i < n and underwater[i]:
                if dd[i] < min_dd:
                    min_dd = dd[i]
                    trough_idx = i
                i += 1
            # i now points to first recovered period (or n if ongoing)
            end = i - 1  # last underwater period
            recovered = i < n and not underwater[i]
            duration = end - start + 1
            episode: _DrawdownEpisode = {
                "start": int(start),
                "end": int(end),
                "trough_idx": int(trough_idx),
                "depth": float(min_dd),
                "duration": int(duration),
                "recovered": bool(recovered),
            }
            episodes.append(episode)
        else:
            i += 1

    return episodes


# ---------------------------------------------------------------------------
# Numba-accelerated drawdown episode detection.
# Attempts to JIT-compile _drawdown_episodes_numba_impl at module load.
# Falls back to pure-python _drawdown_episodes when numba is unavailable.
# ---------------------------------------------------------------------------

try:
    import numba  # noqa: F401
    from numba import njit
except ImportError:  # pragma: no cover
    _drawdown_episodes_numba = _drawdown_episodes
else:

    @njit(cache=False)
    def _drawdown_episodes_numba_impl(
        underwater_arr: NDArray[np.bool_],
        dd_arr: NDArray[np.floating],
        n: int,
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.floating],
        NDArray[np.bool_],
    ]:
        """Numba-compiled drawdown walker.

        Returns arrays of start, end, trough, depth, recovered for each episode.
        """
        max_eps = n // 2 + 1
        starts = np.zeros(max_eps, dtype=np.int64)
        ends = np.zeros(max_eps, dtype=np.int64)
        troughs = np.zeros(max_eps, dtype=np.int64)
        depths_arr = np.zeros(max_eps, dtype=np.float64)
        recovered = np.zeros(max_eps, dtype=np.bool_)

        i = 0
        count = 0
        while i < n:
            if underwater_arr[i]:
                start = i
                trough_idx = start
                min_dd = dd_arr[start]
                while i < n and underwater_arr[i]:
                    if dd_arr[i] < min_dd:
                        min_dd = dd_arr[i]
                        trough_idx = i
                    i += 1
                end = i - 1
                starts[count] = start
                ends[count] = end
                troughs[count] = trough_idx
                depths_arr[count] = min_dd
                recovered[count] = i < n and not underwater_arr[i]
                count += 1
            else:
                i += 1

        return starts, ends, troughs, depths_arr, recovered

    def _drawdown_episodes_numba(
        equity: NDArray[np.floating],
        running_max: NDArray[np.floating],
        dd: NDArray[np.floating],
    ) -> list[_DrawdownEpisode]:
        """Numba-accelerated episode detection, returning standard dict list."""
        n = len(equity)
        underwater = equity < running_max

        starts, ends, troughs, depths_arr, recovered = (
            _drawdown_episodes_numba_impl(
                underwater.astype(np.bool_),
                dd,
                n,
            )
        )

        episodes: list[_DrawdownEpisode] = []
        for k in range(len(starts)):
            if starts[k] == 0 and ends[k] == 0 and k > 0:
                break
            s = int(starts[k])
            e = int(ends[k])
            episode: _DrawdownEpisode = {
                "start": s,
                "end": e,
                "trough_idx": int(troughs[k]),
                "depth": float(depths_arr[k]),
                "duration": e - s + 1,
                "recovered": bool(recovered[k]),
            }
            if s == 0 and e == 0:
                break  # sentinel: empty slot
            episodes.append(episode)

        return episodes


def _analyse_drawdowns(
    r: NDArray[np.floating], return_type: str = "simple"
) -> tuple[
    NDArray[np.floating],  # equity (with prepended 1.0)
    NDArray[np.floating],  # running_max
    NDArray[np.floating],  # drawdown series
    list[list[_DrawdownEpisode]],  # episodes per strategy
]:
    """Full drawdown analysis for all strategy columns.

    Returns:
        (equity, running_max, drawdown_series, all_episodes)
        where all_episodes[col] is the list of episode dicts for strategy col.
    """
    equity = _equity_curve(r, return_type)
    running_max, dd = _drawdown_series(equity)

    all_episodes: list[list[_DrawdownEpisode]] = []
    for col in range(r.shape[1]):
        episodes = _drawdown_episodes_numba(
            equity[:, col], running_max[:, col], dd[:, col]
        )
        all_episodes.append(episodes)

    return equity, running_max, dd, all_episodes


# ---------------------------------------------------------------------------
# Helper: normal distribution functions (no scipy dependency)
# ---------------------------------------------------------------------------

_NORM_PDF_C = 1.0 / np.sqrt(2.0 * np.pi)

# Coefficients for the Acklam quantile (inverse CDF) approximation.
# Acklam, P.J. (2003), "An algorithm for computing the inverse normal
# cumulative distribution function."
_ACKLAM_A1 = -39.6968302866538
_ACKLAM_A2 = 220.9460984245205
_ACKLAM_A3 = -275.9285104469687
_ACKLAM_A4 = 138.3577518672690
_ACKLAM_A5 = -30.6647980661472
_ACKLAM_A6 = 2.50662827745924

_ACKLAM_B1 = -54.4760987982241
_ACKLAM_B2 = 161.5858368580409
_ACKLAM_B3 = -155.6989798598866
_ACKLAM_B4 = 66.8013118877197
_ACKLAM_B5 = -13.2806815528857

_ACKLAM_C1 = -7.78489400243029e-3
_ACKLAM_C2 = -0.322396458041136
_ACKLAM_C3 = -2.40075827716184
_ACKLAM_C4 = -2.54973253934373
_ACKLAM_C5 = 4.37466414146497
_ACKLAM_C6 = 2.93816398269878

_ACKLAM_D1 = 7.78469570904146e-3
_ACKLAM_D2 = 0.322467129070040
_ACKLAM_D3 = 2.44513413714300
_ACKLAM_D4 = 3.75440866190742

_ACKLAM_P_LOW = 0.02425
_ACKLAM_P_HIGH = 1.0 - _ACKLAM_P_LOW


def _norm_pdf(z: float) -> float:
    """Standard normal PDF."""
    return float(_NORM_PDF_C * float(np.exp(-0.5 * z * z)))


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf.

    Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    import math

    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal quantile (inverse CDF).

    Uses the Acklam (2003) rational approximation, accurate to ~1.15e-9
    in the tails. Falls back to a simple approximation for extreme tails.
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")

    # Central region: rational approximation
    if _ACKLAM_P_LOW <= p <= _ACKLAM_P_HIGH:
        q = p - 0.5
        r = q * q
        num = ((((_ACKLAM_A1 * r + _ACKLAM_A2) * r + _ACKLAM_A3) * r
                + _ACKLAM_A4) * r + _ACKLAM_A5) * r + _ACKLAM_A6
        den = ((((_ACKLAM_B1 * r + _ACKLAM_B2) * r + _ACKLAM_B3) * r
                 + _ACKLAM_B4) * r + _ACKLAM_B5) * r + 1.0
        return float(q * num / den)

    # Tail regions
    if p < _ACKLAM_P_LOW:
        # Left tail: p < 0.02425
        r = np.sqrt(-2.0 * np.log(p))
        num = (((_ACKLAM_C1 * r + _ACKLAM_C2) * r + _ACKLAM_C3) * r
               + _ACKLAM_C4) * r + _ACKLAM_C5
        den = (((_ACKLAM_D1 * r + _ACKLAM_D2) * r + _ACKLAM_D3) * r
               + _ACKLAM_D4) * r + 1.0
        return float(num / den)

    # Right tail: p > 0.97575
    r = np.sqrt(-2.0 * np.log(1.0 - p))
    num = (((_ACKLAM_C1 * r + _ACKLAM_C2) * r + _ACKLAM_C3) * r
           + _ACKLAM_C4) * r + _ACKLAM_C5
    den = (((_ACKLAM_D1 * r + _ACKLAM_D2) * r + _ACKLAM_D3) * r
           + _ACKLAM_D4) * r + 1.0
    return float(-num / den)


# ---------------------------------------------------------------------------
# Helper: Cornish-Fisher expansion for VaR
# ---------------------------------------------------------------------------


def _cornish_fisher_z(
    z_alpha: float, gamma1: float, gamma2: float
) -> float:
    """Cornish-Fisher expansion for a modified z-score.

    z_CF = z + gamma1/6*(z^2 - 1) + gamma2/24*(z^3 - 3z)
           - gamma1^2/36*(2z^3 - 5z)

    Args:
        z_alpha: Standard normal quantile at the desired confidence level.
        gamma1: Skewness.
        gamma2: Excess kurtosis.

    Returns:
        Cornish-Fisher adjusted z-score.
    """
    z = z_alpha
    term1 = gamma1 / 6.0 * (z * z - 1.0)
    term2 = gamma2 / 24.0 * (z * z * z - 3.0 * z)
    term3 = gamma1 * gamma1 / 36.0 * (2.0 * z * z * z - 5.0 * z)
    return z + term1 + term2 - term3


# ---------------------------------------------------------------------------
# 2.1 Max Drawdown
# Reference: Pospisil & Vecer (2011), J. Applied Probability, 48(3)
# ---------------------------------------------------------------------------

_MAX_DD_REF = (
    "Pospisil & Vecer (2011), "
    '"Maximum Drawdown of a Brownian Motion," '
    "Journal of Applied Probability, 48(3)"
)


@register_metric(
    name="max_drawdown",
    requires="returns",
    category=("risk", "returns"),
    backend="sequential",
    ref=_MAX_DD_REF,
)
def max_drawdown(
    input_data: ReturnsInput, return_type: str = "simple"
) -> MetricResult:
    """Maximum drawdown — largest peak-to-trough decline.

    Formula (simple returns, default):
        P_t = prod(1 + r_tau), tau=1..t
        MDD = max_t (P_t - max_{tau <= t} P_tau) / max_{tau <= t} P_tau

    Args:
        input_data: A ``ReturnsInput``.
        return_type: ``"simple"`` (default) or ``"log"``.

    Returns:
        MetricResult with max drawdown as a negative float (or array).
    """
    if return_type not in ("simple", "log"):
        raise ValueError(
            f"return_type must be 'simple' or 'log', got {return_type!r}"
        )

    r = input_data.values
    equity = _equity_curve(r, return_type)
    running_max, dd = _drawdown_series(equity)

    arr = np.nanmin(dd, axis=0)  # most negative drawdown

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="max_drawdown",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _MAX_DD_REF,
            "return_type": return_type,
        },
    )


# ---------------------------------------------------------------------------
# 2.2 Longest Drawdown Duration
# Reference: van Hemert et al. (2020, Tactical Asset Allocation, Ch. 5)
# ---------------------------------------------------------------------------

_LONGEST_DD_REF = "van Hemert et al. (2020, Tactical Asset Allocation, Ch. 5)"


@register_metric(
    name="longest_drawdown_duration",
    requires="returns",
    category=("risk", "returns"),
    backend="sequential",
    ref=_LONGEST_DD_REF,
)
def longest_drawdown_duration(
    input_data: ReturnsInput, units: str = "periods"
) -> MetricResult:
    """Longest contiguous underwater period.

    Args:
        input_data: A ``ReturnsInput``.
        units: ``"periods"`` (default) or ``"years"``. When ``"years"``,
            ``periods_per_year`` must be set on the input.

    Returns:
        MetricResult with the longest drawdown duration.
    """
    if units not in ("periods", "years"):
        raise ValueError(
            f"units must be 'periods' or 'years', got {units!r}"
        )

    r = input_data.values
    _, _, _, all_episodes = _analyse_drawdowns(r)

    arr: NDArray[np.floating] = np.zeros(r.shape[1], dtype=np.float64)
    for col in range(r.shape[1]):
        episodes = all_episodes[col]
        if episodes:
            arr[col] = float(max(ep["duration"] for ep in episodes))
        else:
            arr[col] = 0.0

    if units == "years":
        if input_data.periods_per_year is None:
            raise ValueError(
                "units='years' requires periods_per_year on the ReturnsInput"
            )
        arr = arr / float(input_data.periods_per_year)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="longest_drawdown_duration",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _LONGEST_DD_REF,
            "units": units,
        },
    )


# ---------------------------------------------------------------------------
# 2.3 Time to Recovery
# Reference: Bacon (2008, Practical Portfolio Perf. Measurement, 2nd ed., Sec. 7.2)
# ---------------------------------------------------------------------------

_TTR_REF = "Bacon (2008, Practical Portfolio Performance Measurement, 2nd ed., Sec. 7.2)"


@register_metric(
    name="time_to_recovery",
    requires="returns",
    category=("risk", "returns"),
    backend="sequential",
    ref=_TTR_REF,
)
def time_to_recovery(input_data: ReturnsInput) -> MetricResult:
    """Time to recovery for each drawdown episode.

    For each drawdown episode, the time in periods from the peak to the
    first subsequent period where equity returns to or exceeds the
    running-maximum level.

    Returns a dict with keys ``mean``, ``median``, and ``max`` across all
    recovered episodes. Episodes that have not yet recovered are excluded
    from the statistics.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult whose value is a dict with ``mean``, ``median``, ``max``
        recovery times in periods (float for single, arrays for multi-strategy).
    """
    r = input_data.values
    _, _, _, all_episodes = _analyse_drawdowns(r)

    means = np.zeros(r.shape[1], dtype=np.float64)
    medians = np.zeros(r.shape[1], dtype=np.float64)
    maxes = np.zeros(r.shape[1], dtype=np.float64)

    for col in range(r.shape[1]):
        durations = [
            ep["duration"]
            for ep in all_episodes[col]
            if ep["recovered"]
        ]
        if durations:
            means[col] = float(np.mean(durations))
            medians[col] = float(np.median(durations))
            maxes[col] = float(np.max(durations))
        else:
            means[col] = np.nan
            medians[col] = np.nan
            maxes[col] = np.nan

    # Stack as (3, n_strategies): row 0=mean, 1=median, 2=max
    arr: NDArray[np.floating] = np.stack([means, medians, maxes], axis=0)

    if input_data.is_single:
        value: NDArray[np.floating] | float = arr.squeeze(axis=1)  # shape (3,)
    else:
        value = arr

    return MetricResult(
        name="time_to_recovery",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _TTR_REF,
            "output_index": ["mean", "median", "max"],
        },
    )


# ---------------------------------------------------------------------------
# 2.4 Average Drawdown
# Reference: Bacon (2008, Sec. 7.2)
# ---------------------------------------------------------------------------

_AVG_DD_REF = "Bacon (2008, Practical Portfolio Performance Measurement, 2nd ed., Sec. 7.2)"


@register_metric(
    name="average_drawdown",
    requires="returns",
    category=("risk", "returns"),
    backend="sequential",
    ref=_AVG_DD_REF,
)
def average_drawdown(input_data: ReturnsInput) -> MetricResult:
    """Average drawdown — mean peak-to-trough depth across all episodes.

    Formula:
        ADD = (1/K) * sum(depth_k)

    where depth_k is the peak-to-trough decline of the k-th episode
    (a negative percentage).

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with average drawdown depth (negative float or array).
    """
    r = input_data.values
    _, _, _, all_episodes = _analyse_drawdowns(r)

    arr = np.zeros(r.shape[1], dtype=np.float64)
    for col in range(r.shape[1]):
        episodes = all_episodes[col]
        if episodes:
            depths = [ep["depth"] for ep in episodes]
            arr[col] = float(np.mean(depths))
        else:
            arr[col] = 0.0  # no drawdowns = no pain

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="average_drawdown",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _AVG_DD_REF},
    )


# ---------------------------------------------------------------------------
# 2.5 Average Drawdown Duration
# Reference: Bacon (2008, Sec. 7.2)
# ---------------------------------------------------------------------------

_AVG_DD_DUR_REF = "Bacon (2008, Practical Portfolio Performance Measurement, 2nd ed., Sec. 7.2)"


@register_metric(
    name="average_drawdown_duration",
    requires="returns",
    category=("risk", "returns"),
    backend="sequential",
    ref=_AVG_DD_DUR_REF,
)
def average_drawdown_duration(input_data: ReturnsInput) -> MetricResult:
    """Average drawdown duration — mean duration across all episodes.

    Formula:
        T_bar_DD = (1/K) * sum(T_DD^(k))

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with average drawdown duration in periods.
    """
    r = input_data.values
    _, _, _, all_episodes = _analyse_drawdowns(r)

    arr = np.zeros(r.shape[1], dtype=np.float64)
    for col in range(r.shape[1]):
        episodes = all_episodes[col]
        if episodes:
            durations = [ep["duration"] for ep in episodes]
            arr[col] = float(np.mean(durations))
        else:
            arr[col] = 0.0

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="average_drawdown_duration",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _AVG_DD_DUR_REF},
    )


# ---------------------------------------------------------------------------
# 2.6 Ulcer Index
# Reference: Martin & McCann (1989); Martin (1993)
# ---------------------------------------------------------------------------

_ULCER_REF = (
    "Martin & McCann (1989, The Investor's Guide to Fidelity Funds); Martin (1993)"
)


@register_metric(
    name="ulcer_index",
    requires="returns",
    category=("risk", "returns"),
    backend="sequential",
    ref=_ULCER_REF,
)
def ulcer_index(input_data: ReturnsInput) -> MetricResult:
    """Ulcer Index — root-mean-square of percentage drawdowns.

    Formula:
        UI = sqrt((1/n) * sum(d_t^2))

    where d_t = (P_t - max_{tau <= t} P_tau) / max_{tau <= t} P_tau
    is the percentage drawdown at each period.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with Ulcer Index (non-negative float or array).
    """
    r = input_data.values
    equity = _equity_curve(r, "simple")
    _, dd = _drawdown_series(equity)

    # RMS of drawdowns (dd values are ≤ 0, so squared is fine)
    arr = np.sqrt(np.nanmean(dd**2, axis=0))

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="ulcer_index",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _ULCER_REF},
    )


# ---------------------------------------------------------------------------
# 2.7 Downside Deviation
# Reference: Sortino & van der Meer (1991); Sortino & Price (1994)
# ---------------------------------------------------------------------------

_DOWNSIDE_DEV_REF = "Sortino & van der Meer (1991); Sortino & Price (1994)"


@register_metric(
    name="downside_deviation",
    requires="returns",
    category=("risk", "returns"),
    backend="vectorized",
    ref=_DOWNSIDE_DEV_REF,
)
def downside_deviation(
    input_data: ReturnsInput, mar: float = 0.0
) -> MetricResult:
    """Downside deviation (semi-deviation below a minimum acceptable return).

    Formula:
        DD = sqrt((1/n) * sum(min(r_t - mar, 0)^2))

    With mar = 0, this is the standard semi-deviation.

    Args:
        input_data: A ``ReturnsInput``.
        mar: Minimum acceptable return (default 0.0).

    Returns:
        MetricResult with downside deviation.
    """
    r = input_data.values
    below = np.minimum(r - mar, 0.0)  # only returns below MAR
    arr = np.sqrt(np.nanmean(below**2, axis=0))

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="downside_deviation",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _DOWNSIDE_DEV_REF, "mar": mar},
    )


# ---------------------------------------------------------------------------
# 2.8 Upside Deviation
# Reference: Bacon (2008, Sec. 6.4)
# ---------------------------------------------------------------------------

_UPSIDE_DEV_REF = "Bacon (2008, Practical Portfolio Performance Measurement, 2nd ed., Sec. 6.4)"


@register_metric(
    name="upside_deviation",
    requires="returns",
    category=("risk", "returns"),
    backend="vectorized",
    ref=_UPSIDE_DEV_REF,
)
def upside_deviation(
    input_data: ReturnsInput, mar: float = 0.0
) -> MetricResult:
    """Upside deviation — mirror of downside deviation for positive dispersion.

    Formula:
        UD = sqrt((1/n) * sum(max(r_t - mar, 0)^2))

    Args:
        input_data: A ``ReturnsInput``.
        mar: Minimum acceptable return (default 0.0).

    Returns:
        MetricResult with upside deviation.
    """
    r = input_data.values
    above = np.maximum(r - mar, 0.0)
    arr = np.sqrt(np.nanmean(above**2, axis=0))

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="upside_deviation",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _UPSIDE_DEV_REF, "mar": mar},
    )


# ---------------------------------------------------------------------------
# 2.9 VaR (Value at Risk)
# Reference: Litterman (1996); Zangari (1996); Jorion (2006)
# ---------------------------------------------------------------------------

_VAR_REF = (
    "Litterman (1996); Zangari (1996); "
    "Jorion (2006, Value at Risk, 3rd ed., Sec. 5.2)"
)


def _var_historical(
    r: NDArray[np.floating], confidence: float
) -> NDArray[np.floating]:
    """Historical VaR: negative of the alpha-quantile of returns."""
    alpha = 1.0 - confidence
    # Along axis 0: for each strategy column
    raw = -np.nanpercentile(r, alpha * 100.0, axis=0)
    return np.asarray(raw, dtype=np.float64)


def _var_parametric(
    r: NDArray[np.floating], confidence: float
) -> NDArray[np.floating]:
    """Parametric VaR: -(mean + z_alpha * std)."""
    alpha = 1.0 - confidence
    z_alpha = _norm_ppf(alpha)
    mean = np.nanmean(r, axis=0)
    std = np.nanstd(r, axis=0, ddof=1)
    return np.asarray(-(mean + z_alpha * std), dtype=np.float64)


def _var_cornish_fisher(
    r: NDArray[np.floating], confidence: float
) -> NDArray[np.floating]:
    """Cornish-Fisher VaR: parametric with CF-adjusted z-score."""
    alpha = 1.0 - confidence
    z_alpha = _norm_ppf(alpha)

    # Compute per-column skewness and excess kurtosis
    n_strat = r.shape[1]
    var_arr = np.zeros(n_strat, dtype=np.float64)
    for col in range(n_strat):
        col_data = r[:, col]
        valid = ~np.isnan(col_data)
        valid_data = col_data[valid]
        n_eff = len(valid_data)
        if n_eff < 4 or np.std(valid_data, ddof=1) < 1e-15:
            var_arr[col] = np.nan
            continue

        col_mean = np.mean(valid_data)
        col_std = np.std(valid_data, ddof=1)
        z_vals = (valid_data - col_mean) / col_std

        # Skewness (bias-corrected)
        m3 = np.sum(z_vals**3)
        gamma1 = (n_eff / ((n_eff - 1) * (n_eff - 2))) * m3 if n_eff >= 3 else 0.0

        # Excess kurtosis (bias-corrected)
        m4 = np.sum(z_vals**4)
        term_a = n_eff * (n_eff + 1) / ((n_eff - 1) * (n_eff - 2) * (n_eff - 3))
        term_b = 3 * (n_eff - 1) ** 2 / ((n_eff - 2) * (n_eff - 3))
        gamma2 = term_a * m4 - term_b if n_eff >= 4 else 0.0

        z_cf = _cornish_fisher_z(z_alpha, gamma1, gamma2)
        var_arr[col] = -(col_mean + z_cf * col_std)

    return var_arr


@register_metric(
    name="var",
    requires="returns",
    category=("risk", "returns"),
    backend="vectorized",
    ref=_VAR_REF,
)
def var(
    input_data: ReturnsInput,
    method: str = "historical",
    confidence: float = 0.95,
) -> MetricResult:
    """Value at Risk — loss threshold at a given confidence level.

    Three estimation methods:

    - ``"historical"`` (default): empirical percentile of the return distribution.
    - ``"parametric"``: assumes normal distribution, VaR = -(mean + z_alpha * std).
    - ``"cornish_fisher"``: parametric with Cornish-Fisher expansion adjusting
      for skewness and excess kurtosis.

    Args:
        input_data: A ``ReturnsInput``.
        method: Estimation method (``"historical"``, ``"parametric"``,
            or ``"cornish_fisher"``).
        confidence: Confidence level, default 0.95 (95 % VaR).

    Returns:
        MetricResult with VaR (positive float = loss magnitude).
    """
    if method not in ("historical", "parametric", "cornish_fisher"):
        raise ValueError(
            f"method must be 'historical', 'parametric', or 'cornish_fisher', "
            f"got {method!r}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence must be in (0, 1), got {confidence}"
        )

    r = input_data.values

    if method == "historical":
        arr = _var_historical(r, confidence)
    elif method == "parametric":
        arr = _var_parametric(r, confidence)
    else:  # cornish_fisher
        arr = _var_cornish_fisher(r, confidence)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="var",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _VAR_REF,
            "method": method,
            "confidence": confidence,
        },
    )


# ---------------------------------------------------------------------------
# 2.10 CVaR / Expected Shortfall
# Reference: Rockafellar & Uryasev (2000); Acerbi & Tasche (2002)
# ---------------------------------------------------------------------------

_CVAR_REF = (
    "Rockafellar & Uryasev (2000), Journal of Risk, 2(3); Acerbi & Tasche (2002)"
)


def _cvar_historical(
    r: NDArray[np.floating], confidence: float
) -> NDArray[np.floating]:
    """Historical CVaR: mean of returns below the VaR threshold."""
    alpha = 1.0 - confidence
    n_strat = r.shape[1]
    cvar_arr = np.zeros(n_strat, dtype=np.float64)
    for col in range(n_strat):
        col_data = r[:, col]
        valid_data = col_data[~np.isnan(col_data)]
        if len(valid_data) == 0:
            cvar_arr[col] = np.nan
            continue
        threshold = -np.percentile(valid_data, alpha * 100.0)
        tail = valid_data[valid_data <= -threshold]
        if len(tail) > 0:
            cvar_arr[col] = -float(np.mean(tail))
        else:
            cvar_arr[col] = -float(np.min(valid_data))
    return cvar_arr


def _cvar_parametric(
    r: NDArray[np.floating], confidence: float
) -> NDArray[np.floating]:
    """Parametric CVaR (ES) under normality: -(mean - std * phi(z)/alpha)."""
    alpha = 1.0 - confidence
    z_alpha = _norm_ppf(alpha)
    phi_z = _norm_pdf(z_alpha)

    mean = np.nanmean(r, axis=0)
    std = np.nanstd(r, axis=0, ddof=1)

    return np.asarray(-(mean - std * phi_z / alpha), dtype=np.float64)


@register_metric(
    name="cvar",
    requires="returns",
    category=("risk", "returns"),
    backend="vectorized",
    ref=_CVAR_REF,
)
def cvar(
    input_data: ReturnsInput,
    method: str = "historical",
    confidence: float = 0.95,
) -> MetricResult:
    """Conditional Value at Risk (Expected Shortfall).

    The expected loss given that the loss exceeds VaR.

    - ``"historical"`` (default): mean of returns below the empirical VaR threshold.
    - ``"parametric"``: assumes normality, ES = -(mean - std * phi(z_alpha) / alpha).

    Args:
        input_data: A ``ReturnsInput``.
        method: ``"historical"`` or ``"parametric"``.
        confidence: Confidence level, default 0.95.

    Returns:
        MetricResult with CVaR (positive float = expected loss beyond VaR).
    """
    if method not in ("historical", "parametric"):
        raise ValueError(
            f"method must be 'historical' or 'parametric', got {method!r}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    r = input_data.values

    if method == "historical":
        arr = _cvar_historical(r, confidence)
    else:
        arr = _cvar_parametric(r, confidence)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="cvar",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _CVAR_REF,
            "method": method,
            "confidence": confidence,
        },
    )


# ---------------------------------------------------------------------------
# 2.11 Tail Ratio
# Reference: Connor, Goldberg & Korajczyk (2010, Portfolio Risk Analysis, Ch. 9)
# ---------------------------------------------------------------------------

_TAIL_RATIO_REF = (
    "Connor, Goldberg & Korajczyk "
    "(2010, Portfolio Risk Analysis, Ch. 9)"
)


@register_metric(
    name="tail_ratio",
    requires="returns",
    category=("risk", "returns"),
    backend="vectorized",
    ref=_TAIL_RATIO_REF,
)
def tail_ratio(
    input_data: ReturnsInput, tail_cutoff: float = 0.05
) -> MetricResult:
    """Tail ratio — upper-tail mean divided by absolute lower-tail mean.

    Formula:
        TR = E[r | r >= q_{1-alpha}] / |E[r | r <= q_alpha]|

    where alpha = tail_cutoff.

    Args:
        input_data: A ``ReturnsInput``.
        tail_cutoff: Tail fraction, default 0.05 (5 % each tail).

    Returns:
        MetricResult with tail ratio (non-negative, higher = fatter right tail).
    """
    if not 0.0 < tail_cutoff < 0.5:
        raise ValueError(
            f"tail_cutoff must be in (0, 0.5), got {tail_cutoff}"
        )

    r = input_data.values
    n_strat = r.shape[1]
    tr_arr = np.zeros(n_strat, dtype=np.float64)

    for col in range(n_strat):
        col_data = r[:, col]
        valid = col_data[~np.isnan(col_data)]
        if len(valid) < 2:
            tr_arr[col] = np.nan
            continue

        lower_q = np.percentile(valid, tail_cutoff * 100.0)
        upper_q = np.percentile(valid, (1.0 - tail_cutoff) * 100.0)

        lower_tail = valid[valid <= lower_q]
        upper_tail = valid[valid >= upper_q]

        if len(lower_tail) == 0 or len(upper_tail) == 0:
            tr_arr[col] = np.nan
            continue

        upper_mean = float(np.mean(upper_tail))
        lower_mean = float(np.mean(lower_tail))

        if np.abs(lower_mean) < 1e-30:
            tr_arr[col] = np.nan
        else:
            tr_arr[col] = upper_mean / np.abs(lower_mean)

    value: float | NDArray[np.floating]
    value = float(tr_arr[0]) if input_data.is_single else tr_arr

    return MetricResult(
        name="tail_ratio",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _TAIL_RATIO_REF,
            "tail_cutoff": tail_cutoff,
        },
    )


# ---------------------------------------------------------------------------
# 2.12 Common-Sense Ratio
# Reference: Bacon (2008, Sec. 7.5)
# ---------------------------------------------------------------------------

_CSR_REF = "Bacon (2008, Practical Portfolio Performance Measurement, 2nd ed., Sec. 7.5)"


@register_metric(
    name="common_sense_ratio",
    requires="returns",
    category=("risk", "returns"),
    backend="vectorized",
    ref=_CSR_REF,
)
def common_sense_ratio(
    input_data: ReturnsInput, tail_cutoff: float = 0.05
) -> MetricResult:
    """Common-Sense Ratio — tail ratio multiplied by gain-to-pain ratio.

    Formula:
        CSR = TR_alpha * (sum(gains) / |sum(losses)|)

    where TR_alpha is the tail ratio at cutoff alpha.

    Args:
        input_data: A ``ReturnsInput``.
        tail_cutoff: Tail fraction passed to tail ratio, default 0.05.

    Returns:
        MetricResult with common-sense ratio. Returns ``inf`` when there
        are gains but no losses (infinite gain-to-pain), and ``NaN`` when
        both gains and losses are absent or the tail ratio is undefined.
    """
    r = input_data.values
    n_strat = r.shape[1]
    csr_arr = np.zeros(n_strat, dtype=np.float64)

    for col in range(n_strat):
        col_data = r[:, col]
        valid = col_data[~np.isnan(col_data)]
        if len(valid) < 2:
            csr_arr[col] = np.nan
            continue

        # Tail ratio for this column
        lower_q = np.percentile(valid, tail_cutoff * 100.0)
        upper_q = np.percentile(valid, (1.0 - tail_cutoff) * 100.0)
        lower_tail = valid[valid <= lower_q]
        upper_tail = valid[valid >= upper_q]

        if len(lower_tail) == 0 or len(upper_tail) == 0:
            csr_arr[col] = np.nan
            continue

        tr = np.mean(upper_tail) / np.abs(np.mean(lower_tail))

        # Gain-to-pain ratio
        gains = valid[valid > 0]
        losses = valid[valid < 0]
        if len(losses) == 0 or np.abs(np.sum(losses)) < 1e-30:
            gpr = np.inf if len(gains) > 0 else np.nan
        else:
            gpr = np.sum(gains) / np.abs(np.sum(losses))

        csr_arr[col] = tr * gpr

    value: float | NDArray[np.floating]
    value = float(csr_arr[0]) if input_data.is_single else csr_arr

    return MetricResult(
        name="common_sense_ratio",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _CSR_REF,
            "tail_cutoff": tail_cutoff,
        },
    )


# ---------------------------------------------------------------------------
# 2.13 Hill Tail Index (EVT)
# Reference: Hill (1975), Annals of Statistics, 3(5)
# ---------------------------------------------------------------------------

_HILL_REF = (
    'Hill (1975), "A Simple General Approach to Inference About the'
    ' Tail of a Distribution," Annals of Statistics, 3(5)'
)


@register_metric(
    name="hill_tail_index",
    requires="returns",
    category=("risk", "returns"),
    backend="vectorized",
    ref=_HILL_REF,
)
def hill_tail_index(
    input_data: ReturnsInput, tail_fraction: float = 0.10
) -> MetricResult:
    """Hill estimator for the tail index of the return distribution.

    Fits the upper tail (largest positive returns) of the distribution.
    For analysing loss severity, pass negative returns or use
    ``hill_tail_index`` on -returns.

    Formula:
        xi_hat = (1/k) * sum_{i=1}^{k} ln(X_{(i)} / X_{(k+1)})

    where X_{(i)} are descending order statistics and k = tail_fraction * n.

    Args:
        input_data: A ``ReturnsInput``.
        tail_fraction: Fraction of observations in the tail, default 0.10.

    Returns:
        MetricResult with the Hill tail index estimate.
        Values > 0 indicate a heavy tail (Pareto-like), with smaller values
        indicating heavier tails. Returns NaN when the tail threshold crosses
        into non-positive territory (no positive returns in the chosen tail
        fraction) — use -returns to analyse the left (loss) tail instead.

        .. note::

           The estimator analyses the upper (right) tail of returns. For loss
           severity analysis, pass ``-returns`` as input. When NaN is returned
           because the tail threshold is <= 0, the ``meta`` field includes a
           ``note`` key explaining the cause.
    """
    if not 0.0 < tail_fraction <= 0.5:
        raise ValueError(
            f"tail_fraction must be in (0, 0.5], got {tail_fraction}"
        )

    r = input_data.values
    n_strat = r.shape[1]
    hill_arr = np.zeros(n_strat, dtype=np.float64)

    for col in range(n_strat):
        col_data = r[:, col]
        valid = col_data[~np.isnan(col_data)]
        n_valid = len(valid)
        k = max(int(np.floor(tail_fraction * n_valid)), 2)

        if n_valid < k + 1:
            hill_arr[col] = np.nan
            continue

        # Sort descending (largest first) for upper-tail analysis
        sorted_data = np.sort(valid)[::-1]
        x_kp1 = sorted_data[k]  # threshold
        if x_kp1 <= 0:
            # Tail threshold is non-positive — the right-tail Hill estimator
            # requires positive tail values. For loss-severity analysis,
            # pass -returns as the input data.
            hill_arr[col] = np.nan
            continue

        tail_vals = sorted_data[:k]
        hill_arr[col] = float(np.mean(np.log(tail_vals / x_kp1)))

    value: float | NDArray[np.floating]
    value = float(hill_arr[0]) if input_data.is_single else hill_arr

    hill_meta: dict[str, object] = {
        "ref": _HILL_REF,
        "tail_fraction": tail_fraction,
    }
    if np.any(np.isnan(hill_arr)):
        n_nan = int(np.sum(np.isnan(hill_arr)))
        hill_meta["note"] = (
            f"{n_nan} strategy column(s) returned NaN — "
            "tail threshold may be non-positive. "
            "For loss-severity analysis, pass -returns."
        )

    return MetricResult(
        name="hill_tail_index",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta=hill_meta,
    )


# ---------------------------------------------------------------------------
# 2.14 GPD Tail Fit
# Reference: Pickands (1975); Hosking & Wallis (1987); Embrechts et al. (1997)
# ---------------------------------------------------------------------------

_GPD_REF = (
    "Pickands (1975); Hosking & Wallis (1987); "
    "Embrechts, Kluppelberg & Mikosch (1997, Modelling Extremal Events, Springer)"
)


@register_metric(
    name="gpd_tail_fit",
    requires="returns",
    category=("risk", "returns"),
    backend="vectorized",
    ref=_GPD_REF,
)
def gpd_tail_fit(input_data: ReturnsInput) -> MetricResult:
    """Generalized Pareto Distribution fit to the tail of negative returns.

    Fits the GPD to exceedances above the 90th percentile of negative returns
    (i.e., the worst 10 % of returns). Returns the estimated shape (xi) and
    scale (beta) parameters.

    Uses the method of moments (Hosking & Wallis 1987 PWM estimator).

    Requires at least 50 observations for a meaningful fit (at least 5
    exceedances above the 90th-percentile threshold). Returns NaN shape
    and scale when data is insufficient.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult whose value is a dict with ``shape`` (xi) and ``scale``
        (beta) for each strategy.
    """
    r = input_data.values
    n_strat = r.shape[1]

    xi_arr = np.zeros(n_strat, dtype=np.float64)
    beta_arr = np.zeros(n_strat, dtype=np.float64)

    for col in range(n_strat):
        col_data = r[:, col]
        valid = col_data[~np.isnan(col_data)]
        if len(valid) < 20:
            xi_arr[col] = np.nan
            beta_arr[col] = np.nan
            continue

        # Work with negative returns (losses as positive)
        losses = -valid
        threshold = np.percentile(losses, 90.0)
        exceedances = losses[losses > threshold] - threshold

        n_exc = len(exceedances)
        if n_exc < 5:
            xi_arr[col] = np.nan
            beta_arr[col] = np.nan
            continue

        # Method of moments estimator (Hosking & Wallis 1987):
        # Let w = (1/n) * sum((1 - j/(n+1)) * X_(j))
        # More simply: xi ≈ 0.5 * (1 - mean^2 / var_sample)
        # where var_sample is biased variance of exceedances.
        exc_mean = np.mean(exceedances)
        exc_var = np.var(exceedances, ddof=0)  # biased

        if exc_var < 1e-30:
            xi_arr[col] = np.nan
            beta_arr[col] = exc_mean
        else:
            xi_est = 0.5 * (1.0 - exc_mean**2 / exc_var)
            beta_est = 0.5 * exc_mean * (exc_mean**2 / exc_var + 1.0)
            xi_arr[col] = float(xi_est)
            beta_arr[col] = float(max(beta_est, 0.0))  # scale must be > 0

    # Stack as (2, n_strategies): row 0=shape, 1=scale
    arr: NDArray[np.floating] = np.stack([xi_arr, beta_arr], axis=0)

    if input_data.is_single:
        value: NDArray[np.floating] | float = arr.squeeze(axis=1)  # shape (2,)
    else:
        value = arr

    return MetricResult(
        name="gpd_tail_fit",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _GPD_REF,
            "method": "PWM (Hosking & Wallis 1987)",
            "output_index": ["shape", "scale"],
        },
    )


# ---------------------------------------------------------------------------
# 2.15 Risk of Ruin
# Reference: Vince (1990); standard risk-of-ruin literature
# ---------------------------------------------------------------------------

_ROR_REF = "Vince (1990, Portfolio Management Formulas)"


@register_metric(
    name="risk_of_ruin",
    requires="returns",
    category=("risk", "returns"),
    backend="vectorized",
    ref=_ROR_REF,
)
def risk_of_ruin(input_data: ReturnsInput) -> MetricResult:
    """Risk of ruin — normal-approximation probability of 100 % loss.

    Formula:
        P_ruin = Phi(-mean * T / (sigma * sqrt(T)))

    where T = n_periods / periods_per_year (horizon in years), and Phi
    is the standard normal CDF.

    **Warning:** This assumes normally distributed returns and is unreliable
    for fat-tailed (leptokurtic) distributions. The estimate should be
    treated as a lower bound for such distributions.

    Args:
        input_data: A ``ReturnsInput`` with ``periods_per_year`` set.

    Returns:
        MetricResult with ruin probability (0 to 1).

    Raises:
        ValueError: If ``periods_per_year`` is None.
    """
    if input_data.periods_per_year is None:
        raise ValueError(
            "risk_of_ruin requires periods_per_year to be set on the ReturnsInput"
        )

    r = input_data.values
    n = input_data.n_periods
    p = float(input_data.periods_per_year)
    t_years = n / p

    mean = np.nanmean(r, axis=0)
    std = np.nanstd(r, axis=0, ddof=1)

    with np.errstate(invalid="ignore"):
        z = -mean * t_years / (std * np.sqrt(t_years))
        arr = np.array([_norm_cdf(float(zi)) for zi in z], dtype=np.float64)
        arr = np.where(std < 1e-15, np.nan, arr)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="risk_of_ruin",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={
            "ref": _ROR_REF,
            "warning": (
                "Assumes normality — unreliable for fat-tailed returns. "
                "Treat as a lower bound for leptokurtic distributions."
            ),
        },
    )


# ---------------------------------------------------------------------------
# 2.16 Drawdown Volatility
# Reference: Bacon (2008, Sec. 7.3)
# ---------------------------------------------------------------------------

_DD_VOL_REF = "Bacon (2008, Practical Portfolio Performance Measurement, 2nd ed., Sec. 7.3)"


@register_metric(
    name="drawdown_volatility",
    requires="returns",
    category=("risk", "returns"),
    backend="sequential",
    ref=_DD_VOL_REF,
)
def drawdown_volatility(input_data: ReturnsInput) -> MetricResult:
    """Standard deviation of the drawdown time series.

    Formula:
        sigma_DD = std(d_1, ..., d_n)

    where d_t = (P_t - max_{tau <= t} P_tau) / max_{tau <= t} P_tau.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with drawdown volatility (non-negative).
    """
    r = input_data.values
    equity = _equity_curve(r, "simple")
    _, dd = _drawdown_series(equity)

    arr = np.nanstd(dd, axis=0, ddof=1)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="drawdown_volatility",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _DD_VOL_REF},
    )


# ---------------------------------------------------------------------------
# 2.17 Drawdown Periods Count
# Reference: Bacon (2008, Sec. 7.2)
# ---------------------------------------------------------------------------

_DD_COUNT_REF = "Bacon (2008, Practical Portfolio Performance Measurement, 2nd ed., Sec. 7.2)"


@register_metric(
    name="drawdown_periods_count",
    requires="returns",
    category=("risk", "returns"),
    backend="sequential",
    ref=_DD_COUNT_REF,
)
def drawdown_periods_count(input_data: ReturnsInput) -> MetricResult:
    """Number of distinct drawdown episodes.

    An episode begins when the equity curve falls below its running maximum
    and ends when it next returns to the running maximum.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with count of drawdown episodes (int for single,
        array of ints for multi-strategy).
    """
    r = input_data.values
    _, _, _, all_episodes = _analyse_drawdowns(r)

    arr = np.array(
        [len(episodes) for episodes in all_episodes], dtype=np.float64
    )

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="drawdown_periods_count",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _DD_COUNT_REF},
    )


# ---------------------------------------------------------------------------
# 2.18 Current Drawdown
# Reference: Bacon (2008, Sec. 7.2)
# ---------------------------------------------------------------------------

_CURR_DD_REF = "Bacon (2008, Practical Portfolio Performance Measurement, 2nd ed., Sec. 7.2)"


@register_metric(
    name="current_drawdown",
    requires="returns",
    category=("risk", "returns"),
    backend="sequential",
    ref=_CURR_DD_REF,
)
def current_drawdown(input_data: ReturnsInput) -> MetricResult:
    """Drawdown at the most recent observation.

    Formula:
        d_current = (P_n - max_{tau <= n} P_tau) / max_{tau <= n} P_tau

    Returns 0.0 if the equity curve is at a new high.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with current drawdown (≤ 0).
    """
    r = input_data.values
    equity = _equity_curve(r, "simple")
    _, dd = _drawdown_series(equity)

    arr = dd[-1, :]  # last period's drawdown

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="current_drawdown",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _CURR_DD_REF},
    )


# ---------------------------------------------------------------------------
# 2.19 Current Drawdown Duration
# Reference: Bacon (2008, Sec. 7.2)
# ---------------------------------------------------------------------------

_CURR_DD_DUR_REF = "Bacon (2008, Practical Portfolio Performance Measurement, 2nd ed., Sec. 7.2)"


@register_metric(
    name="current_drawdown_duration",
    requires="returns",
    category=("risk", "returns"),
    backend="sequential",
    ref=_CURR_DD_DUR_REF,
)
def current_drawdown_duration(input_data: ReturnsInput) -> MetricResult:
    """Periods elapsed from the most recent running-maximum peak.

    If the equity curve is at a new high (current drawdown = 0), the duration
    is 0.

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with current drawdown duration in periods.
    """
    r = input_data.values
    equity = _equity_curve(r, "simple")
    running_max, dd = _drawdown_series(equity)

    n_strat = r.shape[1]
    arr = np.zeros(n_strat, dtype=np.float64)
    for col in range(n_strat):
        # Find the last time equity was at running max
        at_peak = equity[:, col] >= running_max[:, col] - 1e-15
        if at_peak[-1]:
            arr[col] = 0.0  # currently at peak
        else:
            # Last time we were at peak (search backwards)
            last_peak = int(np.max(np.where(at_peak)[0]))
            arr[col] = float(len(equity) - 1 - last_peak)

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="current_drawdown_duration",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _CURR_DD_DUR_REF},
    )


# ---------------------------------------------------------------------------
# 2.20 Drawdown Total Duration
# Reference: Bacon (2008, Sec. 7.2)
# ---------------------------------------------------------------------------

_DD_TOT_REF = "Bacon (2008, Practical Portfolio Performance Measurement, 2nd ed., Sec. 7.2)"


@register_metric(
    name="drawdown_total_duration",
    requires="returns",
    category=("risk", "returns"),
    backend="sequential",
    ref=_DD_TOT_REF,
)
def drawdown_total_duration(input_data: ReturnsInput) -> MetricResult:
    """Sum of all underwater-period lengths (in periods).

    Formula:
        T_DD_total = sum_{k=1}^{K} T_DD^(k)

    Args:
        input_data: A ``ReturnsInput``.

    Returns:
        MetricResult with total drawdown duration in periods.
    """
    r = input_data.values
    _, _, _, all_episodes = _analyse_drawdowns(r)

    arr = np.zeros(r.shape[1], dtype=np.float64)
    for col in range(r.shape[1]):
        episodes = all_episodes[col]
        if episodes:
            arr[col] = float(sum(ep["duration"] for ep in episodes))
        else:
            arr[col] = 0.0

    value: float | NDArray[np.floating]
    value = float(arr[0]) if input_data.is_single else arr

    return MetricResult(
        name="drawdown_total_duration",
        value=value,
        category=("risk", "returns"),
        periods_per_year=input_data.periods_per_year,
        meta={"ref": _DD_TOT_REF},
    )
