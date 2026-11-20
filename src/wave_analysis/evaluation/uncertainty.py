"""Uncertainty quantification: dependence-aware bootstrap CIs and probabilistic scores.

**Why a block bootstrap?** Consecutive samples from one camera or buoy are
autocorrelated, so the effective sample size is far smaller than the row
count. An i.i.d. bootstrap then produces confidence intervals that are too
narrow. :func:`block_bootstrap_ci` resamples *whole groups* (e.g. days or
events), or contiguous blocks when no grouping is given (moving-block bootstrap,
Künsch 1989), which keeps the dependence structure within each resample.

References
----------
Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary
observations. *Ann. Statist.*, 17(3), 1217-1241. https://doi.org/10.1214/aos/1176347265

Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction,
and estimation. *JASA*, 102(477), 359-378. https://doi.org/10.1198/016214506000001437

Baran, S. & Lerch, S. (2015). Log-normal distribution based ensemble model output
statistics models for probabilistic wind-speed forecasting. *QJRMS*, 141(691),
2289-2299. https://doi.org/10.1002/qj.2521 (closed-form CRPS of a log-normal)

Lee, Y., Barber, R. F. & Willett, R. (2023). Distribution-free inference with
hierarchical data. arXiv:2306.06342 (conformal prediction when groups, here buoys,
are the exchangeable units)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

FloatArray = NDArray[np.float64]
Metric = Callable[[FloatArray, FloatArray], float]


def block_bootstrap_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric: Metric,
    *,
    groups: ArrayLike | None = None,
    block_length: int = 24,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile ``(1-alpha)`` CI of ``metric`` under a block bootstrap.

    Parameters
    ----------
    groups
        Optional group labels (e.g. day or event id). If given, whole groups are
        resampled with replacement (cluster bootstrap). Otherwise contiguous
        blocks of ``block_length`` samples (in input order - sort by time first)
        are resampled (moving-block bootstrap).
    """
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    point = float(metric(a, b))
    rng = np.random.default_rng(seed)
    n = a.size
    reps = np.empty(n_boot)
    if groups is not None:
        g = np.asarray(groups)
        uniq, inv = np.unique(g, return_inverse=True)
        members = [np.flatnonzero(inv == k) for k in range(uniq.size)]
        for i in range(n_boot):
            pick = rng.integers(0, uniq.size, uniq.size)
            idx = np.concatenate([members[k] for k in pick])
            reps[i] = metric(a[idx], b[idx])
    else:
        L = max(1, min(block_length, n))
        n_blocks = int(np.ceil(n / L))
        starts_max = n - L + 1
        for i in range(n_boot):
            starts = rng.integers(0, starts_max, n_blocks)
            idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
            reps[i] = metric(a[idx], b[idx])
    lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_block_bootstrap_ci(
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    metric: Metric,
    *,
    groups: ArrayLike,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``metric(a) - metric(b)`` on the same samples, with a percentile CI from one group resampling.

    Both predictions are scored on the same resampled groups in each draw, so
    the interval is that of the difference, not of two separate estimates.
    """
    a = np.asarray(y_true, dtype=np.float64)
    pa = np.asarray(pred_a, dtype=np.float64)
    pb = np.asarray(pred_b, dtype=np.float64)
    point = float(metric(a, pa) - metric(a, pb))
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(uniq.size)]
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, uniq.size, uniq.size)])
        reps[i] = metric(a[idx], pa[idx]) - metric(a[idx], pb[idx])
    lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def interval_coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Prediction-interval coverage probability (PICP)."""
    y = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def mean_interval_width(lower: ArrayLike, upper: ArrayLike) -> float:
    """Mean prediction-interval width (sharpness); report together with coverage."""
    return float(np.mean(np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)))


def gaussian_nll(y_true: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> float:
    """Mean negative log-likelihood under N(mu, sigma^2)."""
    return float(-np.mean(stats.norm.logpdf(np.asarray(y_true), np.asarray(mu), np.asarray(sigma))))


def gaussian_crps(y_true: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> float:
    """Mean closed-form CRPS for Gaussian predictive distributions (Gneiting & Raftery 2007)."""
    y = np.asarray(y_true, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    z = (y - m) / s
    crps = s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def lognormal_crps(y_true: ArrayLike, mu_log: ArrayLike, sigma_log: ArrayLike) -> FloatArray:
    """CRPS of each log-normal forecast ``LN(mu_log, sigma_log**2)`` against ``y_true``, in ``y``'s unit.

    Closed form of Baran & Lerch (2015). The CRPS is the mean absolute error of the
    whole predicted distribution: for a single-valued forecast it equals the absolute
    error. ``y_true <= 0`` is scored as 0 (all predicted mass lies above it).
    """
    y = np.asarray(y_true, dtype=np.float64)
    m = np.asarray(mu_log, dtype=np.float64)
    s = np.asarray(sigma_log, dtype=np.float64)
    pos = y > 0
    w = np.where(pos, (np.log(np.where(pos, y, 1.0)) - m) / s, -np.inf)
    mean = np.exp(m + s**2 / 2)
    phi_w = stats.norm.cdf(w)
    out = y * (2 * phi_w - 1) - 2 * mean * (
        stats.norm.cdf(w - s) + stats.norm.cdf(s / np.sqrt(2)) - 1
    )
    return np.asarray(out, dtype=np.float64)


def merge_gaussians(
    mus: ArrayLike, sigmas: ArrayLike, axis: int = 0
) -> tuple[FloatArray, FloatArray]:
    """One Gaussian with the first two moments of an equal-weight mixture (a deep ensemble's members).

    Mean of the means; variance = mean of the variances plus the variance of the
    means (Lakshminarayanan et al. 2017).
    """
    m = np.asarray(mus, dtype=np.float64)
    s = np.asarray(sigmas, dtype=np.float64)
    mu = m.mean(axis=axis)
    var = (s**2 + m**2).mean(axis=axis) - mu**2
    return mu, np.sqrt(np.maximum(var, 0.0))


def hierarchical_conformal_threshold(
    scores: ArrayLike, groups: ArrayLike, level: float = 0.9
) -> float:
    """Conformal threshold when whole groups (buoys) are exchangeable (Lee, Barber & Willett 2023).

    The ``level`` quantile of the calibration scores with every group weighted
    equally (each score ``1 / ((K + 1) n_k)``) plus a mass ``1 / (K + 1)`` at infinity
    for the new group. ``{y : score(y) <= threshold}`` then covers a random point of a
    new group with probability at least ``level``, on average over groups; it says
    nothing about any one group. Returns ``inf`` when there are too few groups.
    """
    sc = np.asarray(scores, dtype=np.float64)
    g = np.asarray(groups)
    ok = np.isfinite(sc)
    sc, g = sc[ok], g[ok]
    uniq, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    k = uniq.size
    if k == 0:
        return float("inf")
    w = 1.0 / ((k + 1) * counts[inv])
    order = np.argsort(sc, kind="stable")
    cum = np.cumsum(w[order])
    hit = np.flatnonzero(cum >= level - 1e-12)
    return float(sc[order][hit[0]]) if hit.size else float("inf")


def calibration_curve(
    y_true: ArrayLike, mu: ArrayLike, sigma: ArrayLike, levels: ArrayLike = (0.5, 0.8, 0.9, 0.95)
) -> dict[float, float]:
    """Empirical coverage of central Gaussian intervals at each nominal level.

    A calibrated model has empirical ~= nominal; systematic under-coverage
    means over-confident uncertainty.
    """
    y = np.asarray(y_true, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    out = {}
    for p in np.asarray(levels, dtype=np.float64):
        z = stats.norm.ppf(0.5 + p / 2)
        out[float(p)] = interval_coverage(y, m - z * s, m + z * s)
    return out
