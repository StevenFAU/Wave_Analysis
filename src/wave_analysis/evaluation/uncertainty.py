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
