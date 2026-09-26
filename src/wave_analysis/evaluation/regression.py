r"""Regression and agreement metrics for scalar wave parameters.

All functions take ``(y_true, y_pred)`` and ignore pairs where either value is
non-finite (the count used is reported by :func:`regression_report`).

Normalised RMSE is *ambiguous in the literature*: e.g. Kamagata et al. (2026)
report NRMSE = 28.81 % for Hs with RMSE = 0.5618 m, which equals RMSE divided
by the training **range** of Hs (2.80 - 0.85 = 1.95 m). Other studies divide by
the mean (the oceanographic *scatter index*) or the standard deviation. This
module therefore requires the normaliser to be named explicitly, and
:func:`regression_report` returns all common variants.

References
----------
Bland, J. M. & Altman, D. G. (1986). Statistical methods for assessing agreement
between two methods of clinical measurement. *The Lancet*, 327(8476), 307-310.
https://doi.org/10.1016/S0140-6736(86)90837-8

Mentaschi, L., Besio, G., Cassola, F. & Mazzino, A. (2013). Problems in RMSE-based
wave model validations. *Ocean Modelling*, 72, 53-58.
https://doi.org/10.1016/j.ocemod.2013.08.003
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

FloatArray = NDArray[np.float64]
Normalizer = Literal["mean", "range", "std", "iqr"]


def _clean(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same length")
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean error, prediction minus truth (positive = over-prediction)."""
    a, b = _clean(y_true, y_pred)
    return float(np.mean(b - a))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    a, b = _clean(y_true, y_pred)
    return float(np.mean(np.abs(b - a)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root-mean-square error."""
    a, b = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((b - a) ** 2)))


def centered_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """RMSE after removing each series' mean (error not explained by bias)."""
    a, b = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean(((b - b.mean()) - (a - a.mean())) ** 2)))


def nrmse(y_true: ArrayLike, y_pred: ArrayLike, normalizer: Normalizer) -> float:
    """RMSE divided by a *named* statistic of ``y_true``."""
    a, _ = _clean(y_true, y_pred)
    if normalizer == "mean":
        denom = float(np.mean(a))
    elif normalizer == "range":
        denom = float(np.ptp(a))
    elif normalizer == "std":
        denom = float(np.std(a))
    elif normalizer == "iqr":
        q75, q25 = np.percentile(a, [75, 25])
        denom = float(q75 - q25)
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(f"unknown normalizer {normalizer!r}")
    return rmse(y_true, y_pred) / denom


def scatter_index(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Scatter index: centred RMSE / mean(observed) (standard in wave-model validation)."""
    a, _ = _clean(y_true, y_pred)
    return float(centered_rmse(y_true, y_pred) / np.mean(a))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination :math:`1 - SS_{res}/SS_{tot}` (can be negative)."""
    a, b = _clean(y_true, y_pred)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1.0 - np.sum((a - b) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def _correlatable(a: FloatArray, b: FloatArray) -> bool:
    """Correlation is undefined for fewer than 3 pairs or a constant series (e.g. a mean baseline)."""
    return a.size > 2 and bool(np.ptp(a) > 0) and bool(np.ptp(b) > 0)


def pearson(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Pearson linear correlation (NaN if undefined)."""
    a, b = _clean(y_true, y_pred)
    return float(stats.pearsonr(a, b).statistic) if _correlatable(a, b) else float("nan")


def spearman(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Spearman rank correlation (NaN if undefined)."""
    a, b = _clean(y_true, y_pred)
    return float(stats.spearmanr(a, b).statistic) if _correlatable(a, b) else float("nan")


def regression_slope(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[float, float]:
    """OLS slope and intercept of prediction on truth.

    A slope well below 1 with moderate correlation is the signature of
    *regression dilution* toward the training mean (Kamagata et al. 2026 report
    a slope of 0.12 for Hs) and is not visible in RMSE alone.
    """
    a, b = _clean(y_true, y_pred)
    if a.size < 3 or np.ptp(a) == 0:
        return float("nan"), float("nan")
    res = stats.linregress(a, b)
    return float(res.slope), float(res.intercept)


@dataclass(frozen=True)
class BlandAltman:
    """Bland-Altman agreement statistics (difference = prediction - truth)."""

    n: int
    bias: float
    sd: float
    loa_lower: float
    loa_upper: float
    proportional_slope: float
    proportional_p: float

    def as_dict(self) -> dict[str, float]:
        """Plain dict."""
        return asdict(self)


def bland_altman(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    z: float = 1.96,
    against: Literal["mean", "reference"] = "reference",
) -> BlandAltman:
    """Bias, limits of agreement (bias +/- z*SD), and a proportional-bias test.

    ``proportional_slope`` is the OLS slope of the difference on a magnitude
    axis; a significant slope (small ``proportional_p``) means error grows with
    magnitude - reported for Hs, Tp and Tz by Kamagata et al. (2026).

    The classical Bland-Altman plot uses the pair *mean* as the magnitude
    axis. When one method is far more precise than the other - the usual case
    here, an instrument reference versus a vision estimate - regressing the
    difference on the mean induces a **spurious positive slope** (the
    estimator's noise appears in both axes). The default therefore regresses
    on the reference value (``against="reference"``); use ``"mean"`` only when
    the two methods have comparable error.
    """
    a, b = _clean(y_true, y_pred)
    d = b - a
    m = a if against == "reference" else (a + b) / 2
    sd = float(np.std(d, ddof=1))
    lr = stats.linregress(m, d) if a.size > 2 else None
    return BlandAltman(
        n=int(a.size),
        bias=float(d.mean()),
        sd=sd,
        loa_lower=float(d.mean() - z * sd),
        loa_upper=float(d.mean() + z * sd),
        proportional_slope=float(lr.slope) if lr else float("nan"),
        proportional_p=float(lr.pvalue) if lr else float("nan"),
    )


def regression_report(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """All standard regression metrics in one dict (for experiment logs)."""
    a, b = _clean(y_true, y_pred)
    slope, intercept = regression_slope(a, b)
    return {
        "n": float(a.size),
        "bias": bias(a, b),
        "mae": mae(a, b),
        "rmse": rmse(a, b),
        "crmse": centered_rmse(a, b),
        "nrmse_mean": nrmse(a, b, "mean"),
        "nrmse_range": nrmse(a, b, "range"),
        "nrmse_std": nrmse(a, b, "std"),
        "scatter_index": scatter_index(a, b),
        "r2": r2(a, b),
        "pearson": pearson(a, b),
        "spearman": spearman(a, b),
        "slope": slope,
        "intercept": intercept,
    }
