"""Circular statistics for wave and wind directions.

Directions live on a circle: 359 deg and 1 deg differ by 2 deg, not 358 deg.
Linear RMSE, bias, R^2 and Pearson correlation on raw degrees are therefore
wrong near north (and the reported numbers depend on where the 0/360 cut
falls). Direction results in this project must use these functions.

The same wrap-around argument motivates the sin/cos attitude encoding used by
Yang et al. (2026) for vessel roll/pitch/yaw; :func:`encode_angles` provides it.

References
----------
Fisher, N. I. (1993). *Statistical Analysis of Circular Data*. Cambridge University
Press. https://doi.org/10.1017/CBO9780511564345

Jammalamadaka, S. R. & SenGupta, A. (2001). *Topics in Circular Statistics*.
World Scientific. https://doi.org/10.1142/4031
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def wrap_deg(x: ArrayLike) -> FloatArray:
    """Wrap angles to (-180, 180]."""
    a = np.asarray(x, dtype=np.float64)
    w = np.mod(a + 180.0, 360.0) - 180.0
    return np.where(w == -180.0, 180.0, w)


def angular_difference(y_true: ArrayLike, y_pred: ArrayLike) -> FloatArray:
    """Signed smallest difference ``pred - true`` in degrees, in (-180, 180]."""
    return wrap_deg(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))


def _clean(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def circular_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute angular error [deg]."""
    a, b = _clean(y_true, y_pred)
    return float(np.mean(np.abs(angular_difference(a, b))))


def circular_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root-mean-square angular error [deg]."""
    a, b = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean(angular_difference(a, b) ** 2)))


def circular_mean(angles_deg: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Mean direction [0, 360) via the resultant vector."""
    th = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    w = np.ones_like(th) if weights is None else np.asarray(weights, dtype=np.float64)
    return float(
        np.mod(np.rad2deg(np.arctan2(np.sum(w * np.sin(th)), np.sum(w * np.cos(th)))), 360.0)
    )


def circular_bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Circular mean of the signed angular errors [deg, (-180, 180]]."""
    a, b = _clean(y_true, y_pred)
    return float(wrap_deg(circular_mean(angular_difference(a, b))))


def circular_std(angles_deg: ArrayLike) -> float:
    r"""Circular standard deviation :math:`\sqrt{-2\ln \bar R}` [deg]."""
    th = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    R = np.hypot(np.mean(np.cos(th)), np.mean(np.sin(th)))
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(max(R, 1e-300)))))


def circular_correlation(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Jammalamadaka-SenGupta circular correlation coefficient in [-1, 1].

    .. math::
        \rho = \frac{\sum \sin(a_i-\bar a)\sin(b_i-\bar b)}
                    {\sqrt{\sum \sin^2(a_i-\bar a)\sum \sin^2(b_i-\bar b)}}
    """
    a, b = _clean(y_true, y_pred)
    ar, br = np.deg2rad(a), np.deg2rad(b)
    sa = np.sin(ar - np.deg2rad(circular_mean(a)))
    sb = np.sin(br - np.deg2rad(circular_mean(b)))
    den = np.sqrt(np.sum(sa**2) * np.sum(sb**2))
    return float(np.sum(sa * sb) / den) if den > 0 else float("nan")


def encode_angles(angles_deg: ArrayLike) -> FloatArray:
    """Map angles to ``[sin, cos]`` pairs, shape ``(..., 2)`` - continuous across 0/360."""
    th = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    return np.stack([np.sin(th), np.cos(th)], axis=-1)


def decode_angles(sin_cos: ArrayLike) -> FloatArray:
    """Inverse of :func:`encode_angles` (tolerates un-normalised network outputs)."""
    sc = np.asarray(sin_cos, dtype=np.float64)
    return np.mod(np.rad2deg(np.arctan2(sc[..., 0], sc[..., 1])), 360.0)


def circular_report(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """All direction metrics in one dict."""
    a, b = _clean(y_true, y_pred)
    return {
        "n": float(a.size),
        "circ_bias_deg": circular_bias(a, b),
        "circ_mae_deg": circular_mae(a, b),
        "circ_rmse_deg": circular_rmse(a, b),
        "circ_corr": circular_correlation(a, b),
    }
