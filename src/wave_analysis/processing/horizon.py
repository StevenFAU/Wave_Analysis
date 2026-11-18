"""The sea horizon in one camera view: its tilt and height, as a measure of the platform's motion.

On a buoy (or ship) the camera moves with the hull, so the horizon's tilt in a
view is the hull's roll about that camera's axis, and the horizon's height is
its pitch towards that camera's heading (research question Q-M5; horizon
tracking for ship video: Schwendeman & Thomson 2015). A camera's own mounting
angle adds a constant to both, removed later by subtracting each camera's
median over many images.

Method (no learning): the image is smoothed; in each column the row where it
darkens most steeply going down (bright sky above, darker sea below) is taken
as a horizon candidate; a straight line is fitted to the candidates by RANSAC
(sun glint, clouds and whitecaps give outliers) and refitted by least squares
on its inliers. The fit reports the share of columns that agree with the line
and the brightness step across it, so that views without a clear horizon
(glare, fog, hull in view, darkness) can be left out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class HorizonFit:
    """A horizon line in image coordinates (x to the right, y down, pixels)."""

    angle_deg: float  # positive: the line descends to the right (clockwise in the image)
    row_mid: float  # the line's row at the view's centre column
    inlier_share: float  # share of searched columns within ``tol_px`` of the line
    step: float  # median darkening across the line (grey levels, over 2 * gap rows)
    rms_px: float  # residual of the inliers

    def ok(self, min_inlier_share: float = 0.5, min_step: float = 6.0) -> bool:
        """Whether the fit is clear enough to use: enough agreeing columns and a real brightness step."""
        return (
            np.isfinite(self.angle_deg)
            and self.inlier_share >= min_inlier_share
            and self.step >= min_step
        )


NO_FIT = HorizonFit(np.nan, np.nan, 0.0, 0.0, np.nan)


def detect_horizon(
    gray: ArrayLike,
    *,
    sigma: float = 2.0,
    gap: int = 2,
    margin_x: float = 0.06,
    margin_y: float = 0.08,
    col_step: int = 4,
    tol_px: float = 2.5,
    iterations: int = 300,
    seed: int = 0,
) -> HorizonFit:
    """Fit the sea horizon in one greyscale view (H x W array, any numeric type)."""
    g = gaussian_filter(np.asarray(gray, dtype=np.float32), sigma)
    h, w = g.shape
    # darkening going down: positive where the row above is brighter than the row below
    drop = g[: h - 2 * gap] - g[2 * gap :]
    y0, y1 = int(margin_y * h), int((1 - margin_y) * h) - 2 * gap
    cols = np.arange(int(margin_x * w), int((1 - margin_x) * w), col_step)
    if y1 <= y0 + 2 or cols.size < 4:
        return NO_FIT
    band = drop[y0:y1, cols]
    ys = band.argmax(axis=0) + y0 + gap  # the centre of the step
    strength = band.max(axis=0)
    xs = cols.astype(np.float64)
    ys = ys.astype(np.float64)
    rng = np.random.default_rng(seed)
    best = np.zeros(xs.size, dtype=bool)
    for _ in range(iterations):
        i, j = rng.choice(xs.size, 2, replace=False)
        if xs[i] == xs[j]:
            continue
        slope = (ys[j] - ys[i]) / (xs[j] - xs[i])
        inl = np.abs(ys - (ys[i] + slope * (xs - xs[i]))) <= tol_px
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 4:
        return NO_FIT
    slope, icpt = np.polyfit(xs[best], ys[best], 1)
    inl = np.abs(ys - (icpt + slope * xs)) <= tol_px  # refit's own inliers
    if inl.sum() >= 4:
        slope, icpt = np.polyfit(xs[inl], ys[inl], 1)
    res = ys[inl] - (icpt + slope * xs[inl])
    return HorizonFit(
        angle_deg=float(np.degrees(np.arctan(slope))),
        row_mid=float(icpt + slope * (w / 2)),
        inlier_share=float(inl.mean()),
        step=float(np.median(strength[inl])) if inl.any() else 0.0,
        rms_px=float(np.sqrt(np.mean(res**2))) if inl.any() else np.nan,
    )
