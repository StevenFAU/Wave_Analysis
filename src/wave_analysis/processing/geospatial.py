"""Geodesy helpers: great-circle distance and bearing (spherical Earth)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wave_analysis.physics.constants import EARTH_RADIUS_M

FloatArray = NDArray[np.float64]


def haversine_m(lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike) -> FloatArray:
    """Great-circle distance [m] on a sphere of mean Earth radius (error < 0.5 % vs WGS-84)."""
    p1, p2 = (
        np.deg2rad(np.asarray(lat1, dtype=np.float64)),
        np.deg2rad(np.asarray(lat2, dtype=np.float64)),
    )
    dphi = p2 - p1
    dlmb = np.deg2rad(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return np.asarray(2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1))), dtype=np.float64)


def initial_bearing_deg(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> FloatArray:
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from true north."""
    p1, p2 = (
        np.deg2rad(np.asarray(lat1, dtype=np.float64)),
        np.deg2rad(np.asarray(lat2, dtype=np.float64)),
    )
    dlmb = np.deg2rad(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    x = np.sin(dlmb) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlmb)
    return np.asarray(np.mod(np.rad2deg(np.arctan2(x, y)), 360.0), dtype=np.float64)
