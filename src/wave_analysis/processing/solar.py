"""Solar geometry for illumination annotation of imagery.

Implements the NOAA Global Monitoring Laboratory solar-position equations
(after Meeus, *Astronomical Algorithms*), accurate to ~0.01 deg in elevation
for 1901-2099 - far better than needed to classify illumination regimes or
flag sun-glint geometry.

Illumination categories use the standard twilight definitions:

====================  ===========================
category              solar elevation (deg)
====================  ===========================
``day``               >= 6
``low_sun``           0 to 6   (long shadows, strong glint risk)
``civil_twilight``    -6 to 0
``nautical_twilight`` -12 to -6
``night``             < -12
====================  ===========================

References
----------
NOAA GML Solar Calculator: https://gml.noaa.gov/grad/solcalc/calcdetails.html
Meeus, J. (1998). *Astronomical Algorithms* (2nd ed.). Willmann-Bell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def solar_position(
    times_utc: ArrayLike, latitude: ArrayLike, longitude: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Return ``(elevation_deg, azimuth_deg)`` of the sun; azimuth clockwise from true north.

    Elevation is geometric (no atmospheric refraction correction, which is
    < 0.6 deg and only matters within a degree of the horizon).
    """
    t = pd.to_datetime(np.atleast_1d(np.asarray(times_utc)), utc=True)
    lat = np.deg2rad(np.asarray(latitude, dtype=np.float64))
    lon = np.asarray(longitude, dtype=np.float64)

    jd = t.to_julian_date().to_numpy()
    jc = (jd - 2451545.0) / 36525.0
    l0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = np.deg2rad(m)
    c = (
        np.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + np.sin(2 * mr) * (0.019993 - 0.000101 * jc)
        + np.sin(3 * mr) * 0.000289
    )
    true_long = l0 + c
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * np.sin(np.deg2rad(omega))
    mean_obliq = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obliq = np.deg2rad(mean_obliq + 0.00256 * np.cos(np.deg2rad(omega)))
    decl = np.arcsin(np.sin(obliq) * np.sin(np.deg2rad(app_long)))

    y = np.tan(obliq / 2) ** 2
    l0r = np.deg2rad(l0)
    eot = 4 * np.rad2deg(
        y * np.sin(2 * l0r)
        - 2 * e * np.sin(mr)
        + 4 * e * y * np.sin(mr) * np.cos(2 * l0r)
        - 0.5 * y**2 * np.sin(4 * l0r)
        - 1.25 * e**2 * np.sin(2 * mr)
    )  # minutes
    minutes = (t.hour * 60 + t.minute + t.second / 60 + t.microsecond / 6e7).to_numpy()
    tst = np.mod(minutes + eot + 4 * lon, 1440.0)
    ha = np.deg2rad(np.where(tst / 4 < 0, tst / 4 + 180, tst / 4 - 180))

    cos_zen = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    zen = np.arccos(np.clip(cos_zen, -1, 1))
    elevation = 90.0 - np.rad2deg(zen)
    with np.errstate(invalid="ignore", divide="ignore"):
        az_num = np.sin(lat) * np.cos(zen) - np.sin(decl)
        az_den = np.cos(lat) * np.sin(zen)
        az = np.rad2deg(np.arccos(np.clip(az_num / az_den, -1, 1)))
    azimuth = np.where(ha > 0, np.mod(az + 180, 360), np.mod(540 - az, 360))
    return np.asarray(elevation, dtype=np.float64), np.asarray(azimuth, dtype=np.float64)


def illumination_category(elevation_deg: ArrayLike) -> NDArray[np.str_]:
    """Map solar elevation to an illumination category (see module docstring)."""
    el = np.asarray(elevation_deg, dtype=np.float64)
    return np.select(
        [el >= 6, el >= 0, el >= -6, el >= -12],
        ["day", "low_sun", "civil_twilight", "nautical_twilight"],
        default="night",
    ).astype(np.str_)


def glint_risk(
    sun_elevation_deg: ArrayLike,
    sun_azimuth_deg: ArrayLike,
    camera_azimuth_deg: float,
    camera_hfov_deg: float,
    *,
    max_elevation_deg: float = 35.0,
) -> NDArray[np.bool_]:
    """Whether the sun lies within the camera's horizontal field of view at low elevation.

    Specular sun glint on the sea surface is strongest when the camera looks
    towards a low sun. This is a geometric *risk* annotation; the measured
    glare fraction comes from :func:`wave_analysis.qc.imagery.image_quality`.
    """
    el = np.asarray(sun_elevation_deg, dtype=np.float64)
    daz = np.abs(
        (np.asarray(sun_azimuth_deg, dtype=np.float64) - camera_azimuth_deg + 180) % 360 - 180
    )
    return np.asarray(
        (el > 0) & (el < max_elevation_deg) & (daz <= camera_hfov_deg / 2), dtype=bool
    )
