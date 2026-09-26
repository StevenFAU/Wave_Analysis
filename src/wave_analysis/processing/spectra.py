r"""Spectral wave analysis: containers, moments, bulk parameters, directional distributions.

Design rule (``docs/methodology/spectral_analysis.md``): the *highest-information
representation available upstream is preserved*. A :class:`FrequencySpectrum`
carries :math:`S(f)` together with the four directional Fourier parameters
(:math:`\alpha_1, \alpha_2, r_1, r_2`) when the provider supplies them; bulk
parameters are always *derived* from it, never stored in its place.

Conventions
-----------
* Frequencies in Hz; spectral density in m^2 Hz^-1.
* Directions are **nautical, coming-from, degrees clockwise from true north**
  (NDBC ``MWD``/``alpha1``; CF ``sea_surface_wave_from_direction``).
* Canonical Fourier coefficients use the same nautical angle :math:`\theta`:
  :math:`a_n = \int D\cos n\theta\,d\theta`, :math:`b_n = \int D\sin n\theta\,d\theta`,
  so :math:`\alpha_1 = \operatorname{atan2}(b_1,a_1)`,
  :math:`r_1 = \sqrt{a_1^2+b_1^2}` etc. (CDIP uses this convention directly;
  NDBC publishes :math:`\alpha, r` - see :func:`fourier_from_ndbc`.)

References
----------
Kuik, A. J., van Vledder, G. Ph. & Holthuijsen, L. H. (1988). A method for the
routine analysis of pitch-and-roll buoy wave data. *J. Phys. Oceanogr.*, 18(7),
1020-1034. https://doi.org/10.1175/1520-0485(1988)018<1020:AMFTRA>2.0.CO;2

Lygre, A. & Krogstad, H. E. (1986). Maximum entropy estimation of the
directional distribution in ocean wave spectra. *J. Phys. Oceanogr.*, 16(12),
2052-2060. https://doi.org/10.1175/1520-0485(1986)016<2052:MEEOTD>2.0.CO;2

Young, I. R. (1995). The determination of confidence limits associated with
estimates of the spectral peak frequency. *Ocean Engineering*, 22(7), 669-686.
https://doi.org/10.1016/0029-8018(95)00002-3

NDBC (2023). Measurement descriptions and units.
https://www.ndbc.noaa.gov/faq/measdes.shtml
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

DIRECTIONAL_FIELDS: tuple[str, ...] = ("alpha1", "alpha2", "r1", "r2")


# --------------------------------------------------------------------------- #
# Frequency grid utilities
# --------------------------------------------------------------------------- #


def frequency_bin_edges(frequency: ArrayLike) -> FloatArray:
    r"""Bin edges from bin centres using midpoints, extrapolating the outer edges symmetrically.

    For the NDBC 47-band grid this reproduces the published ``WVHT`` from
    :math:`4\sqrt{m_0}` to within rounding (mean difference 0.0002 m,
    SD 0.005 m over 4,281 records at station 41010 in 2023; see
    ``scripts/validate_ndbc_hm0.py``).
    """
    f = np.asarray(frequency, dtype=np.float64)
    if f.ndim != 1 or f.size < 2:
        raise ValueError("frequency must be 1-D with at least two bins")
    if np.any(np.diff(f) <= 0):
        raise ValueError("frequency must be strictly increasing")
    mid = 0.5 * (f[1:] + f[:-1])
    first = f[0] - (mid[0] - f[0])
    last = f[-1] + (f[-1] - mid[-1])
    return np.concatenate([[max(first, 0.0)], mid, [last]])


def frequency_bandwidths(frequency: ArrayLike) -> FloatArray:
    """Bandwidth of each frequency bin [Hz] (see :func:`frequency_bin_edges`)."""
    return np.diff(frequency_bin_edges(frequency))


# --------------------------------------------------------------------------- #
# Directional coefficient conversions
# --------------------------------------------------------------------------- #


def fourier_from_ndbc(
    alpha1: ArrayLike, alpha2: ArrayLike, r1: ArrayLike, r2: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    r"""Convert NDBC :math:`(\alpha_1,\alpha_2,r_1,r_2)` to canonical nautical :math:`(a_1,b_1,a_2,b_2)`.

    NDBC's spreading function is
    :math:`D(\theta) = \pi^{-1}[\tfrac12 + r_1\cos(\theta-\alpha_1) + r_2\cos 2(\theta-\alpha_2)]`
    with :math:`\alpha` in nautical coming-from degrees, which expands to
    :math:`a_1 = r_1\cos\alpha_1`, :math:`b_1 = r_1\sin\alpha_1`,
    :math:`a_2 = r_2\cos 2\alpha_2`, :math:`b_2 = r_2\sin 2\alpha_2`.
    """
    a1r = np.deg2rad(np.asarray(alpha1, dtype=np.float64))
    a2r = np.deg2rad(np.asarray(alpha2, dtype=np.float64))
    r1a = np.asarray(r1, dtype=np.float64)
    r2a = np.asarray(r2, dtype=np.float64)
    return r1a * np.cos(a1r), r1a * np.sin(a1r), r2a * np.cos(2 * a2r), r2a * np.sin(2 * a2r)


def ndbc_from_fourier(
    a1: ArrayLike, b1: ArrayLike, a2: ArrayLike, b2: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Inverse of :func:`fourier_from_ndbc`; returns ``(alpha1, alpha2, r1, r2)``, angles in [0, 360)."""
    a1a, b1a = np.asarray(a1, dtype=np.float64), np.asarray(b1, dtype=np.float64)
    a2a, b2a = np.asarray(a2, dtype=np.float64), np.asarray(b2, dtype=np.float64)
    alpha1 = np.mod(np.rad2deg(np.arctan2(b1a, a1a)), 360.0)
    alpha2 = np.mod(0.5 * np.rad2deg(np.arctan2(b2a, a2a)), 360.0)
    return alpha1, alpha2, np.hypot(a1a, b1a), np.hypot(a2a, b2a)


def directional_distribution(
    alpha1: ArrayLike,
    alpha2: ArrayLike,
    r1: ArrayLike,
    r2: ArrayLike,
    direction_deg: ArrayLike,
    *,
    method: Literal["fourier", "mem"] = "mem",
) -> FloatArray:
    r"""Estimate the directional spreading function :math:`D(f,\theta)` [rad^-1].

    Parameters
    ----------
    alpha1, alpha2, r1, r2
        Directional parameters, shape ``(..., n_freq)``.
    direction_deg
        Output direction grid (nautical coming-from), shape ``(n_dir,)``.
    method
        ``"fourier"`` - the truncated Fourier series published by NDBC
        (Longuet-Higgins et al. 1963). Simple and linear, but broad and can go
        negative for narrow spreads.
        ``"mem"`` (default) - the maximum-entropy method of Lygre & Krogstad
        (1986), which is non-negative and resolves narrow and bimodal seas far
        better from the same four coefficients.

    Returns
    -------
    numpy.ndarray
        Shape ``(..., n_freq, n_dir)``; integrates to 1 over :math:`2\pi` for
        each frequency (renormalised numerically on the supplied grid).
    """
    th = np.deg2rad(np.asarray(direction_deg, dtype=np.float64))
    a1, b1, a2, b2 = (x[..., None] for x in fourier_from_ndbc(alpha1, alpha2, r1, r2))
    if method == "fourier":
        D = (1.0 / (2 * np.pi)) * (
            1 + 2 * (a1 * np.cos(th) + b1 * np.sin(th) + a2 * np.cos(2 * th) + b2 * np.sin(2 * th))
        )
    elif method == "mem":
        c1 = a1 + 1j * b1
        c2 = a2 + 1j * b2
        with np.errstate(divide="ignore", invalid="ignore"):
            phi1 = (c1 - c2 * np.conj(c1)) / (1 - np.abs(c1) ** 2)
            phi2 = c2 - c1 * phi1
            num = 1 - phi1 * np.conj(c1) - phi2 * np.conj(c2)
            den = np.abs(1 - phi1 * np.exp(-1j * th) - phi2 * np.exp(-2j * th)) ** 2
            D = np.real(num) / (2 * np.pi * den)
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(f"unknown method {method!r}")
    D = np.asarray(D, dtype=np.float64)
    dth = 2 * np.pi / th.size if th.size > 1 else 2 * np.pi
    norm = np.nansum(D, axis=-1, keepdims=True) * dth
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(D / norm, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FrequencySpectrum:
    r"""A time series of (optionally directional) frequency spectra on one frequency grid.

    Attributes
    ----------
    time
        UTC timestamps, ``datetime64[ns]``, shape ``(n_time,)``. The meaning of
        the timestamp (start / end / nominal) is recorded in ``meta["time_reference"]``.
    frequency
        Bin-centre frequencies [Hz], shape ``(n_freq,)``.
    density
        Variance density :math:`S(f)` [m^2/Hz], shape ``(n_time, n_freq)``.
    bandwidth
        Bin widths [Hz]; derived from ``frequency`` if not supplied upstream.
    alpha1, alpha2, r1, r2
        Optional directional parameters, shape ``(n_time, n_freq)``. Angles are
        nautical coming-from degrees; ``r`` is dimensionless in [0, 1]
        (already un-scaled for NDBC historical files).
    separation_frequency
        Optional provider swell/wind-sea separation frequency [Hz], ``(n_time,)``.
    meta
        Free-form provenance (source, station, product, file checksum, ...).
    """

    time: NDArray[np.datetime64]
    frequency: FloatArray
    density: FloatArray
    bandwidth: FloatArray | None = None
    alpha1: FloatArray | None = None
    alpha2: FloatArray | None = None
    r1: FloatArray | None = None
    r2: FloatArray | None = None
    separation_frequency: FloatArray | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_t, n_f = np.shape(self.density)
        if np.shape(self.time) != (n_t,):
            raise ValueError("time must have shape (n_time,)")
        if np.shape(self.frequency) != (n_f,):
            raise ValueError("frequency must have shape (n_freq,)")
        for name in DIRECTIONAL_FIELDS:
            arr = getattr(self, name)
            if arr is not None and np.shape(arr) != (n_t, n_f):
                raise ValueError(f"{name} must have shape (n_time, n_freq) = {(n_t, n_f)}")
        if self.bandwidth is None:
            object.__setattr__(self, "bandwidth", frequency_bandwidths(self.frequency))

    # ------------------------------------------------------------------ #
    @property
    def n_time(self) -> int:
        """Number of spectra."""
        return int(self.density.shape[0])

    @property
    def is_directional(self) -> bool:
        """Whether all four directional parameters are present."""
        return all(getattr(self, n) is not None for n in DIRECTIONAL_FIELDS)

    def moment(
        self, n: int | float, fmin: float | None = None, fmax: float | None = None
    ) -> FloatArray:
        r"""Spectral moment :math:`m_n = \int f^n S(f)\,df` using bin bandwidths.

        Bins with NaN density are *excluded* and the returned value is NaN for
        any record where more than zero in-band bins are missing, so that a
        partially-missing spectrum can never masquerade as a low-energy one.
        """
        assert self.bandwidth is not None
        sel = np.ones_like(self.frequency, dtype=bool)
        if fmin is not None:
            sel &= self.frequency >= fmin
        if fmax is not None:
            sel &= self.frequency <= fmax
        S = self.density[:, sel]
        w = (self.frequency[sel] ** n) * self.bandwidth[sel]
        out = np.nansum(S * w, axis=1)
        out[np.any(np.isnan(S), axis=1)] = np.nan
        return np.asarray(out, dtype=np.float64)

    def select_time(self, mask: NDArray[np.bool_]) -> FrequencySpectrum:
        """Return a new spectrum restricted to ``mask`` along time."""

        def sub(a: FloatArray | None) -> FloatArray | None:
            return None if a is None else a[mask]

        return replace(
            self,
            time=self.time[mask],
            density=self.density[mask],
            alpha1=sub(self.alpha1),
            alpha2=sub(self.alpha2),
            r1=sub(self.r1),
            r2=sub(self.r2),
            separation_frequency=None
            if self.separation_frequency is None
            else self.separation_frequency[mask],
        )

    def to_long_frame(self) -> pd.DataFrame:
        """Tidy (time, frequency) table suitable for Parquet in the standardized layer."""
        n_t, n_f = self.density.shape
        assert self.bandwidth is not None
        data: dict[str, Any] = {
            "time_utc": np.repeat(self.time, n_f),
            "frequency_hz": np.tile(self.frequency, n_t),
            "bandwidth_hz": np.tile(self.bandwidth, n_t),
            "spectral_density_m2_hz": self.density.ravel(),
        }
        for name in DIRECTIONAL_FIELDS:
            arr = getattr(self, name)
            if arr is not None:
                data[name] = arr.ravel()
        df = pd.DataFrame(data)
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        return df


# --------------------------------------------------------------------------- #
# Bulk parameters
# --------------------------------------------------------------------------- #


def _peak_index(S: FloatArray) -> NDArray[np.intp]:
    filled = np.where(np.isnan(S), -np.inf, S)
    return np.asarray(np.argmax(filled, axis=1), dtype=np.intp)


def bulk_parameters(
    spec: FrequencySpectrum,
    fmin: float | None = None,
    fmax: float | None = None,
) -> pd.DataFrame:
    r"""Derive standard integral wave parameters from a spectrum time series.

    Returned columns (NaN when undefined):

    ``Hm0``
        :math:`4\sqrt{m_0}` [m] - spectral significant wave height.
    ``Tp``
        :math:`1/f_p` at the discrete spectral maximum [s] (NDBC ``DPD``).
    ``Tp_smooth``
        Young (1995) weighted peak period :math:`\left(\int f S^4 df / \int S^4 df\right)^{-1}`,
        a far less grid-quantised estimator, recommended as an ML target.
    ``Tm01``
        :math:`m_0/m_1` [s] - mean period.
    ``Tm02``
        :math:`\sqrt{m_0/m_2}` [s] - spectral estimate of the zero-crossing period :math:`T_z`.
    ``Tm_10``
        :math:`m_{-1}/m_0` [s] - energy period :math:`T_e` (wave-power studies).
    ``nu``
        Longuet-Higgins (1975) spectral width :math:`\sqrt{m_0 m_2/m_1^2 - 1}`.
    ``Qp``
        Goda peakedness :math:`2 m_0^{-2}\int f S^2 df`.
    ``steepness_p``
        :math:`s_p = 2\pi H_{m0}/(g T_p^2)`, deep-water peak steepness.

    and, for directional spectra:

    ``Dp``
        :math:`\alpha_1` at the peak frequency [deg] (NDBC ``MWD`` definition).
    ``Dm``
        Energy-weighted mean direction :math:`\operatorname{atan2}(\bar b_1, \bar a_1)` [deg].
    ``spread_m``
        Kuik et al. (1988) circular spread :math:`\sqrt{2(1-\bar r_1)}` [deg].
    ``spread_p``
        Spread at the peak frequency [deg].
    """
    from wave_analysis.physics.constants import G

    m0 = spec.moment(0, fmin, fmax)
    m1 = spec.moment(1, fmin, fmax)
    m2 = spec.moment(2, fmin, fmax)
    m_1 = spec.moment(-1, fmin, fmax)

    sel = np.ones_like(spec.frequency, dtype=bool)
    if fmin is not None:
        sel &= spec.frequency >= fmin
    if fmax is not None:
        sel &= spec.frequency <= fmax
    f = spec.frequency[sel]
    assert spec.bandwidth is not None
    df = spec.bandwidth[sel]
    S = spec.density[:, sel]

    with np.errstate(divide="ignore", invalid="ignore"):
        ip = _peak_index(S)
        fp = f[ip]
        Tp = np.where(np.nanmax(np.where(np.isnan(S), -np.inf, S), axis=1) > 0, 1.0 / fp, np.nan)
        S4 = np.nan_to_num(S) ** 4
        fp_smooth = (S4 * f * df).sum(axis=1) / (S4 * df).sum(axis=1)
        out = pd.DataFrame(
            {
                "time_utc": pd.to_datetime(spec.time, utc=True),
                "m0": m0,
                "Hm0": 4.0 * np.sqrt(m0),
                "Tp": Tp,
                "Tp_smooth": 1.0 / fp_smooth,
                "Tm01": m0 / m1,
                "Tm02": np.sqrt(m0 / m2),
                "Tm_10": m_1 / m0,
                "nu": np.sqrt(m0 * m2 / m1**2 - 1.0),
                "Qp": 2.0 / m0**2 * (np.nan_to_num(S) ** 2 * f * df).sum(axis=1),
            }
        )
        out["steepness_p"] = 2 * np.pi * out["Hm0"] / (G * out["Tp"] ** 2)

        if spec.is_directional:
            assert spec.alpha1 is not None
            assert spec.alpha2 is not None
            assert spec.r1 is not None
            assert spec.r2 is not None
            a1, b1, _, _ = fourier_from_ndbc(
                spec.alpha1[:, sel], spec.alpha2[:, sel], spec.r1[:, sel], spec.r2[:, sel]
            )
            E = np.nan_to_num(S) * df
            valid = ~np.isnan(a1)
            Et = np.where(valid, E, 0.0).sum(axis=1)
            A1 = np.where(valid, a1 * E, 0.0).sum(axis=1) / Et
            B1 = np.where(valid, b1 * E, 0.0).sum(axis=1) / Et
            rows = np.arange(S.shape[0])
            r1p = spec.r1[:, sel][rows, ip]
            out["Dp"] = spec.alpha1[:, sel][rows, ip]
            out["Dm"] = np.mod(np.rad2deg(np.arctan2(B1, A1)), 360.0)
            out["spread_m"] = np.rad2deg(np.sqrt(2.0 * (1.0 - np.hypot(A1, B1))))
            out["spread_p"] = np.rad2deg(np.sqrt(2.0 * (1.0 - r1p)))
    return out


# --------------------------------------------------------------------------- #
# Energy-conserving regridding
# --------------------------------------------------------------------------- #


def regrid_energy_conserving(
    src_edges: ArrayLike, src_density: ArrayLike, dst_edges: ArrayLike
) -> FloatArray:
    r"""Rebin a spectral density onto new frequency bins, conserving variance.

    The source density is treated as piecewise-constant within its bins (the
    representation implied by the provider's band-averaged estimates). The
    variance in each destination bin is the overlap-weighted sum of source
    variances, then divided by the destination width. Total variance
    :math:`m_0` is conserved exactly over the common frequency range, unlike
    point interpolation, which does not conserve :math:`m_0` when grids differ in
    resolution. Destination bins with any portion outside the source range are
    returned as NaN rather than extrapolated.

    ``src_density`` may be 1-D ``(n_src,)`` or 2-D ``(n_time, n_src)``.
    """
    se = np.asarray(src_edges, dtype=np.float64)
    de = np.asarray(dst_edges, dtype=np.float64)
    S = np.atleast_2d(np.asarray(src_density, dtype=np.float64))
    lo = np.maximum(se[None, :-1], de[:-1, None])
    hi = np.minimum(se[None, 1:], de[1:, None])
    overlap = np.clip(hi - lo, 0.0, None)  # (n_dst, n_src)
    missing = np.isnan(S)
    energy = np.where(missing, 0.0, S) @ overlap.T  # (n_time, n_dst)
    # A destination bin is undefined if any overlapping source bin is missing.
    touched_missing = missing.astype(np.float64) @ (overlap.T > 0).astype(np.float64) > 0
    width = np.diff(de)
    out = energy / width[None, :]
    out[touched_missing] = np.nan
    outside = (de[:-1] < se[0] - 1e-12) | (de[1:] > se[-1] + 1e-12)
    out[:, outside] = np.nan
    result: FloatArray = out if np.ndim(src_density) > 1 else out[0]
    return result
