r"""Time-domain (wave-by-wave) analysis of surface-elevation records.

Individual waves are delimited by successive zero crossings of the
de-meaned surface elevation :math:`\eta(t)`. For each wave the height is the
crest-to-trough excursion and the period is the time between the bounding
crossings. Crossing instants are located by linear interpolation between
samples, which removes the :math:`\pm 1/f_s` quantisation of naive sample
indexing.

Sea-state statistics follow the IAHR/PIANC conventions:

* :math:`H_{1/3}` - mean of the highest one-third of wave heights. Following
  Yang et al. (2026, eq. for :math:`H_s`) the count is
  :math:`N_s = \lfloor N/3 \rfloor`; the choice of floor versus round is
  recorded in :class:`ZeroCrossingResult` because it matters for short records.
* :math:`H_{1/10}`, :math:`H_{\max}`, :math:`H_{\text{mean}}`,
  :math:`H_{\text{rms}}`.
* :math:`T_z` (mean zero-crossing period), :math:`T_{1/3}` (mean period of the
  highest third), :math:`T_{H\max}`.

Note that :math:`H_{1/3}` (time domain) and :math:`H_{m0} = 4\sqrt{m_0}`
(spectral) are *different estimators*. For a narrow-banded Gaussian sea they
coincide asymptotically; in real, finite-bandwidth seas :math:`H_{1/3}` is
typically ~5 % lower than :math:`H_{m0}` in deep water, i.e.
:math:`H_{1/3} \approx 0.95\,H_{m0}` (Goda 2010).
Datasets in this project therefore label which estimator produced every
:math:`H_s` value (``hs_estimator`` in ``schemas.observation``).

References
----------
IAHR Working Group on Wave Generation and Analysis (1989). List of sea-state
parameters. *J. Waterway, Port, Coastal, Ocean Eng.*, 115(6), 793-808.
https://doi.org/10.1061/(ASCE)0733-950X(1989)115:6(793)

Goda, Y. (2010). *Random Seas and Design of Maritime Structures* (3rd ed.).
World Scientific. https://doi.org/10.1142/7425
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
CrossingType = Literal["up", "down"]


@dataclass(frozen=True)
class ZeroCrossingResult:
    """Wave-by-wave decomposition of an elevation record and summary statistics.

    All heights in metres, periods in seconds (assuming the input was metres
    and the sample rate Hz).
    """

    crossing: CrossingType
    heights: FloatArray
    periods: FloatArray
    crest_heights: FloatArray
    trough_depths: FloatArray
    crossing_times: FloatArray
    n_third_rule: str = "floor"
    record_duration_s: float = 0.0
    meta: dict[str, float] = field(default_factory=dict)

    @property
    def n_waves(self) -> int:
        """Number of complete waves."""
        return int(self.heights.size)

    def _highest_fraction(self, fraction: float) -> tuple[FloatArray, FloatArray]:
        n = self.n_waves
        m = int(np.floor(n * fraction)) if self.n_third_rule == "floor" else round(n * fraction)
        if m < 1:
            raise ValueError(f"record too short: {n} waves gives 0 waves in highest {fraction:.3g}")
        order = np.argsort(self.heights)[::-1][:m]
        return self.heights[order], self.periods[order]

    @property
    def H_one_third(self) -> float:
        """Mean height of the highest one-third of waves, :math:`H_{1/3}`."""
        return float(np.mean(self._highest_fraction(1.0 / 3.0)[0]))

    @property
    def T_one_third(self) -> float:
        """Mean period of the highest one-third of waves, :math:`T_{1/3}`."""
        return float(np.mean(self._highest_fraction(1.0 / 3.0)[1]))

    @property
    def H_one_tenth(self) -> float:
        """Mean height of the highest one-tenth of waves, :math:`H_{1/10}`."""
        return float(np.mean(self._highest_fraction(0.1)[0]))

    @property
    def H_max(self) -> float:
        """Maximum individual wave height."""
        return float(np.max(self.heights))

    @property
    def T_H_max(self) -> float:
        """Period of the maximum-height wave."""
        return float(self.periods[int(np.argmax(self.heights))])

    @property
    def H_mean(self) -> float:
        """Mean wave height."""
        return float(np.mean(self.heights))

    @property
    def H_rms(self) -> float:
        """Root-mean-square wave height."""
        return float(np.sqrt(np.mean(self.heights**2)))

    @property
    def T_z(self) -> float:
        """Mean zero-crossing period."""
        return float(np.mean(self.periods))

    def summary(self) -> dict[str, float]:
        """Return the standard statistics as a flat dict (for manifests / labels)."""
        return {
            "n_waves": float(self.n_waves),
            "H_one_third": self.H_one_third,
            "H_one_tenth": self.H_one_tenth,
            "H_max": self.H_max,
            "H_mean": self.H_mean,
            "H_rms": self.H_rms,
            "T_z": self.T_z,
            "T_one_third": self.T_one_third,
            "T_H_max": self.T_H_max,
            "record_duration_s": self.record_duration_s,
        }


def _crossing_times(eta: FloatArray, t: FloatArray, crossing: CrossingType) -> FloatArray:
    a, b = eta[:-1], eta[1:]
    idx = (
        np.flatnonzero((a < 0) & (b >= 0))
        if crossing == "up"
        else np.flatnonzero((a > 0) & (b <= 0))
    )
    # Linear interpolation of the zero crossing between samples idx and idx+1.
    frac = a[idx] / (a[idx] - b[idx])
    return np.asarray(t[idx] + frac * (t[idx + 1] - t[idx]), dtype=np.float64)


def zero_crossing_analysis(
    elevation: ArrayLike,
    sample_rate_hz: float,
    *,
    crossing: CrossingType = "up",
    detrend: Literal["mean", "linear", "none"] = "mean",
    n_third_rule: Literal["floor", "round"] = "floor",
) -> ZeroCrossingResult:
    r"""Decompose a surface-elevation record into individual waves.

    Parameters
    ----------
    elevation
        Surface elevation :math:`\eta` [m], uniformly sampled, gap-free.
        Records containing NaN are rejected: gaps must be handled explicitly
        upstream (QARTOD Test 9) rather than silently bridged.
    sample_rate_hz
        Sampling frequency [Hz].
    crossing
        ``"up"`` (zero-upcrossing, as in Yang et al. 2026 and NDBC) or ``"down"``
        (zero-downcrossing, recommended by IAHR for height definitions).
    detrend
        Remove the record mean (default), a least-squares line, or nothing.
    n_third_rule
        How the number of waves in the "highest third" is computed from ``N``.

    Returns
    -------
    ZeroCrossingResult
    """
    eta = np.asarray(elevation, dtype=np.float64).ravel()
    if eta.size < 3:
        raise ValueError("elevation record must contain at least 3 samples")
    if not np.all(np.isfinite(eta)):
        raise ValueError("elevation contains NaN/inf; resolve gaps before wave-by-wave analysis")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    t = np.arange(eta.size, dtype=np.float64) / sample_rate_hz
    if detrend == "mean":
        eta = eta - eta.mean()
    elif detrend == "linear":
        coef = np.polyfit(t, eta, 1)
        eta = eta - np.polyval(coef, t)

    tc = _crossing_times(eta, t, crossing)
    if tc.size < 2:
        raise ValueError("fewer than two zero crossings; no complete wave in record")

    # Sample indices bounding each wave (inclusive of interior samples).
    bounds = np.searchsorted(t, tc)
    heights, periods, crests, troughs = [], [], [], []
    for i in range(tc.size - 1):
        seg = eta[bounds[i] : bounds[i + 1]]
        if seg.size == 0:
            continue
        crest, trough = float(seg.max()), float(seg.min())
        crests.append(max(crest, 0.0))
        troughs.append(max(-trough, 0.0))
        heights.append(max(crest, 0.0) + max(-trough, 0.0))
        periods.append(tc[i + 1] - tc[i])

    return ZeroCrossingResult(
        crossing=crossing,
        heights=np.asarray(heights),
        periods=np.asarray(periods),
        crest_heights=np.asarray(crests),
        trough_depths=np.asarray(troughs),
        crossing_times=tc,
        n_third_rule=n_third_rule,
        record_duration_s=float(t[-1] - t[0]),
    )
