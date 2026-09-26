"""Project-level physical-consistency QC for wave observations.

These complement QARTOD (``qc.qartod``) with checks that follow from wave
physics or from the internal structure of spectral products. They are applied
*after* provider QC and never delete data: every result is a flag array.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from wave_analysis.physics.dispersion import wave_steepness
from wave_analysis.qc.flags import QCFlag, QCTestResult

#: Deep-water limiting steepness of an individual Stokes wave (Michell 1893; H/L ~ 0.142).
MICHELL_LIMIT = 0.142


def steepness_test(
    Hs: ArrayLike,
    Tp: ArrayLike,
    *,
    suspect_above: float = 0.07,
    fail_above: float = MICHELL_LIMIT,
) -> QCTestResult:
    r"""Flag implausible sea-state steepness :math:`s_p = 2\pi H_{m0}/(g T_p^2)`.

    Observed open-ocean :math:`s_p` rarely exceeds ~0.05-0.07 even in young
    wind seas; a *sea-state* steepness at the individual-wave breaking limit
    (~0.142) indicates a corrupt height or period, or a mis-paired record.
    Very young coastal seas can approach the suspect threshold legitimately,
    hence SUSPECT (retain, annotate) rather than FAIL below the hard limit.
    """
    H = np.asarray(Hs, dtype=np.float64)
    T = np.asarray(Tp, dtype=np.float64)
    ok = np.isfinite(H) & np.isfinite(T) & (T > 0)
    s = np.full(H.shape, np.nan)
    s[ok] = wave_steepness(H[ok], T[ok])
    flags = np.full(H.shape, int(QCFlag.PASS), dtype=np.int8)
    flags[~ok] = int(QCFlag.MISSING)
    flags[ok & (s > suspect_above)] = int(QCFlag.SUSPECT)
    flags[ok & (s > fail_above)] = int(QCFlag.FAIL)
    return QCTestResult(
        "steepness",
        "physical",
        flags,
        {"suspect_above": suspect_above, "fail_above": fail_above},
        "Linear-theory deep-water steepness; Michell (1893) limiting steepness",
    )


def spectral_consistency_test(
    reported_hs: ArrayLike, spectral_hm0: ArrayLike, *, abs_tol: float = 0.1, rel_tol: float = 0.1
) -> QCTestResult:
    """SUSPECT when provider bulk Hs disagrees with ``4*sqrt(m0)`` of the provider spectrum.

    Catches product mis-pairing (e.g. summary and spectral files with differing
    timestamps), unit/scale errors, and truncated spectra. Tolerance is
    ``max(abs_tol, rel_tol * reported)``.
    """
    a = np.asarray(reported_hs, dtype=np.float64)
    b = np.asarray(spectral_hm0, dtype=np.float64)
    flags = np.full(a.shape, int(QCFlag.PASS), dtype=np.int8)
    ok = np.isfinite(a) & np.isfinite(b)
    flags[~ok] = int(QCFlag.NOT_EVALUATED)
    tol = np.maximum(abs_tol, rel_tol * np.abs(a))
    flags[ok & (np.abs(a - b) > tol)] = int(QCFlag.SUSPECT)
    return QCTestResult(
        "spectral_consistency",
        "sensor",
        flags,
        {"abs_tol": abs_tol, "rel_tol": rel_tol},
        "Hm0 = 4 sqrt(m0); see docs/methodology/spectral_analysis.md",
    )


def directional_moment_test(
    alpha1: ArrayLike, alpha2: ArrayLike, r1: ArrayLike, r2: ArrayLike
) -> QCTestResult:
    r"""Check directional Fourier parameters for range and realisability.

    FAIL if :math:`r_1, r_2 \notin [0, 1]` or angles outside [0, 360].
    SUSPECT if the moments cannot come from any non-negative distribution:
    for circular moments :math:`c_1, c_2` the Toeplitz matrix
    :math:`[[1, c_1^*, c_2^*],[c_1, 1, c_1^*],[c_2, c_1, 1]]` must be positive
    semi-definite, which reduces to :math:`|c_2 - c_1^2| \le 1 - |c_1|^2`.
    Commonly violated at low-energy frequency bins where estimates are noisy;
    such bins should be down-weighted, not trusted.

    Input arrays are ``(..., n_freq)``; flags have the same shape.
    """
    from wave_analysis.processing.spectra import fourier_from_ndbc

    A1, A2 = np.asarray(alpha1, dtype=np.float64), np.asarray(alpha2, dtype=np.float64)
    R1, R2 = np.asarray(r1, dtype=np.float64), np.asarray(r2, dtype=np.float64)
    flags = np.full(A1.shape, int(QCFlag.PASS), dtype=np.int8)
    ok = np.isfinite(A1) & np.isfinite(A2) & np.isfinite(R1) & np.isfinite(R2)
    flags[~ok] = int(QCFlag.MISSING)
    a1, b1, a2, b2 = fourier_from_ndbc(A1, A2, R1, R2)
    c1, c2 = a1 + 1j * b1, a2 + 1j * b2
    with np.errstate(invalid="ignore"):
        unrealisable = np.abs(c2 - c1**2) > (1 - np.abs(c1) ** 2) + 1e-9
    flags[ok & unrealisable] = int(QCFlag.SUSPECT)
    out_of_range = (
        (R1 < 0) | (R1 > 1) | (R2 < 0) | (R2 > 1) | (A1 < 0) | (A1 > 360) | (A2 < 0) | (A2 > 360)
    )
    flags[ok & out_of_range] = int(QCFlag.FAIL)
    return QCTestResult(
        "directional_moments",
        "sensor",
        flags,
        {},
        "Circular-moment realisability (positive-definite Toeplitz condition)",
    )
