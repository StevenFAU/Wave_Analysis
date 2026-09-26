r"""Linear (Airy) wave theory: dispersion relation and derived kinematics.

The linear dispersion relation for surface gravity waves on water of uniform
depth :math:`h` is

.. math::

    \omega^2 = g k \tanh(k h), \qquad \omega = 2\pi f,

where :math:`k = 2\pi / L` is the wavenumber. It has no closed-form inverse for
:math:`k`, so :func:`wavenumber` uses the explicit approximation of
Guo (2002) as the starting point for Newton iterations, converging to machine
precision in two or three steps for all :math:`kh`.

Limits:

* deep water (:math:`kh \gg 1`): :math:`k = \omega^2/g`, :math:`L_0 = gT^2/2\pi`
  (:math:`\approx 1.56\,T^2` m);
* shallow water (:math:`kh \ll 1`): :math:`c = c_g = \sqrt{gh}`.

References
----------
Dean, R. G. & Dalrymple, R. A. (1991). *Water Wave Mechanics for Engineers and
Scientists*. World Scientific. https://doi.org/10.1142/1232

Guo, J. (2002). Simple and explicit solution of wave dispersion equation.
*Coastal Engineering*, 45(2), 71-74. https://doi.org/10.1016/S0378-3839(02)00039-X

Battjes, J. A. (1974). Surf similarity. *Proc. 14th Int. Conf. Coastal Eng.*,
466-480. https://doi.org/10.9753/icce.v14.26
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wave_analysis.physics.constants import G

FloatArray = NDArray[np.float64]

_GUO_BETA = 2.4908
_KH_DEEP = 20.0  # tanh(20) == 1 to double precision


def _as_depth(depth: ArrayLike | None, shape: tuple[int, ...]) -> FloatArray:
    if depth is None:
        return np.full(shape, np.inf)
    d = np.broadcast_to(np.asarray(depth, dtype=np.float64), shape).copy()
    if np.any(d <= 0):
        raise ValueError("depth must be strictly positive (use None for deep water)")
    return d


def wavenumber(
    frequency: ArrayLike,
    depth: ArrayLike | None = None,
    *,
    g: float = G,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> FloatArray:
    r"""Solve the linear dispersion relation for wavenumber :math:`k` [rad m^-1].

    Parameters
    ----------
    frequency
        Wave frequency :math:`f` [Hz] (not angular frequency). Must be > 0.
    depth
        Still-water depth :math:`h` [m]. ``None`` (default) or ``inf`` gives the
        deep-water solution :math:`k = \omega^2/g`.
    g
        Gravitational acceleration [m s^-2].
    tol
        Relative convergence tolerance on :math:`k` for the Newton iteration.
    max_iter
        Maximum Newton iterations.

    Returns
    -------
    numpy.ndarray
        Wavenumber with the broadcast shape of ``frequency`` and ``depth``.
    """
    f = np.asarray(frequency, dtype=np.float64)
    if np.any(f <= 0):
        raise ValueError("frequency must be strictly positive")
    shape = np.broadcast_shapes(f.shape, np.shape(depth) if depth is not None else ())
    f = np.atleast_1d(np.broadcast_to(f, shape))
    h = np.atleast_1d(_as_depth(depth, shape))
    omega = 2.0 * np.pi * f
    k_deep = omega**2 / g

    k = k_deep.copy()
    finite = np.isfinite(h)
    if np.any(finite):
        hf = h[finite]
        of = omega[finite]
        # Guo (2002) explicit approximation, relative error < 0.75 %.
        x = of * np.sqrt(hf / g)
        with np.errstate(over="ignore"):
            kh0 = x**2 * (1.0 - np.exp(-(x**_GUO_BETA))) ** (-1.0 / _GUO_BETA)
        kf = kh0 / hf
        for _ in range(max_iter):
            kh = np.minimum(kf * hf, _KH_DEEP)
            t = np.tanh(kh)
            residual = g * kf * t - of**2
            # d/dk [g k tanh(kh)] = g tanh(kh) + g k h sech^2(kh)
            deriv = g * t + g * kf * hf * (1.0 - t**2)
            step = residual / deriv
            kf = kf - step
            if np.all(np.abs(step) <= tol * np.abs(kf)):
                break
        k[finite] = kf
    return np.asarray(k.reshape(shape), dtype=np.float64)


def wavelength(frequency: ArrayLike, depth: ArrayLike | None = None, *, g: float = G) -> FloatArray:
    r"""Wavelength :math:`L = 2\pi/k` [m] from linear dispersion."""
    return 2.0 * np.pi / wavenumber(frequency, depth, g=g)


def phase_velocity(
    frequency: ArrayLike, depth: ArrayLike | None = None, *, g: float = G
) -> FloatArray:
    r"""Phase speed :math:`c = \omega/k` [m s^-1]."""
    f = np.asarray(frequency, dtype=np.float64)
    return 2.0 * np.pi * f / wavenumber(f, depth, g=g)


def group_velocity(
    frequency: ArrayLike, depth: ArrayLike | None = None, *, g: float = G
) -> FloatArray:
    r"""Group velocity :math:`c_g = n c` [m s^-1], :math:`n = \tfrac12(1 + 2kh/\sinh 2kh)`.

    Energy (and hence the wave *state* seen by a downstream sensor) propagates
    at :math:`c_g`; this is the relevant speed when estimating the travel-time
    lag between an offshore buoy and a nearshore camera
    (see ``processing.synchronize.propagation_lag``).
    """
    f = np.asarray(frequency, dtype=np.float64)
    k = wavenumber(f, depth, g=g)
    c = 2.0 * np.pi * f / k
    if depth is None:
        return 0.5 * c
    h = np.broadcast_to(np.asarray(depth, dtype=np.float64), k.shape)
    kh = k * h
    n = np.full_like(kh, 0.5)
    shallow = np.isfinite(kh) & (kh < _KH_DEEP)
    n[shallow] = 0.5 * (1.0 + 2.0 * kh[shallow] / np.sinh(2.0 * kh[shallow]))
    return n * c


def deep_water_wavelength(period: ArrayLike, *, g: float = G) -> FloatArray:
    r"""Deep-water wavelength :math:`L_0 = gT^2/2\pi` [m]."""
    T = np.asarray(period, dtype=np.float64)
    return g * T**2 / (2.0 * np.pi)


def wave_steepness(
    height: ArrayLike, period: ArrayLike, depth: ArrayLike | None = None, *, g: float = G
) -> FloatArray:
    r"""Wave steepness :math:`H/L` (dimensionless).

    With ``depth=None`` this is the *deep-water* (fictitious) steepness
    :math:`s_0 = 2\pi H/(g T^2)`, the conventional sea-state descriptor when
    using :math:`H_{m0}` and :math:`T_p` (giving :math:`s_p`). Observed ocean
    values of :math:`s_p` rarely exceed ~0.05-0.07; individual waves break near
    :math:`H/L \approx 1/7` (Miche limit in deep water).
    """
    H = np.asarray(height, dtype=np.float64)
    T = np.asarray(period, dtype=np.float64)
    return np.asarray(H / wavelength(1.0 / T, depth, g=g), dtype=np.float64)


def iribarren_number(
    slope: ArrayLike, height: ArrayLike, period: ArrayLike, *, g: float = G
) -> FloatArray:
    r"""Surf-similarity (Iribarren) parameter :math:`\xi_0 = \tan\beta / \sqrt{H_0/L_0}`.

    Classifies breaker type on a plane beach (Battjes 1974): spilling
    :math:`\xi_0 < 0.5`, plunging :math:`0.5 < \xi_0 < 3.3`, surging/collapsing
    :math:`\xi_0 > 3.3`. Useful for relating visual breaking-type labels to
    physical wave parameters.
    """
    tan_beta = np.asarray(slope, dtype=np.float64)
    H = np.asarray(height, dtype=np.float64)
    L0 = deep_water_wavelength(period, g=g)
    return np.asarray(tan_beta / np.sqrt(H / L0), dtype=np.float64)


def ursell_number(
    height: ArrayLike, period: ArrayLike, depth: ArrayLike, *, g: float = G
) -> FloatArray:
    r"""Ursell number :math:`U = H L^2 / h^3`; linear theory degrades for :math:`U \gtrsim 26`."""
    H = np.asarray(height, dtype=np.float64)
    h = np.asarray(depth, dtype=np.float64)
    L = wavelength(1.0 / np.asarray(period, dtype=np.float64), h, g=g)
    return np.asarray(H * L**2 / h**3, dtype=np.float64)
