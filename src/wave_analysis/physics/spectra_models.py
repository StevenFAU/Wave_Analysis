r"""Parametric frequency spectra, directional spreading functions, and linear sea-surface simulation.

These serve three purposes in the project:

1. **Analytic test oracles** - spectral-moment and parameter code is verified
   against spectra whose integrals are known in closed form.
2. **Synthetic "virtual buoy" labels** - reproducing the digital-twin labelling
   strategy of Yang et al. (2026), in which labels are derived from simulated
   buoy heave by both zero-crossing and spectral methods.
3. **Priors / physics regularisation** - e.g. bounding predicted
   :math:`(H_s, T_p)` pairs by physically plausible steepness.

References
----------
Pierson, W. J. & Moskowitz, L. (1964). A proposed spectral form for fully
developed wind seas. *J. Geophys. Res.*, 69(24), 5181-5190.
https://doi.org/10.1029/JZ069i024p05181

Hasselmann, K. et al. (1973). Measurements of wind-wave growth and swell decay
during the Joint North Sea Wave Project (JONSWAP). *Dtsch. Hydrogr. Z.*, A8(12).

Goda, Y. (1988). Statistical variability of sea state parameters as a function
of wave spectrum. *Coastal Engineering in Japan*, 31(1), 39-52.
https://doi.org/10.1080/05785634.1988.11924482

Longuet-Higgins, M. S., Cartwright, D. E. & Smith, N. D. (1963). Observations
of the directional spectrum of sea waves using the motions of a floating buoy.
*Ocean Wave Spectra*, Prentice-Hall, 111-136.

Tucker, M. J. (1993). Recommended standard for wave data sampling and
near-real-time processing. *Ocean Engineering*, 20(5), 459-474.
https://doi.org/10.1016/0029-8018(93)90015-G  (random-amplitude simulation)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

from wave_analysis.physics.dispersion import wavenumber

FloatArray = NDArray[np.float64]


def pierson_moskowitz(frequency: ArrayLike, Hs: float, Tp: float) -> FloatArray:
    r"""Pierson-Moskowitz spectrum parameterised by :math:`H_s` and :math:`T_p`.

    .. math::
        S(f) = \frac{5}{16} H_s^2 f_p^4 f^{-5} \exp\!\left[-\tfrac54 (f_p/f)^4\right]

    Its zeroth moment over :math:`(0,\infty)` is exactly :math:`H_s^2/16`.
    """
    f = np.asarray(frequency, dtype=np.float64)
    fp = 1.0 / Tp
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        S = (5.0 / 16.0) * Hs**2 * fp**4 * f**-5 * np.exp(-1.25 * (fp / f) ** 4)
    return np.where(f > 0, S, 0.0)


def jonswap(
    frequency: ArrayLike,
    Hs: float,
    Tp: float,
    gamma: float = 3.3,
    sigma_a: float = 0.07,
    sigma_b: float = 0.09,
    *,
    normalize: bool = True,
) -> FloatArray:
    r"""JONSWAP spectrum in the :math:`(H_s, T_p, \gamma)` form of Goda (1988).

    .. math::
        S(f) = \beta_J H_s^2 T_p^{-4} f^{-5}
               \exp\!\left[-1.25 (T_p f)^{-4}\right]
               \gamma^{\exp[-(T_p f - 1)^2 / 2\sigma^2]}

    with :math:`\sigma = \sigma_a` for :math:`f \le f_p` and :math:`\sigma_b`
    otherwise. Goda's :math:`\beta_J` is accurate to a few percent; with
    ``normalize=True`` the result is rescaled so that its trapezoidal integral
    on the supplied grid equals :math:`H_s^2/16` exactly (making the frequency
    truncation explicit rather than hidden).
    """
    f = np.asarray(frequency, dtype=np.float64)
    fp = 1.0 / Tp
    beta = (0.0624 / (0.230 + 0.0336 * gamma - 0.185 / (1.9 + gamma))) * (
        1.094 - 0.01915 * np.log(gamma)
    )
    sigma = np.where(f <= fp, sigma_a, sigma_b)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        r = np.exp(-((f / fp - 1.0) ** 2) / (2.0 * sigma**2))
        S = beta * Hs**2 * Tp**-4 * f**-5 * np.exp(-1.25 * (Tp * f) ** -4) * gamma**r
    S = np.where(f > 0, S, 0.0)
    if normalize and f.size > 1:
        m0 = float(np.trapezoid(S, f))
        if m0 > 0:
            S = S * (Hs**2 / 16.0) / m0
    return np.asarray(S, dtype=np.float64)


def cos2s_spreading(direction_deg: ArrayLike, mean_direction_deg: float, s: float) -> FloatArray:
    r"""Longuet-Higgins et al. (1963) :math:`\cos^{2s}` directional spreading function.

    .. math::
        D(\theta) = \frac{2^{2s-1}\,\Gamma(s+1)^2}{\pi\,\Gamma(2s+1)}
                    \cos^{2s}\!\left(\frac{\theta-\theta_m}{2}\right)

    normalised so that :math:`\int_0^{2\pi} D\,d\theta = 1` (units rad^-1). Its
    first two normalised Fourier moments are :math:`r_1 = s/(s+1)` and
    :math:`r_2 = s(s-1)/[(s+1)(s+2)]` (see :func:`cos2s_moments`).
    """
    theta = np.deg2rad(np.asarray(direction_deg, dtype=np.float64) - mean_direction_deg)
    log_c = (2 * s - 1) * np.log(2.0) + 2 * gammaln(s + 1) - np.log(np.pi) - gammaln(2 * s + 1)
    return np.asarray(np.exp(log_c) * np.abs(np.cos(theta / 2.0)) ** (2 * s), dtype=np.float64)


def cos2s_moments(s: float) -> tuple[float, float]:
    """Return the normalised first/second circular moments ``(r1, r2)`` of a cos-2s spread."""
    return s / (s + 1.0), s * (s - 1.0) / ((s + 1.0) * (s + 2.0))


def simulate_surface_elevation(
    frequency: ArrayLike,
    spectral_density: ArrayLike,
    duration_s: float,
    sample_rate_hz: float,
    *,
    seed: int | None = None,
    random_amplitude: bool = True,
) -> tuple[FloatArray, FloatArray]:
    r"""Simulate a Gaussian surface-elevation record (a "virtual buoy") from :math:`S(f)`.

    Linear superposition of independent components,
    :math:`\eta(t) = \sum_i a_i \cos(2\pi f_i t + \phi_i)` with
    :math:`\phi_i \sim U[0, 2\pi)`.

    With ``random_amplitude=True`` (default) :math:`a_i` is Rayleigh distributed
    with :math:`E[a_i^2] = 2 S(f_i)\Delta f_i`, which yields a genuinely Gaussian
    process; the deterministic-amplitude alternative under-represents
    sampling variability of short records (Tucker 1993) and would make derived
    labels unrealistically stable.

    Returns
    -------
    t, eta
        Time [s] and elevation [m].
    """
    f = np.asarray(frequency, dtype=np.float64)
    S = np.asarray(spectral_density, dtype=np.float64)
    if f.shape != S.shape:
        raise ValueError("frequency and spectral_density must have the same shape")
    df = np.gradient(f)
    rng = np.random.default_rng(seed)
    variance = 2.0 * S * df
    if random_amplitude:
        amp = np.sqrt(variance / 2.0) * np.sqrt(-2.0 * np.log(1.0 - rng.random(f.size)))
        # E[amp^2] = variance: Rayleigh with sigma^2 = variance/2.
    else:
        amp = np.sqrt(variance)
    phase = rng.uniform(0.0, 2.0 * np.pi, f.size)
    t = np.arange(round(duration_s * sample_rate_hz)) / sample_rate_hz
    eta = (amp[None, :] * np.cos(2.0 * np.pi * f[None, :] * t[:, None] + phase[None, :])).sum(
        axis=1
    )
    return t, eta


def simulate_directional_field(
    frequency: ArrayLike,
    spectral_density: ArrayLike,
    direction_deg: ArrayLike,
    spreading: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    t: float | ArrayLike,
    *,
    depth: float | None = None,
    seed: int | None = None,
) -> FloatArray:
    r"""Simulate a linear, short-crested sea surface :math:`\eta(x, y, t)`.

    Components are placed on the ``frequency x direction`` grid with variance
    :math:`2 S(f) D(f,\theta)\Delta f \Delta\theta`. ``direction_deg`` follows
    the oceanographic *coming-from*, clockwise-from-north convention (as NDBC
    MWD); ``x`` is east and ``y`` north in metres.

    ``spreading`` is either ``D(theta)`` of shape ``(n_dir,)`` (frequency
    independent) or ``D(f, theta)`` of shape ``(n_freq, n_dir)``, in rad^-1.

    The output has shape ``(n_t, n_y, n_x)``. Intended for synthetic imagery and
    for testing spatial wave-field inference (project Level 7 task), not for
    engineering-grade phase-resolved simulation (no nonlinearity, no
    refraction).
    """
    f = np.asarray(frequency, dtype=np.float64)
    S = np.asarray(spectral_density, dtype=np.float64)
    th = np.deg2rad(np.asarray(direction_deg, dtype=np.float64))
    D = np.asarray(spreading, dtype=np.float64)
    if D.ndim == 1:
        D = np.broadcast_to(D, (f.size, th.size))
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    ts = np.atleast_1d(np.asarray(t, dtype=np.float64))

    df = np.gradient(f)
    dth = np.gradient(np.unwrap(th)) if th.size > 1 else np.array([2 * np.pi])
    var = 2.0 * S[:, None] * D * df[:, None] * np.abs(dth)[None, :]
    rng = np.random.default_rng(seed)
    amp = np.sqrt(var / 2.0) * np.sqrt(-2.0 * np.log(1.0 - rng.random(var.shape)))
    phase = rng.uniform(0.0, 2.0 * np.pi, var.shape)
    k = wavenumber(f, depth)
    # Propagation direction is opposite to the coming-from direction.
    kx = -k[:, None] * np.sin(th)[None, :]
    ky = -k[:, None] * np.cos(th)[None, :]
    omega = 2.0 * np.pi * f[:, None]

    X, Y = np.meshgrid(xs, ys)
    out = np.zeros((ts.size, ys.size, xs.size))
    amp_f, kx_f, ky_f = amp.ravel(), kx.ravel(), ky.ravel()
    om_f = np.broadcast_to(omega, amp.shape).ravel()
    ph_f = phase.ravel()
    keep = amp_f > 1e-6 * amp_f.max()
    for a, kxx, kyy, om, ph in zip(
        amp_f[keep], kx_f[keep], ky_f[keep], om_f[keep], ph_f[keep], strict=True
    ):
        spatial = kxx * X + kyy * Y + ph
        out += a * np.cos(spatial[None, :, :] - om * ts[:, None, None])
    return out
