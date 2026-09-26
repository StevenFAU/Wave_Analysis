"""Linear wave theory, zero-crossing analysis and parametric spectra against analytic results."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wave_analysis.physics import G
from wave_analysis.physics.dispersion import (
    deep_water_wavelength,
    group_velocity,
    iribarren_number,
    phase_velocity,
    wave_steepness,
    wavelength,
    wavenumber,
)
from wave_analysis.physics.spectra_models import (
    cos2s_moments,
    cos2s_spreading,
    jonswap,
    pierson_moskowitz,
    simulate_directional_field,
    simulate_surface_elevation,
)
from wave_analysis.physics.timeseries import zero_crossing_analysis

# ----------------------------------------------------------------------------- dispersion


def test_deep_water_limit():
    T = np.array([4.0, 8.0, 12.0])
    L = wavelength(1 / T)
    np.testing.assert_allclose(L, G * T**2 / (2 * np.pi), rtol=1e-12)
    np.testing.assert_allclose(L, 1.5613 * T**2, rtol=1e-3)  # textbook 1.56 T^2


def test_shallow_water_limit():
    h = 0.5
    f = 0.01  # kh << 1
    np.testing.assert_allclose(phase_velocity(f, h), np.sqrt(G * h), rtol=1e-3)
    np.testing.assert_allclose(group_velocity(f, h), np.sqrt(G * h), rtol=1e-3)


def test_deep_water_group_velocity_is_half_phase():
    f = 0.1
    np.testing.assert_allclose(group_velocity(f), 0.5 * phase_velocity(f), rtol=1e-12)


@settings(max_examples=200, deadline=None)
@given(
    f=st.floats(min_value=0.02, max_value=1.0),
    h=st.floats(min_value=0.2, max_value=5000.0),
)
def test_dispersion_residual_is_machine_precision(f, h):
    k = wavenumber(f, h)
    omega = 2 * np.pi * f
    residual = G * k * np.tanh(k * h) - omega**2
    assert abs(residual) <= 1e-9 * omega**2


def test_known_intermediate_depth_value():
    # Dean & Dalrymple (1991) style check: T = 10 s, h = 10 m -> L ~ 92.3 m.
    assert wavelength(0.1, 10.0) == pytest.approx(92.3, abs=0.2)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        wavenumber(0.0)
    with pytest.raises(ValueError):
        wavenumber(0.1, -1.0)


def test_steepness_and_iribarren():
    s = wave_steepness(2.0, 10.0)
    assert s == pytest.approx(2.0 / deep_water_wavelength(10.0), rel=1e-12)
    xi = iribarren_number(0.1, 1.0, 8.0)
    assert xi == pytest.approx(0.1 / np.sqrt(1.0 / deep_water_wavelength(8.0)))


# ----------------------------------------------------------------------------- spectra models


@pytest.mark.parametrize(("Hs", "Tp"), [(1.0, 8.0), (3.5, 12.0), (0.4, 4.0)])
def test_pierson_moskowitz_m0(Hs, Tp):
    f = np.linspace(0.005, 3.0, 60000)
    m0 = np.trapezoid(pierson_moskowitz(f, Hs, Tp), f)
    assert 4 * np.sqrt(m0) == pytest.approx(Hs, rel=2e-3)


def test_jonswap_normalised_and_peaked():
    f = np.linspace(0.02, 0.6, 2000)
    S = jonswap(f, 2.0, 10.0, gamma=3.3)
    assert 4 * np.sqrt(np.trapezoid(S, f)) == pytest.approx(2.0, rel=1e-10)
    assert f[np.argmax(S)] == pytest.approx(0.1, abs=1e-3)
    # gamma = 1 reduces to Pierson-Moskowitz shape
    pm = pierson_moskowitz(f, 2.0, 10.0)
    j1 = jonswap(f, 2.0, 10.0, gamma=1.0)
    np.testing.assert_allclose(j1 / j1.max(), pm / pm.max(), atol=1e-10)


@pytest.mark.parametrize("s", [1.0, 5.0, 20.0])
def test_cos2s_normalisation_and_moments(s):
    theta = np.arange(0, 360, 0.25)
    D = cos2s_spreading(theta, 45.0, s)
    th = np.deg2rad(theta)
    dth = np.deg2rad(0.25)
    assert np.sum(D) * dth == pytest.approx(1.0, rel=1e-6)
    a1 = np.sum(D * np.cos(th - np.deg2rad(45))) * dth
    a2 = np.sum(D * np.cos(2 * (th - np.deg2rad(45)))) * dth
    r1, r2 = cos2s_moments(s)
    assert a1 == pytest.approx(r1, rel=1e-6)
    assert a2 == pytest.approx(r2, rel=1e-6)


# ----------------------------------------------------------------------------- zero crossing


def test_zero_crossing_monochromatic():
    fs, T, H = 10.0, 8.0, 2.0
    t = np.arange(0, 800, 1 / fs)
    eta = (H / 2) * np.sin(2 * np.pi * t / T)
    zc = zero_crossing_analysis(eta, fs)
    # 100 periods; the crossing at t=0 is not detectable (no preceding sample), so 98 complete waves.
    assert zc.n_waves == 98
    assert zc.H_one_third == pytest.approx(H, rel=1e-3)
    assert zc.T_z == pytest.approx(T, rel=1e-6)  # interpolated crossings: no sample quantisation
    assert zc.H_max == pytest.approx(H, rel=1e-3)


def test_zero_crossing_matches_spectral_hs_statistically():
    """For a Gaussian sea H1/3 ~= 0.95 Hm0 (Goda 2010); check across many random records."""
    f = np.linspace(0.03, 0.5, 400)
    S = jonswap(f, 2.0, 10.0)
    ratios = []
    for seed in range(8):
        _, eta = simulate_surface_elevation(f, S, 3600, 2.0, seed=seed)
        zc = zero_crossing_analysis(eta, 2.0)
        ratios.append(zc.H_one_third / 2.0)
    assert 0.88 < np.mean(ratios) < 1.0


def test_zero_crossing_rejects_nan_and_short():
    with pytest.raises(ValueError, match="NaN"):
        zero_crossing_analysis([0.0, np.nan, 1.0, -1.0], 1.0)
    with pytest.raises(ValueError):
        zero_crossing_analysis([1.0, 2.0, 3.0], 1.0)


def test_down_crossing_option():
    t = np.arange(0, 100, 0.1)
    zc = zero_crossing_analysis(np.sin(2 * np.pi * t / 10), 10.0, crossing="down")
    assert zc.crossing == "down"
    assert zc.T_z == pytest.approx(10.0, rel=1e-6)


def test_simulated_variance_matches_spectrum():
    f = np.linspace(0.03, 0.5, 300)
    S = jonswap(f, 1.5, 9.0)
    vars_ = [np.var(simulate_surface_elevation(f, S, 7200, 2.0, seed=s)[1]) for s in range(10)]
    assert np.mean(vars_) == pytest.approx((1.5 / 4) ** 2, rel=0.08)


def test_directional_field_shape_and_variance():
    f = np.linspace(0.05, 0.3, 30)
    S = jonswap(f, 1.0, 8.0)
    theta = np.arange(0, 360, 10.0)
    D = cos2s_spreading(theta, 270.0, 10.0)
    x = np.linspace(0, 200, 21)
    eta = simulate_directional_field(f, S, theta, D, x, x, [0.0, 1.0], seed=1)
    assert eta.shape == (2, 21, 21)
    assert np.isfinite(eta).all()
