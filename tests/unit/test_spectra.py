"""Spectral container, bulk parameters, directional reconstruction and regridding."""

from __future__ import annotations

import numpy as np
import pytest

from wave_analysis.physics.spectra_models import cos2s_moments, cos2s_spreading, jonswap
from wave_analysis.processing.spectra import (
    FrequencySpectrum,
    bulk_parameters,
    directional_distribution,
    fourier_from_ndbc,
    frequency_bandwidths,
    frequency_bin_edges,
    ndbc_from_fourier,
    regrid_energy_conserving,
)


def _spec(Hs=2.0, Tp=10.0, n=3, directional=False, mean_dir=90.0, s=10.0):
    f = np.linspace(0.03, 0.5, 200)
    S = np.tile(jonswap(f, Hs, Tp), (n, 1))
    kw = {}
    if directional:
        r1, r2 = cos2s_moments(s)
        kw = {
            "alpha1": np.full_like(S, mean_dir),
            "alpha2": np.full_like(S, mean_dir),
            "r1": np.full_like(S, r1),
            "r2": np.full_like(S, r2),
        }
    t = np.array(
        ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"], dtype="datetime64[ns]"
    )[:n]
    return FrequencySpectrum(time=t, frequency=f, density=S, **kw)


def test_bin_edges_and_bandwidths():
    f = np.array([0.02, 0.0325, 0.0375, 0.1, 0.11])
    e = frequency_bin_edges(f)
    assert e.size == f.size + 1
    assert np.all(np.diff(e) > 0)
    np.testing.assert_allclose(frequency_bandwidths(f).sum(), e[-1] - e[0])
    with pytest.raises(ValueError):
        frequency_bin_edges([0.1, 0.05])


def test_bulk_parameters_recover_hs_and_tp():
    b = bulk_parameters(_spec())
    assert b["Hm0"].iloc[0] == pytest.approx(2.0, rel=5e-3)
    assert b["Tp"].iloc[0] == pytest.approx(10.0, rel=0.03)
    assert b["Tp_smooth"].iloc[0] == pytest.approx(10.0, rel=0.03)
    # Moment-period ordering is guaranteed for non-negative spectra: Tm02 <= Tm01 <= Tm-10.
    assert (b["Tm02"] <= b["Tm01"]).all()
    assert (b["Tm01"] <= b["Tm_10"]).all()
    assert b["steepness_p"].iloc[0] == pytest.approx(2 * np.pi * 2.0 / (9.80665 * 100), rel=0.06)


def test_missing_bin_makes_moment_nan():
    sp = _spec(n=1)
    dens = sp.density.copy()
    dens[0, 50] = np.nan
    sp2 = FrequencySpectrum(time=sp.time, frequency=sp.frequency, density=dens)
    assert np.isnan(sp2.moment(0)[0])
    assert np.isfinite(sp2.moment(0, fmax=sp.frequency[40])[0])


def test_directional_bulk_parameters():
    b = bulk_parameters(_spec(directional=True, mean_dir=300.0, s=10.0))
    assert b["Dm"].iloc[0] == pytest.approx(300.0, abs=1e-6)
    r1, _ = cos2s_moments(10.0)
    assert b["spread_m"].iloc[0] == pytest.approx(np.rad2deg(np.sqrt(2 * (1 - r1))), rel=1e-6)


def test_fourier_roundtrip():
    rng = np.random.default_rng(0)
    a1, a2 = rng.uniform(0, 360, 50), rng.uniform(0, 180, 50)
    r1, r2 = rng.uniform(0, 1, 50), rng.uniform(0, 1, 50)
    back = ndbc_from_fourier(*fourier_from_ndbc(a1, a2, r1, r2))
    np.testing.assert_allclose(back[0], a1, atol=1e-9)
    np.testing.assert_allclose(np.mod(back[1], 180), np.mod(a2, 180), atol=1e-9)
    np.testing.assert_allclose(back[2], r1, atol=1e-12)
    np.testing.assert_allclose(back[3], r2, atol=1e-12)


@pytest.mark.parametrize("method", ["fourier", "mem"])
@pytest.mark.parametrize("s", [2.0, 8.0, 30.0])
def test_directional_distribution_reproduces_input_moments(method, s):
    """Both estimators must be normalised and reproduce the four input moments."""
    theta = np.arange(0, 360, 1.0)
    r1, r2 = cos2s_moments(s)
    D = directional_distribution([200.0], [200.0], [r1], [r2], theta, method=method)[0]
    th = np.deg2rad(theta)
    dth = np.deg2rad(1.0)
    assert np.sum(D) * dth == pytest.approx(1.0, rel=1e-9)
    m1 = np.sum(D * np.exp(1j * th)) * dth
    m2 = np.sum(D * np.exp(2j * th)) * dth
    assert abs(m1) == pytest.approx(r1, rel=1e-3)
    assert abs(m2) == pytest.approx(r2, rel=1e-3)
    assert np.rad2deg(np.angle(m1)) % 360 == pytest.approx(200.0, abs=0.1)


def test_mem_is_nonnegative_and_sharper_than_fourier_for_narrow_spread():
    theta = np.arange(0, 360, 1.0)
    r1, r2 = cos2s_moments(40.0)
    Df = directional_distribution([0.0], [0.0], [r1], [r2], theta, method="fourier")[0]
    Dm = directional_distribution([0.0], [0.0], [r1], [r2], theta, method="mem")[0]
    assert Dm.min() >= 0
    assert Dm.max() > Df.max()
    truth = cos2s_spreading(theta, 0.0, 40.0)
    assert np.abs(Dm - truth).mean() < np.abs(Df - truth).mean()


def test_regrid_conserves_energy():
    f_src = np.linspace(0.03, 0.5, 48)
    S = jonswap(f_src, 1.5, 8.0)
    e_src = frequency_bin_edges(f_src)
    e_dst = np.linspace(e_src[0], e_src[-1], 21)
    out = regrid_energy_conserving(e_src, S, e_dst)
    assert np.sum(out * np.diff(e_dst)) == pytest.approx(np.sum(S * np.diff(e_src)), rel=1e-12)


def test_regrid_outside_range_is_nan_and_nan_propagates_locally():
    e_src = np.array([0.1, 0.2, 0.3, 0.4])
    S = np.array([[1.0, np.nan, 3.0]])
    out = regrid_energy_conserving(e_src, S, np.array([0.05, 0.1, 0.15, 0.35, 0.4]))
    assert np.isnan(out[0, 0])  # below source range
    assert out[0, 1] == pytest.approx(1.0)
    assert np.isnan(out[0, 2])  # overlaps the missing bin
    assert out[0, 3] == pytest.approx(3.0)


def test_container_shape_validation():
    t = np.array(["2024-01-01"], dtype="datetime64[ns]")
    with pytest.raises(ValueError):
        FrequencySpectrum(time=t, frequency=np.array([0.1, 0.2]), density=np.ones((1, 3)))
    with pytest.raises(ValueError):
        FrequencySpectrum(
            time=t, frequency=np.array([0.1, 0.2]), density=np.ones((1, 2)), r1=np.ones((2, 2))
        )


def test_long_frame_and_select():
    sp = _spec(directional=True)
    df = sp.to_long_frame()
    assert len(df) == sp.n_time * sp.frequency.size
    assert {"alpha1", "r2", "bandwidth_hz"} <= set(df.columns)
    sub = sp.select_time(np.array([True, False, True]))
    assert sub.n_time == 2 and sub.r1 is not None and sub.r1.shape[0] == 2
