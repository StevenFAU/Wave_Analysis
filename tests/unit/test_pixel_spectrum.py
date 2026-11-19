"""The pixel spectrum of bar M5 on synthetic brightness series."""

from __future__ import annotations

import numpy as np

from wave_analysis.processing.pixel_spectrum import (
    BREAKING_FEATURES,
    FEATURES,
    FEATURES_V2,
    block_means,
    coherence_spectrum,
    otsu,
    physics_v2,
    pixel_spectrum,
)


def _series(period_s: float, n: int = 2368, seed: int = 0) -> np.ndarray:
    """1,000 blocks: 300 follow waves of one period over a red background, the rest noise or constant."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 4.0
    x = rng.normal(0, 3, (n, 1000))
    x[:, :600] += np.cumsum(rng.normal(0, 0.5, (n, 600)), axis=0)  # slow drift (clouds, exposure)
    x[:, :300] += 20 * np.sin(2 * np.pi * t[:, None] / period_s + rng.uniform(0, 6.3, 300))
    x[:, 900:] = 0.0  # a mask: no power, dropped
    return x


def test_peak_above_a_red_background() -> None:
    for period in (6.0, 9.3, 13.0):
        r = pixel_spectrum(_series(period))
        assert abs(r["features"]["T_pix_s"] - period) / period < 0.05
        assert r["n_live"] == 900 and r["n_water"] == 180
        assert set(FEATURES) <= set(r["features"])
        fr = [r["features"][k] for k in FEATURES if k.startswith("frac_")]
        assert np.isclose(sum(fr), 1.0, atol=0.05)


def test_water_is_the_wave_blocks() -> None:
    r = pixel_spectrum(_series(9.3))
    assert np.mean(np.array(r["water"]) < 300) > 0.95


def test_short_and_empty_series() -> None:
    r = pixel_spectrum(_series(9.3, n=64))  # one 16-s window: coarse, but defined
    assert np.isfinite(r["features"]["T_pix_s"])
    r = pixel_spectrum(np.zeros((256, 10)))
    assert r["n_live"] == 0 and np.isnan(r["features"]["T_pix_s"])


def test_block_means() -> None:
    g = np.arange(2 * 8 * 10, dtype=np.uint8).reshape(2, 8, 10)
    b = block_means(g)
    assert b.shape == (2, 2, 2)
    assert np.isclose(b[0, 0, 0], g[0, :4, :4].mean())


# --- recipe v2 (bars M6-M7) ---


def _view(period_s: float, n: int = 2368, seed: int = 0, exposure: bool = True) -> np.ndarray:
    """[T, 40, 60] blocks: sky (red drift) on top, a wave train crossing the sea below, breakers in the
    lowest rows (1-s rise, 2-s foam decay), all scaled by a slow auto-exposure gain."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 4.0
    x = 80 + rng.normal(0, 2, (n, 40, 60))
    x[:, :20] += np.cumsum(rng.normal(0, 0.3, (n, 20, 60)), axis=0)
    phase = 2 * np.pi * np.arange(60)[None, None, :] / 40 + 0.05 * np.arange(20)[None, :, None]
    x[:, 20:] += 15 * np.sin(2 * np.pi * t[:, None, None] / period_s - phase)
    ph = (t % period_s) / period_s  # each breaker: a 1-s rise, then foam decaying over 2 s
    rise = 1.0 / period_s
    x[:, 32:] += (
        60 * np.where(ph < rise, ph / rise, np.exp(-(ph - rise) * period_s / 2.0))[:, None, None]
    )
    if exposure:
        x *= (1 + 0.15 * np.sin(2 * np.pi * t / 200))[:, None, None]
    return x


def test_v2_periods_through_exposure_changes() -> None:
    for period in (6.0, 9.3, 13.0):
        f = physics_v2(_view(period))["features"]  # T_pix: the cleaned spectrum; T_on: the onsets
        assert abs(f["T_pix_s"] - period) / period < 0.05
        assert abs(f["T_on_s"] - period) / period < 0.05
        assert set(FEATURES_V2) <= set(f) and set(BREAKING_FEATURES) <= set(f)
        assert f["peak_prominence"] > 1.0


def test_coherence_of_a_wave_and_of_noise() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(2368) / 4.0
    fb = np.arange(6, 43) / 128
    noise = coherence_spectrum(rng.normal(size=(2368, 300)), fb)
    assert noise.max() < 0.2
    wave = np.sin(2 * np.pi * t[:, None] / 9.0 - rng.uniform(0, 6.3, 300)) + 0.3 * rng.normal(
        size=(2368, 300)
    )
    c = coherence_spectrum(wave, fb)
    assert c[np.argmin(np.abs(fb - 1 / 9.0))] > 0.8


def test_breaking_features() -> None:
    f = physics_v2(_view(8.0))["features"]
    assert 0 < f["foam_frac"] < 1 and 0 < f["foam_band_width"] <= 1
    assert 0 < f["onset_rate_per_min"] <= 60 / 8.0 + 1  # at most one onset per wave per block
    assert 0 < f["foam_persistence_s"] < 8.0
    assert 0.5 < f["foam_separability"] < 1
    calm = physics_v2(_view(8.0)[:, :32])["features"]  # no breaker rows: no persistent foam band
    assert calm["foam_band_width"] < f["foam_band_width"]


def test_otsu() -> None:
    rng = np.random.default_rng(2)
    v = np.concatenate([rng.normal(60, 5, 5000), rng.normal(200, 10, 1000)])
    thr, eta = otsu(v)  # any split of the gap is optimal; it must separate the two modes
    assert np.mean(v[:5000] < thr) > 0.99 and np.all(v[5000:] > thr)
    assert eta > 0.95
    _, eta1 = otsu(rng.normal(100, 10, 6000))
    assert abs(eta1 - 2 / np.pi) < 0.03


def test_v2_short_and_empty() -> None:
    assert np.isfinite(physics_v2(_view(9.3, n=64))["features"]["T_pix_s"])
    assert np.isnan(physics_v2(np.zeros((256, 4, 4)))["features"]["T_pix_s"])
