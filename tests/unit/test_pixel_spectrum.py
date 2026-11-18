"""The pixel spectrum of bar M5 on synthetic brightness series."""

from __future__ import annotations

import numpy as np

from wave_analysis.processing.pixel_spectrum import FEATURES, block_means, pixel_spectrum


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
