"""Test 6's per-camera adaptation (scripts/sensing_site_adaptation.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import sensing_site_adaptation as sa  # noqa: E402


def _cameras(seed: int = 0):
    """Two cameras whose features follow the sea state alike; the second is offset and stretched."""
    rng = np.random.default_rng(seed)
    n = 2000
    era5 = rng.uniform(0.5, 4.0, n)
    camera = np.repeat(np.array(["a", "b"]), n // 2)
    x = np.column_stack([era5, rng.normal(size=n)]) + rng.normal(scale=0.1, size=(n, 2))
    x[camera == "b"] = 2.0 * x[camera == "b"] + 5.0
    return x, camera, era5, camera == "a"


def test_binned_adaptation_removes_the_offset_and_keeps_the_sea_state():
    x, camera, era5, train = _cameras()
    out, info = sa.adapt(x, camera, era5, train, "scaled")
    assert info["cameras"] == 2 and not info["unadapted"]
    b = camera == "b"
    # camera b now looks like camera a at the same sea state, and still follows it
    # offset from the sea state, as camera a shows it (a: feature 0 = ERA5 + noise)
    assert abs((out[b, 0] - era5[b]).mean() - (x[~b, 0] - era5[~b]).mean()) < 0.05
    assert np.corrcoef(out[b, 0], era5[b])[0, 1] > 0.9
    np.testing.assert_allclose(out[~b], x[~b], atol=0.05)  # the reference camera barely moves


def test_plain_standardisation_ignores_the_bins():
    x, camera, era5, train = _cameras()
    out, _ = sa.adapt(x, camera, era5, train, "plain")
    b = camera == "b"
    np.testing.assert_allclose(out[b].mean(0), x[~b].mean(0), atol=1e-8)
