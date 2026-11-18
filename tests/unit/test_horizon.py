"""Sea-horizon fit in one view (wave_analysis.processing.horizon)."""

from __future__ import annotations

import numpy as np
import pytest

from wave_analysis.processing.horizon import detect_horizon


def _view(
    angle_deg: float, row: float, *, h: int = 270, w: int = 480, glare: bool = False
) -> np.ndarray:
    """Bright sky over a darker, textured sea, split by a straight tilted line."""
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:h, 0:w]
    line = row + np.tan(np.radians(angle_deg)) * (x - w / 2)
    img = np.where(y < line, 190.0, 70.0) + rng.normal(0, 6, (h, w))
    if glare:  # a bright glint column on the sea, as from a low sun
        img[:, 200:260] = np.where(y[:, 200:260] >= line[:, 200:260], 250.0, img[:, 200:260])
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("angle", [-6.0, -1.5, 0.0, 2.0, 7.0])
def test_recovers_angle_and_row(angle: float):
    f = detect_horizon(_view(angle, 140.0))
    assert f.ok()
    assert f.angle_deg == pytest.approx(angle, abs=0.3)
    assert f.row_mid == pytest.approx(140.0, abs=1.5)


def test_glare_columns_are_outliers():
    f = detect_horizon(_view(3.0, 120.0, glare=True))
    assert f.ok()
    assert f.angle_deg == pytest.approx(3.0, abs=0.4)
    assert f.inlier_share < 1.0


def test_no_horizon_is_rejected():
    flat = np.random.default_rng(0).normal(100, 6, (270, 480)).clip(0, 255).astype(np.uint8)
    assert not detect_horizon(flat).ok()
