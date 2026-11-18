"""The pieces of the shore-video probe that need no weights."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wave_analysis.models.video_probe import (  # noqa: E402
    AttentiveProbe,
    MeanProbe,
    arm_order,
    pool_tokens,
    square_tiles,
)


def test_square_tiles_cover_both_edges() -> None:
    f = np.arange(2 * 4 * 7 * 3, dtype=np.uint8).reshape(2, 4, 7, 3)
    t = square_tiles(f)
    assert t.shape == (2, 2, 4, 4, 3)
    assert np.array_equal(t[0], f[:, :, :4])
    assert np.array_equal(t[1], f[:, :, 3:])
    with pytest.raises(ValueError):
        square_tiles(f.transpose(0, 2, 1, 3))


def test_arm_orders() -> None:
    assert list(arm_order("video", 5, "c")) == [0, 1, 2, 3, 4]
    assert list(arm_order("still", 64, "c")) == [32] * 64
    s = arm_order("shuffled", 64, "clip-a")
    assert sorted(s) == list(range(64)) and list(s) != list(range(64))
    assert list(s) == list(arm_order("shuffled", 64, "clip-a"))  # fixed per clip
    assert list(s) != list(arm_order("shuffled", 64, "clip-b"))
    with pytest.raises(ValueError):
        arm_order("reversed", 4, "c")


def test_pool_tokens_averages_blocks() -> None:
    x = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(1, 32, 1)  # grid t=2, h=4, w=4
    p = pool_tokens(x, (2, 4, 4), (1, 2, 2))
    assert p.shape == (1, 4, 1)
    first = x.reshape(2, 4, 4)[:, :2, :2].mean()
    assert torch.isclose(p[0, 0, 0], first)
    with pytest.raises(ValueError):
        pool_tokens(x, (2, 4, 5), (1, 2, 2))


@pytest.mark.parametrize("cls", ["attentive", "mean"])
def test_heads_shapes(cls: str) -> None:
    x = torch.randn(3, 2, 4, 16, 32)  # clips, tiles, layers, tokens, dim
    head = (
        AttentiveProbe(4, 32, 2 * 16, 2, d=16, heads=2)
        if cls == "attentive"
        else MeanProbe(4, 32, 2, d=16)
    )
    assert head(x).shape == (3, 2)
