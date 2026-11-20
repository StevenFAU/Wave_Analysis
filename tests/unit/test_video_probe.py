"""The pieces of the shore-video probe that need no weights."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wave_analysis.models.video_probe import (  # noqa: E402
    AttentiveProbe,
    MeanProbe,
    RecordPooler,
    StackedProbes,
    arm_order,
    centred_windows,
    energy_mean,
    geometric_mean,
    midpoint_window,
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


def test_midpoint_window() -> None:
    assert midpoint_window(600.0, 37) == 18  # 300 s lies in window 18 (288-304 s)
    assert midpoint_window(20.65, 1) == 0
    assert midpoint_window(33.0, 2) == 1  # 16.5 s lies in the second window
    assert midpoint_window(600.0, 15) == 14  # fewer frames than the length: the last window
    with pytest.raises(ValueError):
        midpoint_window(10.0, 0)


def test_centred_windows() -> None:
    assert centred_windows(18, 37) == (13, 24)  # 11 windows, 176 s around the midpoint
    assert centred_windows(2, 37) == (0, 11)  # shifted inside the clip
    assert centred_windows(36, 37) == (26, 37)
    assert centred_windows(3, 7) == (0, 7)  # fewer than 11: all of them


def test_averages() -> None:
    h = np.array([1.0, 2.0, 2.0])
    assert np.isclose(energy_mean(h), np.sqrt(3.0))
    assert np.isclose(geometric_mean(np.array([2.0, 8.0])), 4.0)
    assert energy_mean(h) >= geometric_mean(h)


def test_record_pooler_ignores_padding() -> None:
    torch.manual_seed(0)
    head = RecordPooler(4, 32, 2, max_windows=8, d=16, heads=2, dropout=0.0).eval()
    x = torch.randn(1, 8, 2, 4, 32)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, :5] = True
    a = head(x, mask)
    x2 = x.clone()
    x2[0, 5:] = 99.0  # padded windows must not matter
    assert a.shape == (1, 2)
    assert torch.allclose(a, head(x2, mask), atol=1e-5)


def test_stacked_probes_match_separate_training() -> None:
    """Three probes trained in lock step equal three trained alone (AdamW + OneCycleLR)."""
    import copy

    def make(seed: int) -> AttentiveProbe:
        torch.manual_seed(seed)
        return AttentiveProbe(4, 16, 2 * 8, 2, d=16, heads=2, dropout=0.0)

    gen = torch.Generator().manual_seed(1)
    x = torch.randn(40, 2, 4, 8, 16, generator=gen)
    y = torch.randn(40, 2, generator=gen)
    # copy 0 steps every time, copy 1 skips steps 1 and 3, copy 2 sees a different batch size
    plan = [([0, 1, 2], 6), ([0, 2], 6), ([0, 1, 2], 6), ([0], 6), ([1, 2], 6), ([0, 1, 2], 6)]
    totals = [sum(1 for ids, _ in plan if c in ids) for c in range(3)]
    alone = [make(s) for s in (0, 1, 2)]
    stacked = StackedProbes([copy.deepcopy(m) for m in alone], lr=3e-3, weight_decay=0.05, total_steps=totals)
    opts = [torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.05) for m in alone]
    scheds = [torch.optim.lr_scheduler.OneCycleLR(o, max_lr=3e-3, total_steps=t, pct_start=0.1)
              for o, t in zip(opts, totals, strict=True)]  # fmt: skip
    for k, (ids, bs) in enumerate(plan):
        rows = [torch.arange(c * 5 + k, c * 5 + k + bs) % 40 for c in ids]
        for c, r in zip(ids, rows, strict=True):
            alone[c].train()
            loss = torch.nn.functional.smooth_l1_loss(alone[c](x[r]), y[r])
            opts[c].zero_grad()
            loss.backward()
            opts[c].step()
            scheds[c].step()
        stacked.step(ids, torch.stack([x[r] for r in rows]), torch.stack([y[r] for r in rows]))
    for c in range(3):
        mine = stacked.state(c)
        for k, v in alone[c].state_dict().items():
            a, b = mine[k], v
            if k.endswith("in_proj_bias"):  # the key bias has no gradient (softmax ignores a shift):
                a, b = torch.cat([a[:16], a[32:]]), torch.cat([b[:16], b[32:]])  # Adam steps on rounding
            assert torch.allclose(a, b, atol=1e-6), (c, k)
        alone[c].eval()
    out = stacked.predict([0, 1, 2], x[:5])
    for c in range(3):
        assert torch.allclose(out[c], alone[c](x[:5]), atol=1e-5)
