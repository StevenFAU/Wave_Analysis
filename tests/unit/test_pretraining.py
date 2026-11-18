"""Domain pretraining helpers (wave_analysis.pretraining): buoy views and DINOv3 weight names."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from wave_analysis.pretraining.buoy_views import BuoyViews, ViewMetadata, crop_view, view_table
from wave_analysis.pretraining.dinov3_weights import (
    dinov3_to_timm,
    teacher_checkpoint,
    timm_to_dinov3,
)


def _package(tmp_path, n_views: int = 6, w: int = 48, h: int = 10):
    """Three strips: two training (stations B, A), one test; each view filled with its index."""
    rows = []
    for i, (station, part) in enumerate([("B", "train"), ("A", "train"), ("C", "test")]):
        arr = np.zeros((h, w * n_views, 3), np.uint8)
        for v in range(n_views):
            arr[:, v * w : (v + 1) * w] = 30 + 10 * v
        path = f"images/{i}.png"
        (tmp_path / "images").mkdir(exist_ok=True)
        Image.fromarray(arr).save(tmp_path / path)
        row = {"image": path, "station_id": station, "split_station": part, "Hs_label_m": 9.9}
        for v in range(n_views):
            usable = not (i == 0 and v == 5)  # strip 0's last view is dark
            row[f"view_mean_{v}"] = 30.0 + 10 * v if usable else 2.0
            row[f"view_std_{v}"] = 10.0 if usable else 0.1
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)
    return tmp_path


def test_view_table_keeps_usable_training_views(tmp_path):
    t = view_table(_package(tmp_path), "train")
    assert len(t) == 11  # 5 usable + 6 usable; the test strip is left out
    assert set(t["station_id"]) == {"A", "B"}
    assert "Hs_label_m" not in t.columns  # no label is read
    with pytest.raises(ValueError):
        view_table(tmp_path, "val")


def test_buoy_views_items_and_station_index(tmp_path):
    ds = BuoyViews(str(_package(tmp_path)), "train")
    assert ds.stations == ["A", "B"]
    image, (label, meta) = ds[0]
    assert label == 0 and isinstance(meta, ViewMetadata) and meta.station == 1  # strip 0 is B
    assert image.size == (48, 10)
    assert np.asarray(image)[0, 0, 0] == 30  # view 0
    image4, _ = ds[4]
    assert np.asarray(image4)[0, 0, 0] == 70  # view 4
    plain = BuoyViews(str(tmp_path), "train", with_metadata=False)
    assert plain[0][1] == 0


def test_crop_view_splits_strip_evenly():
    strip = Image.new("RGB", (60, 4))
    assert crop_view(strip, 5, n_views=6).size == (10, 4)


def _timm_like() -> dict[str, torch.Tensor]:
    return {
        "cls_token": torch.randn(1, 1, 8),
        "reg_token": torch.randn(1, 4, 8),
        "blocks.0.gamma_1": torch.randn(8),
        "blocks.0.gamma_2": torch.randn(8),
        "blocks.0.attn.qkv.weight": torch.randn(24, 8),
        "norm.weight": torch.randn(8),
    }


def test_weight_names_round_trip():
    sd = _timm_like()
    d = timm_to_dinov3(sd)
    assert {"storage_tokens", "blocks.0.ls1.gamma", "blocks.0.ls2.gamma"} <= set(d)
    ckpt = teacher_checkpoint(sd)
    teacher = dict(ckpt["teacher"])
    teacher["backbone.mask_token"] = torch.zeros(1, 8)
    teacher["backbone.rope_embed.periods"] = torch.ones(2)
    teacher["backbone.blocks.0.attn.qkv.bias"] = torch.zeros(24)
    teacher["dino_head.mlp.0.weight"] = torch.randn(4, 8)
    back = dinov3_to_timm(teacher)
    assert set(back) == set(sd)
    assert all(torch.equal(back[k], sd[k]) for k in sd)


def test_nonzero_qkv_bias_is_refused():
    teacher = {f"backbone.{k}": v for k, v in timm_to_dinov3(_timm_like()).items()}
    teacher["backbone.blocks.0.attn.qkv.bias"] = torch.ones(24)
    with pytest.raises(ValueError, match="qkv"):
        dinov3_to_timm(teacher)
