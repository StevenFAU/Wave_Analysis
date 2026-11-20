"""Test 3's video-half trainer end to end on a tiny synthetic package (CPU, one epoch)."""

import importlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

Image = pytest.importorskip("PIL.Image")

torch = pytest.importorskip("torch")

SPECTRUM = ["log_T_pix", "log_T_m01", "log_T_m02", "frac_25_16s", "frac_16_12s", "frac_12_9s",
            "frac_9_6s", "frac_6_3s", "log10_band_power", "rate_share"]  # fmt: skip


def _package(tmp: Path, n_days: int = 15, per_day: int = 2, k: int = 12, dim: int = 32):
    rng = np.random.default_rng(0)
    rows, labs = [], []
    for d in range(n_days):
        hs_day = [1.4, 2.2, 3.0][d % 3]
        for i in range(per_day):
            cid = f"cam-2024-01-{d + 1:02d}-{i:06d}Z"
            hs = hs_day * (1 + 0.05 * rng.standard_normal())
            rows.append({"clip_id": cid, "camera": "cam", "local_day": f"2024-01-{d + 1:02d}", "fold": d % 5,
                         "K": k, "duration_s": 16.0 * k, "Hs_label_m": hs, "Tp_label_s": 9.0, "Dp_label_deg": 90.0,
                         "pred_hs_clip": hs, "pred_tp_clip": 9.0, "pred_hs_window": hs, "pred_hs_window_image": hs,
                         **{c: rng.standard_normal() for c in SPECTRUM}})  # fmt: skip
            labs.append({"clip_id": cid, **{f"{t}_{c}": v for t in ["good", "incl_unevaluated"] for c, v in
                         [("mean_height_m", 0.6 * hs), ("h_third_m", 0.92 * hs), ("h_tenth_m", 1.15 * hs),
                          ("tz_s", 6.0), ("wave_count", 250)]}})  # fmt: skip
            a = rng.standard_normal((k, 2, 4, 128, dim)).astype(np.float16)
            (tmp / "cache" / "cam").mkdir(parents=True, exist_ok=True)
            np.save(tmp / "cache" / "cam" / f"{cid}.video.npy", a)
            (tmp / "kim" / "cam").mkdir(parents=True, exist_ok=True)
            np.save(
                tmp / "kim" / "cam" / f"{cid}.npy",
                rng.standard_normal((90, 1024)).astype(np.float16),
            )
    labs = labs[1:]  # one clip without a per-wave label, as on real data (class column then float)
    (tmp / "pkg").mkdir()
    pd.DataFrame(rows).to_csv(tmp / "pkg" / "interim.csv", index=False)
    pd.DataFrame(labs).to_csv(tmp / "pkg" / "labels.csv", index=False)


def test_kim_frames(tmp_path):
    m = _module()
    z = tmp_path / "c.zip"
    with zipfile.ZipFile(z, "w") as f:
        for i in range(64 * 2):
            p = tmp_path / "f.jpg"
            Image.fromarray(np.full((256, 341, 3), i % 255, np.uint8)).save(p)
            f.write(p, f"{i:04d}.jpg")
    x = m.kim_frames(z, duration_s=40.0, k=2, seconds=20.0, side=224)
    assert x.shape == (40, 3, 224, 224)  # 20 s at 2 fps, centre square, three grey channels
    assert np.allclose(x[:, 0] * 0.229 + 0.485, x[:, 1] * 0.224 + 0.456, atol=1e-5)


def _module():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("sensing_shore_test3")


def test_test3_end_to_end(tmp_path):
    m = _module()
    _package(tmp_path)
    cfg = Path(__file__).resolve().parents[2] / "configs/experiments/sensing_shore_v1.yaml"
    text = (
        cfg.read_text()
        .replace("hidden: 2000", "hidden: 8")
        .replace("seeds: [0, 1, 2, 3, 4]", "seeds: [0]")
    )
    (tmp_path / "cfg.yaml").write_text(text)
    rc = m.main(["--config", str(tmp_path / "cfg.yaml"), "--package", str(tmp_path / "pkg"),
                 "--out", str(tmp_path / "out"), "--windows", str(tmp_path / "zips"),
                 "--window-cache", str(tmp_path / "cache"), "--kim-cache", str(tmp_path / "kim"),
                 "--epochs", "1", "--device", "cpu", "--num-workers", "0"])  # fmt: skip
    assert rc == 0
    res = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert res["kamagata"]["n_clips"] == 30 and "T3-S verdict" in res["kamagata"]
    assert res["kim"]["n_clips"] == 29 and res["kim"]["assessable"]
    p = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert p["kamagata_pred_hs_clip"].gt(0).all() and p["kamagata_pred_hs_window"].gt(0).all()
