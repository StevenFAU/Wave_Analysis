"""Breaker-type classification on VWBT_9000: feature cache, heads, metric and bars."""

import argparse
import importlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

SITES = ["SiteA", "SiteB", "SiteC", "SiteD", "SiteE", "SiteF"]
CLASSES = ["Spilling", "Plunging", "Surging"]


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module


def fake_split(per_site=6):
    """Six sites, per_site clips each (two days per site), as the official CSV lays it out."""
    clips = []
    for s, site in enumerate(SITES):
        for k in range(per_site):
            label = CLASSES[(k + s) % 3]
            day = 1 + k % 2
            ts = f"2024010{day}T{10 + k:02d}0000Z"
            clips.append(
                {
                    "filename": f"{label}-{s * 100 + k:04d}-cam{s}-{ts.replace('T', '').replace(':', '')}.mp4",
                    "site": site,
                    "station": f"ST{s}",
                    "database_camera": f"cam{s}",
                    "tensor_camera": f"Camera_{s + 1:02d}",
                    "true_label": label,
                    "y_true": CLASSES.index(label),
                    "timestamp_utc": f"2024-01-0{day}T{10 + k:02d}:00:00Z",
                }
            )
    clips = pd.DataFrame(clips)
    rows = []
    for f, held in enumerate(SITES, 1):
        for _, c in clips.iterrows():
            if c.site == held:
                split = "test"
            else:
                split = "val" if c.filename.split("-")[1].endswith(("4", "5")) else "train"
            rows.append(dict(fold=f, held_out_site=held, held_out_station="X", split=split, **c))
    return pd.DataFrame(rows)


def test_official_split_is_checked(scripts, tmp_path):
    bf = scripts("breaker_features")
    split = fake_split()
    split.to_csv(tmp_path / "ok.csv", index=False, encoding="utf-8-sig")
    assert len(bf.official_split(tmp_path / "ok.csv")) == 6 * 36
    bad = split.copy()
    leak = bad.index[(bad.fold == 1) & (bad.site == "SiteA")][0]
    bad.loc[leak, "split"] = "train"  # the held-out site seen in training
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        bf.official_split(tmp_path / "bad.csv")
    wrong = split.copy()
    wrong.loc[0, "true_label"] = (
        "Surging" if wrong.loc[0, "true_label"] != "Surging" else "Spilling"
    )
    wrong.to_csv(tmp_path / "wrong.csv", index=False)
    with pytest.raises(ValueError, match="label"):
        bf.official_split(tmp_path / "wrong.csv")


def test_set_b_frames_found_by_tensor_camera_and_sorted_numerically(scripts):
    bf = scripts("breaker_features")
    clips = bf.clip_table(fake_split())
    first = clips.iloc[0]
    assert first.key_b.split("-")[2].startswith("Camera_")
    names = [f"Set_B/{k}/frame_{i:06d}.jpg" for k in clips.key_b for i in range(1, 61)][::-1]
    names.append(f"Set_B/{clips.key_b[1]}/frame_000061.jpg")  # as two v5 folders
    members, found = bf.set_b_members(names, clips)
    assert [bf.frame_number(n) for n in members[0]] == list(range(1, 61))
    assert [bf.frame_number(n) for n in members[1]] == list(range(1, 61))
    assert found[:3] == [60, 61, 60]
    by_name = [
        f"Set_B/{f.removesuffix('.mp4')}/frame_{i:06d}.jpg"
        for f in clips.filename
        for i in range(1, 61)
    ]
    assert bf.set_b_members(by_name, clips)[0][0][0].endswith(
        clips.filename[0][:-4] + "/frame_000001.jpg"
    )
    with pytest.raises(ValueError, match="no Set B folder"):
        bf.set_b_members(names[60:], clips.iloc[::-1].reset_index(drop=True))
    with pytest.raises(ValueError, match="expected 60"):
        bf.set_b_members(names[1:], clips)


def test_fit_length(scripts):
    bf = scripts("breaker_features")
    f = np.arange(4)[:, None, None, None] * np.ones((4, 2, 2, 3), np.uint8)
    assert bf.fit_length(f, 6)[:, 0, 0, 0].tolist() == [0, 1, 2, 3, 3, 3]
    assert bf.fit_length(f, 3)[:, 0, 0, 0].tolist() == [0, 1, 2]


def test_set_a_decodes_every_frame_once(scripts, tmp_path):
    cv2 = pytest.importorskip("cv2")
    bf = scripts("breaker_features")
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (224, 224))
    if not w.isOpened():
        pytest.skip("no MPEG-4 writer in this OpenCV")
    for i in range(150):
        w.write(np.full((224, 224, 3), i, np.uint8))
    w.release()
    with zipfile.ZipFile(tmp_path / "Set_A.zip", "w") as z:
        z.write(path, "Set_A/clip.mp4")
    x, i, n, fps = bf.SetA(tmp_path / "Set_A.zip", ["Set_A/clip.mp4"], str(tmp_path))[0]
    assert x.shape == (150, 3, 224, 224) and x.dtype == torch.uint8
    assert n == 150 and fps == pytest.approx(5.0)
    assert x[0].float().mean() < x[-1].float().mean()  # frames in order
    assert list(tmp_path.glob("tmp*.mp4")) == []  # the decoder's copy is removed


def test_macro_f1_over_present_classes(scripts):
    hb = scripts("train_breaker_heads")
    y = np.array([2, 2, 2, 2, 0])
    p = np.array([2, 2, 1, 2, 2])
    conf = hb.confusion(y, p)
    # Surging: tp 3, fp 1, fn 1 -> 0.75; Spilling: tp 0 -> 0; Plunging absent: not averaged
    assert hb.macro_f1(conf) == pytest.approx((0.75 + 0.0) / 2)
    assert hb.macro_f1(np.stack([conf, np.eye(3) * 5])).tolist() == pytest.approx([0.375, 1.0])
    assert hb.accuracy(conf) == pytest.approx(0.6)


def test_orderfree_head_ignores_frame_order_and_temporal_does_not(scripts):
    hb = scripts("train_breaker_heads")
    torch.manual_seed(0)
    x = torch.randn(2, 30, 16)
    perm = torch.randperm(30)
    free = hb.ConvHead(16, kernel=1).eval()
    temporal = hb.ConvHead(16, kernel=5).eval()
    assert torch.allclose(free(x), free(x[:, perm]), atol=1e-5)
    assert not torch.allclose(temporal(x), temporal(x[:, perm]), atol=1e-4)
    assert temporal(x).shape == (2, 3)


def test_yin_crnn_matches_their_layers(scripts):
    hb = scripts("train_breaker_heads")
    m = hb.YinCRNN(512)
    assert m(torch.randn(3, 59, 512)).shape == (3, 3)
    lstm = 4 * 512 * (512 + 512 + 2) * 3  # three layers of 512, input 512
    fc = (512 * 1024 + 1024) + 2 * 1024 + (1024 * 768 + 768) + 2 * 768 + (768 * 512 + 512)
    dec = (512 * 256 + 256) + (256 * 3 + 3)
    assert sum(p.numel() for p in m.parameters()) == lstm + fc + dec


def test_yin_frames_follow_their_numbering(scripts):
    hb = scripts("train_breaker_heads")
    from_one = " ".join(f"frame_{i:06d}.jpg" for i in range(1, 61))
    from_zero = " ".join(f"frame_{i:06d}.jpg" for i in range(60))
    assert hb.yin_frames(from_one).tolist() == list(range(59))
    assert hb.yin_frames(from_zero).tolist() == list(range(1, 60))


def test_iribarren_and_battjes(scripts):
    hb = scripts("train_breaker_heads")
    clips = pd.DataFrame({"filename": ["a", "b"], "site": ["SiteC", "SiteD"]})
    era5 = pd.DataFrame(
        {
            "filename": ["a", "b"],
            "wave_height": [1.0, 0.5],
            "wave_period": [6.0, 12.0],
            "wind_speed": [5.0, 2.0],
        }
    )
    e = hb.physics_features(clips, era5)
    l0 = 9.81 * 36 / (2 * np.pi)
    assert e.xi0[0] == pytest.approx(0.02 / np.sqrt(1.0 / l0))
    assert hb.battjes(np.array([0.2, 0.5, 3.3, 4.0])).tolist() == [0, 1, 1, 2]


def test_site_day_bootstrap_keeps_each_sites_day_count(scripts):
    hb = scripts("train_breaker_heads")
    clusters = pd.DataFrame({"site": ["A"] * 3 + ["B"] * 5, "site_day": list("abcdefgh")})
    w = hb.site_day_draws(clusters, n_draws=50, seed=1)
    assert (w[:, :3].sum(1) == 3).all() and (w[:, 3:].sum(1) == 5).all()
    assert (w == hb.site_day_draws(clusters, n_draws=50, seed=1)).all()


def test_scores_and_bars(scripts):
    hb = scripts("train_breaker_heads")
    rng = np.random.default_rng(0)
    n = 600
    site = np.array(SITES)[np.arange(n) % 6]
    day = rng.integers(0, 10, n)
    y = rng.integers(0, 3, n)
    rows = []
    for arm, acc in [("dinov3_temporal", 0.9), ("yin_crnn_b2hz", 0.5), ("dinov3_orderfree", 0.9)]:
        for seed in (0, 1):
            p = np.where(rng.random(n) < acc, y, rng.integers(0, 3, n))
            rows.append(
                pd.DataFrame(
                    {
                        "protocol": "loso",
                        "arm": arm,
                        "seed": seed,
                        "fold": 1,
                        "site": site,
                        "site_day": [f"{s} {d}" for s, d in zip(site, day, strict=True)],
                        "label": np.array(CLASSES)[y],
                        "pred": np.array(CLASSES)[p],
                    }
                )
            )
    out = hb.score_tables(pd.concat(rows), n_draws=200)
    b1 = out["comparisons"]["Y-B1"]
    # accuracy 0.93 against 0.67 with three balanced classes: macro-F1 about 0.27 higher
    assert b1["met"] and 0.2 < b1["pooled"]["delta"] < 0.35 and b1["pooled"]["ci95"][0] > 0
    assert not out["comparisons"]["Y-B2"]["met"]
    assert out["site_days"] == 60
    a = out["arms"]["dinov3_temporal"]
    assert sum(np.sum(v["confusion_summed_over_seeds"]) for v in a["per_site"].values()) == 2 * n


def test_fit_fold_learns_and_picks_the_best_validation_epoch(scripts):
    hb = scripts("train_breaker_heads")
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    y = np.arange(150) % 3
    x = rng.normal(size=(150, 20, 16)).astype(np.float32)
    x[:, :, :4] += 2.0 * y[:, None, None]
    fold = {
        "fold": 1,
        "test_site": "SiteA",
        "train": np.arange(90),
        "val": np.arange(90, 120),
        "test": np.arange(120, 150),
    }
    prob, best, hist = hb.fit_fold("dinov3_temporal", torch.tensor(x), torch.tensor(y), fold, 0, 15)
    assert (prob.argmax(1) == y[120:]).mean() > 0.9
    accs = [h["val_acc"] for h in hist]
    assert best == 1 + int(np.argmax(accs))  # the first epoch with the best validation accuracy


def test_fit_and_score_end_to_end_on_a_fake_cache(scripts, tmp_path):
    hb = scripts("train_breaker_heads")
    torch.set_num_threads(2)
    split = fake_split(per_site=8)
    src = tmp_path / "src"
    src.mkdir()
    split.to_csv(src / "loso_split_by_database_filename.csv", index=False, encoding="utf-8-sig")
    clips = scripts("breaker_features").clip_table(split)
    n = len(clips)
    pd.DataFrame(
        {
            "filename": clips.filename,
            "wave_height": 1.0,
            "wave_direction": 90.0,
            "wave_period": np.linspace(4, 10, n),
            "wind_speed": 5.0,
            "wind_direction": 0.0,
        }
    ).to_csv(src / "ERA5_offshore_wave_wind_by_clip.csv", index=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    clips["b_frames"] = " ".join(f"frame_{i:06d}.jpg" for i in range(1, 61))
    clips.to_csv(cache / "clips.csv", index=False)
    rng = np.random.default_rng(0)
    for name, t, d in [("a_dinov3", 150, 24), ("a_resnet18", 150, 8), ("b_resnet18", 60, 8)]:
        np.save(cache / f"{name}.npy", rng.normal(size=(n, t, d)).astype(np.float16))
    out = tmp_path / "run"
    args = argparse.Namespace(
        cache=cache,
        source=src,
        out=out,
        device="cpu",
        seeds=[0, 1],
        folds=[1, 4],
        arms=["yin_crnn_b2hz", "dinov3_temporal", "dinov3_orderfree", *hb.PHYSICS],
        random5=["dinov3_orderfree"],
        max_epochs=1,
    )
    hb.fit(args)
    pred = pd.read_csv(out / "predictions.csv")
    loso = pred[pred.protocol.eq("loso")]
    assert set(loso.site) == {
        "SiteA",
        "SiteD",
    }  # the two folds asked for: only their held-out sites
    assert (loso.fold.map(dict(enumerate(SITES, 1))) == loso.site).all()
    assert loso.groupby(["arm", "seed", "filename"]).size().eq(1).all()
    assert loso.groupby("arm").filename.nunique().eq(16).all()
    assert set(loso[loso.arm.eq("dinov3_temporal")].seed) == {0, 1}
    assert (
        pred[pred.protocol.eq("random5")].filename.nunique() == n
    )  # every clip once in Yin's protocol
    hb.score(argparse.Namespace(run=out, draws=50))
    s = json.loads((out / "score.json").read_text())
    assert {"Y-B1", "Y-B2"} <= set(s["comparisons"]) and s["clips"] == 16
    assert len(s["arms"]["dinov3_temporal"]["macro_f1_pooled_per_seed"]) == 2
    assert set(s["random5"]) == {"dinov3_orderfree"}
