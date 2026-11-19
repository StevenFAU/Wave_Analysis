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


# ---------------------------------------------------------------- bar Y-B3: V-JEPA 2


def test_vjepa2_windows_cover_each_tubelet_once_from_the_nearest_window(scripts):
    bf = scripts("breaker_features")
    win, pos = bf.stitch_plan()
    assert len(win) == 75 and np.bincount(win).tolist() == [27, 21, 27]
    first = [bf.VJ_STARTS[w] + 2 * p for w, p in zip(win, pos, strict=True)]
    assert first == list(range(0, 150, 2))  # every 2-frame tubelet once, in order
    assert pos.min() >= 0 and pos.max() <= 31
    centres = np.array(bf.VJ_STARTS) + 31.5
    for k, w in enumerate(win):  # the nearest window centre, ties to the earlier window
        d = np.abs(centres - (2 * k + 0.5))
        assert w == np.flatnonzero(d == d.min())[0]
    with pytest.raises(ValueError, match="gap"):
        bf.stitch_plan(starts=(0, 70, 86))
    with pytest.raises(ValueError, match="span"):
        bf.stitch_plan(starts=(0, 42, 84))
    with pytest.raises(ValueError, match="tubelet"):
        bf.stitch_plan(starts=(0, 43, 86))


def test_vjepa2_versions_keep_the_pixels_and_drop_the_order(scripts):
    bf = scripts("breaker_features")
    video = bf.version_frames("video", "a.mp4")
    assert video.shape == (3, 64) and (video == np.arange(64) + np.c_[[0, 42, 86]]).all()
    shuffled = bf.version_frames("shuffled", "a.mp4")
    assert (shuffled == bf.version_frames("shuffled", "a.mp4")).all()  # fixed per clip
    assert not (shuffled == bf.version_frames("shuffled", "b.mp4")).all()  # differs by clip
    perm = np.r_[shuffled[0], shuffled[1, 22:], shuffled[2, 20:]]  # one permutation of 150
    assert sorted(perm.tolist()) == list(range(150))
    assert (shuffled[1, :22] == shuffled[0, 42:]).all()  # windows cut from the same order
    still = bf.version_frames("still", "a.mp4")
    assert (still == np.c_[[32, 74, 118]]).all()  # each window's middle frame
    with pytest.raises(ValueError):
        bf.version_frames("reversed", "a.mp4")


def test_vjepa2_tokens_are_the_spatial_mean_of_the_encoder_output(scripts, monkeypatch):
    transformers = pytest.importorskip("transformers")
    bf = scripts("breaker_features")
    cfg = transformers.VJEPA2Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        crop_size=32,
        image_size=32,
        patch_size=16,
        tubelet_size=2,
        frames_per_clip=4,
        pred_hidden_size=16,
        pred_num_hidden_layers=1,
        pred_num_attention_heads=2,
        mlp_ratio=2,
    )
    torch.manual_seed(0)
    model = transformers.VJEPA2Model(cfg)
    monkeypatch.setattr(bf, "VJ_SIDE", 32)
    monkeypatch.setattr(bf, "VJ_WINDOW", 4)
    enc = bf.VJepa2("cpu", model=model)
    x = torch.randint(0, 256, (3, 4, 3, 24, 24), dtype=torch.uint8)
    tok = enc.tokens(x)
    assert tok.shape == (3, 2, 32)  # 4 frames -> 2 tubelets, one token each
    v = torch.nn.functional.interpolate(
        x.flatten(0, 1).float(), size=(32, 32), mode="bilinear", align_corners=False
    ).view(3, 4, 3, 32, 32)
    mean = torch.tensor(bf.MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(bf.STD).view(1, 1, 3, 1, 1)
    with torch.no_grad():
        out = model(pixel_values_videos=(v / 255 - mean) / std, skip_predictor=True)
    expect = out.last_hidden_state.view(3, 2, 4, 32).mean(2)  # (t, h, w) grid, h * w averaged
    assert torch.allclose(tok, expect, atol=1e-5)
    monkeypatch.setattr(bf, "VJ_SIDE", 48)
    with pytest.raises(ValueError, match="crop"):
        bf.VJepa2("cpu", model=model)


class _FrameMeanEncoder:
    """Stands in for V-JEPA 2: each tubelet's token is the mean pixel value of its frames."""

    tubelet, dim = 2, 2

    def __init__(self, device):
        pass

    def tokens(self, x):
        m = x.float().mean((2, 3, 4)).view(x.shape[0], -1, 2).mean(-1)
        return torch.stack([m, -m], -1)


class _FlatClips(torch.utils.data.Dataset):
    """Stands in for SetA: frame i of every clip is filled with the value i."""

    def __init__(self, zip_path, members, tmpdir):
        self.members = members

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        x = torch.arange(150, dtype=torch.uint8).view(150, 1, 1, 1).expand(150, 3, 4, 4)
        return x.contiguous(), i, 150, 5.0


def test_vjepa2_pass_stitches_each_version(scripts, monkeypatch, tmp_path):
    bf = scripts("breaker_features")
    monkeypatch.setattr(bf, "VJepa2", _FrameMeanEncoder)
    monkeypatch.setattr(bf, "SetA", _FlatClips)
    src = tmp_path / "src"
    src.mkdir()
    fake_split(per_site=2).to_csv(
        src / "loso_split_by_database_filename.csv", index=False, encoding="utf-8-sig"
    )
    (src / "Set_A.zip").write_bytes(b"")
    out = tmp_path / "vj"
    args = argparse.Namespace(
        source=src, out=out, device="cpu", batch=4, workers=0, tmp=None, limit=0, encoders="vjepa2"
    )
    bf.run(args)
    clips = pd.read_csv(out / "clips.csv")
    video = np.load(out / "a_vjepa2_video.npy")
    assert video.shape == (12, 75, 2) and video.dtype == np.float16
    assert (video[:, :, 0] == np.arange(75) * 2 + 0.5).all()  # frames 2k, 2k + 1 in order
    assert (video[:, :, 1] == -video[:, :, 0]).all()
    shuffled = np.load(out / "a_vjepa2_shuffled.npy")[:, :, 0]
    for i, name in enumerate(clips.filename):
        perm = scripts("wave_analysis.models.video_probe").arm_order("shuffled", 150, name)
        assert (shuffled[i] == (perm[0::2] + perm[1::2]) / 2).all()
    still = np.load(out / "a_vjepa2_still.npy")[0, :, 0]
    assert (still == np.repeat([32, 74, 118], [27, 21, 27])).all()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["stitch"]["tubelets_per_window"] == [27, 21, 27]
    assert meta["set_a"]["frames_decoded_values"] == [150] and all(meta["finite_sample"].values())


def _vjepa_cache(tmp_path, scripts, per_site=8):
    split = fake_split(per_site=per_site)
    src = tmp_path / "src"
    src.mkdir()
    split.to_csv(src / "loso_split_by_database_filename.csv", index=False, encoding="utf-8-sig")
    clips = scripts("breaker_features").clip_table(split)
    n = len(clips)
    pd.DataFrame(
        {
            "filename": clips.filename,
            "wave_height": 1.0,
            "wave_period": np.linspace(4, 10, n),
            "wind_speed": 5.0,
        }
    ).to_csv(src / "ERA5_offshore_wave_wind_by_clip.csv", index=False)
    cache = tmp_path / "vj"
    cache.mkdir()
    clips.to_csv(cache / "clips.csv", index=False)
    rng = np.random.default_rng(0)
    for v in ("video", "shuffled", "still"):
        np.save(cache / f"a_vjepa2_{v}.npy", rng.normal(size=(n, 75, 16)).astype(np.float16))
    return src, cache, n


def test_vjepa2_arms_fit_and_score_beside_another_run(scripts, tmp_path):
    hb = scripts("train_breaker_heads")
    torch.set_num_threads(2)
    src, cache, n = _vjepa_cache(tmp_path, scripts)
    vj = [a for a in hb.NEURAL if a.startswith("vjepa2")]
    assert hb.default_arms(cache) == [*vj, *hb.PHYSICS]  # no DINOv3 or ResNet in this cache
    assert hb.BARS["Y-B3"] == ("vjepa2_video", "vjepa2_shuffled")
    assert hb.NEURAL["vjepa2_video"]["kernel"] == hb.NEURAL["dinov3_temporal"]["kernel"]
    out = tmp_path / "run"
    common = {
        "source": src,
        "device": "cpu",
        "seeds": [0],
        "folds": [],
        "random5": [],
        "max_epochs": 1,
    }
    hb.fit(argparse.Namespace(cache=cache, out=out, arms=vj, **common))
    pred = pd.read_csv(out / "predictions.csv")
    assert set(pred.arm) == set(vj) and pred.groupby("arm").filename.nunique().eq(n).all()
    # another run's arms (as Y-B1's) are added; arms this run has are not replaced
    other = pred[pred.arm.eq("vjepa2_video")].assign(arm="dinov3_temporal")
    dup = pred[pred.arm.eq("vjepa2_still")].assign(pred="Surging")
    pd.concat([other, dup]).to_csv(tmp_path / "other.csv", index=False)
    hb.score(argparse.Namespace(run=out, draws=50, with_=[tmp_path / "other.csv"]))
    s = json.loads((out / "score.json").read_text())
    assert list(s["merged_from"].values()) == [["dinov3_temporal"]]
    assert {"Y-B3", "vjepa2_video - dinov3_temporal", "vjepa2_video - vjepa2_still"} <= set(
        s["comparisons"]
    )
    assert s["comparisons"]["vjepa2_video - dinov3_temporal"]["pooled"]["delta"] == 0
    own = pred[pred.arm.eq("vjepa2_still")]  # not the other file's all-surging copy
    f1 = hb.macro_f1(hb.confusion(own.label.map(CLASSES.index), own.pred.map(CLASSES.index)))
    assert s["arms"]["vjepa2_still"]["macro_f1_pooled"] == pytest.approx(f1)
    assert isinstance(s["comparisons"]["Y-B3"]["met"], bool)


def test_scoring_refuses_arms_on_different_clips(scripts):
    hb = scripts("train_breaker_heads")
    base = pd.DataFrame(
        {
            "protocol": "loso",
            "seed": 0,
            "fold": 1,
            "site": "SiteA",
            "site_day": ["SiteA 1", "SiteA 2"],
            "label": "Spilling",
            "pred": "Spilling",
        }
    )
    a = base.assign(arm="vjepa2_video", filename=["x", "y"])
    b = base.assign(arm="vjepa2_shuffled", filename=["x", "z"])
    with pytest.raises(ValueError, match="different clips"):
        hb.score_tables(pd.concat([a, b]), n_draws=10)
