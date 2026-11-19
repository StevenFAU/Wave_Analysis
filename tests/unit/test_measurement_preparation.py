"""Regression tests for measurement definitions and held-out scoring."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module


def test_peak_switch_and_rotation_invariant_width(scripts, monkeypatch):
    labels = scripts("measurement_labels")
    freq = np.round(labels.CENTERS + 1e-8, 3)
    n = len(freq)
    t = pd.date_range("2026-10-01", periods=2, freq="30min", tz="UTC")
    spectra = np.full((2, n), 0.001)
    spectra[0, 15] = 1
    spectra[1, 25] = 2
    summary = pd.DataFrame({"time_utc": t, "Hs_m": [1.0, 2.0]})
    strips = pd.DataFrame({"sample_id": ["strip"], "time_utc": [t[0] + pd.Timedelta("2min")]})

    def make_spectrum(rotation):
        den = pd.DataFrame(spectra, index=t + pd.Timedelta("10min"))
        den["sep"] = 0.1
        return pd.concat(
            {
                "swden": den,
                "swr1": pd.DataFrame(np.full((2, n), 0.7), index=den.index),
                "swdir": pd.DataFrame(np.full((2, n), rotation), index=den.index),
            },
            axis=1,
        )

    monkeypatch.setattr(labels, "spec_records", lambda *a: summary)
    results = []
    for yaw in [5, 240]:
        monkeypatch.setattr(
            labels, "spectral_table", lambda *a, yaw=yaw: (make_spectrum(yaw), freq)
        )
        df, _ = labels.station_labels(Path("."), "41004", strips, 20)
        r = df.iloc[0]
        assert r.tp_nearest_s == pytest.approx(1 / freq[15])
        assert r.tp_window_spectrum_s == pytest.approx(1 / freq[25])
        assert r.n_complete_spectra == 2
        assert r.wind_r1 == pytest.approx(0.7)
        assert r.slope_finite_depth >= r.slope_rms
        assert r[labels.BANDS].sum() == pytest.approx(spectra.mean(0) @ labels.WIDTHS)
        results.append(r.wind_spread_deg)
    assert results[0] == pytest.approx(results[1])


def test_era5_preserves_land_and_partial_band(scripts, tmp_path):
    xr = pytest.importorskip("xarray")
    module = scripts("measurement_era5_spectrum")
    times = pd.date_range("2026-10-01", periods=3, freq="h")
    arr = np.zeros((3, 30, 24, 1, 1))
    arr[2] = np.nan
    grid = xr.DataArray(
        arr,
        coords={
            "time": times,
            "frequency": np.arange(1, 31),
            "direction": np.arange(1, 25),
            "latitude": [1.0],
            "longitude": [2.0],
        },
        dims=["time", "frequency", "direction", "latitude", "longitude"],
        name="d2fd",
    )
    path = tmp_path / "spectra.nc"
    grid.to_netcdf(path)
    meta = pd.Series({"station_id": "x", "grid_latitude": 1.0, "grid_longitude": 2.0})
    lab = pd.DataFrame(
        {
            "station_id": "x",
            "time_utc": pd.to_datetime(["2026-10-01T00:30Z", "2026-10-01T02:00Z"]),
            "sample_id": ["sea", "land"],
        }
    )
    out = module.decode(path, meta, lab)
    assert out.sample_id.tolist() == ["sea"]
    assert "prior_band_0_m2" not in out
    assert out.prior_band_1_m2.iloc[0] == pytest.approx(2 * np.pi * (0.155 - 0.095))


def test_crest_matching_and_fold_training(scripts, tmp_path):
    pytest.importorskip("torch")
    m = scripts("measurement_period")
    assert m.event_scores(np.array([5.0, 12.0, 20.0]), np.array([5.25, 12.0, 18.0, 20.0]), 0.5) == {
        "hits": 3,
        "observed": 3,
        "called": 4,
        "interval_abs_error_sum_s": 0.25,
        "matched_intervals": 1,
    }
    for i in [0, 1]:
        x = np.sin(np.arange(192)[:, None] / 4 + np.arange(64)[None, :] / 64).astype(np.float32)
        np.save(tmp_path / f"clip{i}.npy", x)
    index = pd.DataFrame(
        {
            "clip_id": ["clip0", "clip1"],
            "local_day": ["2026-11-01", "2026-11-03"],
            "camera": ["x", "y"],
        }
    )
    index.to_csv(tmp_path / "index.csv", index=False)
    ann = index[["clip_id"]].assign(reader="A", start_s=0, end_s=48, crest_seconds="8 16 24 32 40")
    ann.to_csv(tmp_path / "annotations.csv", index=False)
    from argparse import Namespace

    out = m.crest(
        Namespace(
            stacks=tmp_path,
            annotations=tmp_path / "annotations.csv",
            reader="A",
            split="date",
            device="cpu",
            epochs=1,
            tolerance=0.5,
        )
    )
    assert len(out) == 4 and set(out.arm) == {"cnn", "detector"}


def test_period_cache_joins_ids(scripts, tmp_path):
    pytest.importorskip("torch")
    from argparse import Namespace

    m = scripts("measurement_period")
    ids = ["a", "b", "c", "d"]
    clips = pd.DataFrame(
        {
            "clip_id": ids,
            "local_day": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "camera": ["a", "b", "a", "b"],
            "Tm02_label_s": [4.0, 5.0, 6.0, 7.0],
        }
    )
    clips.to_csv(tmp_path / "clips.csv", index=False)
    pixel = clips[["clip_id"]].copy()
    for col in ["log_T_pix", "log_T_m01", "log_T_m02", "rate_share", "log10_band_power", "T_m02_s"]:
        pixel[col] = [3.0, 5.0, 4.0, 6.0]
    pixel.to_csv(tmp_path / "pixel_spectrum.csv", index=False)
    cache = tmp_path / "features"
    cache.mkdir()
    clips.iloc[::-1].to_csv(cache / "clips.csv", index=False)
    for arm in ["video", "still", "shuffled"]:
        np.save(cache / (arm + ".npy"), np.ones((4, 2, 2, 3, 5), np.float16))
    out = m.tm02(Namespace(package=tmp_path, features=cache, split="site"))
    assert out.fold.nunique() == 2 and np.isfinite(out["video"]).all()
    assert out.groupby("camera").fold.nunique().eq(1).all()


def test_stereo_propagating_wave(scripts):
    m = scripts("measurement_stereo")
    side = m.SIDE
    x = np.arange(side)[:, None, None] * m.DX
    t = np.arange(100)[None, None, :] / 5
    wave = np.broadcast_to(np.cos(2 * np.pi * (x / 4 - t / 5)), (side, side, 100))
    values, power = m.surface_features(wave, np.ones(wave.shape, bool), 5)
    assert values == pytest.approx([4, 5, 0.8])
    assert power.sum() == pytest.approx(1)


def test_s3_controls_use_matching_oof_records(scripts, tmp_path):
    from argparse import Namespace

    m = scripts("measurement_review")
    frame = pd.DataFrame(
        {
            "sample_id": [f"id-{i}" for i in range(10)],
            "latitude": 34,
            "longitude": -77,
            "sun_azimuth_deg": 160,
            "sun_elevation_deg": 35,
            "time_utc": pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"),
            "station_group": [f"g{i % 5}" for i in range(10)],
            "mwd_label_deg": 359,
            "wind_r1": 0.4,
            "tp_label_s": 6,
            "tp_nearest_s": 5,
            "tp_window_spectrum_s": 7,
            "tp_peak_range_s": 4,
        }
    )
    for k in range(5):
        frame[f"split_fold{k}"] = np.where(np.arange(10) % 5 == k, "test", "train")
    frame.to_csv(tmp_path / "labels.csv", index=False)
    pred = (
        frame[["sample_id"]]
        .iloc[::-1]
        .assign(pred_dir_deg=1, dir_agreement=0.4, pred_tp=6, era5_tp=7)
    )
    path = tmp_path / "oof.csv"
    pred.to_csv(path, index=False)
    args = Namespace(task="direction", package=tmp_path, oof=path)
    assert m.scores(args)["all"]["delta"] == pytest.approx(2)
    args.task = "peaks"
    assert m.scores(args)["switch_gt_3s"]["tp_nearest_s"]["delta"] == pytest.approx(-1)


def test_foam_fraction_on_held_out_water(scripts, tmp_path):
    from argparse import Namespace

    from PIL import Image

    views = []
    for i in range(3):
        mask = np.ones((20, 20), np.uint8)
        mask[:2] = 255
        mask[10:, :4] = 2
        rgb = np.where(mask == 2, 255, 60).astype(np.uint8)
        Image.fromarray(rgb).save(tmp_path / f"view{i}.png")
        Image.fromarray(mask).save(tmp_path / f"mask{i}.png")
        views.append(
            {
                "sample_id": str(i),
                "view": 0,
                "illumination": "day",
                "image": f"view{i}.png",
                "mask": f"mask{i}.png",
                "station_group": i,
            }
        )
    table = pd.DataFrame(views)
    table.to_csv(tmp_path / "views.csv", index=False)
    table.to_csv(tmp_path / "labels.csv", index=False)
    m = scripts("measurement_annotation")
    out = tmp_path / "probes.csv"
    m.foam_probe(Namespace(packet=tmp_path, labels=tmp_path / "labels.csv", out=out))
    result = pd.read_csv(out)
    assert len(result) == 3 and np.allclose(result.visible_foam_fraction, 40 / 360)
    assert np.isfinite(result.color_texture).all()
