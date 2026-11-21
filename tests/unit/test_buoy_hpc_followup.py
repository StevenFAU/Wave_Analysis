"""Calendar and geographic boundaries, and identical rows for the two swell views."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import buoy_hpc_followup as hpc
from wave_analysis.datasets.splits import early_fit_late_test


def test_calendar_uses_only_early_fit_and_late_held_out_rows():
    roles = np.repeat(["train", "val", "test"], 4)
    times = ["2026-10-28T23:59Z", *hpc.DATES, "2026-11-01T01:00Z"] * 3
    df = pd.DataFrame({"time_utc": times, "role": roles, "station_group": roles})
    kept = early_fit_late_test(df, "role", *hpc.DATES)
    assert kept.index.tolist() == [0, 4, 10, 11]
    df.loc[8, "station_group"] = "train"
    with pytest.raises(ValueError, match="multiple split roles"):
        early_fit_late_test(df, "role", *hpc.DATES)


def test_swell_uses_identical_eligible_train_and_test_strips():
    role = np.array(["train"] * 4 + ["val"] + ["test"] * 4)
    lab = pd.DataFrame(
        {
            "sample_id": list("abcdefghi"),
            "wind_speed_mps": np.arange(9.0),
            "wind_gust_mps": np.arange(9.0) + 1,
            "station_id": "s",
            "station_group": "g",
            "cv_fold": 0,
            "band_0_m2": [1, 2, 3, 4, 1, 0, 1, 1, 0],
            "band_1_m2": [1] * 8 + [0],
            "band_2_m2": 0,
            "band_3_m2": 0,
        }
    ).set_index("sample_id")
    x = np.arange(18.0).reshape(9, 2)
    w = x.copy()
    w[[1, 6]] = np.nan
    out, coverage = hpc.swell_fold(lab, x, w, role)
    assert out.sample_id.tolist() == ["f", "h"]
    assert out.observed.tolist() == [0, 4]
    assert coverage == {"train_total": 4, "train_matched": 3, "test_total": 4, "test_matched": 2}
    lab.loc[["b", "e", "h"], "band_0_m2"] = 3000  # excluded training row, validation and test
    changed, _ = hpc.swell_fold(lab, x, w, role)
    pd.testing.assert_frame_equal(out.drop(columns="observed"), changed.drop(columns="observed"))


def test_feature_provenance_and_row_alignment(tmp_path):
    rows = pd.DataFrame({"sample_id": ["b", "a"], "role": ["test", "train"]})
    rows.to_csv(tmp_path / "feature_rows.csv", index=False)
    np.save(tmp_path / "features.npy", np.array([[8.0], [3.0]]))
    meta = {
        "fold": 0,
        "seed": 0,
        "backbone": hpc.BACKBONE,
        "smoke": False,
        "water_only": True,
        "saved_features": True,
        "package_sha256": "labels",
    }
    (tmp_path / "run.json").write_text(json.dumps(meta))
    labels = pd.DataFrame({"split_fold0": ["train", "test"]}, index=["a", "b"])
    features, _ = hpc.view_features(tmp_path, labels, 0, True, "labels")
    assert features.ravel().tolist() == [3.0, 8.0]
    with pytest.raises(ValueError, match="wrong feature run"):
        hpc.view_features(tmp_path, labels, 0, False, "labels")
    labels.loc["a", "split_fold0"] = "val"
    with pytest.raises(ValueError, match="sample ids or role"):
        hpc.view_features(tmp_path, labels, 0, True, "labels")


def test_heights_report_requires_later_test_predictions_from_every_fold(tmp_path, monkeypatch):
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"{k}-{day}",
                "station_id": f"4{k}",
                "station_group": f"g{k}",
                "cv_fold": k,
                "time_utc": stamp,
                "wind_wave_label_m": 1 + k / 10,
            }
            for k in range(5)
            for day, stamp in (("early", "2026-10-27T12:00Z"), ("late", "2026-11-01T12:00Z"))
        ]
    )
    for k in range(5):
        labels[f"split_fold{k}"] = np.select(
            [labels.cv_fold == k, labels.cv_fold == (k + 1) % 5], ["test", "val"], default="train"
        )
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)
    monkeypatch.setattr(hpc, "LABELS", labels_path)
    prep = tmp_path / "prep"
    prep.mkdir()
    ref = labels[labels.sample_id.str.endswith("late")][[*hpc.KEYS, "wind_wave_label_m"]].rename(
        columns={"wind_wave_label_m": "observed"}
    )
    ref.assign(cv_fold=ref.cv_fold.astype(str), forecast=ref.observed + 0.3).to_parquet(
        prep / "forecast.parquet", index=False
    )
    (prep / "summary.json").write_text(
        json.dumps(
            {
                "date_split": hpc.DATES,
                "eligibility_archive_sha256": hpc.ARCHIVE_SHA,
                "forecast_sha256": hpc.sha256_file(prep / "forecast.parquet"),
                "source_sha256": {"labels": hpc.sha256_file(labels_path)},
            }
        )
    )
    runs = []
    for k in range(5):
        run = tmp_path / f"fold{k}"
        run.mkdir()
        runs.append(run)
        settings = {"backbone": hpc.BACKBONE, "pretrained": True, "epochs": 10}
        meta = {
            "date_split": hpc.DATES,
            "split": run.name,
            "seed": 0,
            "aux": "none",
            "lp_epochs": 2,
            "backbone_lr_scale": 0.1,
            "settings": settings,
            "config": "ndbc_buoycam_heights_v2cv",
        }
        (run / "metrics.json").write_text(json.dumps({"run": meta}))
        role = f"split_fold{k}"
        expected = early_fit_late_test(labels, role, *hpc.DATES)
        selected = expected[expected[role].isin(["val", "test"])]
        selected.assign(
            partition=selected[role],
            true_wind_wave_m=selected.wind_wave_label_m,
            pred_wind_wave_m=selected.wind_wave_label_m,
        ).to_csv(run / "predictions.csv", index=False)
    dest = tmp_path / "result"
    dest.mkdir()
    result = hpc.heights(dest, prep, runs)
    assert result["strips"] == 5
    assert result["primary_delta_rmse_m"] == pytest.approx(-0.15)
    pred = pd.read_csv(runs[0] / "predictions.csv")
    pred[pred.partition == "val"].to_csv(runs[0] / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="camera partitions"):
        hpc.heights(dest, prep, runs)
