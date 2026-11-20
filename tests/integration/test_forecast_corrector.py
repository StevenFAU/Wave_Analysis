"""Smoke test the registered forecast fit from its command-line entry point."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_forecast_corrector_predicts_held_groups(tmp_path):
    pytest.importorskip("sklearn", reason="requires the forecast dependency group")
    hours = pd.date_range("2026-10-01", periods=24, freq="h", tz="UTC")
    labels = pd.DataFrame(
        [
            (f"{station}-{i}", station, f"group-{station}", i_fold, timestamp)
            for i_fold, station in enumerate(("A", "B", "C"))
            for i, timestamp in enumerate(hours)
        ],
        columns=["sample_id", "station_id", "station_group", "cv_fold", "time_utc"],
    )
    matched = labels.assign(
        bulletin_present=True,
        latitude=30.0,
        longitude=-70.0,
        hs_m=1.0,
        systems=0.0,
        extra_systems=0.0,
    )
    for i in range(1, 7):
        for field in ("present", "wind", "hs_m", "tp_s", "direction_deg"):
            matched[f"system{i}_{field}"] = 0
    era = labels[["station_id", "time_utc"]].assign(
        era5_hs_m=1.0,
        era5_windsea_m=0.6,
        era5_swell_m=0.8,
    )
    labels["split_pair01"] = labels.station_id.map({"A": "train", "B": "val", "C": "test"})
    labels = labels.assign(
        Hs_label_m=1.1, wind_wave_label_m=0.55, wind_speed_mps=4.0, wind_gust_mps=6.0
    )
    # The CLI reads the timestamp from CSV and joins it to the saved GFS match.
    matched["time_utc"] = labels.time_utc.astype(str)
    labels.to_csv(tmp_path / "labels.csv", index=False)
    matched.to_parquet(tmp_path / "matched.parquet")
    era.to_parquet(tmp_path / "era.parquet")
    pd.DataFrame({"station_id": ["A", "B", "C"], "depth_m": [20.0, 30.0, 40.0]}).to_csv(
        tmp_path / "depths.csv", index=False
    )
    result = subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/sensing_forecast_corrector.py",
            "--labels",
            "labels.csv",
            "--matched",
            "matched.parquet",
            "--era",
            "era.parquet",
            "--depths",
            "depths.csv",
            "--splits",
            "pair01",
            "--out",
            "pred.parquet",
        ],
        cwd=tmp_path,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    preds = pd.read_parquet(tmp_path / "pred.parquet")
    assert set(preds.station_id) == {"C"}
    assert set(preds.target) == {"Hs_label_m", "wind_wave_label_m"}
    assert len(preds) == 2 * len(hours)
    assert np.isfinite(preds[["model", "model_wind", "era5_wind"]]).all().all()
