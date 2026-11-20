"""Read-only checks for the wave-measurement research report. Run from any directory."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = (
    Path(os.getenv("WAVE_ANALYSIS_DATA", Path(__file__).resolve().parents[1] / "data"))
    / "processed"
)
BUOY = PROCESSED / "packages" / "ndbc_buoycam_spectral_v3cv"
SHORE = PROCESSED / "packages" / "webcoos_video_v1"
HORIZON = PROCESSED / "horizon" / "ndbc_buoycam_sea_state_v0"


def slope_pilot(labels: pd.DataFrame) -> None:
    """Use the original training partition only, with its existing geographic folds."""
    labels = labels.loc[
        labels["split_station"].eq("train"),
        ["sample_id", "station_id", "station_group", "cv_fold", "wind_speed_mps"],
    ]
    views = pd.read_parquet(HORIZON / "horizons.parquet").query("ok").copy()
    views["roll_sq"] = views["angle_deg"] ** 2
    geometry = views.groupby("sample_id").agg(
        n=("view", "size"), roll=("roll_sq", "mean"), row_sd=("row_frac", "std")
    )
    geometry = geometry.query("n >= 3").copy()
    geometry["roll"] = np.sqrt(geometry["roll"])
    target = pd.read_parquet(HORIZON / "strips.parquet", columns=["sample_id", "slope_rms"])
    data = (
        labels.merge(geometry, on="sample_id", validate="1:1")
        .merge(target, on="sample_id", validate="1:1")
        .dropna()
    )
    data["wind_sq"] = data["wind_speed_mps"] ** 2
    wind = ["wind_speed_mps", "wind_sq"]
    features = {
        "mean": [],
        "geometry": ["roll", "row_sd"],
        "linear_wind": ["wind_speed_mps"],
        "quadratic_wind": wind,
        "quadratic_wind_geometry": [*wind, "roll", "row_sd"],
    }
    print(
        "Slope pilot rows, stations, groups:",
        len(data),
        data.station_id.nunique(),
        data.station_group.nunique(),
    )
    assert data.groupby("station_group").cv_fold.nunique().eq(1).all()
    y = data["slope_rms"].to_numpy()
    losses = pd.DataFrame(index=data.index)
    for name, columns in features.items():
        design = np.column_stack((np.ones(len(data)), data[columns].to_numpy()))
        prediction = np.empty(len(data))
        for fold in sorted(data.cv_fold.unique()):
            train = data.cv_fold.ne(fold).to_numpy()
            coef = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
            prediction[~train] = design[~train] @ coef
        losses[name] = (prediction - y) ** 2
    print("Out-of-fold RMS slope RMSE:", np.sqrt(losses.mean()).to_dict())
    grouped = losses.groupby(data.station_group)
    n = grouped.size().to_numpy()
    draws = np.random.default_rng(0).integers(len(n), size=(2000, len(n)))
    totals = grouped.sum()

    def bootstrap_rmse(name: str) -> np.ndarray:
        return np.sqrt(totals[name].to_numpy()[draws].sum(axis=1) / n[draws].sum(axis=1))

    delta = bootstrap_rmse("quadratic_wind_geometry") - bootstrap_rmse("quadratic_wind")
    print("Increment from geometry, paired 95% interval:", np.quantile(delta, [0.025, 0.975]))


def main() -> None:
    """Print package coverage, the peak-period grid check, and the development pilot."""
    labels = pd.read_csv(BUOY / "labels.csv", dtype={"station_id": str}, low_memory=False)
    frequencies = np.array(json.loads((BUOY / "frequencies.json").read_text())["frequency_hz"])
    peaks = labels["tp_label_s"].dropna().to_numpy()
    distance = np.abs(peaks[:, None] - 1 / frequencies).min(axis=1)
    print(
        "Total strips, finite peak labels, over 0.25 s from any grid period:",
        len(labels),
        len(peaks),
        int((distance > 0.25).sum()),
    )
    print("Label counts:", labels[["tm10_label_s", "wsf_20", "mwd_label_deg"]].count().to_dict())
    clips = pd.read_csv(SHORE / "clips.csv")
    windows = pd.read_csv(SHORE / "windows.csv")
    for name, rows in [("clips", clips), ("windowed", clips[clips.clip_id.isin(windows.clip_id)])]:
        print(
            name,
            "clips, local dates, camera-days:",
            len(rows),
            rows.local_day.nunique(),
            rows.camera_day.nunique(),
        )
    slope_pilot(labels)


if __name__ == "__main__":
    main()
