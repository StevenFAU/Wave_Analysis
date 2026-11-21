"""Count where strips leave the two buoy follow-up comparisons (docs/experiments/hpc_buoy_followup.md).

heights (A): for each fold, the camera's held-out test strips from the later window, those
with a wind-wave label, and those in the fixed scored population, from the five camera
runs and the scored predictions of `buoy_hpc_followup.py heights`.

swell (B): applies the eligibility masks of buoy_hpc_followup.swell_fold one cause at a
time, for each fold's training and test buoys.

Both read finished outputs only; they fit nothing and do not change the scored rows.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from buoy_hpc_followup import LABELS, PKGS, RUNS, read_labels, sha256_file, view_features
from measurement_visibility import wind_features


def heights(runs, scored):
    rows = pd.read_parquet(scored / "predictions.parquet")
    rows["cv_fold"] = rows.cv_fold.astype(str)
    times = pd.to_datetime(read_labels(LABELS).set_index("sample_id").time_utc, utc=True)
    end = pd.to_datetime(rows.time_utc, utc=True).max()
    out = []
    for run in runs:
        k = json.loads((run / "metrics.json").read_text())["run"]["split"].removeprefix("fold")
        pred = read_labels(run / "predictions.csv")
        test = pred[pred.partition == "test"]
        kept = rows[rows.cv_fold == k]
        if not set(kept.sample_id) <= set(test.sample_id):
            raise ValueError(f"scored strips outside the camera test partition: {run}")
        unscored = test[test.true_wind_wave_m.notna() & ~test.sample_id.isin(kept.sample_id)]
        after = int((times.loc[unscored.sample_id] > end).sum())
        out.append(
            {
                "fold": int(k),
                "run": run.name,
                "val_strips": int((pred.partition == "val").sum()),
                "test_strips": len(test),
                "test_buoys": test.station_id.nunique(),
                "test_with_wind_wave_label": int(test.true_wind_wave_m.notna().sum()),
                # labelled but unscored: after the eligibility archive ends, or within it
                # but lacking a forecast input or measured wind
                "unscored_after_archive_end": after,
                "unscored_missing_inputs": len(unscored) - after,
                "scored": len(kept),
                "scored_buoys": kept.station_id.nunique(),
                "scored_groups": kept.station_group.nunique(),
                "scored_first": kept.time_utc.min(),
                "scored_last": kept.time_utc.max(),
            }
        )
    df = pd.DataFrame(out).sort_values("fold")
    # Geographic folds share no buoy, so per-fold test buoys add up.
    total = {
        "fold": "all",
        **df.select_dtypes("number").drop(columns="fold").sum().to_dict(),
        "scored_buoys": rows.station_id.nunique(),
        "scored_groups": rows.station_group.nunique(),
        "scored_first": rows.time_utc.min(),
        "scored_last": rows.time_utc.max(),
    }
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def swell(water_dir, whole_dir):
    path = PKGS / "ndbc_buoycam_slope_v1/labels.csv"
    label_sha = sha256_file(path)
    labels = read_labels(path).set_index("sample_id")
    rows = []
    for k in range(5):
        lab = labels[labels[f"split_fold{k}"].isin(["train", "val", "test"])]
        whole, _ = view_features(whole_dir / f"fold{k}", labels, k, False, label_sha)
        water, _ = view_features(water_dir / f"fold{k}", labels, k, True, label_sha)
        band = lab[[f"band_{j}_m2" for j in range(4)]].to_numpy(float)
        spectrum = np.isfinite(band).all(1) & (band >= 0).all(1) & (band.sum(1) > 0)
        wind = np.isfinite(wind_features(lab)).all(1)
        whole_ok, water_ok = np.isfinite(whole).all(1), np.isfinite(water).all(1)
        kept = spectrum & wind & whole_ok & water_ok
        role = lab[f"split_fold{k}"].to_numpy()
        for r in ("train", "test"):
            m = role == r
            rows.append(
                {
                    "fold": k,
                    "role": r,
                    "strips": int(m.sum()),
                    "spectrum_ok": int((m & spectrum).sum()),
                    "wind_ok": int((m & wind).sum()),
                    "whole_view_ok": int((m & whole_ok).sum()),
                    "water_crop_ok": int((m & water_ok).sum()),
                    "lost_only_to_water_crop": int(
                        (m & spectrum & wind & whole_ok & ~water_ok).sum()
                    ),
                    "kept": int((m & kept).sum()),
                    "kept_zero_swell_band": int((m & kept & (band[:, 0] == 0)).sum()),
                }
            )
    df = pd.DataFrame(rows)
    total = df.groupby("role", sort=False).sum(numeric_only=True).drop(columns="fold")
    return pd.concat([df, total.reset_index().assign(fold="all")], ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="cmd", required=True)
    h = commands.add_parser("heights")
    h.add_argument("--runs", type=Path, nargs=5, required=True)
    h.add_argument("--scored", type=Path, default=RUNS / "buoy-heights-followup-v1")
    s = commands.add_parser("swell")
    s.add_argument("--water", type=Path, required=True)
    s.add_argument("--whole", type=Path, default=RUNS / "claims-inputs/t5-s0")
    for sub in (h, s):
        sub.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    df = heights(a.runs, a.scored) if a.cmd == "heights" else swell(a.water, a.whole)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
