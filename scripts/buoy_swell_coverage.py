"""Count why strips leave the swell-crop comparison (docs/experiments/hpc_buoy_followup.md, B).

Applies the same eligibility masks as buoy_hpc_followup.swell_fold, one cause at a time,
for each fold's training and test buoys. It reads only labels and cached features; it
fits nothing and does not change the scored rows.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from buoy_hpc_followup import PKGS, RUNS, read_labels, sha256_file, view_features
from measurement_visibility import wind_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--water", type=Path, required=True)
    p.add_argument("--whole", type=Path, default=RUNS / "claims-inputs/t5-s0")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    path = PKGS / "ndbc_buoycam_slope_v1/labels.csv"
    label_sha = sha256_file(path)
    labels = read_labels(path).set_index("sample_id")
    rows = []
    for k in range(5):
        lab = labels[labels[f"split_fold{k}"].isin(["train", "val", "test"])]
        whole, _ = view_features(a.whole / f"fold{k}", labels, k, False, label_sha)
        water, _ = view_features(a.water / f"fold{k}", labels, k, True, label_sha)
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
    df = pd.concat([df, total.reset_index().assign(fold="all")], ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
