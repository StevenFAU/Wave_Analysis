"""Repeat historical stage-2 cross-fitting with the new forecast comparators."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from buoycam_partition_fusion import GATE, PartitionFusion, features

CAMERA = [
    "cam_hs",
    "cam_ww",
    "cam_sw",
    "cam_sigma_hs",
    "cam_sigma_ww",
    "usable_views",
    "view_grey",
    "view_contrast",
]


def degrade(test, cuts, seed):
    copy = test.copy()
    bins = np.searchsorted(cuts, test.cmp_hs)
    rng = np.random.default_rng(seed)
    for _, group in test.groupby(["station_id", bins]):
        idx = rng.permutation(group.index)
        copy.loc[group.index, CAMERA] = test.loc[idx, CAMERA].to_numpy()
    return copy


def diagnostic(frame, forecast, baseline):
    wide = forecast.pivot(index="sample_id", columns="target", values=baseline).rename(
        columns={"Hs_label_m": "cmp_hs", "wind_wave_label_m": "cmp_ww"}
    )
    d = frame.join(wide, on="sample_id").dropna(
        subset=["cmp_hs", "cmp_ww", "true_hs_m", "true_wind_wave_m", *GATE, *CAMERA]
    )
    pieces = []
    for k, test in d.groupby("cv_fold"):
        train = d[d.cv_fold != k]
        m = PartitionFusion(True, "comparator", transfer=True).fit(train)
        if not np.isfinite(m.theta).all():
            raise ValueError("invalid gate fit")
        cuts = np.quantile(train.cmp_hs, [0.25, 0.5, 0.75])
        intact = m.predict(test)
        degraded = np.mean([m.predict(degrade(test, cuts, s)) for s in range(3)], axis=0)
        for i, (truth, cmp, cam) in enumerate(
            [("true_hs_m", "cmp_hs", "cam_hs"), ("true_wind_wave_m", "cmp_ww", "cam_ww")]
        ):
            pieces.append(
                test[["sample_id", "station_group", "station_id", "cv_fold"]].assign(
                    target=truth,
                    baseline=baseline,
                    observed=test[truth],
                    camera=test[cam],
                    forecast=test[cmp],
                    fusion=intact[i],
                    permuted=degraded[i],
                    gate_converged=m.converged,
                )
            )
    return pd.concat(pieces, ignore_index=True)


def summarize(d):
    columns = ["forecast", "camera", "fusion", "permuted"]
    for key, g in d.groupby(["baseline", "target"]):
        se = g[columns].sub(g.observed, axis=0).pow(2).groupby(g.station_group)
        total, n = se.sum().to_numpy(), se.size().to_numpy()
        idx = np.random.default_rng(0).integers(len(n), size=(2000, len(n)))
        rms = np.sqrt(total.sum(0) / n.sum())
        boots = np.sqrt(total[idx].sum(1) / n[idx].sum(1)[:, None])
        print(
            key,
            len(g),
            len(n),
            dict(zip(columns, np.round(rms, 4), strict=True)),
            "fusion minus forecast 95%",
            np.quantile(boots[:, 2] - boots[:, 0], [0.025, 0.975]).round(4),
            "fusion minus permuted 95%",
            np.quantile(boots[:, 2] - boots[:, 3], [0.025, 0.975]).round(4),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["labels", "oof", "forecast", "out"]:
        p.add_argument("--" + n, required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    f = features(
        pd.read_csv(a.oof, dtype={"station_id": str}), pd.read_csv(a.labels, low_memory=False)
    )
    fc = pd.read_parquet(a.forecast)
    out = pd.concat([diagnostic(f, fc, b) for b in ["model", "model_wind"]], ignore_index=True)
    out.to_parquet(a.out, index=False)
    summarize(out)


if __name__ == "__main__":
    main()
