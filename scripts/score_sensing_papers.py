"""Paired sealed-station comparison for three seeds of a paper arm and DINOv3."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensemble(paths, reference):
    frames = []
    seeds = set()
    arm = None
    package = None
    for path in paths:
        data = pd.read_csv(path / "predictions.csv", dtype={"station_id": str})
        if reference:
            run = json.loads((path / "metrics.json").read_text())["run"]
            if (
                run["settings"]["backbone"] != "timm:convnext_tiny.dinov3_lvd1689m"
                or run["aux"] != "none"
                or run["config"] != "ndbc_buoycam_heights_v2"
                or run.get("limit_per_partition")
            ):
                raise ValueError("expected a full images-only DINOv3 height run")
            data = data[data.partition == "test"]
            cols = ["pred_hs_m"]
        else:
            run = json.loads((path / "settings.json").read_text())
            if (arm is not None and run["arm"] != arm) or (
                package is not None and run["labels_sha256"] != package
            ):
                raise ValueError("mixed paper arms or packages")
            arm, package = run["arm"], run["labels_sha256"]
            cols = (
                ["pred_hs_m"]
                if arm.startswith("yang")
                else sorted(data.filter(regex=r"^prob_\d+$").columns)
            )
        if run["split"] != "station" or run["seed"] in seeds:
            raise ValueError("expected independent sealed-station seeds")
        seeds.add(run["seed"])
        data = data.set_index("sample_id", verify_integrity=True).sort_index()
        if frames and not data.index.equals(frames[0].index):
            raise ValueError("seed coverage changed")
        if not reference and frames and not data.target.equals(frames[0].target):
            raise ValueError("seed labels changed")
        frames.append(data)
    if seeds != {0, 1, 2}:
        raise ValueError("seeds 0,1,2 required")
    merged = frames[0].copy()
    merged[cols] = np.mean([d[cols].to_numpy() for d in frames], axis=0)
    return arm, merged


def compare(paper, ref):
    arm, p = ensemble(paper, False)
    _, r = ensemble(ref, True)
    d = p.join(
        r[["true_hs_m", "pred_hs_m", "station_id"]].rename(
            columns={"pred_hs_m": "reference", "station_id": "reference_station"}
        ),
        how="inner",
    )
    if d.empty or not d.station_id.eq(d.reference_station).all():
        raise ValueError("unmatched stations")
    if arm == "choi":
        yr = (d.true_hs_m > 3.05).astype(int)
        expected = (d.true_hs_m < 1.52) | (d.true_hs_m > 3.05)
        reference = (d.reference > 2.285).astype(int)
    elif arm == "kim":
        ha = 0.631 * d.true_hs_m
        expected = (ha >= 0.75) & (ha < 3.25)
        yr = np.digitize(ha, [1.25, 1.75, 2.25, 2.75])
        reference = np.digitize(0.631 * d.reference, [1.25, 1.75, 2.25, 2.75])
    else:
        expected = d.true_hs_m.notna()
        yr = d.true_hs_m
        reference = d.reference
    d = d.loc[expected].copy()
    if len(d) != len(expected) or not np.allclose(d.target, yr, atol=1e-4):
        raise ValueError("endpoint differs across arms")
    regression = arm.startswith("yang")
    if regression:
        d["project_score"] = (reference - d.target) ** 2
        d["paper_score"] = (d.pred_hs_m - d.target) ** 2
    else:
        d["project_score"] = (reference == yr).astype(float)
        d["paper_score"] = (d.filter(regex=r"^prob_\d+$").to_numpy().argmax(1) == yr).astype(float)
    groups = d.groupby("station_id")[["project_score", "paper_score"]]
    n = groups.size().to_numpy()
    sums = groups.sum().to_numpy()
    values = sums.sum(0) / n.sum()
    sel = np.random.default_rng(0).integers(len(n), size=(2000, len(n)))
    boot = sums[sel].sum(1) / n[sel].sum(1)[:, None]
    if regression:
        values, boot = np.sqrt(values), np.sqrt(boot)
    return {
        "arm": arm,
        "rows": len(d),
        "paper_eligible": len(p),
        "reference_eligible": len(r),
        "buoys": len(n),
        "metric": "RMSE_m" if regression else "accuracy",
        "project": values[0],
        "paper": values[1],
        "delta": values[0] - values[1],
        "delta_ci": np.quantile(boot[:, 0] - boot[:, 1], [0.025, 0.975]).tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, nargs=3, required=True)
    parser.add_argument("--reference", type=Path, nargs=3, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.paper, args.reference), indent=2))
