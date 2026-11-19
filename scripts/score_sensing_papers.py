"""Paired sealed-station comparison for three seeds of a paper arm and DINOv3.

With ``--control`` instead of ``--reference``: Yang with attitude against Yang
without attitude, on the same rows (RMSE with minus RMSE without).
"""

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
        if not data.sample_id.is_unique:
            raise ValueError("duplicate sample IDs")
        data = data.set_index("sample_id").sort_index()
        if frames and not data.index.equals(frames[0].index):
            raise ValueError("seed coverage changed")
        if not reference and frames and not data.target.equals(frames[0].target):
            raise ValueError("seed labels changed")
        frames.append(data)
    if seeds != {0, 1, 2}:
        raise ValueError("seeds 0,1,2 required")
    merged = frames[0].copy()
    merged[cols] = np.mean([d[cols].to_numpy() for d in frames], axis=0)
    merged.attrs["labels_sha256"] = package
    return arm, merged


def paired(d, regression):
    """Pooled metric of ``project_score`` and ``paper_score`` and a whole-buoy bootstrap of their difference."""
    groups = d.groupby("station_id")[["project_score", "paper_score"]]
    n = groups.size().to_numpy()
    sums = groups.sum().to_numpy()
    values = sums.sum(0) / n.sum()
    sel = np.random.default_rng(0).integers(len(n), size=(2000, len(n)))
    boot = sums[sel].sum(1) / n[sel].sum(1)[:, None]
    if regression:
        values, boot = np.sqrt(values), np.sqrt(boot)
    ci = np.quantile(boot[:, 0] - boot[:, 1], [0.025, 0.975]).tolist()
    return values, ci, len(n)


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
    values, ci, buoys = paired(d, regression)
    return {
        "arm": arm,
        "rows": len(d),
        "paper_eligible": len(p),
        "reference_eligible": len(r),
        "buoys": buoys,
        "metric": "RMSE_m" if regression else "accuracy",
        "project": values[0],
        "paper": values[1],
        "delta": values[0] - values[1],
        "delta_ci": ci,
    }


def compare_attitude(with_attitude, without_attitude):
    """Yang with minus without attitude: RMSE on identical strips, whole-buoy interval."""
    arm_a, a = ensemble(with_attitude, False)
    arm_b, b = ensemble(without_attitude, False)
    if (arm_a, arm_b) != ("yang", "yang-no-attitude"):
        raise ValueError("expected Yang with attitude, then Yang without")
    if a.attrs["labels_sha256"] != b.attrs["labels_sha256"]:
        raise ValueError("mixed packages")
    if not a.index.equals(b.index) or not a.station_id.equals(b.station_id):
        raise ValueError("the two arms cover different strips")
    if not np.allclose(a.target, b.target):
        raise ValueError("endpoint differs across arms")
    d = a[["station_id"]].assign(
        project_score=(a.pred_hs_m - a.target) ** 2,
        paper_score=(b.pred_hs_m - b.target) ** 2,
    )
    values, ci, buoys = paired(d, True)
    return {
        "arm": "yang vs yang-no-attitude",
        "rows": len(d),
        "buoys": buoys,
        "metric": "RMSE_m",
        "with_attitude": values[0],
        "without_attitude": values[1],
        "delta": values[0] - values[1],
        "delta_ci": ci,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, nargs=3, required=True)
    other = parser.add_mutually_exclusive_group(required=True)
    other.add_argument("--reference", type=Path, nargs=3, help="DINOv3 runs, seeds 0-2")
    other.add_argument("--control", type=Path, nargs=3, help="Yang without attitude, seeds 0-2")
    args = parser.parse_args()
    result = (
        compare(args.paper, args.reference)
        if args.reference
        else compare_attitude(args.paper, args.control)
    )
    print(json.dumps(result, indent=2))
