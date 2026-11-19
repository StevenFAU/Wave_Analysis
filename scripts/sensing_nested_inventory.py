#!/usr/bin/env python
"""Which station groups every cached prediction's model trained, validated and tested on, per outer fold.

    python scripts/sensing_nested_inventory.py --package <v2cvn package> \\
        --camera runs/<15 fold runs> [--inner runs/<30 pair runs>] [--station runs/<sealed runs>]

For outer fold k the fusion gate is calibrated on strips of the other folds j. The
table says, for the models whose predictions on fold j would be used, how many of
fold k's groups each one trained on (``train``), picked its epoch on (``val``) or
predicted (``test``). The roles come from the package's split columns; the cached
runs' own ``predictions.csv`` (validation and test strips) and ``metrics.json``
(split, seed) are checked against them, so the table records what the runs did,
not what they were meant to do. The forecast corrector of test 1 uses the same
``split_fold<j>`` columns (``scripts/sensing_forecast_corrector.py``), so its rows
are those of the camera's fold models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def run_meta(run: Path) -> dict:
    m = json.loads((run / "metrics.json").read_text())["run"]
    return {"split": m["split"], "seed": int(m["seed"]), "config": m["config"]}


def checked_roles(runs: list[Path], labels: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    """Per split: the groups in each role, after checking every run's val and test strips."""
    out: dict[str, dict[str, set[str]]] = {}
    group = labels.set_index("sample_id")["station_group"]
    for r in runs:
        meta = run_meta(r)
        col = f"split_{meta['split']}"
        roles = {p: set(labels.loc[labels[col] == p, "station_group"]) for p in ("train", "val", "test")}
        pred = pd.read_csv(r / "predictions.csv", usecols=["sample_id", "partition"])
        for p in ("val", "test"):
            seen = set(group.loc[pred.loc[pred["partition"] == p, "sample_id"]])
            if seen != roles[p]:
                raise SystemExit(f"{r.name}: {p} groups differ from {col}")
            if (pred["partition"] == p).sum() != (labels[col] == p).sum():
                raise SystemExit(f"{r.name}: {p} strips differ from {col}")
        if roles["train"] & (roles["val"] | roles["test"]) or roles["val"] & roles["test"]:
            raise SystemExit(f"{col}: a group holds two roles")
        out.setdefault(meta["split"], roles)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--camera", type=Path, nargs="+", required=True)
    ap.add_argument("--inner", type=Path, nargs="*", default=[])
    ap.add_argument("--station", type=Path, nargs="*", default=[])
    args = ap.parse_args(argv)
    labels = pd.read_csv(args.package / "labels.csv", dtype={"station_id": str, "cv_fold": str}, low_memory=False)
    labels = labels[labels["cv_fold"].notna()]
    folds = sorted(labels["cv_fold"].astype(int).unique())
    groups = {k: set(labels.loc[labels["cv_fold"].astype(int) == k, "station_group"]) for k in folds}
    cam = checked_roles(args.camera, labels)
    inner = checked_roles(args.inner, labels) if args.inner else {}
    seeds = sorted({run_meta(r)["seed"] for r in args.camera})

    def count(roles: dict[str, set[str]], k: int) -> str:
        return "/".join(str(len(groups[k] & roles[p])) for p in ("train", "val", "test"))

    print("| Outer fold k | Groups (buoys, strips) | Calibration fold j | Cached fold-j models: fold-k groups train/val/test"
          + (" | Inner pair {j,k} model: fold-k groups train/val/test |" if inner else " |"))
    print("|---|---|---|---|" + ("---|" if inner else ""))
    for k in folds:
        lk = labels[labels["cv_fold"].astype(int) == k]
        head = f"{k} | {len(groups[k])} ({lk['station_id'].nunique()}, {len(lk):,})"
        for i, j in enumerate(f for f in folds if f != k):
            row = f"| {head if i == 0 else ' | '} | {j} | {count(cam[f'fold{j}'], k)}"
            if inner:
                row += f" | {count(inner[f'pair{min(j, k)}{max(j, k)}'], k)}"
            print(row + " |")
    print(f"\nCached camera runs checked: {len(args.camera)} (seeds {seeds}); "
          f"every fold-j model's test groups are fold j's and nothing else.")
    for r in args.station:
        meta = run_meta(r)
        col = f"split_{meta['split']}"
        roles = {p: set(labels.loc[labels[col] == p, "station_group"]) for p in ("train", "val", "test")}
        print(f"{r.name}: per outer fold, groups train/val/test: "
              + ", ".join(f"fold {k} {count(roles, k)}" for k in folds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
