"""Score buoy-camera training runs against the bar fixed before training.

``docs/experiments/ndbc_buoycam_sea_state.md``, "The bar": on the test
partition the camera must beat ERA5 at each buoy **on the same strips**, in
balanced accuracy *and* macro-F1, with the 95 % interval of the gain above zero
(paired bootstrap over whole groups: buoys for the station split, buoy-days
for the time split, as in the config). The time split is also held against
each buoy's own majority class. The comparison uses the strips a baseline
covers (-1 in baselines.csv: ERA5 lags real time) and reports that coverage.

    python scripts/buoycam_bar.py runs/<run> [runs/<run> ...]

Writes ``bar.json`` into each run folder and prints one line per comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wave_analysis.config import load_yaml
from wave_analysis.evaluation.classification import classification_report
from wave_analysis.evaluation.uncertainty import paired_block_bootstrap_ci

METRICS = ("balanced_accuracy", "f1_macro")
#: Baselines the bar holds each split against (columns of the package's baselines.csv).
AGAINST = {"station": ("era5_station",), "time": ("era5_time", "station_majority_time")}


def metric(name: str, k: int):
    def f(y: np.ndarray, p: np.ndarray) -> float:
        return float(
            classification_report(y.astype(np.int64), p.astype(np.int64), n_classes=k)[name]
        )

    return f


def score(run: Path, root: Path) -> dict:
    meta = json.loads((run / "metrics.json").read_text())["run"]
    split, name = meta["split"], meta["config"]
    cfg = load_yaml(root / "configs" / "experiments" / f"{name}.yaml")
    pkg = root / "data" / "processed" / "packages" / name
    boot = cfg["evaluation"]["bootstrap"]
    gcol = boot["group_col"][split] if isinstance(boot["group_col"], dict) else boot["group_col"]
    k = len(cfg["task"]["classes"]["names"])

    pred = pd.read_csv(run / "predictions.csv")
    labels = pd.read_csv(pkg / "labels.csv", usecols=["sample_id", gcol])
    base = pd.read_csv(pkg / "baselines.csv")
    df = (
        pred.drop(columns=[gcol], errors="ignore")
        .merge(labels, on="sample_id")
        .merge(base, on="sample_id")
    )
    if len(df) != len(pred):
        raise SystemExit(f"{run}: {len(pred) - len(df)} predictions have no row in the package")

    out = {
        "run": run.name,
        "split": split,
        "group_col": gcol,
        "n_boot": int(boot["n_boot"]),
        "alpha": float(boot["alpha"]),
        "seed": int(boot["seed"]),
        "comparisons": [],
    }
    for part in [p for p in ("test", "test_later") if (df["partition"] == p).any()]:
        d = df[df["partition"] == part]
        for b in AGAINST[split]:
            ok = (d[b].fillna(-1) >= 0).to_numpy()  # -1: no baseline for this strip
            e = d[ok]
            row = {
                "partition": part,
                "against": b,
                "n": int(ok.sum()),
                "coverage": float(ok.mean()),
                "n_groups": int(e[gcol].nunique()),
            }
            for m in METRICS:
                gain, lo, hi = paired_block_bootstrap_ci(
                    e["true"],
                    e["pred"],
                    e[b],
                    metric(m, k),
                    groups=e[gcol],
                    n_boot=int(boot["n_boot"]),
                    alpha=float(boot["alpha"]),
                    seed=int(boot["seed"]),
                )
                row[m] = {
                    "model": metric(m, k)(e["true"].to_numpy(), e["pred"].to_numpy()),
                    "baseline": metric(m, k)(e["true"].to_numpy(), e[b].to_numpy()),
                    "gain": gain,
                    "ci": [lo, hi],
                }
            row["met"] = all(row[m]["ci"][0] > 0 for m in METRICS)
            out["comparisons"].append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    for run in args.runs:
        res = score(run, root)
        (run / "bar.json").write_text(json.dumps(res, indent=2) + "\n")
        for c in res["comparisons"]:
            parts = [
                f"{m} {c[m]['model']:.3f} vs {c[m]['baseline']:.3f}, gain {c[m]['gain']:+.3f} "
                f"[{c[m]['ci'][0]:+.3f}, {c[m]['ci'][1]:+.3f}]"
                for m in METRICS
            ]
            print(
                f"{run.name} {c['partition']} vs {c['against']} ({c['n']:,} strips, "
                f"{c['coverage']:.0%} covered, {c['n_groups']} {res['group_col']}): "
                f"{'; '.join(parts)} -> {'MET' if c['met'] else 'not met'}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
