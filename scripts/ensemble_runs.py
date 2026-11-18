"""Average the class probabilities of several runs of one configuration (seeds) into one judged prediction.

    python scripts/ensemble_runs.py --out runs/<name> runs/<seed 0> runs/<seed 1> runs/<seed 2>

The runs must share the configuration, the split and the strips. Writes
``<out>/predictions.csv`` (the mean of each ``p_<class>`` over the runs, and
``pred`` = the class with the highest mean) and ``<out>/metrics.json`` holding
the run metadata (members, their seeds and commits), so that
``scripts/buoycam_bar.py`` and ``scripts/buoycam_swell_check.py`` score it as
they score a single run. One seed moved the v0 test macro-F1 by 0.12, which
is why the v1 bars judge the average of three.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if len(args.runs) < 2:
        raise SystemExit("give at least two runs")

    metas = [json.loads((r / "metrics.json").read_text())["run"] for r in args.runs]
    for key in ("config", "split"):
        if len({m[key] for m in metas}) != 1:
            raise SystemExit(f"the runs differ in {key}: {[m[key] for m in metas]}")
    preds = [
        pd.read_csv(r / "predictions.csv").sort_values("sample_id").reset_index(drop=True)
        for r in args.runs
    ]
    first = preds[0]
    pcols = [c for c in first.columns if c.startswith("p_")]
    for r, p in zip(args.runs[1:], preds[1:], strict=True):
        if not p["sample_id"].equals(first["sample_id"]) or not p["true"].equals(first["true"]):
            raise SystemExit(f"{r} has other strips or labels than {args.runs[0]}")
    mean = np.mean([p[pcols].to_numpy() for p in preds], axis=0)
    out = first.drop(columns=pcols).copy()
    out["pred"] = mean.argmax(axis=1)
    for j, c in enumerate(pcols):
        out[c] = mean[:, j]

    commits = []
    for r in args.runs:
        prov = r / "provenance.yaml"
        git = (yaml.safe_load(prov.read_text()) or {}).get("git", {}) if prov.exists() else {}
        commits.append(git.get("commit"))
    meta = dict(metas[0])
    meta.update(
        seed="ensemble",
        members=[r.name for r in args.runs],
        member_seeds=[m.get("seed") for m in metas],
        member_commits=commits,
        combine="mean_probability",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "predictions.csv", index=False, float_format="%.6g")
    (args.out / "metrics.json").write_text(json.dumps({"run": meta}, indent=2) + "\n")
    agree = np.mean([(p["pred"].to_numpy() == out["pred"].to_numpy()).mean() for p in preds])
    print(
        f"{args.out}: {len(out):,} strips, mean of {len(preds)} runs (seeds {meta['member_seeds']}); "
        f"members agree with the average on {agree:.0%} of strips"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
