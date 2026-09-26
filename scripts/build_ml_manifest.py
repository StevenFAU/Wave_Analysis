#!/usr/bin/env python
"""Build an ML manifest from a synchronised sample table and an experiment config.

    uv run python scripts/build_ml_manifest.py configs/experiments/baseline_hs.yaml samples.parquet

The config's ``task.filters`` are applied in order (each excluded sample is
recorded with the rule that excluded it), then the configured split protocol
is applied and audited for leakage. Outputs go to ``data/manifests/ml/<name>``
together with a provenance record.
"""

import argparse

import pandas as pd

from wave_analysis.config import load_yaml, repo_root
from wave_analysis.datasets import (
    build_manifest,
    build_provenance,
    check_leakage,
    extreme_condition_split,
    grouped_split,
    holdout_groups_split,
    time_block_split,
    write_provenance,
)


def make_split(df: pd.DataFrame, spec: dict):
    protocol = spec["protocol"]
    if protocol == "iid_grouped":
        return grouped_split(
            df,
            spec["group_cols"],
            spec.get("fractions", (0.7, 0.15, 0.15)),
            seed=spec.get("seed", 0),
        )
    if protocol == "cross_time":
        return time_block_split(
            df,
            spec["time_col"],
            val_start=spec["val_start"],
            test_start=spec["test_start"],
            embargo=spec.get("embargo", "24h"),
            group_col=spec.get("event_col"),
        )
    if protocol in ("cross_station", "cross_region"):
        return holdout_groups_split(
            df,
            spec["group_col"],
            test_groups=spec["test_groups"],
            val_groups=spec.get("val_groups", []),
            protocol=protocol,
        )
    if protocol == "extreme":
        return extreme_condition_split(
            df,
            spec["value_col"],
            quantile=spec.get("quantile", 0.9),
            group_col=spec.get("group_col"),
            seed=spec.get("seed", 0),
        )
    raise ValueError(f"unknown protocol {protocol!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("samples")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    df = pd.read_parquet(args.samples)
    split = make_split(df, cfg["split"])
    built = build_manifest(
        df,
        id_col=cfg["task"]["id_col"],
        target_cols=cfg["task"]["targets"],
        filters=cfg["task"].get("filters", []),
        split=split,
    )
    audit = cfg["split"].get("audit", {})
    report = check_leakage(df.loc[df.index.isin(split.labels.index)], split, **audit)
    name = cfg["name"]
    out = repo_root() / "data" / "manifests" / "ml" / name
    paths = built.write(out, name)
    write_provenance(
        build_provenance(
            artifact_id=name,
            inputs=[args.samples, args.config],
            outputs=list(paths.values()),
            config=cfg,
            notes=f"leakage audit ok={report.ok}: {report}",
        ),
        out / f"{name}.provenance.yaml",
    )
    print(
        f"{name}: {len(built.manifest)} samples, {len(built.exclusions)} excluded; split {split.summary()}; "
        f"leakage ok={report.ok}"
    )
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
