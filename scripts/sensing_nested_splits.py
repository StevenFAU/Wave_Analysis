#!/usr/bin/env python
"""Inner splits for test 1's nested fusion: every pair of the five outer folds held out together.

    python scripts/sensing_nested_splits.py [--src <v2cv package>] [--out <package dir>]

Fixed 2026-11-19, before any inner model was trained or any nested score computed
(``docs/experiments/sensing_model_bar.md``, "Amendment: nested calibration").

The historical fusion gate for outer fold k was fitted on the other folds' out-of-fold
predictions, but the models that made those predictions had trained or validated on
fold k's groups. Here a calibration prediction for a strip of fold j, used to fit the
gate that will be applied to fold k, comes from a model that saw neither fold j nor
fold k. One model per unordered pair {a, b} of outer folds does this for both orders:
it trains on the other three folds and predicts a and b (10 pairs instead of 20
fold-by-fold inner models).

* **Groups and folds:** the v2cv package's (``cv_fold``; 43 station groups, 79 buoys).
* **Validation inside the pool:** the three remaining folds are the pool; the epoch is
  picked on validation groups drawn from the pool by ``buoycam_cv_folds.draw_val``
  (same draws, admissibility and swell-share rule as the outer folds). Its target keeps
  the outer folds' validation-to-pool ratio, 0.15 / (1 - 1/5) = 0.1875 of the pool's
  strips, and the draws use ``numpy`` seed ``100 + 10 a + b``.
* **Roles:** ``split_pair<ab>`` is ``test`` on folds a and b, ``val`` and ``train`` in the
  pool, ``excluded`` where v2cv excludes the strip.

Writes ``<out>`` (default ``data/processed/packages/ndbc_buoycam_heights_v2cvn``): every
file of v2cv as a hard link except ``labels.csv`` (v2cv's text plus ten ``split_pair<ab>``
columns), ``SHA256SUMS`` (rewritten) and a new ``pairs.csv`` (one row per group: its
fold and its role in each pair). Choose a new output directory for each build.
The package is published after the source manifest checksums have been verified.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from buoycam_cv_folds import N_FOLDS, VAL_SHARE, draw_val  # scripts/ is on sys.path
from wave_analysis.config import data_dir
from wave_analysis.datasets.splits import TEST, TRAIN, VAL
from wave_analysis.ingest.checksum import sha256_file

SEED = 100
VAL_OF_POOL = VAL_SHARE / (1 - 1 / N_FOLDS)
PAIRS = list(itertools.combinations(range(N_FOLDS), 2))


def pair_name(a: int, b: int) -> str:
    return f"pair{a}{b}"


def pair_roles(t: pd.DataFrame) -> tuple[dict[str, dict[str, str]], pd.DataFrame]:
    """Role of every group in every pair split, and one summary row per pair."""
    fold_of = t.groupby("station_group")["cv_fold"].first().astype(int).to_dict()
    role: dict[str, dict[str, str]] = {g: {} for g in fold_of}
    summary = []
    for a, b in PAIRS:
        name = pair_name(a, b)
        pool = sorted(g for g, f in fold_of.items() if f not in (a, b))
        pool_strips = int(t["station_group"].isin(pool).sum())
        target = round(VAL_OF_POOL * pool_strips)
        val, info = draw_val(t, pool, target, np.random.default_rng(SEED + 10 * a + b))
        for g, f in fold_of.items():
            role[g][name] = TEST if f in (a, b) else VAL if g in val else TRAIN
        tt = t[t["cv_fold"].astype(int).isin([a, b])]
        summary.append({"pair": name, "test_folds": f"{a},{b}", "test_strips": len(tt),
                        "pool_strips": pool_strips, "val_target": target, **info,
                        "train_strips": pool_strips - info["val_strips"]})  # fmt: skip
    return role, pd.DataFrame(summary)


def build_package(src: Path, out: Path) -> tuple[int, int, pd.DataFrame]:
    labels = pd.read_csv(
        src / "labels.csv", dtype={"station_id": str, "cv_fold": str}, low_memory=False
    )
    use = labels["cv_fold"].notna().to_numpy() & (labels["cv_fold"].astype(str) != "")
    t = labels.loc[use].copy()
    role, summary = pair_roles(t)

    if (src / "pairs.csv").is_file():
        raise ValueError("source already contains pairs.csv; supply the outer-fold package")
    for f in src.rglob("*"):
        if f.is_file() and f.relative_to(src) not in (Path("labels.csv"), Path("SHA256SUMS")):
            dest = out / f.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.hardlink_to(f)
    grp = labels["station_group"].to_numpy()
    names = [pair_name(a, b) for a, b in PAIRS]
    cols = {
        f"split_{n}": [role[g][n] if u else "excluded" for g, u in zip(grp, use, strict=True)]
        for n in names
    }
    rows = (src / "labels.csv").read_text().splitlines()
    if len(rows) != len(labels) + 1:
        raise SystemExit("labels.csv has quoted line breaks: append the columns another way")
    text = [rows[0] + "," + ",".join(cols)] + [
        r + "," + ",".join(cols[c][i] for c in cols) for i, r in enumerate(rows[1:])
    ]
    (out / "labels.csv").write_text("\n".join(text) + "\n")
    folds = pd.read_csv(src / "folds.csv")
    pairs = folds[["station_group", "stations", "strips", "swell_share", "cv_fold"]].assign(
        **{f"role_{n}": folds["station_group"].map(lambda g, n=n: role[g][n]) for n in names}
    )
    pairs.to_csv(out / "pairs.csv", index=False, float_format="%.4f")
    lines = []
    seen = set()
    for line in (src / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(None, 1)
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path in seen:
            raise ValueError(f"invalid or duplicate package path {name}")
        seen.add(path)
        if path == Path("labels.csv"):
            actual = sha256_file(src / path)
            line = f"{sha256_file(out / path)}  {name}"
        else:
            actual = sha256_file(out / path)
        if actual != digest:
            raise ValueError(f"checksum mismatch for {name}")
        lines.append(line)
    if not {Path("labels.csv"), Path("folds.csv")} <= seen:
        raise ValueError("source manifest must include labels.csv and folds.csv")
    prefix = "./" if any(" ./" in line for line in lines) else ""
    lines.append(f"{sha256_file(out / 'pairs.csv')}  {prefix}pairs.csv")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return len(t), t["station_group"].nunique(), summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--src", type=Path, default=data_dir("processed") / "packages/ndbc_buoycam_heights_v2cv"
    )
    ap.add_argument(
        "--out", type=Path, default=data_dir("processed") / "packages/ndbc_buoycam_heights_v2cvn"
    )
    args = ap.parse_args(argv)
    src, out = args.src, args.out
    src = src.resolve(strict=True)
    target = out.resolve()
    if target.is_relative_to(src) or src.is_relative_to(target):
        ap.error("source and output directories must be separate and must not contain each other")
    if out.exists() or out.is_symlink():
        ap.error(f"output already exists: {out}; choose a new package directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp) / "package"
        stage.mkdir()
        strips, groups, summary = build_package(src, stage)
        stage.rename(out)
    print(f"{out}: {strips:,} strips, {groups} groups; validation {VAL_OF_POOL:.4f} of each pool")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
