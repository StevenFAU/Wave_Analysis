#!/usr/bin/env python
"""Grouped cross-validation for the buoy-camera heights: five folds over the station groups.

Rank 1 of ``reports/Wave camera architectures beyond backbones.md``: every buoy is a
test buoy once, predicted by a model that saw neither it nor any buoy within 300 km.
Fixed 2026-11-18, before any run on these folds (``docs/experiments/
ndbc_buoycam_sea_state.md``, "Grouped cross-validation").

* **Groups:** the station split's own (``station_group``: buoys under 300 km apart,
  chained): 43 groups at 79 buoys.
* **Folds, balanced by strips:** groups in decreasing order of strips (ties by name),
  each to the fold with the fewest strips so far (ties to the lower fold). The US
  West Coast group (9,440 strips) is a fold on its own; the others come out level.
* **Validation inside each fold's training groups:** fold k's model picks its epoch
  on validation groups drawn from the four other folds, never from its own test
  groups. Draws (``numpy`` seed ``SEED + k``): the training groups in random order,
  taken until the validation strips reach ``VAL_SHARE`` of all strips (a group larger
  than that target never goes to validation). A draw is admissible with at most
  ``VAL_MAX_OVER`` times the target and at least ``VAL_MIN_BUOYS`` buoys. Of
  ``N_DRAWS`` draws, the admissible one whose swell share is closest to that of its
  training strips is kept (ties: closest to the target size). Swell share is
  SwH² / (SwH² + WWH²) over the strips with both. This uses training labels only;
  it answers the station split's flaw, validation buoys with far less swell (0.40)
  than the test buoys (0.67).

Writes ``data/processed/packages/ndbc_buoycam_heights_v2cv``: every file of
``ndbc_buoycam_heights_v2`` (hard links: no second copy of the images) except
``labels.csv``, which gains ``cv_fold`` (the fold whose test buoys hold the strip) and
``split_fold0`` .. ``split_fold4`` (train, val or test in that fold), and a new
``folds.csv`` (one row per group: its buoys, strips, swell share, fold and role in
each fold). ``SHA256SUMS`` is rewritten for both.

    python scripts/buoycam_cv_folds.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir
from wave_analysis.datasets.splits import TEST, TRAIN, VAL

SRC = "ndbc_buoycam_heights_v2"
OUT = "ndbc_buoycam_heights_v2cv"
N_FOLDS = 5
VAL_SHARE = 0.15
VAL_MAX_OVER = 1.15
VAL_MIN_BUOYS = 8
N_DRAWS = 5000
SEED = 0


def swell_share(t: pd.DataFrame) -> float:
    sw, ww = t["swell_label_m"].to_numpy(float) ** 2, t["wind_wave_label_m"].to_numpy(float) ** 2
    ok = np.isfinite(sw) & np.isfinite(ww)
    return float(sw[ok].sum() / (sw[ok].sum() + ww[ok].sum()))


def assign_folds(strips: pd.Series, n_folds: int = N_FOLDS) -> dict[str, int]:
    """Largest group first, each to the fold with the fewest strips so far (ties: lower fold)."""
    order = sorted(strips.index, key=lambda g: (-int(strips[g]), str(g)))
    load = [0] * n_folds
    out: dict[str, int] = {}
    for g in order:
        k = min(range(n_folds), key=lambda i: (load[i], i))
        out[g] = k
        load[k] += int(strips[g])
    return out


def draw_val(
    t: pd.DataFrame, pool: list[str], target: int, rng: np.random.Generator
) -> tuple[list[str], dict]:
    """Validation groups from ``pool`` (see the module text); returns them and the draw's numbers."""
    by_g = dict(list(t.groupby("station_group")))
    size = {g: len(by_g[g]) for g in pool}
    buoys = {g: by_g[g]["station_id"].nunique() for g in pool}
    eligible = [g for g in pool if size[g] <= target]
    best: tuple[float, float] | None = None
    chosen: list[str] = []
    for _ in range(N_DRAWS):
        val: list[str] = []
        n = 0
        for g in rng.permutation(eligible):
            if n >= target:
                break
            val.append(str(g))
            n += size[g]
        if n < target or n > VAL_MAX_OVER * target or sum(buoys[g] for g in val) < VAL_MIN_BUOYS:
            continue
        tv = t[t["station_group"].isin(val)]
        tr = t[t["station_group"].isin(pool) & ~t["station_group"].isin(val)]
        key = (abs(swell_share(tv) - swell_share(tr)), abs(n - target))
        if best is None or key < best:
            best, chosen = key, sorted(val)
    if best is None:
        raise SystemExit("no admissible validation draw: loosen VAL_MAX_OVER or VAL_MIN_BUOYS")
    tv = t[t["station_group"].isin(chosen)]
    tr = t[t["station_group"].isin(pool) & ~t["station_group"].isin(chosen)]
    return chosen, {
        "val_strips": len(tv),
        "val_buoys": int(tv["station_id"].nunique()),
        "val_swell_share": swell_share(tv),
        "train_swell_share": swell_share(tr),
    }


def main() -> int:
    src, out = data_dir("processed") / "packages" / SRC, data_dir("processed") / "packages" / OUT
    labels = pd.read_csv(src / "labels.csv", dtype={"station_id": str}, low_memory=False)
    use = labels["split_station"].isin([TRAIN, VAL, TEST]).to_numpy()
    t = labels.loc[use]
    strips = t.groupby("station_group").size()
    fold_of = assign_folds(strips)
    target = round(VAL_SHARE * len(t))
    role = {g: [""] * N_FOLDS for g in strips.index}
    summary = []
    for k in range(N_FOLDS):
        test = [g for g, f in fold_of.items() if f == k]
        pool = sorted(g for g, f in fold_of.items() if f != k)
        val, info = draw_val(t, pool, target, np.random.default_rng(SEED + k))
        for g in strips.index:
            role[g][k] = TEST if g in test else VAL if g in val else TRAIN
        tt = t[t["station_group"].isin(test)]
        summary.append({"fold": k, "test_groups": len(test), "test_buoys": int(tt["station_id"].nunique()),
                        "test_strips": len(tt), "test_swell_share": swell_share(tt), **info,
                        "train_strips": len(t) - len(tt) - info["val_strips"]})  # fmt: skip

    if out.exists():
        shutil.rmtree(out)
    for f in src.rglob("*"):
        if f.is_file() and f.name not in ("labels.csv", "SHA256SUMS"):
            dest = out / f.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.hardlink_to(f)
    grp = labels["station_group"].to_numpy()
    cols = {
        "cv_fold": np.array([str(fold_of[g]) if u else "" for g, u in zip(grp, use, strict=True)])
    }
    for k in range(N_FOLDS):
        cols[f"split_fold{k}"] = np.array(
            [role[g][k] if u else "excluded" for g, u in zip(grp, use, strict=True)]
        )
    # Append the columns to v2's text as it is, so no other value is rewritten.
    rows = (src / "labels.csv").read_text().splitlines()
    if len(rows) != len(labels) + 1:
        raise SystemExit("labels.csv has quoted line breaks: append the columns another way")
    names = list(cols)
    text = [rows[0] + "," + ",".join(names)] + [
        r + "," + ",".join(cols[c][i] for c in names) for i, r in enumerate(rows[1:])
    ]
    (out / "labels.csv").write_text("\n".join(text) + "\n")
    folds = pd.DataFrame(
        [{"station_group": g, "stations": " ".join(sorted(t.loc[t["station_group"] == g, "station_id"].unique())),
          "strips": int(strips[g]), "swell_share": swell_share(t[t["station_group"] == g]),
          "cv_fold": fold_of[g], **{f"role_fold{k}": role[g][k] for k in range(N_FOLDS)}}
         for g in sorted(strips.index, key=lambda g: (fold_of[g], -int(strips[g])))]
    )  # fmt: skip
    folds.to_csv(out / "folds.csv", index=False, float_format="%.4f")
    sums = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in ("labels.csv", "folds.csv")
    }
    lines = []
    for line in (src / "SHA256SUMS").read_text().splitlines():
        _sha, name = line.split(None, 1)
        lines.append(
            f"{sums['labels.csv']}  {name.strip()}"
            if name.strip().lstrip("./") == "labels.csv"
            else line
        )
    prefix = "./" if any(" ./" in x for x in lines) else ""
    lines.append(f"{sums['folds.csv']}  {prefix}folds.csv")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    s = pd.DataFrame(summary)
    print(f"{out}: {len(t):,} strips, {strips.size} groups, {t['station_id'].nunique()} buoys; "
          f"validation target {target:,} strips")  # fmt: skip
    print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
