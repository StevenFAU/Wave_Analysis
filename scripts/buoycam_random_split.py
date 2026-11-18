#!/usr/bin/env python
"""Protocol check R: the v2 heights package with a random split of strips (Choi 2020's and Kim 2023's protocol).

Not a bar and not our result: it shows how much the published protocol adds. Choi et al.
2020 split 44,620 examples at random with the same buoys and days on both sides; Kim et
al. 2023 split frames at random within 30 one-hour sessions. Our results hold out whole
buoys. Fixed 2026-11-18, before any run on this split (``docs/experiments/
ndbc_buoycam_sea_state.md``, "Protocol check R").

Writes ``data/processed/packages/ndbc_buoycam_heights_v2r``: every file of
``ndbc_buoycam_heights_v2`` (hard links, no second copy of the images) except
``labels.csv``, which gains ``split_random``: the strips the station split uses (train,
val or test there) dealt at random (seed 0) to train 80 %, val 10 % and test 10 %, so
neighbouring hours of one buoy and day fall on both sides, as in those papers.
``SHA256SUMS`` is rewritten for the new ``labels.csv``.

    python scripts/buoycam_random_split.py
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
OUT = "ndbc_buoycam_heights_v2r"
SHARES = {TRAIN: 0.8, VAL: 0.1, TEST: 0.1}
SEED = 0


def main() -> int:
    src, out = data_dir("processed") / "packages" / SRC, data_dir("processed") / "packages" / OUT
    if out.exists():
        shutil.rmtree(out)
    for f in src.rglob("*"):
        if f.is_file() and f.name not in ("labels.csv", "SHA256SUMS"):
            dest = out / f.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.hardlink_to(f)
    labels = pd.read_csv(src / "labels.csv", dtype={"station_id": str}, low_memory=False)
    use = labels["split_station"].isin([TRAIN, VAL, TEST]).to_numpy()
    idx = np.flatnonzero(use)
    perm = np.random.default_rng(SEED).permutation(idx)
    n_tr, n_va = round(SHARES[TRAIN] * len(idx)), round(SHARES[VAL] * len(idx))
    split = np.full(len(labels), "excluded", dtype=object)
    split[perm[:n_tr]], split[perm[n_tr : n_tr + n_va]], split[perm[n_tr + n_va :]] = (
        TRAIN,
        VAL,
        TEST,
    )
    labels["split_random"] = split
    # Append the column to v2's text as it is, so no other value is rewritten.
    rows = (src / "labels.csv").read_text().splitlines()
    if len(rows) != len(labels) + 1:
        raise SystemExit("labels.csv has quoted line breaks: append the column another way")
    text = [rows[0] + ",split_random"] + [r + "," + v for r, v in zip(rows[1:], split, strict=True)]
    (out / "labels.csv").write_text("\n".join(text) + "\n")
    sha = hashlib.sha256((out / "labels.csv").read_bytes()).hexdigest()
    lines = []
    for line in (src / "SHA256SUMS").read_text().splitlines():
        _sha, name = line.split(None, 1)
        lines.append(
            f"{sha}  {name.strip()}" if name.strip().lstrip("./") == "labels.csv" else line
        )
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    t = labels.loc[use]
    print(f"{out}: {len(idx):,} strips; " + ", ".join(f"{p} {int((split == p).sum()):,}" for p in SHARES)
          + f"; test strips at {t.loc[t['split_random'] == TEST, 'station_id'].nunique()} buoys")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
