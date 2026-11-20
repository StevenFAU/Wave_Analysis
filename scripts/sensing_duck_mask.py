"""Measure how much an IR breaking mask tells us about an already tracked wave."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(waves, masks):
    if masks.shape != (len(waves), 64, 64) or waves.wave_id.duplicated().any():
        raise ValueError("expected one ordered 64 by 64 mask per unique wave")
    if not np.allclose(masks.mean(axis=(1, 2)), waves.mask_fraction, atol=1e-6):
        raise ValueError("mask order differs from wave table")
    quadrants = masks.reshape(-1, 2, 32, 2, 32).mean(axis=(2, 4)).reshape(-1, 4)
    x = np.column_stack((masks.mean(axis=(1, 2)), masks.std(axis=(1, 2)), quadrants)).astype(float)
    if waves.utc_day.nunique() != 2 or waves.groupby("run_id").utc_day.nunique().max() != 1:
        raise ValueError("expected two separate dates with disjoint runs")
    output = []
    for day in sorted(waves.utc_day.unique()):
        for target in ("height_m", "period_s"):
            y = waves[target].to_numpy()
            valid = np.isfinite(y) & (y > 0) & np.isfinite(x).all(axis=1)
            train, test = valid & waves.utc_day.ne(day), valid & waves.utc_day.eq(day)
            sd = x[train].std(axis=0)
            z = (x - x[train].mean(axis=0)) / np.where(sd > 0, sd, 1)
            reference = y[train].mean()
            coef = np.linalg.solve(
                z[train].T @ z[train] + 10 * np.eye(z.shape[1]), z[train].T @ (y[train] - reference)
            )
            output.append(
                waves.loc[test, ["utc_day", "run_id", "wave_id"]].assign(
                    target=target,
                    observed=y[test],
                    mean_pred=reference,
                    mask_pred=reference + z[test] @ coef,
                )
            )
    return pd.concat(output, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("package", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    d = evaluate(pd.read_csv(a.package / "waves.csv"), np.load(a.package / "masks.npy"))
    d.to_csv(a.out, index=False, mode="x")
    for keys in [["target", "utc_day"], ["target"]]:
        for key, g in d.groupby(keys):
            print(
                key,
                len(g),
                *(
                    round(np.sqrt(((g[c] - g.observed) ** 2).mean()), 4)
                    for c in ("mean_pred", "mask_pred")
                ),
            )


if __name__ == "__main__":
    main()
