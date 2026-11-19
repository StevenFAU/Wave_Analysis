"""Compare radiance and stereo elevation on the same resolvable space-time scales."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import fftfreq, rfftfreq, rfftn
from scipy.signal import fftconvolve

from measurement_buoy_probe import ridge
from measurement_references import read_stereo

SIDE = 96
DX = 0.25
TARGETS = ["wavelength_m", "period_s", "phase_speed_mps"]


def surface_features(surface, valid, fps):
    """Average tapered k-f power across 20 s blocks. k is cycles/m and f is Hz."""
    length = round(20 * fps)
    segs = surface.shape[2] // length
    k = fftfreq(SIDE, DX)
    radial = np.hypot(k[:, None], k[None, :])
    radial_edges = np.arange(1.5, 25) / (SIDE * DX)
    freq = rfftfreq(length, 1 / fps)
    keep = (freq >= 0.1) & (freq <= 0.7)
    bands = np.searchsorted(radial_edges, radial, side="right") - 1
    spatial = (bands >= 0) & (bands < len(radial_edges) - 1)
    bins = bands[spatial]
    count = np.bincount(bins, minlength=len(radial_edges) - 1)
    taper = np.hanning(SIDE).astype("f")[:, None] * np.hanning(SIDE).astype("f")[None, :]
    power = np.zeros((len(count), keep.sum()))
    for i in range(segs):
        sl = slice(i * length, (i + 1) * length)
        mask = valid[:, :, sl]
        block = np.where(mask, surface[:, :, sl], 0).astype("f")
        mean = block.sum(axis=2) / np.maximum(mask.sum(axis=2), 1)
        block -= mean.astype("f")[:, :, None]
        block *= mask * taper[:, :, None] * np.hanning(length).astype("f")
        p = abs(rfftn(block)[:, :, keep][spatial]) ** 2
        for j in range(p.shape[1]):
            power[:, j] += np.bincount(bins, weights=p[:, j], minlength=len(count)) / count
    if not np.isfinite(power).all() or power.sum() <= 0:
        raise ValueError("no resolvable wave power")
    a, b = np.unravel_index(power.argmax(), power.shape)
    wavelength = 2 / (radial_edges[a] + radial_edges[a + 1])
    frequency = freq[keep][b]
    return [wavelength, 1 / frequency, wavelength * frequency], power / power.sum()


def features(index, out, limit):
    indexed = pd.read_csv(index).sort_values(["burst", "minute", "archive"])
    # A burst can occur in both source archives. Take its first minute exactly once.
    records = indexed.drop_duplicates("burst")
    if limit:
        records = records.head(limit)
    rows = []
    for r in records.itertuples():
        im, z = read_stereo(r)
        valid = np.isfinite(im) & np.isfinite(z)
        # Choose the best-covered patch using mask geometry alone.
        m = valid.mean(2)

        coverage = fftconvolve(m, np.ones((SIDE, SIDE)) / SIDE**2, mode="valid")
        y, x = np.unravel_index(coverage.argmax(), coverage.shape)
        row = {"burst": r.burst, "day": r.day, "member": r.member, "coverage": coverage[y, x]}
        if row["coverage"] >= 0.95:
            patch = np.s_[y : y + SIDE, x : x + SIDE, :]
            fps = im.shape[2] / 60
            for name, surface in [("radiance", im), ("stereo", z)]:
                values, power = surface_features(surface[patch], valid[patch], fps)
                row.update({f"{name}_{k}": v for k, v in zip(TARGETS, values, strict=True)})
                if name == "stereo":
                    ki, fi = np.unravel_index(power.argmax(), power.shape)
                    row["peak_inside_k_band"] = bool(0 < ki < power.shape[0] - 1)
                    row["peak_inside_f_band"] = bool(0 < fi < power.shape[1] - 1)
                    elevation = np.where(valid[patch], surface[patch], np.nan)
                    row["stereo_temporal_rms_m"] = np.sqrt(
                        np.mean(np.nanvar(elevation[np.isfinite(elevation).any(axis=2)], axis=1))
                    )
                if name == "radiance":
                    row.update({f"power_{i:03d}": v for i, v in enumerate(power.ravel())})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False, mode="x")


def compare(table, out):
    data = pd.read_csv(table)
    cols = data.filter(regex="^power_").columns
    data = data.dropna(subset=[*cols, *["stereo_" + k for k in TARGETS]]).copy()
    if data.day.nunique() < 2 or data.burst.duplicated().any():
        raise ValueError("need unique bursts on two or more usable days")
    for target in [*TARGETS, "temporal_rms_m"]:
        eligible = np.ones(len(data), dtype=bool)
        if target in ("wavelength_m", "phase_speed_mps"):
            eligible &= data.peak_inside_k_band.eq(True).to_numpy()
        if target in ("period_s", "phase_speed_mps"):
            eligible &= data.peak_inside_f_band.eq(True).to_numpy()
        data["eligible_" + target] = eligible
        for day in data.day.unique():
            test = eligible & data.day.eq(day)
            train = eligible & data.day.ne(day)
            if not test.any() or train.sum() < 2:
                continue
            truth = data.loc[train, "stereo_" + target].to_numpy()
            data.loc[test, "mean_" + target] = truth.mean()
            data.loc[test, "ridge_" + target] = np.maximum(
                0, ridge(data.loc[train, cols].to_numpy(), truth, data.loc[test, cols].to_numpy())
            )
    data.drop(columns=cols).to_csv(out, index=False, mode="x")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="task", required=True)
    a = sub.add_parser("features")
    a.add_argument("--index", type=Path, required=True)
    a.add_argument("--limit", type=int)
    a = sub.add_parser("compare")
    a.add_argument("--features", type=Path, required=True)
    for a in sub.choices.values():
        a.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.task == "features":
        features(args.index, args.out, args.limit)
    else:
        compare(args.features, args.out)
