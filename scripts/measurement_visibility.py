"""Test 5: the visibility curve and the virtual anemometer, from saved frozen features.

    scripts/measurement_visibility.py --package <portable package> --bins <buoys-bins/labels.csv> \\
        --era5 <era5_wind_strips.csv> --probes <probe dir> ... --out <dir>

Each probe directory is one fold and one seed of ``measurement_buoy_probe.py
--save-features`` (``features.npy``, ``feature_rows.csv``, ``run.json``). Every ridge is
fitted on the fold's training strips only (penalty 10 on standardised inputs, as the
slope probe) and predicts the fold's test buoys; a target gets its own ridge (its own
head). The three seeds' predictions are averaged. Bar T5 in
``docs/experiments/wave_measurements.md``:

- **Visibility curve.** The energy of each of the 46 bins and of the four bands, as an
  equivalent height 4 sqrt(E) in metres. Arms: the camera's features alone, the buoy's
  wind (speed, gust, their squares and product: "wind-conditioned"), both, and the
  training mean. R^2 per bin at unseen buoys, pooled over the five folds, with a 95 %
  interval from a bootstrap over the 43 station groups.
- **Virtual anemometer.** The buoy's wind speed and gust (m/s) from the camera's
  features alone, against ERA5's 10-m wind (gust) at the buoy corrected by a line
  fitted on the training strips: RMSE and the paired interval of the difference.
  Beside it: camera + ERA5, and the same with the camera's features permuted within
  buoy and ERA5-wind decile on the test strips (the control that the camera's share is
  about the strip, not the buoy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve

from measurement_buoy_probe import wind_features

N_BINS = 46
BOOT = 2000


def ridge_multi(x, y, xt, penalty=10.0):
    """``measurement_buoy_probe.ridge`` for several targets at once (``y``: n x k)."""
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-8] = 1
    z = (x - mean) / scale
    ym = y.mean(0)
    w = solve(z.T @ z + penalty * np.eye(z.shape[1]), z.T @ (y - ym), assume_a="pos")
    return (xt - mean) / scale @ w + ym


def r2_boot(y, preds, groups, rng):
    """Pooled R^2 of each prediction and its 95 % interval over groups (resampled whole)."""
    g, inv = np.unique(groups, return_inverse=True)
    n = np.bincount(inv)
    sy = np.bincount(inv, y)
    syy = np.bincount(inv, y * y)
    sse = {k: np.bincount(inv, (p - y) ** 2) for k, p in preds.items()}
    sel = rng.integers(len(g), size=(BOOT, len(g)))

    def r2(e, idx=None):
        if idx is None:
            tot = syy.sum() - sy.sum() ** 2 / n.sum()
            return 1 - e.sum() / tot
        nn, s1, s2 = n[idx].sum(1), sy[idx].sum(1), syy[idx].sum(1)
        return 1 - e[idx].sum(1) / (s2 - s1**2 / nn)

    out = {}
    for k, e in sse.items():
        b = r2(e, sel)
        out[k] = {"r2": float(r2(e)), "ci": np.quantile(b, [0.025, 0.975]).tolist()}
        out[k]["_boot"] = b
    return out


def rmse_pair(y, a, b, groups, rng):
    g, inv = np.unique(groups, return_inverse=True)
    n = np.bincount(inv)
    ea, eb = np.bincount(inv, (a - y) ** 2), np.bincount(inv, (b - y) ** 2)
    sel = rng.integers(len(g), size=(BOOT, len(g)))
    d = np.sqrt(ea[sel].sum(1) / n[sel].sum(1)) - np.sqrt(eb[sel].sum(1) / n[sel].sum(1))
    ra, rb = np.sqrt(ea.sum() / n.sum()), np.sqrt(eb.sum() / n.sum())
    return {
        "rows": int(n.sum()),
        "groups": len(g),
        "candidate": float(ra),
        "comparator": float(rb),
        "delta": float(ra - rb),
        "delta_ci": np.quantile(d, [0.025, 0.975]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--bins", type=Path, required=True, help="labels.csv with bin_00_m2 ...")
    ap.add_argument("--era5", type=Path, required=True)
    ap.add_argument("--probes", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    bins = [f"bin_{k:02d}_m2" for k in range(N_BINS)]
    bands = [f"band_{j}_m2" for j in range(4)]
    lab = pd.read_csv(args.package / "labels.csv", dtype={"station_id": str}, low_memory=False)
    lab = lab.merge(
        pd.read_csv(args.bins, usecols=["sample_id", *bins]), on="sample_id", validate="1:1"
    ).merge(pd.read_csv(args.era5), on="sample_id", how="left", validate="1:1")
    lab = lab.set_index("sample_id")
    energy = lab[bins + bands].to_numpy(float)
    energy[lab[bins].sum(1).to_numpy() <= 0] = np.nan  # all-zero spectra: dropouts
    heights = pd.DataFrame(4 * np.sqrt(energy), index=lab.index, columns=bins + bands)
    wind = pd.DataFrame(wind_features(lab.reset_index()), index=lab.index)
    meta = [json.loads((p / "run.json").read_text()) for p in args.probes]
    folds = sorted({m["fold"] for m in meta})
    seeds = sorted({m["seed"] for m in meta})
    if folds != list(range(5)) or len(meta) != 5 * len(seeds):
        raise ValueError("one probe per fold and seed")
    if any(m["smoke"] or m.get("water_only") or not m.get("saved_features") for m in meta):
        raise ValueError("full whole-view probes with saved features required")
    preds = []  # one frame per fold: seed-mean predictions on its test strips
    seed_rows = []
    for k in folds:
        per_seed = []
        for p, m in zip(args.probes, meta, strict=True):
            if m["fold"] != k:
                continue
            rows = pd.read_csv(p / "feature_rows.csv")
            x = np.load(p / "features.npy").astype(np.float64)
            ids = rows.sample_id.to_numpy()
            role = rows.role.to_numpy()
            tr, te = role == "train", role == "test"
            out = pd.DataFrame(index=ids[te])
            h = heights.loc[ids].to_numpy()
            w = wind.loc[ids].to_numpy()
            ok_h = np.isfinite(h).all(1)
            ok_w = np.isfinite(w).all(1)
            # visibility curve: one ridge per bin (multi-output solves each column alone)
            for arm, feat, okx in [
                ("camera", x, np.ones(len(x), bool)),
                ("wind", w, ok_w),
                ("camera_wind", np.column_stack([x, w]), ok_w),
            ]:
                fit = tr & ok_h & okx
                take = te & okx
                pr = np.full((te.sum(), h.shape[1]), np.nan)
                pr[take[te]] = np.maximum(0, ridge_multi(feat[fit], h[fit], feat[take]))
                out[[f"{arm}__{c}" for c in heights.columns]] = pr
            out[[f"mean__{c}" for c in heights.columns]] = h[tr & ok_h].mean(0)
            # virtual anemometer
            for target, era in [
                ("wind_speed_mps", "era5_wind_mps"),
                ("wind_gust_mps", "era5_gust_mps"),
            ]:
                y = lab.loc[ids, target].to_numpy(float)
                e = lab.loc[ids, era].to_numpy(float)
                fit = tr & np.isfinite(y) & np.isfinite(e)
                out[f"camera__{target}"] = np.maximum(0, ridge_multi(x[fit], y[fit, None], x[te]))[
                    :, 0
                ]
                coef = np.polyfit(e[fit], y[fit], 1)
                out[f"era5__{target}"] = np.maximum(0, np.polyval(coef, e[te]))
                xe = np.column_stack([x, e])
                fte = te & np.isfinite(e)
                pr = np.full(te.sum(), np.nan)
                pr[fte[te]] = np.maximum(0, ridge_multi(xe[fit], y[fit, None], xe[fte]))[:, 0]
                out[f"camera_era5__{target}"] = pr
                # permutation control on the test strips: shuffle the camera's features
                # within buoy and ERA5-wind decile, keep ERA5
                test = pd.DataFrame({"st": lab.loc[ids[te], "station_id"].to_numpy(), "e": e[te]})
                test["bin"] = pd.qcut(test.e.rank(method="first"), 10, labels=False)
                rng = np.random.default_rng(1000 * k + m["seed"])
                perm = np.arange(te.sum())
                for _, idx in test.groupby(["st", "bin"]).indices.items():
                    perm[idx] = rng.permutation(perm[idx])
                xp = np.column_stack([x[te][perm], e[te]])
                okp = np.isfinite(e[te])
                pr = np.full(te.sum(), np.nan)
                pr[okp] = np.maximum(0, ridge_multi(xe[fit], y[fit, None], xp[okp]))[:, 0]
                out[f"camera_era5_permuted__{target}"] = pr
            out["seed"] = m["seed"]
            per_seed.append(out)
        both = pd.concat(per_seed)
        seed_rows.append(both.assign(fold=k))
        preds.append(both.drop(columns="seed").groupby(level=0).mean().assign(fold=k))
    pred = pd.concat(preds)
    data = lab.loc[pred.index]
    groups = data.station_group.to_numpy()
    rng = np.random.default_rng(0)
    curve = []
    ok_h = (
        heights.loc[pred.index].notna().all(1).to_numpy()
        & wind.loc[pred.index].notna().all(1).to_numpy()
    )
    for c in heights.columns:
        y = heights.loc[pred.index, c].to_numpy()[ok_h]
        r = r2_boot(
            y, {a: pred[f"{a}__{c}"].to_numpy()[ok_h] for a in ["camera", "wind", "camera_wind", "mean"]},
            groups[ok_h], rng,
        )  # fmt: skip
        gain = r["camera_wind"]["_boot"] - r["wind"]["_boot"]
        curve.append(
            {
                "target": c,
                **{f"r2_{a}": r[a]["r2"] for a in r},
                **{f"r2_{a}_lo": r[a]["ci"][0] for a in r},
                **{f"r2_{a}_hi": r[a]["ci"][1] for a in r},
                "gain_over_wind": r["camera_wind"]["r2"] - r["wind"]["r2"],
                "gain_lo": float(np.quantile(gain, 0.025)),
                "gain_hi": float(np.quantile(gain, 0.975)),
                "rows": int(ok_h.sum()),
            }
        )
    curve = pd.DataFrame(curve)
    report = {"visibility_rows": int(ok_h.sum()), "seeds": seeds}
    for target in ["wind_speed_mps", "wind_gust_mps"]:
        y = data[target].to_numpy(float)
        ok = np.isfinite(y) & pred[f"era5__{target}"].notna().to_numpy()
        ok &= pred[f"camera_era5__{target}"].notna().to_numpy()
        a = {
            arm: pred[f"{arm}__{target}"].to_numpy()[ok]
            for arm in ["camera", "era5", "camera_era5", "camera_era5_permuted"]
        }
        report[target] = {
            "camera_vs_era5": rmse_pair(y[ok], a["camera"], a["era5"], groups[ok], rng),
            "camera_era5_vs_era5": rmse_pair(y[ok], a["camera_era5"], a["era5"], groups[ok], rng),
            "permuted_vs_era5": rmse_pair(
                y[ok], a["camera_era5_permuted"], a["era5"], groups[ok], rng
            ),
        }
        # seed spread of the camera alone
        sr = pd.concat(seed_rows)
        report[target]["camera_rmse_by_seed"] = {
            int(s): float(
                np.sqrt(np.nanmean((g[f"camera__{target}"] - data.loc[g.index, target]) ** 2))
            )
            for s, g in sr.groupby("seed")
        }
    args.out.mkdir(parents=True)
    curve.to_csv(args.out / "visibility_curve.csv", index=False, float_format="%.4f")
    pred.to_parquet(args.out / "predictions.parquet")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
