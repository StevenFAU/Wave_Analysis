"""Does the buoy camera see the wind sea but not the swell? (research question Q-M8)

An exploratory check on a trained run's saved predictions; nothing is trained.
The questions were fixed on 2026-11-18, before the answer was looked at:

1. **Too calm with swell.** Within each true sea-state class, is the camera's
   share of calls one class or more too calm higher on strips where more of the
   wave energy is swell? Swell share = SwH² / (SwH² + WWH²), NDBC's own split
   of each record's spectrum into swell and wind waves. Strips are split into
   thirds of swell share *within* each class (so a bigger sea does not pass for
   more swell); the gain is the top third minus the bottom third, averaged over
   classes by size. ERA5 at the buoy, which models swell, is the contrast. The
   same is repeated within each buoy and class, which holds the buoy's place,
   light and camera fixed.
2. **What the calls follow.** Standardised slopes of the camera's expected
   class (sum of k * p_k) on log wind-wave height and log swell height, beside
   the same fit for the true class, which depends on both by construction.
3. **Per buoy.** Does a buoy's mean swell share predict how far the camera
   under-calls there (Spearman over buoys)? And do the test buoys carry more
   swell than the training buoys?

Intervals: bootstrap over whole buoys, with the config's draws and seed.

    python scripts/buoycam_swell_check.py runs/<run> [runs/<run> ...]

Writes ``swell_check.json`` into each run folder and prints a short summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from wave_analysis.config import load_yaml

FLOOR_M = 0.05  # heights below this are set to it before the logarithm


def boot(
    df: pd.DataFrame,
    fn: Callable[[pd.DataFrame], float],
    group: str,
    n: int,
    seed: int,
    alpha: float,
) -> list[float]:
    """Percentile interval of ``fn`` over resampled whole groups."""
    keys = df[group].unique()
    parts = {k: df[df[group] == k] for k in keys}
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n):
        pick = rng.choice(keys, size=keys.size, replace=True)
        reps.append(fn(pd.concat([parts[k] for k in pick], ignore_index=True)))
    lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def too_calm_gain(d: pd.DataFrame, pred: str, by: tuple[str, ...] = ("true",)) -> float:
    """Top-third minus bottom-third too-calm share within groups (class, or buoy and class), weighted by size."""
    gains, weights = [], []
    for _, c in d[d["true"] >= 1].groupby(list(by)):  # a smooth sea cannot be called too calm
        if len(c) < 30:
            continue
        q = c["share"].rank(pct=True, method="first")
        top, bot = c[q > 2 / 3], c[q <= 1 / 3]
        gains.append((top[pred] < top["true"]).mean() - (bot[pred] < bot["true"]).mean())
        weights.append(len(c))
    return float(np.average(gains, weights=weights)) if gains else float("nan")


def slopes(d: pd.DataFrame, y: str) -> tuple[float, float]:
    """Standardised OLS slopes of ``y`` on log wind-wave and log swell height."""
    x = np.column_stack(
        [np.log(d["wwh"].clip(lower=FLOOR_M)), np.log(d["swh"].clip(lower=FLOOR_M))]
    )
    x = (x - x.mean(0)) / x.std(0)
    yy = d[y].to_numpy(float)
    coef, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(len(d)), x]), (yy - yy.mean()) / yy.std(), rcond=None
    )
    return float(coef[1]), float(coef[2])


def check(run: Path, root: Path) -> dict:
    meta = json.loads((run / "metrics.json").read_text())["run"]
    split, name = meta["split"], meta["config"]
    cfg = load_yaml(root / "configs" / "experiments" / f"{name}.yaml")
    pkg = root / "data" / "processed" / "packages" / name
    b = cfg["evaluation"]["bootstrap"]
    n_boot, seed, alpha = int(b["n_boot"]), int(b["seed"]), float(b["alpha"])
    names = list(cfg["task"]["classes"]["names"])

    pred = pd.read_csv(run / "predictions.csv")
    lab = pd.read_csv(
        pkg / "labels.csv",
        usecols=[
            "sample_id",
            "station_id",
            f"split_{split}",
            "swell_height_m",
            "wind_wave_height_m",
        ],
    )
    base = pd.read_csv(pkg / "baselines.csv", usecols=["sample_id", f"era5_{split}"])
    d = (
        pred.drop(columns=["station_id"], errors="ignore")
        .merge(lab, on="sample_id")
        .merge(base, on="sample_id")
    )
    d = d.rename(
        columns={"swell_height_m": "swh", "wind_wave_height_m": "wwh", f"era5_{split}": "era5"}
    )
    d["expected"] = sum(k * d[f"p_{n}"] for k, n in enumerate(names))
    d = d[d["swh"].notna() & d["wwh"].notna()].copy()
    d["share"] = d["swh"] ** 2 / (d["swh"] ** 2 + d["wwh"] ** 2)
    t = d[d["partition"] == "test"]
    e = t[t["era5"] >= 0]
    ci = {"group": "station_id", "n": n_boot, "seed": seed, "alpha": alpha}

    out: dict = {
        "run": run.name,
        "split": split,
        "n_test": len(t),
        "n_test_buoys": int(t["station_id"].nunique()),
    }
    # 1. too calm with swell: camera, and ERA5 on the strips it covers
    out["too_calm_gain_camera"] = {
        "gain": too_calm_gain(t, "pred"),
        "ci": boot(t, lambda x: too_calm_gain(x, "pred"), **ci),
    }
    out["too_calm_gain_era5"] = {
        "gain": too_calm_gain(e, "era5"),
        "ci": boot(e, lambda x: too_calm_gain(x, "era5"), **ci),
    }
    # the same within each buoy and class: the buoy's place, light and camera are held fixed
    wb = ("station_id", "true")
    out["too_calm_gain_camera_within_buoy"] = {
        "gain": too_calm_gain(t, "pred", wb),
        "ci": boot(t, lambda x: too_calm_gain(x, "pred", wb), **ci),
    }
    out["too_calm_by_third"] = {}
    for who, frame, col in (("camera", t, "pred"), ("era5", e, "era5")):
        rows = []
        for _, c in frame[frame["true"] >= 1].groupby("true"):
            q = c["share"].rank(pct=True, method="first")
            rows.append(
                pd.DataFrame({"third": np.ceil(q * 3).clip(1, 3), "calm": c[col] < c["true"]})
            )
        r = pd.concat(rows).groupby("third")["calm"].mean()
        out["too_calm_by_third"][who] = {f"third_{int(k)}": float(v) for k, v in r.items()}
    # 2. what the calls follow
    cw, cs = slopes(t, "expected")
    lw, ls = slopes(t, "true")
    out["slopes"] = {
        "camera": {
            "wind_wave": cw,
            "swell": cs,
            "difference_ci": boot(t, lambda x: np.subtract(*slopes(x, "expected")), **ci),
        },
        "true_class": {"wind_wave": lw, "swell": ls},
    }
    # 3. per buoy, over all buoys held out of training (val and test for the station split)
    held = d[d["partition"] != "train"]
    pb = held.groupby("station_id").apply(
        lambda g: pd.Series(
            {
                "bias": float((g["pred"] - g["true"]).mean()),
                "share": float(g["share"].mean()),
                "n": len(g),
            }
        ),
        include_groups=False,
    )
    rho, p = stats.spearmanr(pb["share"], pb["bias"])
    labs_all = lab[lab["swell_height_m"].notna() & lab["wind_wave_height_m"].notna()].copy()
    labs_all["share"] = labs_all["swell_height_m"] ** 2 / (
        labs_all["swell_height_m"] ** 2 + labs_all["wind_wave_height_m"] ** 2
    )
    by_part = labs_all.groupby(f"split_{split}")["share"].mean()
    out["per_buoy"] = {
        "n_buoys": len(pb),
        "spearman_share_vs_bias": float(rho),
        "p": float(p),
        "buoys": pb.round(3).reset_index().to_dict(orient="records"),
    }
    out["mean_swell_share_by_partition"] = {str(k): float(v) for k, v in by_part.items()}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    for run in args.runs:
        r = check(run, root)
        (run / "swell_check.json").write_text(json.dumps(r, indent=2) + "\n")
        c, e, s = r["too_calm_gain_camera"], r["too_calm_gain_era5"], r["slopes"]
        print(f"== {run.name} ({r['n_test']:,} test strips, {r['n_test_buoys']} buoys)")
        print(
            f"  1. too-calm share, most-swell third minus least-swell third (within classes): camera "
            f"{c['gain']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]; ERA5 {e['gain']:+.3f} "
            f"[{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"
        )
        w = r["too_calm_gain_camera_within_buoy"]
        print(
            f"     within each buoy and class: camera {w['gain']:+.3f} [{w['ci'][0]:+.3f}, {w['ci'][1]:+.3f}]"
        )
        print(
            "     by third (least -> most swell): camera "
            + " ".join(f"{v:.2f}" for v in r["too_calm_by_third"]["camera"].values())
            + "; ERA5 "
            + " ".join(f"{v:.2f}" for v in r["too_calm_by_third"]["era5"].values())
        )
        print(
            f"  2. standardised slopes on log wind-wave / log swell height: camera {s['camera']['wind_wave']:.2f} / "
            f"{s['camera']['swell']:.2f} (difference CI [{s['camera']['difference_ci'][0]:+.2f}, "
            f"{s['camera']['difference_ci'][1]:+.2f}]); true class {s['true_class']['wind_wave']:.2f} / "
            f"{s['true_class']['swell']:.2f}"
        )
        pb, sh = r["per_buoy"], r["mean_swell_share_by_partition"]
        print(
            f"  3. over {pb['n_buoys']} held-out buoys, Spearman(swell share, camera bias) = "
            f"{pb['spearman_share_vs_bias']:+.2f} (p {pb['p']:.3f}); mean swell share by partition: "
            + ", ".join(f"{k} {v:.2f}" for k, v in sh.items())
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
