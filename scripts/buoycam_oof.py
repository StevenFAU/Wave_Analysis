#!/usr/bin/env python
"""Out-of-fold scores of the buoy-camera heights under grouped cross-validation, and bar B2-CV.

    python scripts/buoycam_oof.py --arm A runs/<fold x seed runs> \\
        [--arm dinov3-gentle runs/<fold x seed runs>] --out runs/<name>

Choices fixed on 2026-11-18, before any run on the folds (``docs/experiments/
ndbc_buoycam_sea_state.md``, "Grouped cross-validation"; config
``ndbc_buoycam_heights_v2cv.yaml``):

* **Out of fold.** Each arm's runs cover the five folds (``split`` fold0..fold4 in
  ``metrics.json``) with the same seeds in each. A fold's seeds are merged: the point
  is the mean of their log heights (as v2); the distribution is one log-normal with
  the first two moments of the seeds' (mean of means; mean of variances plus
  variance of means). The test strips of the five folds together are every strip
  once, each predicted by models that saw neither its buoy nor any buoy within 300 km.
* **Scores per height** over all 79 buoys, 95 % intervals from a bootstrap over whole
  buoys: RMSE, bias, R², Pearson, MAE and the **CRPS** of the merged distribution
  (in metres: the mean absolute error of the whole distribution; for a single number
  it is the absolute error, so it compares directly with the MAE). The **seed spread**:
  each seed's own out-of-fold RMSE (its five fold models). Every paper's measure
  (``compare_papers.score_arm``) on the out-of-fold heights.
* **90 % intervals** (buoy-level split conformal): score |log y - mu| / sigma; fold
  k's threshold from the out-of-fold scores of the other four folds' buoys, each
  buoy weighted equally (:func:`hierarchical_conformal_threshold`). Coverage (the
  share of true values inside; 0.90 is right) over all strips, as a mean over buoys,
  per buoy and per sea-state class (WMO, of the true height), and mean width in
  metres. Beside it: the distribution head's own mu +/- 1.645 sigma (uncalibrated),
  the same conformal rule calibrated on each fold's own validation buoys, and an
  unnormalised score (|log y - log point|) to show what the spread adds.
* **Camera against ERA5**, out of fold: ERA5's three heights at the buoy with a line
  fitted on each fold's training strips.
* **Bar B2-CV** (two arms, the second judged against the first): the second arm's
  RMSE minus the first's, H_s **and** wind-wave height, each 95 % interval below zero,
  paired bootstrap over whole buoys (all 79). Beside it: the same over whole station
  groups (43; buoys of one group share weather) and the CRPS difference.

Writes ``<out>/oof_<arm>.csv`` (every strip's out-of-fold heights, distribution and
interval; the input of the fusion in stage 2), ``oof.json`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from compare_papers import HEIGHTS, LABELS, score_arm  # scripts/ is on sys.path
from wave_analysis.config import data_dir, load_yaml, repo_root
from wave_analysis.datasets.sea_state import linear_correction, nowcast_at, sea_state_class
from wave_analysis.evaluation.regression import mae, pearson, r2, rmse
from wave_analysis.evaluation.uncertainty import (
    block_bootstrap_ci,
    hierarchical_conformal_threshold,
    lognormal_crps,
    merge_gaussians,
    paired_block_bootstrap_ci,
)

ERA5 = {"hs": "era5_hs_m", "wind_wave": "era5_windsea_m", "swell": "era5_swell_m"}
Z90 = 1.6448536269514722
#: The station split's sealed test buoys: their out-of-fold numbers sit beside the headline.
SEALED_TEST = ("41002", "41043", "41047", "41049", "42057", "44065", "46059", "46066", "46085",
               "51000", "51002", "51003")  # fmt: skip


def fold_runs(runs: list[Path]) -> tuple[dict[str, list[Path]], dict]:
    """Runs grouped by fold; every fold must hold the same seeds and one config."""
    by: dict[str, dict[int, Path]] = defaultdict(dict)
    metas = []
    for r in runs:
        m = json.loads((r / "metrics.json").read_text())["run"]
        metas.append(m)
        if m["seed"] in by[m["split"]]:
            raise SystemExit(f"two runs of {m['split']} with seed {m['seed']}: {r}")
        by[m["split"]][int(m["seed"])] = r
    if len({(m["config"], m.get("aux"), m["settings"]["backbone"]) for m in metas}) != 1:
        raise SystemExit("the runs of an arm differ in config, aux or backbone")
    seeds = {k: sorted(v) for k, v in by.items()}
    if len({tuple(s) for s in seeds.values()}) != 1:
        raise SystemExit(f"folds hold different seeds: {seeds}")
    return {k: [v[s] for s in sorted(v)] for k, v in sorted(by.items())}, metas[0]


def merge_fold(runs: list[Path]) -> pd.DataFrame:
    """One fold's val and test strips: seed-mean point, merged distribution, per-seed points."""
    frames = [pd.read_csv(r / "predictions.csv", dtype={"station_id": str}).sort_values(["partition", "sample_id"])
              .reset_index(drop=True) for r in runs]  # fmt: skip
    base = frames[0][["sample_id", "partition", "station_id"]].copy()
    for r, f in zip(runs[1:], frames[1:], strict=True):
        if not f["sample_id"].equals(base["sample_id"]):
            raise SystemExit(f"{r} has other strips than {runs[0]}")
    for h in HEIGHTS:
        base[f"true_{h}_m"] = frames[0][f"true_{h}_m"]
        logs = np.array([np.log(f[f"pred_{h}_m"].to_numpy()) for f in frames])
        base[f"pred_{h}_m"] = np.exp(logs.mean(axis=0))
        for i, lg in enumerate(logs):
            base[f"seed{i}_{h}_m"] = np.exp(lg)
        if f"mu_{h}_log" in frames[0]:
            mu, sd = merge_gaussians(
                [f[f"mu_{h}_log"].to_numpy() for f in frames],
                [f[f"sigma_{h}_log"].to_numpy() for f in frames],
            )
            base[f"mu_{h}_log"], base[f"sigma_{h}_log"] = mu, sd
    return base


def era5_by_fold(oof: pd.DataFrame, labels: pd.DataFrame, era: pd.DataFrame) -> pd.DataFrame:
    """ERA5's three heights at each strip, corrected by a line on the strip's fold's training strips."""
    lab = labels.copy()
    lab["time_utc"] = pd.to_datetime(lab["time_utc"], utc=True)
    raw = {}
    for h, col in ERA5.items():
        x = np.full(len(lab), np.nan)
        for sid, gr in era.groupby("station_id"):
            rows = (lab["station_id"] == str(sid)).to_numpy()
            if rows.any() and col in gr:
                x[rows] = nowcast_at(
                    lab.loc[rows, "time_utc"], gr.set_index("time_utc")[col].dropna(), max_gap="3h"
                )
        raw[h] = x
    out = pd.DataFrame({"sample_id": lab["sample_id"]})
    fold = lab["cv_fold"].astype(str).to_numpy()
    for h in HEIGHTS:
        y = lab[LABELS[h]].to_numpy(dtype=float)
        v = np.full(len(lab), np.nan)
        for k in sorted(set(fold) - {"", "nan"}):
            a, b = linear_correction(raw[h], y, (lab[f"split_fold{k}"] == "train").to_numpy())
            rows = fold == k
            v[rows] = a + b * raw[h][rows]
        out[f"era5_{h}_m"] = v
        out[f"era5_{h}_raw_m"] = raw[h]
    return oof.merge(out, on="sample_id", how="left")


def conformal(oof: pd.DataFrame, val: pd.DataFrame, h: str, level: float) -> dict[str, Any]:
    """90 % intervals for height ``h`` (see the module text); adds ``lo_/hi_<h>_m`` to ``oof``."""
    y = oof[f"true_{h}_m"].to_numpy(dtype=float)
    ok = np.isfinite(y)
    ly = np.log(np.maximum(y, 0.05))
    mu, sd = oof[f"mu_{h}_log"].to_numpy(), oof[f"sigma_{h}_log"].to_numpy()
    score = np.where(ok, np.abs(ly - mu) / sd, np.nan)
    score_u = np.where(ok, np.abs(ly - np.log(oof[f"pred_{h}_m"].to_numpy())), np.nan)
    fold, buoy = oof["cv_fold"].to_numpy(), oof["station_id"].to_numpy()
    q, qu, qv = np.full(len(oof), np.nan), np.full(len(oof), np.nan), np.full(len(oof), np.nan)
    thresholds = {}
    for k in sorted(set(fold)):
        rows, other = fold == k, (fold != k) & ok
        q[rows] = hierarchical_conformal_threshold(score[other], buoy[other], level)
        qu[rows] = hierarchical_conformal_threshold(score_u[other], buoy[other], level)
        v = val[(val["cv_fold"] == k) & val[f"true_{h}_m"].notna()]
        sv = (
            np.abs(np.log(np.maximum(v[f"true_{h}_m"].to_numpy(), 0.05)) - v[f"mu_{h}_log"])
            / v[f"sigma_{h}_log"]
        )
        qv[rows] = hierarchical_conformal_threshold(sv.to_numpy(), v["station_id"].to_numpy(), level)  # fmt: skip
        thresholds[k] = {"other_folds": float(q[rows][0]), "own_val_buoys": float(qv[rows][0]),
                         "unnormalised": float(qu[rows][0])}  # fmt: skip
    lo, hi = np.exp(mu - q * sd), np.exp(mu + q * sd)
    oof[f"lo_{h}_m"], oof[f"hi_{h}_m"] = lo, hi
    yy = np.maximum(y, 0.05)  # the model's floor: a true 0.0 counts as covered if the floor is

    def summary(lo_: np.ndarray, hi_: np.ndarray) -> dict[str, Any]:
        inside = ((yy >= lo_) & (yy <= hi_))[ok]
        per_buoy = pd.Series(inside).groupby(buoy[ok]).mean()
        return {"coverage": float(inside.mean()), "coverage_buoy_mean": float(per_buoy.mean()),
                "mean_width_m": float(np.mean((hi_ - lo_)[ok])),
                "median_width_m": float(np.median((hi_ - lo_)[ok]))}  # fmt: skip

    inside = ((yy >= lo) & (yy <= hi))[ok]
    per_buoy = pd.Series(inside).groupby(buoy[ok]).mean().sort_values()
    bins = ["smooth", "slight", "moderate", "rough_or_more"]
    cls = sea_state_class(y[ok], [0.5, 1.25, 2.5])
    by_bin = {bins[c]: {"n": int((cls == c).sum()), "coverage": float(inside[cls == c].mean())}
              for c in range(4) if (cls == c).any()}  # fmt: skip
    point = oof[f"pred_{h}_m"].to_numpy()
    return {
        "level": level,
        "thresholds_by_fold": thresholds,
        "conformal": summary(lo, hi)
        | {
            "per_buoy": {str(b): float(c) for b, c in per_buoy.items()},
            "buoys_below_0.80": int((per_buoy < 0.8).sum()),
            "buoys_below_0.50": int((per_buoy < 0.5).sum()),
            "worst_buoy": [str(per_buoy.index[0]), float(per_buoy.iloc[0])],
            "by_sea_state": by_bin,
        },
        "head_uncalibrated": summary(np.exp(mu - Z90 * sd), np.exp(mu + Z90 * sd)),
        "conformal_own_val_buoys": summary(np.exp(mu - qv * sd), np.exp(mu + qv * sd)),
        "conformal_unnormalised": summary(point * np.exp(-qu), point * np.exp(qu)),
    }


def ci(values: np.ndarray, groups: np.ndarray, kw: dict) -> list[float]:
    """Mean of per-strip values with a bootstrap interval over groups."""
    p, lo, hi = block_bootstrap_ci(np.zeros_like(values), values, lambda _a, b: float(np.mean(b)), groups=groups, **kw)  # fmt: skip
    return [p, lo, hi]


def score(oof: pd.DataFrame, seeds: int, kw: dict, bounds: list[float], level: float,
          val: pd.DataFrame) -> dict[str, Any]:  # fmt: skip
    g = oof["station_id"].to_numpy()
    out: dict[str, Any] = {"n": len(oof), "buoys": int(oof["station_id"].nunique()), "seeds": seeds,
                           "heights": {}, "papers": score_arm(oof, kw, bounds)}  # fmt: skip
    sealed = oof["station_id"].isin(SEALED_TEST).to_numpy()
    for h in HEIGHTS:
        ok = oof[f"true_{h}_m"].notna().to_numpy()
        y, p, gg = oof[f"true_{h}_m"].to_numpy()[ok], oof[f"pred_{h}_m"].to_numpy()[ok], g[ok]
        r: dict[str, Any] = {"n": int(ok.sum())}
        for name, fn in (("rmse_m", rmse), ("r2", r2), ("pearson", pearson), ("mae_m", mae)):
            r[name] = list(block_bootstrap_ci(y, p, fn, groups=gg, **kw))
        r["bias_m"] = float(np.mean(p - y))
        r["seed_rmse_m"] = [rmse(y, oof[f"seed{i}_{h}_m"].to_numpy()[ok]) for i in range(seeds)]
        r["by_fold_rmse_m"] = {str(k): rmse(y[f], p[f]) for k in sorted(oof["cv_fold"].unique())
                               if (f := (oof["cv_fold"].to_numpy()[ok] == k)).any()}  # fmt: skip
        s = sealed[ok]
        r["sealed_12_buoys_rmse_m"] = rmse(y[s], p[s])
        if f"mu_{h}_log" in oof:
            c = lognormal_crps(
                y, oof[f"mu_{h}_log"].to_numpy()[ok], oof[f"sigma_{h}_log"].to_numpy()[ok]
            )
            r["crps_m"] = ci(c, gg, kw)
            oof.loc[ok, f"crps_{h}_m"] = c
            r["intervals"] = conformal(oof, val, h, level)
        e = oof[f"era5_{h}_m"].to_numpy()[ok]
        both = np.isfinite(e)
        d, lo, hi = paired_block_bootstrap_ci(
            y[both], p[both], e[both], rmse, groups=gg[both], **kw
        )
        r["vs_era5"] = {"n": int(both.sum()), "camera_rmse_m": rmse(y[both], p[both]),
                        "era5_rmse_m": rmse(y[both], e[both]), "camera_mae_m": mae(y[both], p[both]),
                        "era5_mae_m": mae(y[both], e[both]), "difference_m": [d, lo, hi]}  # fmt: skip
        out["heights"][h] = r
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", nargs="+", action="append", required=True, metavar=("NAME", "RUN"),
                    help="an arm's name, then its runs (every fold x seed); the second arm is judged against the first")  # fmt: skip
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--era5", type=Path, default=data_dir("processed") / "era5/buoycam/era5_at_buoys.parquet"
    )
    args = ap.parse_args(argv)
    arms = {a[0]: [Path(r) for r in a[1:]] for a in args.arm}
    oofs, vals, metas, n_seeds = {}, {}, {}, {}
    smoke = False
    for name, runs in arms.items():
        by_fold, metas[name] = fold_runs(runs)
        smoke |= any(json.loads((r / "metrics.json").read_text())["run"].get("limit_per_partition")
                     for r in runs)  # fmt: skip
        parts = []
        for k, rs in by_fold.items():
            d = merge_fold(rs)
            d["cv_fold"] = k.removeprefix("fold")
            parts.append(d)
        n_seeds[name] = len(next(iter(by_fold.values())))
        allp = pd.concat(parts, ignore_index=True)
        oofs[name] = (
            allp[allp["partition"] == "test"].sort_values("sample_id").reset_index(drop=True)
        )
        vals[name] = allp[allp["partition"] == "val"].reset_index(drop=True)
        if oofs[name]["sample_id"].duplicated().any():
            raise SystemExit(f"{name}: a strip is in two folds' test buoys")
    meta = next(iter(metas.values()))
    cfg = load_yaml(repo_root() / "configs/experiments" / f"{meta['config']}.yaml")
    b = cfg["evaluation"]["bootstrap"]
    kw = {"n_boot": int(b["n_boot"]), "alpha": float(b["alpha"]), "seed": int(b["seed"])}
    bounds = [float(x) for x in cfg["task"]["classes"]["bounds_m"]]
    level = float(cfg.get("intervals", {}).get("level", 0.9))
    labels = pd.read_csv(repo_root() / cfg["package"]["out"] / "labels.csv",
                         dtype={"station_id": str, "cv_fold": str}, low_memory=False)  # fmt: skip
    era = pd.read_parquet(args.era5)
    n_all = int(labels["cv_fold"].notna().sum())
    res: dict[str, Any] = {"config": meta["config"], "arms": {}, "runs": {n: [str(r) for r in rs] for n, rs in arms.items()}}  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    for name, oof in oofs.items():
        if len(oof) != n_all and not smoke:
            raise SystemExit(f"{name}: {len(oof):,} out-of-fold strips, the package has {n_all:,}")
        oof = era5_by_fold(oof, labels, era)
        oof = oof.merge(labels[["sample_id", "station_group", "time_utc"]], on="sample_id")
        res["arms"][name] = score(oof, n_seeds[name], kw, bounds, level, vals[name])
        res["arms"][name]["backbone"] = metas[name]["settings"]["backbone"]
        oofs[name] = oof
        oof.to_csv(args.out / f"oof_{name}.csv", index=False, float_format="%.5f")
    names = list(oofs)
    if len(names) == 2:
        first, second = names
        a, c = oofs[first], oofs[second]
        if not a["sample_id"].equals(c["sample_id"]):
            raise SystemExit("the two arms have other strips")
        bar: dict[str, Any] = {"first": first, "second": second}
        for h in ("hs", "wind_wave", "swell"):
            ok = a[f"true_{h}_m"].notna().to_numpy()
            y = a[f"true_{h}_m"].to_numpy()[ok]
            pa, pc = a[f"pred_{h}_m"].to_numpy()[ok], c[f"pred_{h}_m"].to_numpy()[ok]
            r = {"by_buoy": list(paired_block_bootstrap_ci(y, pc, pa, rmse, groups=a["station_id"].to_numpy()[ok], **kw)),
                 "by_group": list(paired_block_bootstrap_ci(y, pc, pa, rmse, groups=a["station_group"].to_numpy()[ok], **kw))}  # fmt: skip
            if f"crps_{h}_m" in a and f"crps_{h}_m" in c:
                r["crps_by_buoy"] = ci(c[f"crps_{h}_m"].to_numpy()[ok] - a[f"crps_{h}_m"].to_numpy()[ok],
                                       a["station_id"].to_numpy()[ok], kw)  # fmt: skip
            bar[h] = r
        bar["met"] = all(bar[h]["by_buoy"][2] < 0 for h in ("hs", "wind_wave"))
        res["b2_cv"] = bar
    (args.out / "oof.json").write_text(json.dumps(res, indent=1, default=str))

    def f3(v: list[float]) -> str:
        return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"

    lines = [f"# Grouped cross-validation, out of fold ({', '.join(names)})"
             + (" - SMOKE TEST, not a result" if smoke else "") + "\n"]  # fmt: skip
    for name, r in res["arms"].items():
        lines.append(
            f"## {name} ({r['backbone']}; {r['seeds']} seeds per fold; {r['n']:,} strips, {r['buoys']} buoys)\n"
        )
        for h, v in r["heights"].items():
            lines.append(f"- **{h}**: RMSE {f3(v['rmse_m'])} m, MAE {v['mae_m'][0]:.3f}, CRPS "
                         + (f3(v["crps_m"]) if "crps_m" in v else "-")
                         + f" m, bias {v['bias_m']:+.3f}, R2 {v['r2'][0]:.3f}, Pearson {v['pearson'][0]:.3f}; "
                         f"seeds alone {', '.join(f'{x:.3f}' for x in v['seed_rmse_m'])}; folds "
                         + ", ".join(f"{k} {x:.3f}" for k, x in v["by_fold_rmse_m"].items())
                         + f"; sealed 12 buoys {v['sealed_12_buoys_rmse_m']:.3f}; ERA5 {v['vs_era5']['era5_rmse_m']:.3f} "
                         f"(camera minus ERA5 {f3(v['vs_era5']['difference_m'])}, n {v['vs_era5']['n']:,})")  # fmt: skip
            if "intervals" in v:
                iv = v["intervals"]
                cf = iv["conformal"]
                lines.append(
                    f"  - 90 % conformal: coverage {cf['coverage']:.3f} (buoy mean {cf['coverage_buoy_mean']:.3f}), "
                    f"mean width {cf['mean_width_m']:.2f} m; buoys below 0.80: {cf['buoys_below_0.80']}, worst "
                    f"{cf['worst_buoy'][0]} {cf['worst_buoy'][1]:.2f}; by sea state "
                    + ", ".join(f"{k} {x['coverage']:.2f} (n {x['n']:,})" for k, x in cf["by_sea_state"].items())
                    + f"; head alone {iv['head_uncalibrated']['coverage']:.3f}; own val buoys "
                    f"{iv['conformal_own_val_buoys']['coverage']:.3f} (width {iv['conformal_own_val_buoys']['mean_width_m']:.2f}); "
                    f"unnormalised {iv['conformal_unnormalised']['coverage']:.3f} (width {iv['conformal_unnormalised']['mean_width_m']:.2f})"
                )  # fmt: skip
        p = r["papers"]
        lines += [f"- papers: Kamagata RMSE {p['hs']['rmse_m']['value']:.3f}, R2 {p['hs']['r2']['value']:.3f}, "
                  f"Pearson {p['hs']['pearson']['value']:.3f}; Choi {p['choi2020_accuracy']['value']:.1%}; "
                  f"Kim {p['kim2023_accuracy']['value']:.1%} [{p['kim2023_accuracy']['ci'][0]:.1%}, {p['kim2023_accuracy']['ci'][1]:.1%}]; "
                  f"WMO balanced accuracy {p['wmo_classes']['balanced_accuracy']:.3f}", ""]  # fmt: skip
    if "b2_cv" in res:
        v = res["b2_cv"]
        lines.append(f"## Bar B2-CV: {v['second']} minus {v['first']} (RMSE, m)\n")
        for h in ("hs", "wind_wave", "swell"):
            lines.append(f"- {h}: by buoy {v[h]['by_buoy'][0]:+.4f} [{v[h]['by_buoy'][1]:+.4f}, {v[h]['by_buoy'][2]:+.4f}]; "
                         f"by group [{v[h]['by_group'][1]:+.4f}, {v[h]['by_group'][2]:+.4f}]"
                         + (f"; CRPS {v[h]['crps_by_buoy'][0]:+.4f} [{v[h]['crps_by_buoy'][1]:+.4f}, {v[h]['crps_by_buoy'][2]:+.4f}]"
                            if "crps_by_buoy" in v[h] else ""))  # fmt: skip
        lines.append(f"\n**Bar B2-CV {'met' if v['met'] else 'not met'}.**")
    text = "\n".join(lines) + "\n"
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
