#!/usr/bin/env python
"""Stage 3 scores: the camera's periods, direction and wind-sea spectrum against ERA5, out of fold.

    python scripts/buoycam_spectral_oof.py --runs runs/<15 stage-3 runs> \\
        --stage1-oof runs/<stage-1 oof>/oof_<winning arm>.csv --out runs/<name>

Choices fixed on 2026-11-18, before any stage-3 run (``docs/experiments/
ndbc_buoycam_sea_state.md``, "Stage 3"). The runs cover the five folds with the same
seeds; a fold's seeds are merged (scalars: mean of logs; direction: mean of unit
vectors; shape: mean of shares), and the five folds' test strips are every strip once.

* **ERA5 at the buoy**, interpolated to each strip (no value across 3 h; directions
  through their cosine and sine). Periods are corrected by a line fitted on each fold's
  training strips, as the heights; directions are used as they are.
* **Bar S3, one line per output** (camera minus ERA5, paired bootstrap over whole
  buoys, 95 % interval below zero to pass):
  P1 peak period T_p against ERA5 ``pp1d`` (RMSE, s); P2 energy period Tm-1,0 against
  ``mwp`` (RMSE, s); P3 mean direction against ``mwd`` (mean absolute angle, degrees).
* **H_s checked for harm**: the stage-3 model's out-of-fold H_s RMSE minus stage 1's
  winning arm's on the same strips; harm if the interval lies above zero.
* **Beside, not part of the bar:** APD against ``mp2``; NDBC's wind-wave period
  against ``mpww``; the wind sea's Tm-1,0 against ``mpww``, from the direct output and
  from the predicted shape (sum of share / frequency); NDBC's swell period against
  ``mpts``; the direction by sun height (the sun is the compass) and against NDBC's
  ``.spec`` MWD; the shape's mean Kullback-Leibler divergence and Hellinger distance
  from the buoy's, against each fold's mean training shape (the no-camera guess); the
  heights.

Writes ``<out>/spectral.json``, ``report.md`` and ``oof_spectral.csv``.
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

from wave_analysis.config import data_dir, load_yaml, repo_root
from wave_analysis.datasets.sea_state import linear_correction, nowcast_at
from wave_analysis.evaluation.regression import rmse
from wave_analysis.evaluation.uncertainty import block_bootstrap_ci, paired_block_bootstrap_ci

#: Camera output -> (label column, ERA5 column, corrected by a line)
PAIRS = {
    "tp": ("tp_label_s", "era5_tp_s"),
    "tm10": ("tm10_label_s", "era5_tm10_s"),
    "apd": ("apd_label_s", "era5_tm02_s"),
    "wwp": ("wwp_label_s", "era5_windsea_tm10_s"),
    "tm10_ww": ("tm10_ww_label_s", "era5_windsea_tm10_s"),
    "tm10_ww_shape": ("tm10_ww_label_s", "era5_windsea_tm10_s"),
    "swp": ("swp_label_s", "era5_swell_tm10_s"),
}
BAR = {"P1": "tp", "P2": "tm10"}
N_BINS = 46


def circ_err(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(p) - np.asarray(y) + 180.0) % 360.0 - 180.0)


def circ_mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(circ_err(y, p)))


def merge(runs: list[Path]) -> pd.DataFrame:
    """Every fold's val and test strips, its seeds merged."""
    by: dict[str, list[Path]] = defaultdict(list)
    for r in runs:
        by[json.loads((r / "metrics.json").read_text())["run"]["split"]].append(r)
    if len({len(v) for v in by.values()}) != 1:
        raise SystemExit(
            f"folds hold different numbers of seeds: { {k: len(v) for k, v in by.items()} }"
        )
    parts = []
    for k, rs in sorted(by.items()):
        fr = [pd.read_csv(r / "predictions.csv", dtype={"station_id": str}).sort_values(["partition", "sample_id"])
              .reset_index(drop=True) for r in rs]  # fmt: skip
        base = fr[0][["sample_id", "partition", "station_id"]].copy()
        if any(not f["sample_id"].equals(base["sample_id"]) for f in fr[1:]):
            raise SystemExit(f"{k}: seeds have other strips")
        names = [c.removeprefix("pred_") for c in fr[0].columns if c.startswith("pred_")
                 and not c.startswith(("pred_wsf_", "pred_dir_"))]  # fmt: skip
        for nm in names:
            base[f"pred_{nm}"] = np.exp(np.mean([np.log(f[f"pred_{nm}"]) for f in fr], axis=0))
            base[f"true_{nm}"] = fr[0][f"true_{nm}"]
        if "pred_dir_rel_cos" in fr[0]:
            c = np.mean([f["pred_dir_rel_cos"] for f in fr], axis=0)
            s = np.mean([f["pred_dir_rel_sin"] for f in fr], axis=0)
            base["pred_dir_rel_deg"] = np.degrees(np.arctan2(s, c))
            base["dir_agreement"] = np.hypot(c, s)  # 1 when the seeds agree
        shp = np.mean(
            [f[[f"pred_wsf_{j:02d}" for j in range(N_BINS)]].to_numpy() for f in fr], axis=0
        )
        for j in range(N_BINS):
            base[f"pred_wsf_{j:02d}"] = shp[:, j]
        base["cv_fold"] = k.removeprefix("fold")
        base["n_seeds"] = len(rs)
        parts.append(base)
    return pd.concat(parts, ignore_index=True)


def era5_at(lab: pd.DataFrame, era: pd.DataFrame, col: str, circular: bool = False) -> np.ndarray:
    """ERA5 ``col`` interpolated to every strip of ``lab`` (directions through cos and sin)."""
    out = np.full(len(lab), np.nan)
    for sid, gr in era.groupby("station_id"):
        rows = (lab["station_id"] == str(sid)).to_numpy()
        if not rows.any() or col not in gr:
            continue
        ser = gr.set_index("time_utc")[col].dropna()
        t = lab.loc[rows, "time_utc"]
        if circular:
            r = np.deg2rad(ser)
            c = nowcast_at(t, np.cos(r), max_gap="3h")
            s = nowcast_at(t, np.sin(r), max_gap="3h")
            out[rows] = np.degrees(np.arctan2(s, c)) % 360
        else:
            out[rows] = nowcast_at(t, ser, max_gap="3h")
    return out


def compare(
    y: np.ndarray, cam: np.ndarray, era: np.ndarray, g: np.ndarray, metric: Any, kw: dict
) -> dict:
    ok = np.isfinite(y) & np.isfinite(cam) & np.isfinite(era)
    d, lo, hi = paired_block_bootstrap_ci(y[ok], cam[ok], era[ok], metric, groups=g[ok], **kw)
    return {"n": int(ok.sum()), "buoys": len(set(g[ok])), "camera": float(metric(y[ok], cam[ok])),
            "era5": float(metric(y[ok], era[ok])), "camera_minus_era5": [d, lo, hi], "camera_better": bool(hi < 0)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--stage1-oof", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs/experiments/ndbc_buoycam_spectral_v3cv.yaml",
    )
    ap.add_argument(
        "--era5", type=Path, default=data_dir("processed") / "era5/buoycam/era5_at_buoys.parquet"
    )
    args = ap.parse_args(argv)  # fmt: skip
    cfg = load_yaml(args.config)
    b = cfg["evaluation"]["bootstrap"]
    kw = {"n_boot": int(b["n_boot"]), "alpha": float(b["alpha"]), "seed": int(b["seed"])}
    pkg = repo_root() / cfg["package"]["out"]
    freq = np.asarray(json.loads((pkg / "frequencies.json").read_text())["frequency_hz"])
    lab = pd.read_csv(
        pkg / "labels.csv", dtype={"station_id": str, "cv_fold": str}, low_memory=False
    )
    lab["time_utc"] = pd.to_datetime(lab["time_utc"], utc=True)
    allp = merge(args.runs)
    oof = allp[allp["partition"] == "test"].drop(columns="partition").reset_index(drop=True)
    smoke = len(oof) != int(lab["cv_fold"].notna().sum())
    oof = oof.merge(lab, on=["sample_id", "station_id"], how="left", suffixes=("", "_lab"))
    g = oof["station_id"].to_numpy()
    era = pd.read_parquet(args.era5)
    shapes = oof[[f"pred_wsf_{j:02d}" for j in range(N_BINS)]].to_numpy()
    oof["pred_tm10_ww_shape"] = (shapes / freq).sum(axis=1)
    oof["pred_dir_deg"] = (oof["pred_dir_rel_deg"] + oof["sun_azimuth_deg"]) % 360
    res: dict[str, Any] = {"n": len(oof), "buoys": int(oof["station_id"].nunique()), "smoke": smoke,
                           "seeds": int(oof["n_seeds"].iloc[0]), "periods": {}, "bar": {}}  # fmt: skip
    fold = oof["cv_fold"].astype(str).to_numpy()
    lab_fold = lab["cv_fold"].astype(str)
    for name, (lcol, ecol) in PAIRS.items():
        raw_all = era5_at(lab, era, ecol)
        corr = np.full(len(oof), np.nan)
        idx = (
            pd.Series(np.arange(len(lab)), index=lab["sample_id"]).loc[oof["sample_id"]].to_numpy()
        )
        for k in sorted(set(fold)):
            a, bb = linear_correction(
                raw_all, lab[lcol].to_numpy(float), (lab[f"split_fold{k}"] == "train").to_numpy()
            )
            rows = fold == k
            corr[rows] = a + bb * raw_all[idx[rows]]
        oof[f"era5_{name}"] = corr
        y = oof[lcol].to_numpy(float)
        res["periods"][name] = compare(y, oof[f"pred_{name}"].to_numpy(float), corr, g, rmse, kw)
    del lab_fold
    oof["era5_dir_deg"] = era5_at(oof, era, "era5_mwd_deg", circular=True)
    y = oof["mwd_label_deg"].to_numpy(float)
    res["direction"] = compare(
        y, oof["pred_dir_deg"].to_numpy(), oof["era5_dir_deg"].to_numpy(), g, circ_mae, kw
    )
    res["direction_vs_ndbc_mwd"] = compare(oof["mwd_ndbc_label_deg"].to_numpy(float), oof["pred_dir_deg"].to_numpy(),
                                           oof["era5_dir_deg"].to_numpy(), g, circ_mae, kw)  # fmt: skip
    hi_sun = (oof["sun_elevation_deg"] > 10).to_numpy()
    res["direction_by_sun"] = {
        lab_: compare(y[m], oof["pred_dir_deg"].to_numpy()[m], oof["era5_dir_deg"].to_numpy()[m], g[m], circ_mae, kw)
        for lab_, m in (("sun above 10 deg", hi_sun), ("sun at most 10 deg", ~hi_sun)) if m.sum() > 100
    }  # fmt: skip
    res["bar"] = {"P1": res["periods"]["tp"], "P2": res["periods"]["tm10"], "P3": res["direction"]}
    # Shape: KL(buoy || camera) and Hellinger, against each fold's mean training shape.
    true_s = oof[[f"wsf_{j:02d}" for j in range(N_BINS)]].to_numpy(float)
    oks = np.isfinite(true_s).all(axis=1)
    base = np.full_like(true_s, np.nan)
    for k in sorted(set(fold)):
        tr = (lab[f"split_fold{k}"] == "train").to_numpy()
        ms = np.nanmean(
            lab.loc[tr, [f"wsf_{j:02d}" for j in range(N_BINS)]].to_numpy(float), axis=0
        )
        base[fold == k] = ms / ms.sum()
    eps = 1e-6

    def kl(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.sum(np.where(t > 0, t * (np.log(t + eps) - np.log(p + eps)), 0.0), axis=1)

    def hel(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.sqrt(
            0.5 * np.sum((np.sqrt(np.clip(t, 0, None)) - np.sqrt(np.clip(p, 0, None))) ** 2, axis=1)
        )

    res["shape"] = {}
    for nm, fn in (("kl", kl), ("hellinger", hel)):
        cam, bas = fn(true_s[oks], shapes[oks]), fn(true_s[oks], base[oks])
        d, lo, hi = paired_block_bootstrap_ci(np.zeros(oks.sum()), cam, bas, lambda _a, v: float(np.mean(v)),
                                              groups=g[oks], **kw)  # fmt: skip
        res["shape"][nm] = {
            "camera": float(cam.mean()),
            "mean_training_shape": float(bas.mean()),
            "camera_minus_mean": [d, lo, hi],
        }
    # Heights, and H_s harm against stage 1's winner on the same strips.
    res["heights"] = {}
    for h, lcol in (
        ("hs", "Hs_label_m"),
        ("wind_wave", "wind_wave_label_m"),
        ("swell", "swell_label_m"),
    ):
        y = oof[lcol].to_numpy(float)
        ok = np.isfinite(y)
        res["heights"][h] = {
            "rmse": list(
                block_bootstrap_ci(y[ok], oof[f"pred_{h}"].to_numpy()[ok], rmse, groups=g[ok], **kw)
            )
        }
    s1 = pd.read_csv(args.stage1_oof, dtype={"station_id": str})[
        ["sample_id", "pred_hs_m", "pred_wind_wave_m"]
    ]
    m = oof[
        ["sample_id", "station_id", "Hs_label_m", "wind_wave_label_m", "pred_hs", "pred_wind_wave"]
    ].merge(s1, on="sample_id")
    for h, (lcol, c3, c1) in {
        "hs": ("Hs_label_m", "pred_hs", "pred_hs_m"),
        "wind_wave": ("wind_wave_label_m", "pred_wind_wave", "pred_wind_wave_m"),
    }.items():
        ok = m[lcol].notna().to_numpy()
        d, lo, hi = paired_block_bootstrap_ci(m[lcol].to_numpy()[ok], m[c3].to_numpy()[ok], m[c1].to_numpy()[ok], rmse,
                                              groups=m["station_id"].to_numpy()[ok], **kw)  # fmt: skip
        res["heights"][h]["stage3_minus_stage1"] = [d, lo, hi]
        res["heights"][h]["harm"] = bool(lo > 0)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "spectral.json").write_text(json.dumps(res, indent=1, default=str))
    keep = ["sample_id", "station_id", "cv_fold", "time_utc", "sun_elevation_deg", "sun_azimuth_deg",
            "pred_dir_deg", "era5_dir_deg", "mwd_label_deg", "dir_agreement", "pred_tm10_ww_shape"]  # fmt: skip
    keep += [
        c
        for c in oof.columns
        if c.startswith(("pred_", "era5_")) and c not in keep and not c.startswith("pred_wsf_")
    ]
    oof[keep].to_csv(args.out / "oof_spectral.csv", index=False, float_format="%.4f")

    def line(nm: str, v: dict, unit: str) -> str:
        d = v["camera_minus_era5"]
        return (f"- {nm}: camera {v['camera']:.2f} {unit}, ERA5 {v['era5']:.2f}; difference {d[0]:+.2f} "
                f"[{d[1]:+.2f}, {d[2]:+.2f}] ({v['n']:,} strips, {v['buoys']} buoys)")  # fmt: skip

    lines = [f"# Stage 3: periods, direction and wind-sea spectrum, out of fold ({res['seeds']} seeds per fold)"
             + (" - SMOKE TEST" if smoke else "") + "\n", "## Bar S3\n",
             line("P1 peak period (pp1d), RMSE", res["bar"]["P1"], "s"),
             line("P2 energy period Tm-1,0 (mwp), RMSE", res["bar"]["P2"], "s"),
             line("P3 mean direction (mwd), mean abs angle", res["bar"]["P3"], "deg"), "",
             "## Beside\n"]  # fmt: skip
    lines += [
        line(f"{k}, RMSE", v, "s") for k, v in res["periods"].items() if k not in ("tp", "tm10")
    ]
    lines += [line(f"direction, {k}", v, "deg") for k, v in res["direction_by_sun"].items()]
    lines.append(line("direction against NDBC .spec MWD", res["direction_vs_ndbc_mwd"], "deg"))
    for nm, v in res["shape"].items():
        d = v["camera_minus_mean"]
        lines.append(f"- wind-sea shape {nm}: camera {v['camera']:.3f}, mean training shape {v['mean_training_shape']:.3f}; "
                     f"difference {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")  # fmt: skip
    for h, v in res["heights"].items():
        extra = (
            f"; minus stage 1 {v['stage3_minus_stage1'][0]:+.3f} [{v['stage3_minus_stage1'][1]:+.3f}, {v['stage3_minus_stage1'][2]:+.3f}]"
            if "stage3_minus_stage1" in v
            else ""
        )
        lines.append(
            f"- {h}: RMSE {v['rmse'][0]:.3f} m [{v['rmse'][1]:.3f}, {v['rmse'][2]:.3f}]{extra}"
        )
    lines.append("\n" + ", ".join(f"{k} {'met' if v['camera_better'] else 'not met'}" for k, v in res["bar"].items())
                 + f"; H_s harm: {'yes' if res['heights']['hs'].get('harm') else 'no'}")  # fmt: skip
    text = "\n".join(lines) + "\n"
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
