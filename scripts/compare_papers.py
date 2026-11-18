"""Every published measure, and the v2 bars, from the buoy-camera wave-heights runs.

    python scripts/compare_papers.py --arm images runs/<A seed 0> runs/<A seed 1> runs/<A seed 2> \\
        [--arm horizon runs/<B seed 0> ...] --out runs/<name>

Each arm is judged by the mean of its seeds' log heights (a geometric mean of the
heights). On the test partition (the 12 buoys held out of training) it reports,
with 95 % intervals from a bootstrap over whole buoys:

* **Kamagata et al. 2026** (coastal video, held-out H_s RMSE 0.586 m, R² 0.107,
  Pearson 0.436): RMSE, R² and Pearson of H_s, and the regression slope.
* **Choi et al. 2020** (NDBC buoy cameras, 84 %): accuracy on calm (< 1.52 m) vs
  rough (> 3.05 m) strips, the predicted H_s split at 2.285 m.
* **Kim et al. 2023** (one fixed camera, 78.3 % snapshot): accuracy in Kim's five
  classes of 0.5 m from 0.75 to 3.25 m (strips outside left out). Kim's label is an
  "average wave height" whose definition is not verified; here it is H_s.
* The WMO classes of v0 (balanced accuracy, macro-F1), from the predicted H_s.
* **Bar V2-2**: RMSE of each height against ERA5's (total, wind-wave, swell; line
  fitted on the training strips), paired over the same strips.
* **Bar V2-1** (two arms): the second arm's RMSE minus the first's, for H_s and the
  wind-wave height (Yang et al. 2026's image + attitude question on real data).

Writes ``<out>/compare.json`` and ``<out>/report.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir, load_yaml, repo_root
from wave_analysis.datasets.sea_state import linear_correction, nowcast_at, sea_state_class
from wave_analysis.evaluation.classification import classification_report
from wave_analysis.evaluation.regression import pearson, r2, regression_slope, rmse
from wave_analysis.evaluation.uncertainty import block_bootstrap_ci, paired_block_bootstrap_ci

HEIGHTS = ("hs", "wind_wave", "swell")
LABELS = {"hs": "Hs_label_m", "wind_wave": "wind_wave_label_m", "swell": "swell_label_m"}
ERA5 = {"hs": "era5_hs_m", "wind_wave": "era5_windsea_m", "swell": "era5_swell_m"}
CHOI_LOW, CHOI_HIGH = 1.52, 3.05
KIM_EDGES = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25]
PUBLISHED = {
    "kamagata2026": {"rmse_m": 0.586, "r2": 0.107, "pearson": 0.436},
    "choi2020": {"accuracy": 0.84},
    "kim2023": {"accuracy_snapshot": 0.783, "accuracy_video_180s": 0.932},
}


def arm_predictions(runs: list[Path]) -> tuple[pd.DataFrame, dict]:
    """The seeds' predictions merged; ``pred_*_m`` is the geometric mean over the seeds."""
    metas = [json.loads((r / "metrics.json").read_text())["run"] for r in runs]
    if len({(m["config"], m["split"], m.get("aux")) for m in metas}) != 1:
        raise SystemExit(f"the runs of an arm differ in config, split or aux: {runs}")
    frames = [
        pd.read_csv(r / "predictions.csv", dtype={"station_id": str}).sort_values("sample_id")
        for r in runs
    ]
    base = frames[0].reset_index(drop=True)
    for r, f in zip(runs[1:], frames[1:], strict=True):
        if not f["sample_id"].reset_index(drop=True).equals(base["sample_id"]):
            raise SystemExit(f"{r} has other strips than {runs[0]}")
    for h in HEIGHTS:
        logs = np.mean([np.log(f[f"pred_{h}_m"].to_numpy()) for f in frames], axis=0)
        base[f"pred_{h}_m"] = np.exp(logs)
    return base, metas[0]


def ci(y: np.ndarray, p: np.ndarray, fn: Any, g: np.ndarray, kw: dict) -> dict:
    point, lo, hi = block_bootstrap_ci(y, p, fn, groups=g, **kw)
    return {"value": float(point), "ci": [float(lo), float(hi)]}


def acc(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))


def score_arm(d: pd.DataFrame, kw: dict, bounds: list[float]) -> dict:
    t = d[d["partition"] == "test"]
    g = t["station_id"].to_numpy()
    out: dict[str, Any] = {"n_test": len(t), "test_buoys": int(t["station_id"].nunique())}
    for h in HEIGHTS:
        ok = t[f"true_{h}_m"].notna().to_numpy()
        y, p, gg = t[f"true_{h}_m"].to_numpy()[ok], t[f"pred_{h}_m"].to_numpy()[ok], g[ok]
        out[h] = {
            "n": int(ok.sum()),
            "rmse_m": ci(y, p, rmse, gg, kw),
            "bias_m": float(np.mean(p - y)),
            "r2": ci(y, p, r2, gg, kw),
            "pearson": ci(y, p, pearson, gg, kw),
            "slope": float(regression_slope(y, p)[0]),
        }
    y, p = t["true_hs_m"].to_numpy(), t["pred_hs_m"].to_numpy()
    c = (y < CHOI_LOW) | (y > CHOI_HIGH)
    out["choi2020_accuracy"] = ci(
        (y[c] > CHOI_HIGH).astype(float),
        (p[c] > (CHOI_LOW + CHOI_HIGH) / 2).astype(float),
        acc,
        g[c],
        kw,
    )
    out["choi2020_n"] = int(c.sum())
    k = (y >= KIM_EDGES[0]) & (y < KIM_EDGES[-1])
    yk, pk = np.digitize(y[k], KIM_EDGES[1:-1]), np.clip(np.digitize(p[k], KIM_EDGES[1:-1]), 0, 4)
    out["kim2023_accuracy"] = ci(yk.astype(float), pk.astype(float), acc, g[k], kw)
    out["kim2023_n"] = int(k.sum())
    yc, pc = sea_state_class(y, bounds), sea_state_class(p, bounds)
    rep = classification_report(yc, pc, n_classes=len(bounds) + 1)
    out["wmo_classes"] = {
        "balanced_accuracy": float(rep["balanced_accuracy"]),
        "f1_macro": float(rep["f1_macro"]),
    }
    return out


def era5_in_metres(d: pd.DataFrame, labels: pd.DataFrame, era: pd.DataFrame) -> pd.DataFrame:
    """ERA5's three heights at each strip (interpolated, no value across 3 h), corrected by a line on the training strips."""
    lab = labels.copy()
    lab["time_utc"] = pd.to_datetime(lab["time_utc"], utc=True)
    for h, col in ERA5.items():
        x = np.full(len(lab), np.nan)
        for sid, gr in era.groupby("station_id"):
            rows = (lab["station_id"] == str(sid)).to_numpy()
            if rows.any() and col in gr:
                x[rows] = nowcast_at(
                    lab.loc[rows, "time_utc"], gr.set_index("time_utc")[col].dropna(), max_gap="3h"
                )
        train = (lab["split_station"] == "train").to_numpy()
        a, b = linear_correction(x, lab[LABELS[h]].to_numpy(dtype=float), train)
        lab[f"era5_{h}_m"] = a + b * x
    return d.merge(
        lab[["sample_id", *[f"era5_{h}_m" for h in HEIGHTS]]], on="sample_id", how="left"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", nargs="+", action="append", required=True, metavar=("NAME", "RUN"),
                    help="an arm's name, then its runs (seeds); give --arm twice for the V2-1 comparison")  # fmt: skip
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--era5", type=Path, default=data_dir("processed") / "era5/buoycam/era5_at_buoys.parquet"
    )
    args = ap.parse_args(argv)
    arms = {a[0]: [Path(r) for r in a[1:]] for a in args.arm}
    preds, metas = {}, {}
    for name, runs in arms.items():
        preds[name], metas[name] = arm_predictions(runs)
    meta = next(iter(metas.values()))
    cfg = load_yaml(repo_root() / "configs/experiments" / f"{meta['config']}.yaml")
    b = cfg["evaluation"]["bootstrap"]
    kw = {"n_boot": int(b["n_boot"]), "alpha": float(b["alpha"]), "seed": int(b["seed"])}
    bounds = [float(x) for x in cfg["task"]["classes"]["bounds_m"]]
    labels = pd.read_csv(
        repo_root() / cfg["package"]["out"] / "labels.csv", dtype={"station_id": str}
    )
    era = pd.read_parquet(args.era5)

    res: dict[str, Any] = {
        "published": PUBLISHED,
        "arms": {},
        "v2_2_vs_era5": {},
        "seeds": {n: [str(r) for r in rs] for n, rs in arms.items()},
    }
    for name, d in preds.items():
        res["arms"][name] = score_arm(d, kw, bounds)
        e = era5_in_metres(d, labels, era)
        t = e[e["partition"] == "test"]
        res["v2_2_vs_era5"][name] = {}
        for h in HEIGHTS:
            ok = (t[f"true_{h}_m"].notna() & t[f"era5_{h}_m"].notna()).to_numpy()
            y, pc, pe = (t[c].to_numpy()[ok] for c in (f"true_{h}_m", f"pred_{h}_m", f"era5_{h}_m"))
            diff, lo, hi = paired_block_bootstrap_ci(
                y, pc, pe, rmse, groups=t["station_id"].to_numpy()[ok], **kw
            )
            res["v2_2_vs_era5"][name][h] = {"n": int(ok.sum()), "camera_rmse_m": rmse(y, pc), "era5_rmse_m": rmse(y, pe),
                                            "difference_m": [float(diff), float(lo), float(hi)], "camera_better": bool(hi < 0)}  # fmt: skip
    names = list(preds)
    if len(names) == 2:
        a, bname = names
        da, db = preds[a], preds[bname]
        if not da["sample_id"].equals(db["sample_id"]):
            raise SystemExit("the two arms have other strips")
        t = da["partition"] == "test"
        res["v2_1"] = {"first": a, "second": bname}
        for h in ("hs", "wind_wave"):
            ok = (t & da[f"true_{h}_m"].notna()).to_numpy()
            diff, lo, hi = paired_block_bootstrap_ci(
                da[f"true_{h}_m"].to_numpy()[ok], db[f"pred_{h}_m"].to_numpy()[ok], da[f"pred_{h}_m"].to_numpy()[ok],
                rmse, groups=da["station_id"].to_numpy()[ok], **kw,
            )  # fmt: skip
            res["v2_1"][h] = {
                "rmse_second_minus_first_m": [float(diff), float(lo), float(hi)],
                "second_better": bool(hi < 0),
            }
        res["v2_1"]["met"] = all(res["v2_1"][h]["second_better"] for h in ("hs", "wind_wave"))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "compare.json").write_text(json.dumps(res, indent=1, default=str))
    lines = [f"# Buoy-camera heights against published results ({', '.join(names)})\n"]
    for name, r in res["arms"].items():
        hs = r["hs"]
        lines += [
            f"## {name} ({len(arms[name])} seeds averaged; {r['n_test']:,} test strips, {r['test_buoys']} buoys)\n",
            f"- Kamagata 2026 (RMSE 0.586 m, R2 0.107, Pearson 0.436): RMSE {hs['rmse_m']['value']:.3f} m "
            f"[{hs['rmse_m']['ci'][0]:.3f}, {hs['rmse_m']['ci'][1]:.3f}], R2 {hs['r2']['value']:.3f} "
            f"[{hs['r2']['ci'][0]:.3f}, {hs['r2']['ci'][1]:.3f}], Pearson {hs['pearson']['value']:.3f} "
            f"[{hs['pearson']['ci'][0]:.3f}, {hs['pearson']['ci'][1]:.3f}], slope {hs['slope']:.2f}, bias {hs['bias_m']:+.3f} m",
            f"- Choi 2020 (84 %): {r['choi2020_accuracy']['value']:.1%} [{r['choi2020_accuracy']['ci'][0]:.1%}, "
            f"{r['choi2020_accuracy']['ci'][1]:.1%}] on {r['choi2020_n']:,} calm/rough strips",
            f"- Kim 2023 (78.3 % snapshot): {r['kim2023_accuracy']['value']:.1%} [{r['kim2023_accuracy']['ci'][0]:.1%}, "
            f"{r['kim2023_accuracy']['ci'][1]:.1%}] on {r['kim2023_n']:,} strips in 0.75-3.25 m",
            f"- WMO classes from predicted H_s: balanced accuracy {r['wmo_classes']['balanced_accuracy']:.3f}, "
            f"macro-F1 {r['wmo_classes']['f1_macro']:.3f}",
            *[f"- {h}: RMSE {r[h]['rmse_m']['value']:.3f} m, bias {r[h]['bias_m']:+.3f} m, R2 {r[h]['r2']['value']:.3f}, "
              f"Pearson {r[h]['pearson']['value']:.3f} (n {r[h]['n']:,})" for h in HEIGHTS],
            "- V2-2 against ERA5 (RMSE camera minus ERA5, interval): "
            + "; ".join(f"{h} {v['camera_rmse_m']:.3f} vs {v['era5_rmse_m']:.3f} m, {v['difference_m'][0]:+.3f} "
                        f"[{v['difference_m'][1]:+.3f}, {v['difference_m'][2]:+.3f}]" for h, v in res['v2_2_vs_era5'][name].items()),
            "",
        ]  # fmt: skip
    if "v2_1" in res:
        v = res["v2_1"]
        lines += [f"## V2-1: {v['second']} minus {v['first']} (RMSE, m)\n"] + [
            f"- {h}: {v[h]['rmse_second_minus_first_m'][0]:+.3f} [{v[h]['rmse_second_minus_first_m'][1]:+.3f}, "
            f"{v[h]['rmse_second_minus_first_m'][2]:+.3f}]" for h in ("hs", "wind_wave")
        ] + [f"\n**V2-1 {'met' if v['met'] else 'not met'}.**"]  # fmt: skip
    text = "\n".join(lines) + "\n"
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
