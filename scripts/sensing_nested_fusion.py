#!/usr/bin/env python
"""Test 1, nested: does the camera add to forecast + measured wind when the gate never sees the outer test groups?

    python scripts/sensing_nested_fusion.py --labels <v2cvn labels.csv> --oof <oof_dinov3-gentle.csv> \\
        --forecast runs/forecast-baseline-oof-v1.parquet \\
        --inner-forecast runs/forecast-baseline-inner-v1.parquet \\
        --inner-runs runs/buoy-pair*-nested-gentle --out runs/forecast-fusion-nested-v1

Fixed before any nested run (``docs/experiments/sensing_model_bar.md``, "Amendment,
2026-11-19: nested calibration"). For outer fold k every combiner is calibrated on the
strips of the four other folds j, each carrying predictions of the inner models of the
pair {j, k} (``scripts/sensing_nested_splits.py``), which saw neither j nor k; it is
then applied once to fold k with the cached outer predictions (the fold-k camera
ensemble and the fold-k forecast corrector).

* **Test 1 arms**, on the outer strips with every input and both targets, for the
  forecast with measured wind (the bar's comparator) and without: the forecast
  alone, the camera alone, the gated wind-sea swap (``PartitionFusion``, as the
  historical diagnostic), the swap with the camera vector permuted within buoy and
  calibration-defined quartile of the forecast's total (seeds 0-2, predictions
  averaged; movable fraction recorded), and the variance-weighted wind-sea energy.
* **Intervals:** paired 95 % percentile intervals of RMSE differences from 2,000
  resamples of whole station groups (RNG seed 0); sensitivities by whole buoy and
  with each group's mean squared error weighted equally. Bar: the upper end for
  gated swap minus forecast + measured wind below zero for H_s and for wind-wave height.
* **S2 check:** ``buoycam_partition_fusion.fit_fold`` calibrated the same way, the
  gated swap against its ERA5 + wind ridge, intervals as S2's script (whole buoys and
  groups, 1,000 resamples, seed 0). A check beside S2, not a bar.
* **Historical, beside it:** the same code with each outer fold's combiners calibrated
  on the other folds' outer out-of-fold predictions (the superseded arrangement), so
  the two differ only in where the calibration predictions come from.

``--inner-camera <csv>`` replaces ``--inner-runs`` with a table already merged
(``pair``, ``sample_id`` and the camera columns), for checks.

Writes ``<out>/rows.parquet`` (every scored strip and arm, both arrangements),
``<out>/summary.json`` and ``<out>/report.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buoycam_oof import merge_fold  # scripts/ is on sys.path
from buoycam_partition_fusion import (
    BAR,
    COMPARATOR,
    GATE,
    PartitionFusion,
    features,
    fit_fold,
    inputs_ok,
    predict_fold,
)
from sensing_fusion_diagnostic import CAMERA
from sensing_nested_splits import PAIRS, pair_name
from wave_analysis.evaluation.regression import rmse
from wave_analysis.evaluation.uncertainty import paired_block_bootstrap_ci

CAM_COLS = ["pred_hs_m", "pred_wind_wave_m", "pred_swell_m", "sigma_hs_log", "sigma_wind_wave_log"]
TARGETS = [("true_hs_m", "cmp_hs", "cam_hs"), ("true_wind_wave_m", "cmp_ww", "cam_ww")]
FC_NAMES = {"Hs_label_m": "cmp_hs", "wind_wave_label_m": "cmp_ww"}
BASELINES = ["model_wind", "model"]
ARMS = ["forecast", "camera", "fusion", "degraded", "varweight"]
CONTRASTS = [
    ("fusion", "forecast"),
    ("varweight", "forecast"),
    ("fusion", "degraded"),
    ("camera", "forecast"),
]
N_BOOT, MIN_BIN, PERM_SEEDS = 2000, 30, (0, 1, 2)
PAIR_NAMES = [pair_name(a, b) for a, b in PAIRS]


# ----------------------------------------------------------------------------- inputs


def inner_camera(runs: list[Path], oof: pd.DataFrame) -> pd.DataFrame:
    """Each pair's three seeds merged as in stage 1; its test strips with the outer table's other columns."""
    by: dict[str, dict[int, Path]] = {}
    for r in runs:
        m = json.loads((r / "metrics.json").read_text())["run"]
        if m["seed"] in by.setdefault(m["split"], {}):
            raise SystemExit(f"two runs of {m['split']} seed {m['seed']}")
        by[m["split"]][int(m["seed"])] = r
    if sorted(by) != PAIR_NAMES or any(sorted(v) != [0, 1, 2] for v in by.values()):
        raise SystemExit(
            f"need seeds 0-2 of {PAIR_NAMES}; have { {k: sorted(v) for k, v in by.items()} }"
        )
    frames = []
    for split in PAIR_NAMES:
        t = merge_fold([by[split][s] for s in (0, 1, 2)])
        t = t[t["partition"] == "test"]
        frames.append(t[["sample_id", *CAM_COLS, "true_hs_m"]].assign(pair=split))
    cam = pd.concat(frames, ignore_index=True)
    out = attach(cam, oof)
    bad = ~np.isclose(out["true_hs_m_run"], out["true_hs_m"], equal_nan=True)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} inner strips disagree with the outer table's labels")
    return out.drop(columns="true_hs_m_run")


def attach(cam: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Inner camera columns joined to every non-camera column of the outer table."""
    own = ("pred_", "seed", "mu_", "sigma_", "crps_", "lo_", "hi_")
    base = oof.drop(columns=[c for c in oof if c.startswith(own) or c == "partition"])
    out = cam.merge(base, on="sample_id", how="left", validate="m:1", suffixes=("_run", ""))
    a, b = out["pair"].str[4].astype(int), out["pair"].str[5].astype(int)
    fold = out["cv_fold"].astype(int)
    if not ((fold == a) | (fold == b)).all():
        raise SystemExit("an inner prediction lies outside its pair's two folds")
    return out


def inner_features(inner: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    f = features(inner, labels)
    f["pair"] = inner["pair"].to_numpy()
    return f


def with_forecast(f: pd.DataFrame, fc: pd.DataFrame, baseline: str, by_pair: bool) -> pd.DataFrame:
    """``f`` with the corrector's total and wind-wave height as ``cmp_hs`` and ``cmp_ww``."""
    keys = ["split", "sample_id"] if by_pair else ["sample_id"]
    wide = fc.set_index([*keys, "target"])[baseline].unstack("target").rename(columns=FC_NAMES)
    wide = wide.reset_index()
    if by_pair:
        wide = wide.rename(columns={"split": "pair"})
    return f.merge(
        wide, on=["pair", "sample_id"] if by_pair else ["sample_id"], how="left", validate="m:1"
    )


def complete(f: pd.DataFrame) -> pd.DataFrame:
    need = ["cmp_hs", "cmp_ww", "true_hs_m", "true_wind_wave_m", *GATE, *CAMERA]
    return f.dropna(subset=need)


def calibration(inner: pd.DataFrame, k: str) -> pd.DataFrame:
    """Strips of the folds other than k, each predicted by the inner model of its pair with k."""
    a, b = inner["pair"].str[4], inner["pair"].str[5]
    return inner[((a == k) | (b == k)) & (inner["cv_fold"] != k)]


# ----------------------------------------------------------------------------- arms


def degrade(test: pd.DataFrame, cuts: np.ndarray, seed: int) -> tuple[pd.DataFrame, float]:
    """The camera vector permuted within buoy and forecast-total quartile; share of strips moved."""
    copy = test.copy()
    bins = np.searchsorted(cuts, test["cmp_hs"])
    rng = np.random.default_rng(seed)
    moved = 0
    for _, group in test.groupby(["station_id", bins]):
        idx = rng.permutation(group.index)
        copy.loc[group.index, CAMERA] = test.loc[idx, CAMERA].to_numpy()
        moved += int((idx != group.index).sum())
    return copy, moved / len(test)


class VarWeight:
    """Wind-sea energy C² and W² combined by inverse mean squared residual per forecast-total quartile."""

    def fit(self, cal: pd.DataFrame, cuts: np.ndarray) -> VarWeight:
        self.cuts = cuts
        y2 = cal["true_wind_wave_m"].to_numpy() ** 2
        rc = (cal["cam_ww"].to_numpy() ** 2 - y2) ** 2
        rm = (cal["cmp_ww"].to_numpy() ** 2 - y2) ** 2
        bins = np.searchsorted(cuts, cal["cmp_hs"])
        pooled = (float(rc.mean()), float(rm.mean()))
        self.v = {
            b: (float(rc[bins == b].mean()), float(rm[bins == b].mean()))
            if (bins == b).sum() >= MIN_BIN
            else pooled
            for b in range(len(cuts) + 1)
        }
        return self

    def predict(self, f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        b = np.searchsorted(self.cuts, f["cmp_hs"])
        vc = np.array([self.v[x][0] for x in b])
        vm = np.array([self.v[x][1] for x in b])
        c2, w2, t2 = (f[c].to_numpy() ** 2 for c in ("cam_ww", "cmp_ww", "cmp_hs"))
        e = (c2 / vc + w2 / vm) / (1 / vc + 1 / vm)
        return np.sqrt(np.maximum(t2 - w2, 0) + e), np.sqrt(e)

    def describe(self) -> dict[str, Any]:
        return {
            f"q{b}": {"camera_weight": round((1 / vc) / (1 / vc + 1 / vm), 3)}
            for b, (vc, vm) in self.v.items()
        }


def test1(
    outer: pd.DataFrame, cal_of: Any, baseline: str, arrangement: str
) -> tuple[pd.DataFrame, dict]:
    """Every arm on every outer fold; ``cal_of(k)`` gives fold k's calibration strips."""
    pieces, fits = [], {}
    for k, test in outer.groupby("cv_fold"):
        cal = cal_of(k)
        m = PartitionFusion(True, "comparator", transfer=True).fit(cal)
        if not np.isfinite(m.theta).all():
            raise ValueError(f"invalid gate fit, fold {k}")
        cuts = np.quantile(cal["cmp_hs"], [0.25, 0.5, 0.75])
        vw = VarWeight().fit(cal, cuts)
        intact, weighted = m.predict(test), vw.predict(test)
        runs, moved = [], []
        for s in PERM_SEEDS:
            d, share = degrade(test, cuts, s)
            runs.append(m.predict(d))
            moved.append(share)
        degraded = np.mean(runs, axis=0)
        fits[k] = {"calibration_strips": len(cal), "test_strips": len(test), "swap": m.describe(),
                   "mean_gate_test": round(float(m.gate_values(test).mean()), 3),
                   "variance_weights": vw.describe(), "movable_fraction": round(float(np.mean(moved)), 4)}  # fmt: skip
        for i, (truth, cmp, cam) in enumerate(TARGETS):
            pieces.append(
                test[["sample_id", "station_group", "station_id", "cv_fold"]].assign(
                    arrangement=arrangement,
                    baseline=baseline,
                    target=truth,
                    observed=test[truth],
                    forecast=test[cmp],
                    camera=test[cam],
                    fusion=intact[i],
                    degraded=degraded[i],
                    varweight=weighted[i],
                )
            )
    return pd.concat(pieces, ignore_index=True), fits


# ----------------------------------------------------------------------------- scores


def unit_intervals(g: pd.DataFrame, unit: str, equal: bool) -> dict[str, list[float]]:
    """Point and 95 % interval of each contrast, resampling whole ``unit``s (seed 0)."""
    err2 = g[ARMS].sub(g["observed"], axis=0).pow(2).groupby(g[unit])
    se, n = err2.sum().to_numpy(), err2.size().to_numpy()
    idx = np.random.default_rng(0).integers(len(n), size=(N_BOOT, len(n)))

    def score(ix: np.ndarray) -> np.ndarray:
        if equal:
            return np.sqrt((se / n[:, None])[ix].mean(axis=-2))
        return np.sqrt(se[ix].sum(axis=-2) / n[ix].sum(axis=-1)[..., None])

    point, boots = score(np.arange(len(n))), score(idx)
    col = {a: i for i, a in enumerate(ARMS)}
    out = {"rmse": dict(zip(ARMS, np.round(point, 4).tolist(), strict=True))}
    for a, b in CONTRASTS:
        diff = boots[:, col[a]] - boots[:, col[b]]
        lo, hi = np.quantile(diff, [0.025, 0.975])
        out[f"{a} - {b}"] = [
            round(float(point[col[a]] - point[col[b]]), 4),
            round(float(lo), 4),
            round(float(hi), 4),
        ]
    return out


def summarize(rows: pd.DataFrame) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for (arr, base, target), g in rows.groupby(["arrangement", "baseline", "target"]):
        r = {"strips": len(g), "groups": g["station_group"].nunique(), "buoys": g["station_id"].nunique(),
             "by_group": unit_intervals(g, "station_group", False),
             "by_buoy": unit_intervals(g, "station_id", False),
             "equal_group": unit_intervals(g, "station_group", True)}  # fmt: skip
        r["rmse"] = r["by_group"].pop("rmse")
        for key in ("by_buoy", "equal_group"):
            r[key + "_rmse"] = r[key].pop("rmse")
        r["endpoint_met"] = bool(r["by_group"]["fusion - forecast"][2] < 0)
        res.setdefault(arr, {}).setdefault(base, {})[target] = r
    for arr, v in res.items():
        res[arr]["model_wind"]["bar_met"] = all(
            v["model_wind"][t]["endpoint_met"] for t in ("true_hs_m", "true_wind_wave_m")
        )
    return res


def s2_check(outer: pd.DataFrame, cal_of: Any) -> dict[str, Any]:
    """Stage-2 combiners per outer fold, calibrated by ``cal_of(k)``; the gated swap against ERA5 + wind."""
    ok = inputs_ok(outer)
    preds: dict[str, dict[str, np.ndarray]] = {}
    for k in sorted(outer["cv_fold"].unique()):
        test = (outer["cv_fold"] == k).to_numpy() & ok
        cal = cal_of(k)
        fit = cal[
            inputs_ok(cal)
            & cal["true_hs_m"].notna().to_numpy()
            & cal["true_wind_wave_m"].notna().to_numpy()
        ]
        for name, pr in predict_fold(fit_fold(fit), outer[test]).items():
            for h, v in pr.items():
                preds.setdefault(name, {}).setdefault(h, np.full(len(outer), np.nan))[test] = v
    out: dict[str, Any] = {}
    kw = {"n_boot": 1000, "alpha": 0.05, "seed": 0}
    for h, col in (("hs", "true_hs_m"), ("wind_wave", "true_wind_wave_m")):
        y = outer[col].to_numpy(float)
        sel = ok & np.isfinite(y)
        g, grp = outer.loc[sel, "station_id"].to_numpy(), outer.loc[sel, "station_group"].to_numpy()
        base = preds[COMPARATOR][h][sel]
        out[h] = {"strips": int(sel.sum()), "buoys": len(set(g)),
                  "rmse": {n: round(rmse(y[sel], preds[n][h][sel]), 4) for n in (BAR, COMPARATOR, "camera alone")},
                  "swap_minus_comparator_by_buoy": [round(x, 4) for x in paired_block_bootstrap_ci(y[sel], preds[BAR][h][sel], base, rmse, groups=g, **kw)],
                  "swap_minus_comparator_by_group": [round(x, 4) for x in paired_block_bootstrap_ci(y[sel], preds[BAR][h][sel], base, rmse, groups=grp, **kw)]}  # fmt: skip
    out["met_by_buoy"] = all(
        out[h]["swap_minus_comparator_by_buoy"][2] < 0 for h in ("hs", "wind_wave")
    )
    return out


# ----------------------------------------------------------------------------- report


def report(summary: dict[str, Any], fits: dict, s2: dict) -> str:
    names = {"true_hs_m": "Significant height", "true_wind_wave_m": "Wind-wave height"}
    lines = ["# Test 1, nested calibration\n"]
    for arr in ("nested", "historical"):
        lines.append(f"## {arr}\n")
        for base in BASELINES:
            for t in ("true_hs_m", "true_wind_wave_m"):
                r = summary[arr][base][t]
                rm, bg = r["rmse"], r["by_group"]
                lines.append(
                    f"- {base}, {names[t]} ({r['strips']:,} strips, {r['groups']} groups): "
                    + ", ".join(f"{a} {rm[a]:.4f}" for a in ARMS)
                    + "; " + "; ".join(f"{c} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for c, v in bg.items())
                    + f"; by buoy fusion - forecast [{r['by_buoy']['fusion - forecast'][1]:+.4f}, {r['by_buoy']['fusion - forecast'][2]:+.4f}]"
                    + f"; equal-group [{r['equal_group']['fusion - forecast'][1]:+.4f}, {r['equal_group']['fusion - forecast'][2]:+.4f}]"
                )  # fmt: skip
        lines.append(f"\nBar met ({arr}): {summary[arr]['model_wind']['bar_met']}\n")
        for k, v in fits[arr]["model_wind"].items():
            lines.append(f"- fold {k}: {json.dumps(v)}")
        lines.append("")
    lines.append("## S2 check\n")
    for arr in ("nested", "historical"):
        for h in ("hs", "wind_wave"):
            v = s2[arr][h]
            lines.append(
                f"- {arr} {h} ({v['strips']:,} strips): {v['rmse']}; swap - ERA5 + wind by buoy {v['swap_minus_comparator_by_buoy']}, by group {v['swap_minus_comparator_by_group']}"
            )
        lines.append(
            f"- {arr}: S2 criterion (by buoy) {'met' if s2[arr]['met_by_buoy'] else 'not met'}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for n in ("labels", "oof", "forecast", "inner-forecast", "out"):
        ap.add_argument("--" + n, required=True, type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--inner-runs", type=Path, nargs="+")
    src.add_argument("--inner-camera", type=Path)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    labels = pd.read_csv(args.labels, low_memory=False)
    oof = pd.read_csv(args.oof, dtype={"station_id": str})
    if args.inner_runs:
        inner = inner_camera(args.inner_runs, oof)
    else:
        cam = pd.read_csv(args.inner_camera, dtype={"station_id": str})
        inner = attach(cam[["sample_id", "pair", *CAM_COLS]], oof)
    f_out, f_in = features(oof, labels), inner_features(inner, labels)
    fc_out, fc_in = pd.read_parquet(args.forecast), pd.read_parquet(args.inner_forecast)

    rows, fits = [], {"nested": {}, "historical": {}}
    for base in BASELINES:
        outer = complete(with_forecast(f_out, fc_out, base, by_pair=False))
        inn = complete(with_forecast(f_in, fc_in, base, by_pair=True))
        for arr, cal_of in (("nested", lambda k, i=inn: calibration(i, k)),
                            ("historical", lambda k, o=outer: o[o["cv_fold"] != k])):  # fmt: skip
            r, fits[arr][base] = test1(outer, cal_of, base, arr)
            rows.append(r)
    rows_df = pd.concat(rows, ignore_index=True)
    summary = summarize(rows_df)
    s2 = {
        "nested": s2_check(f_out, lambda k: calibration(f_in, k)),
        "historical": s2_check(f_out, lambda k: f_out[f_out["cv_fold"] != k]),
    }
    args.out.mkdir(parents=True)
    rows_df.to_parquet(args.out / "rows.parquet", index=False)
    (args.out / "summary.json").write_text(
        json.dumps({"test1": summary, "fits": fits, "s2": s2}, indent=1)
    )
    text = report(summary, fits, s2)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
