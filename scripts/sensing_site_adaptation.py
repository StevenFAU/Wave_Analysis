#!/usr/bin/env python
"""Test 6: label-free site adaptation of cached camera features, and a few-shot calibration curve.

    python scripts/sensing_site_adaptation.py --labels <v2cv labels.csv> --oof <oof_dinov3-gentle.csv> \\
        --features <probe run, seed 0> <seed 1> <seed 2> --fold-runs <15 DINOv3-gentle fold runs> \\
        --out runs/<name>

Fixed before any run in ``docs/experiments/sensing_site_adaptation.md`` (the bar, every
arm and every number below are defined there).

* **Features:** each fold model's pooled 256-dimensional head input for every strip,
  saved by ``scripts/measurement_buoy_probe.py --save-features`` (``fold<k>/features.npy``
  and ``feature_rows.csv``; ``run.json`` names the checkpoint).
* **Adaptation (A2):** per camera (buoy), its offset and within-bin spread against the
  training cameras at the same ERA5 total-height quintile, removed:
  x' = r_c + (x - a_c) sqrt(u_c / v_c). A3 keeps the scale at 1; A4 ignores the bins.
* **Heads:** A1-A4 refit LayerNorm (no affine) + ridge onto each log height on the fold's
  training strips, penalty chosen on its validation strips; A5 keeps the model's own head
  with only non-training cameras adapted; A0 is the stage-1 out-of-fold table.
* **Few-shot:** shrunk log bias from k labelled strips in each buoy's first 14 days,
  scored from day 17 on; lambda chosen on validation buoys.

CPU only. Writes ``<out>/predictions.parquet``, ``fewshot.parquet``, ``summary.json`` and
``report.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buoycam_oof import merge_fold  # scripts/ is on sys.path
from wave_analysis.evaluation.uncertainty import hierarchical_conformal_threshold

FLOOR = 0.05
N_BINS, MIN_BIN = 5, 5
PENALTIES = (0.1, 1.0, 10.0, 100.0, 1000.0)
TARGETS = {"hs": "Hs_label_m", "wind_wave": "wind_wave_label_m", "swell": "swell_label_m"}
ENDPOINTS = ("hs", "wind_wave")
SEED_SPREAD = {"hs": 0.016, "wind_wave": 0.0075}
UNDER_COVERED = ("46081", "46042", "46047", "44013", "42002", "46013", "46069", "46022", "46014",
                 "44008", "45002")  # fmt: skip
KS, DRAWS, LAMBDAS = (1, 3, 10, 30, 100), range(10), (0.0, 1.0, 3.0, 10.0, 30.0)
WINDOW_DAYS, GAP_DAYS, MIN_EVAL = 14, 17, 30
N_BOOT = 2000
REFIT = ("A1", "A2", "A3", "A4")


# ----------------------------------------------------------------------------- adaptation


def layer_norm(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    return (x - mu) / np.sqrt(x.var(axis=1, keepdims=True) + 1e-5)


def adapt(
    x: np.ndarray, camera: np.ndarray, era5: np.ndarray, train: np.ndarray, mode: str
) -> tuple[np.ndarray, dict]:
    """Every camera's features mapped onto the training cameras' (see the module text)."""
    out = x.copy()
    info = {"cameras": 0, "unadapted": []}
    if mode == "plain":
        r, u = x[train].mean(0), np.maximum(x[train].var(0), 1e-6)
        for c in np.unique(camera):
            m = camera == c
            a, v = x[m].mean(0), np.maximum(x[m].var(0), 1e-6)
            out[m] = r + (x[m] - a) * np.sqrt(u / v)
            info["cameras"] += 1
        return out, info
    edges = np.quantile(era5[train & np.isfinite(era5)], np.linspace(0, 1, N_BINS + 1)[1:-1])
    b = np.where(np.isfinite(era5), np.searchsorted(edges, era5), -1)
    ref_mu = np.stack([x[train & (b == j)].mean(0) for j in range(N_BINS)])
    ref_var = np.stack([x[train & (b == j)].var(0) for j in range(N_BINS)])
    for c in np.unique(camera):
        m = camera == c
        n = np.array([(m & (b == j)).sum() for j in range(N_BINS)])
        ok = n >= MIN_BIN
        if not ok.any():
            info["unadapted"].append(str(c))
            continue
        w = np.where(ok, n, 0) / n[ok].sum()
        a = sum(w[j] * x[m & (b == j)].mean(0) for j in range(N_BINS) if ok[j])
        r = (w[:, None] * ref_mu).sum(0)
        out[m] = x[m] - a + r
        if mode == "scaled":
            v = sum(w[j] * x[m & (b == j)].var(0) for j in range(N_BINS) if ok[j])
            u = (w[:, None] * ref_var).sum(0)
            out[m] = r + (x[m] - a) * np.sqrt(np.maximum(u, 1e-6) / np.maximum(v, 1e-6))
        info["cameras"] += 1
    return out, info


def ridge_head(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray
) -> tuple[np.ndarray, float]:
    """LayerNorm (no affine) + ridge onto ``y``; penalty by validation log-RMSE; predictions for every row."""
    z = layer_norm(x)
    fit = train & np.isfinite(y)
    mu, sd = z[fit].mean(0), z[fit].std(0)
    sd[sd < 1e-8] = 1.0
    zs = (z - mu) / sd
    ym = y[fit].mean()
    e, vec = np.linalg.eigh(zs[fit].T @ zs[fit])
    proj = vec.T @ (zs[fit].T @ (y[fit] - ym))
    best = None
    for alpha in PENALTIES:
        w = vec @ (proj / (e + alpha))
        pred = zs @ w + ym
        vv = val & np.isfinite(y)
        score = float(np.sqrt(np.mean((pred[vv] - y[vv]) ** 2)))
        if best is None or score < best[0]:
            best = (score, alpha, pred)
    return best[2], best[1]


def own_head(x: np.ndarray, run: Path) -> np.ndarray:
    """The checkpoint's own head (LayerNorm with affine, linear) on ``x``: log heights, (n, 3)."""
    import torch

    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    g, b = sd["head.0.weight"].numpy(), sd["head.0.bias"].numpy()
    w, c = sd["head.1.weight"].numpy(), sd["head.1.bias"].numpy()
    z = layer_norm(x) * g + b
    return (z @ w.T + c) * np.asarray(ck["std"]) + np.asarray(ck["mean"])


def fold_seed(
    feat_dir: Path, k: int, lab: pd.DataFrame, era5: pd.Series, runs: dict
) -> dict[str, Any]:
    """All arms' log heights for every strip of fold ``k`` from one seed's features."""
    meta = json.loads((feat_dir / f"fold{k}" / "run.json").read_text())
    rows = pd.read_csv(feat_dir / f"fold{k}" / "feature_rows.csv")
    x = np.load(feat_dir / f"fold{k}" / "features.npy").astype(np.float64)
    t = lab.set_index("sample_id").loc[rows["sample_id"]].reset_index()
    role = t[f"split_fold{k}"].to_numpy()
    if not (role == rows["role"].to_numpy()).all():
        raise SystemExit(f"{feat_dir.name} fold {k}: roles differ from the package")
    train, val = role == "train", role == "val"
    cam, e5 = t["station_id"].to_numpy(), era5.loc[rows["sample_id"]].to_numpy(float)
    feats = {"A1": (x, {})}
    feats["A2"] = adapt(x, cam, e5, train, "scaled")
    feats["A3"] = adapt(x, cam, e5, train, "mean")
    feats["A4"] = adapt(x, cam, e5, train, "plain")
    out: dict[str, Any] = {"sample_id": rows["sample_id"].to_numpy(), "role": role,
                           "seed": meta["seed"], "checkpoint": meta["checkpoint_run"], "logs": {}, "penalty": {},
                           "adapted": {a: {"cameras": f[1].get("cameras"), "unadapted": f[1].get("unadapted")}
                                       for a, f in feats.items() if f[1]}}  # fmt: skip
    for arm in REFIT:
        xa = feats[arm][0]
        cols = []
        for h, col in TARGETS.items():
            y = np.log(np.maximum(t[col].to_numpy(float), FLOOR))
            pred, alpha = ridge_head(xa, y, train, val)
            cols.append(pred)
            out["penalty"][f"{arm}_{h}"] = alpha
        out["logs"][arm] = np.column_stack(cols)
    run = runs[(k, meta["seed"])]
    if run.name != meta["checkpoint_run"]:
        raise SystemExit(f"features of {meta['checkpoint_run']} but run {run.name}")
    out["logs"]["A0f"] = own_head(x, run)
    x5 = np.where(train[:, None], x, feats["A2"][0])
    out["logs"]["A5"] = own_head(x5, run)
    return out


# ----------------------------------------------------------------------------- scores


def contrasts(
    df: pd.DataFrame, pairs: list[tuple[str, str]], unit: str, equal: bool = False
) -> dict[str, list[float]]:
    """Point and 95 % interval of RMSE(a) - RMSE(b), resampling whole ``unit``s (2,000, seed 0)."""
    arms = sorted({a for p in pairs for a in p})
    err = df[arms].sub(df["y"], axis=0).pow(2).groupby(df[unit])
    se, n = err.sum().to_numpy(), err.size().to_numpy()
    idx = np.random.default_rng(0).integers(len(n), size=(N_BOOT, len(n)))

    def score(ix: np.ndarray) -> np.ndarray:
        if equal:
            return np.sqrt((se / n[:, None])[ix].mean(axis=-2))
        return np.sqrt(se[ix].sum(axis=-2) / n[ix].sum(axis=-1)[..., None])

    point, boots = score(np.arange(len(n))), score(idx)
    col = {a: i for i, a in enumerate(arms)}
    res = {"rmse": {a: round(float(point[col[a]]), 4) for a in arms}}
    for a, b in pairs:
        d = boots[:, col[a]] - boots[:, col[b]]
        lo, hi = np.quantile(d, [0.025, 0.975])
        res[f"{a} - {b}"] = [
            round(float(point[col[a]] - point[col[b]]), 4),
            round(float(lo), 4),
            round(float(hi), 4),
        ]
    return res


def coverage(
    frame: pd.DataFrame, mu: np.ndarray, level: float = 0.90
) -> tuple[float, dict[str, float]]:
    """Stage 1's buoy-level conformal coverage of H_s for log-normal centres ``mu``."""
    y = frame["y"].to_numpy(float)
    sd = frame["sigma"].to_numpy(float)
    ly = np.log(np.maximum(y, FLOOR))
    score = np.abs(ly - mu) / sd
    fold, buoy = frame["cv_fold"].to_numpy(), frame["station_id"].to_numpy()
    q = np.full(len(frame), np.nan)
    for k in np.unique(fold):
        q[fold == k] = hierarchical_conformal_threshold(score[fold != k], buoy[fold != k], level)
    yy = np.maximum(y, FLOOR)
    inside = (yy >= np.exp(mu - q * sd)) & (yy <= np.exp(mu + q * sd))
    per = pd.Series(inside).groupby(buoy).mean()
    return float(inside.mean()), {b: round(float(per[b]), 3) for b in UNDER_COVERED if b in per}


# ----------------------------------------------------------------------------- few-shot


def windows(lab: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(lab["time_utc"], utc=True)
    age = t - t.groupby(lab["station_id"]).transform("min")
    return lab.assign(
        window=age < pd.Timedelta(days=WINDOW_DAYS), evaluate=age >= pd.Timedelta(days=GAP_DAYS)
    )


def draws(frame: pd.DataFrame, k: int) -> dict[str, list[np.ndarray]]:
    """Per included buoy, ten index draws of k labelled window strips (positions in ``frame``)."""
    out = {}
    for sid in frame["station_id"].unique():
        win = np.flatnonzero((frame["station_id"] == sid).to_numpy() & frame["window"].to_numpy())
        ev = int(((frame["station_id"] == sid) & frame["evaluate"]).sum())
        if len(win) < k or ev < MIN_EVAL:
            continue
        out[sid] = [
            np.random.default_rng([s, k, zlib.crc32(sid.encode())]).choice(win, k, replace=False)
            for s in DRAWS
        ]
    return out


def corrected_se(
    frame: pd.DataFrame, logp: np.ndarray, k: int, lam: float, d: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Mean squared error over draws on each included buoy's evaluation strips, and the per-draw biases."""
    y = frame["y"].to_numpy(float)
    ly = np.log(np.maximum(y, FLOOR))
    se = np.full(len(frame), np.nan)
    bias = {}
    for sid, picks in d.items():
        ev = ((frame["station_id"] == sid) & frame["evaluate"]).to_numpy()
        acc = np.zeros(ev.sum())
        bs = []
        for p in picks:
            b = float(np.sum(ly[p] - logp[p]) / (k + lam))
            bs.append(b)
            acc += (np.exp(logp[ev] + b) - y[ev]) ** 2
        se[ev] = acc / len(picks)
        bias[sid] = bs
    return se, bias


def choose_lambda(frame: pd.DataFrame, logp: np.ndarray) -> float:
    """Lambda with the lowest RMSE averaged over the five k, on ``frame`` (validation buoys)."""
    ds = {k: draws(frame, k) for k in KS}
    best = None
    for lam in LAMBDAS:
        r = []
        for k in KS:
            se, _ = corrected_se(frame, logp, k, lam, ds[k])
            if np.isfinite(se).any():
                r.append(np.sqrt(np.nanmean(se)))
        score = float(np.mean(r))
        if best is None or score < best[0]:
            best = (score, lam)
    return best[1]


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--features", type=Path, nargs=3, required=True)
    ap.add_argument("--fold-runs", type=Path, nargs=15, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    lab = pd.read_csv(args.labels, dtype={"station_id": str}, low_memory=False)
    lab = lab[lab["cv_fold"].notna()].reset_index(drop=True)
    lab["cv_fold"] = lab["cv_fold"].astype(int)
    oof = (
        pd.read_csv(args.oof, dtype={"station_id": str})
        .set_index("sample_id")
        .loc[lab["sample_id"]]
    )
    era5 = oof["era5_hs_raw_m"]
    runs = {}
    for r in args.fold_runs:
        m = json.loads((r / "metrics.json").read_text())["run"]
        runs[(int(m["split"][4:]), int(m["seed"]))] = r

    # every arm, fold and seed; test and validation log heights
    test_logs: dict[str, dict[int, list[np.ndarray]]] = {}
    val_logs: dict[int, dict[str, list[np.ndarray]]] = {}
    fits = []
    ids = {}
    for k in range(5):
        for fd in args.features:
            r = fold_seed(fd, k, lab, era5, runs)
            if k in ids and not np.array_equal(ids[k][0], r["sample_id"]):
                raise SystemExit(f"fold {k}: the seeds' feature rows are in different orders")
            fits.append({"fold": k, "seed": r["seed"], "checkpoint": r["checkpoint"], "penalty": r["penalty"],
                         "adapted": r["adapted"]})  # fmt: skip
            ids[k] = r["sample_id"], r["role"]
            for arm, lg in r["logs"].items():
                test_logs.setdefault(arm, {}).setdefault(k, []).append(lg[r["role"] == "test"])
                val_logs.setdefault(k, {}).setdefault(arm, []).append(lg[r["role"] == "val"])
            print(f"fold {k} seed {r['seed']} done", flush=True)

    # seed-mean log heights per strip, out of fold
    pred = pd.DataFrame(
        {"sample_id": np.concatenate([ids[k][0][ids[k][1] == "test"] for k in range(5)])}
    )
    for arm, by in test_logs.items():
        stack = np.concatenate([np.mean(by[k], axis=0) for k in range(5)])
        for i, h in enumerate(TARGETS):
            pred[f"{arm}_{h}"] = np.exp(stack[:, i])
    base = lab.set_index("sample_id").loc[pred["sample_id"]]
    pred["station_id"], pred["station_group"] = (
        base["station_id"].to_numpy(),
        base["station_group"].to_numpy(),
    )
    pred["cv_fold"] = base["cv_fold"].to_numpy()
    o = oof.loc[pred["sample_id"]]
    for h in TARGETS:
        pred[f"A0_{h}"] = o[f"pred_{h}_m"].to_numpy()
        pred[f"y_{h}"] = base[TARGETS[h]].to_numpy(float)

    # the bar
    summary: dict[str, Any] = {"fits": fits, "endpoints": {}}
    pairs = [
        ("A2", "A0"),
        ("A2", "A1"),
        ("A1", "A0"),
        ("A3", "A0"),
        ("A4", "A0"),
        ("A5", "A0"),
        ("A0f", "A0"),
    ]
    for h in ENDPOINTS:
        d = pred.assign(
            y=pred[f"y_{h}"],
            **{a: pred[f"{a}_{h}"] for a in ("A0", "A0f", "A1", "A2", "A3", "A4", "A5")},
        )
        d = d[d["y"].notna()]
        r = {"strips": len(d), "buoys": d["station_id"].nunique(), "groups": d["station_group"].nunique(),
             "by_group": contrasts(d, pairs, "station_group"),
             "by_buoy": contrasts(d, pairs[:2], "station_id"),
             "equal_group": contrasts(d, pairs[:2], "station_group", equal=True)}  # fmt: skip
        g = r["by_group"]
        r["pass"] = {
            "gain_at_least_seed_spread": g["A2 - A0"][0] <= -SEED_SPREAD[h],
            "interval_below_zero": g["A2 - A0"][2] < 0,
            "beats_head_matched_control": g["A2 - A1"][2] < 0,
            "gain_at_least_0.016": g["A2 - A0"][0] <= -0.016,
        }
        r["endpoint_met"] = all(
            r["pass"][c]
            for c in (
                "gain_at_least_seed_spread",
                "interval_below_zero",
                "beats_head_matched_control",
            )
        )
        west = d[d["cv_fold"] == 0]
        r["west_coast_fold0"] = {"strips": len(west), "buoys": west["station_id"].nunique(),
                                 **contrasts(west, [("A2", "A0"), ("A2", "A1"), ("A1", "A0")], "station_id")}  # fmt: skip
        summary["endpoints"][h] = r
    summary["test6_met"] = all(summary["endpoints"][h]["endpoint_met"] for h in ENDPOINTS)

    # conformal coverage of H_s at the eleven buoys, before and after
    c = pred.assign(
        y=pred["y_hs"], sigma=o["sigma_hs_log"].to_numpy(), mu0=o["mu_hs_log"].to_numpy()
    )
    c = c[c["y"].notna()]
    before = coverage(c, c["mu0"].to_numpy())
    after = coverage(
        c, c["mu0"].to_numpy() + np.log(c["A2_hs"].to_numpy()) - np.log(c["A0_hs"].to_numpy())
    )
    summary["coverage_hs"] = {"before": {"overall": round(before[0], 4), "under_covered": before[1]},
                              "after_A2": {"overall": round(after[0], 4), "under_covered": after[1]}}  # fmt: skip

    # few-shot curve on A0 and A2
    win = windows(lab).set_index("sample_id")
    shots, lambdas = [], {}
    cov_fs: dict[str, Any] = {}
    for h in ENDPOINTS:
        for arm in ("A0", "A2"):
            lam_by_fold = {}
            for k in range(5):
                # validation buoys of fold k: their own fold-k predictions
                vid = ids[k][0][ids[k][1] == "val"]
                if arm == "A0":
                    vm = merge_fold([runs[(k, s)] for s in range(3)])
                    vm = vm[vm["partition"] == "val"].set_index("sample_id").loc[vid]
                    vlog = np.log(vm[f"pred_{h}_m"].to_numpy())
                else:
                    vlog = np.mean(val_logs[k]["A2"], axis=0)[:, list(TARGETS).index(h)]
                vf = win.loc[vid].reset_index()
                vf["y"] = vf[TARGETS[h]].to_numpy(float)
                keep = vf["y"].notna().to_numpy()
                lam_by_fold[k] = choose_lambda(vf[keep].reset_index(drop=True), vlog[keep])
            lambdas[f"{arm}_{h}"] = lam_by_fold
            f = pred[["sample_id", "station_id", "station_group", "cv_fold"]].assign(
                y=pred[f"y_{h}"]
            )
            f = f.join(win[["window", "evaluate", "time_utc"]], on="sample_id")
            f = f[f["y"].notna()].reset_index(drop=True)
            logp = np.log(pred.set_index("sample_id").loc[f["sample_id"], f"{arm}_{h}"].to_numpy())
            for kk in KS:
                se = np.full(len(f), np.nan)
                biases = {}
                for k in range(5):
                    sub = (f["cv_fold"] == k).to_numpy()
                    fk = f[sub].reset_index(drop=True)
                    s, b = corrected_se(fk, logp[sub], kk, lam_by_fold[k], draws(fk, kk))
                    se[sub] = s
                    biases.update(b)
                inc = np.isfinite(se)
                base_se = (np.exp(logp) - f["y"].to_numpy()) ** 2
                shots.append(
                    f.loc[inc, ["sample_id", "station_id", "station_group", "cv_fold"]].assign(
                        target=h, arm=arm, k=kk, se_k=se[inc], se_0=base_se[inc]
                    )
                )
                if h == "hs":
                    # coverage after the correction on the evaluation strips, averaged over draws
                    fe = f[inc].reset_index(drop=True)
                    oo = o.loc[fe["sample_id"]]
                    mu = oo["mu_hs_log"].to_numpy() + (
                        logp[inc] - np.log(oo["pred_hs_m"].to_numpy())
                    )
                    fe = fe.assign(sigma=oo["sigma_hs_log"].to_numpy())
                    covs, per = [], []
                    for di in DRAWS:
                        shift = np.array([biases[s][di] for s in fe["station_id"]])
                        cv, pb = coverage(fe, mu + shift)
                        covs.append(cv)
                        per.append(pb)
                    base_cov = coverage(fe, mu)
                    cov_fs[f"{arm}_k{kk}"] = {"overall_before": round(base_cov[0], 4), "overall_after": round(float(np.mean(covs)), 4),
                                              "under_covered_before": base_cov[1],
                                              "under_covered_after": {b: round(float(np.mean([p[b] for p in per if b in p])), 3) for b in per[0]}}  # fmt: skip
    fs = pd.concat(shots, ignore_index=True)
    curve = []
    for (h, arm, kk), g in fs.groupby(["target", "arm", "k"]):
        r = contrasts(
            g.assign(y=0.0, K=np.sqrt(g["se_k"]), Z=np.sqrt(g["se_0"])),
            [("K", "Z")],
            "station_group",
        )
        curve.append({"target": h, "arm": arm, "k": kk, "buoys": g["station_id"].nunique(), "strips": len(g),
                      "rmse_k0": r["rmse"]["Z"], "rmse_k": r["rmse"]["K"], "diff": r["K - Z"]})  # fmt: skip
    summary["fewshot"] = {"lambda": lambdas, "curve": curve, "coverage_hs": cov_fs}
    for h in ENDPOINTS:
        ok = [
            c
            for c in curve
            if c["target"] == h
            and c["arm"] == "A0"
            and c["diff"][0] <= -SEED_SPREAD[h]
            and c["diff"][2] < 0
        ]
        summary["fewshot"][f"smallest_k_{h}_A0"] = min((c["k"] for c in ok), default=None)

    args.out.mkdir(parents=True)
    pred.to_parquet(args.out / "predictions.parquet", index=False)
    fs.to_parquet(args.out / "fewshot.parquet", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    lines = ["# Test 6: label-free site adaptation\n"]
    for h in ENDPOINTS:
        r = summary["endpoints"][h]
        lines.append(f"## {h} ({r['strips']:,} strips, {r['buoys']} buoys, {r['groups']} groups)\n")
        lines.append(f"- RMSE: {r['by_group']['rmse']}")
        lines += [f"- {c}: {v}" for c, v in r["by_group"].items() if c != "rmse"]
        lines.append(f"- by buoy: {r['by_buoy']}; equal-group: {r['equal_group']}")
        lines.append(f"- pass: {r['pass']}; endpoint met: {r['endpoint_met']}")
        lines.append(f"- West Coast fold 0: {r['west_coast_fold0']}\n")
    lines.append(f"**Test 6 {'met' if summary['test6_met'] else 'not met'}.**\n")
    lines.append(f"Coverage H_s: {json.dumps(summary['coverage_hs'])}\n")
    lines.append("## Few-shot\n")
    lines.append(f"lambda: {json.dumps(lambdas)}")
    for c in curve:
        lines.append(
            f"- {c['target']} {c['arm']} k={c['k']}: {c['buoys']} buoys, {c['strips']:,} strips, k=0 {c['rmse_k0']:.4f} -> {c['rmse_k']:.4f}, diff {c['diff']}"
        )
    lines.append(
        f"\nsmallest k beyond seed spread (A0): hs {summary['fewshot']['smallest_k_hs_A0']}, wind_wave {summary['fewshot']['smallest_k_wind_wave_A0']}"
    )
    for key, v in cov_fs.items():
        lines.append(f"- coverage {key}: {json.dumps(v)}")
    text = "\n".join(lines) + "\n"
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
