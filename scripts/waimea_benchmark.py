#!/usr/bin/env python
"""First Waimea Bay benchmark: leakage-safe splits and the baselines a camera model must beat.

P1 of ``docs/operations/collection_plan.md``; run after ``scripts/waimea_pairs.py``.
The experiment is declared in ``configs/experiments/waimea_hs_v0.yaml``.

1. **Manifest.** The paired images (trusted capture time, CDIP 106 record
   centred within 30 min) with an exclusion ledger for the rest.
2. **Splits.** ``day``: whole HST days, both cameras of a day together (they
   share a label). ``week``: whole ISO weeks (HST), because a swell lasts 2-4
   days and neighbouring days are correlated across a day split. ``year``:
   train 2009-2011, validate 2012, test 2013, with a 72 h embargo.
3. **Leakage audit** of each split: no HST day (or week) in two partitions, no
   train and test images closer than 6 h (across cameras). Adjacent days and
   the inter-day correlation of H_s are reported as well. Image hashes are
   *not* used: a fixed camera produces near-identical coarse hashes on any two
   calm days, so a hash match there means similar sea state, not the same
   event (the hash distances are reported to show this).
4. **Baselines**, each fitted on train (+ val for the final fit; val chooses
   the ridge penalty) and scored on test under both label rules of ADR 0010
   (window mean, nearest record; Q-M7):

   ``mean``, ``climatology_month``
       Training mean; training mean per HST calendar month.
   ``persistence_<k>h``
       The last CDIP 106 record centred at least k h before the image: what a
       buoy that stopped reporting k hours ago would give. Not a camera
       competitor, but the scale of how predictable the label is.
   ``era5_raw``, ``era5_linear``
       ERA5 significant wave height at the nearest sea point (40 km offshore),
       linearly interpolated to the capture time; raw, and with a linear
       correction fitted on training data (Q-M6's nowcast).
   ``ridge_whitewater``
       Ridge on one number per image: the white-water fraction of the water
       blocks chosen from each camera's view (``waimea_timing_audit.ROI``).
   ``ridge_blocks``
       Ridge on all 16 x 16 block statistics (white fraction and mean luma,
       caption excluded); one model per camera.
   ``era5_linear+ridge_blocks``
       ``era5_linear`` plus a ridge on the blocks fitted to its residual (Q-M6:
       does the image add anything to the nowcast?).

   ``label_rule_difference`` scores the nearest label against the window-mean
   label: the disagreement between two defensible labels, a floor for any
   RMSE.

Metrics: bias (prediction minus label), MAE, RMSE, scatter index, R^2,
Pearson r, with 95 % intervals for RMSE and R^2 from a bootstrap over HST days;
per sea-state bin for the ``day`` split; and paired RMSE differences between
baselines (:data:`COMPARISONS`), bootstrapped over the same days so that the
shared day-to-day variation cancels.

Outputs in ``data/processed/benchmarks/waimea_cdip106_v0/``: ``manifest.*``
(with one split column per protocol), ``predictions.parquet``,
``results.csv``, ``strata.csv``, ``comparisons.csv``, ``summary.json``,
``provenance.yaml``. Kept in Git: the per-day split table
``data/manifests/ml/waimea_cdip106_v0.days.csv`` and copies of results, strata,
comparisons, summary and provenance in ``data/manifests/processed/``.

Usage::

    uv run python scripts/waimea_benchmark.py
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from waimea_timing_audit import ROI  # scripts/ is on sys.path when run as a script
from wave_analysis.config import data_dir, load_yaml, repo_root
from wave_analysis.datasets.builder import build_manifest
from wave_analysis.datasets.provenance import build_provenance, write_provenance
from wave_analysis.datasets.splits import (
    TEST,
    TRAIN,
    VAL,
    SplitResult,
    check_leakage,
    grouped_split,
    time_block_split,
)
from wave_analysis.evaluation import regression as R
from wave_analysis.evaluation.uncertainty import block_bootstrap_ci
from wave_analysis.models.baselines.statistical import (
    ClimatologyBaseline,
    MeanBaseline,
    PersistenceBaseline,
    RidgeBaseline,
)
from wave_analysis.qc.imagery import difference_hash

CONFIG_PATH = repo_root() / "configs" / "experiments" / "waimea_hs_v0.yaml"
HST = pd.Timedelta(hours=10)  # HST = UTC - 10 h, no daylight saving
CAMERAS = ("beachcam_003", "beachcam_004")
METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "bias": R.bias,
    "mae": R.mae,
    "rmse": R.rmse,
    "scatter_index": R.scatter_index,
    "r2": R.r2,
    "pearson": R.pearson,
}
#: Paired comparisons (candidate, reference): RMSE difference with a day bootstrap.
COMPARISONS = (
    ("era5_linear+ridge_blocks", "era5_linear"),
    ("ridge_blocks", "era5_linear"),
    ("ridge_blocks", "climatology_month"),
    ("era5_linear", "climatology_month"),
)
Frame = pd.DataFrame


def _ns(t: pd.Series | pd.Index) -> np.ndarray:
    """Datetimes as int64 nanoseconds (inputs mix micro- and nanosecond units)."""
    return np.asarray(pd.DatetimeIndex(t).as_unit("ns").asi8)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def load_samples() -> tuple[Frame, np.ndarray]:
    """Paired-image table with day keys and the image feature matrix (same row order)."""
    s = pd.read_parquet(data_dir("processed") / "pairs" / "waimea_cdip106" / "samples.parquet")
    audit = data_dir("interim") / "waimea_audit"
    imgs = pd.read_parquet(audit / "images.parquet", columns=["name", "features_ok"])
    feats = np.load(audit / "features.npz")
    pos = pd.Series(np.arange(len(imgs)), index=imgs["name"])
    s["feature_row"] = s["name"].map(pos)
    s["features_ok"] = s["feature_row"].map(imgs["features_ok"]).fillna(False).astype(bool)
    s["paired"] = s["exclusion"].isna()
    local = s["time_utc"].dt.tz_convert(None) - HST
    s["hst_day"] = local.dt.strftime("%Y-%m-%d")
    iso = local.dt.isocalendar()
    s["hst_week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    s["hst_month"] = local.dt.month
    rows = s["feature_row"].fillna(0).astype(int).to_numpy()
    white = feats["block_white"][rows].reshape(len(s), -1)
    luma = feats["block_luma"][rows].reshape(len(s), -1)
    roi = np.zeros(len(s))
    for cam, blocks in ROI.items():
        m = (s["camera"] == cam).to_numpy()
        bw = feats["block_white"][rows[m]]
        roi[m] = np.concatenate([bw[:, r, c].reshape(m.sum(), -1) for r, c in blocks], 1).mean(1)
    s["whitewater_fraction"] = roi
    s["dhash"] = [
        np.uint64(difference_hash(t)) if ok else np.uint64(0)
        for t, ok in zip(feats["thumb"][rows], s["features_ok"], strict=True)
    ]
    return s, np.hstack([white, luma]).astype(np.float64)


def load_series(path: str, variable: str) -> pd.Series:
    """One variable of a standardized bulk table, indexed by its time."""
    d = pd.read_parquet(path, columns=["time_utc", "variable", "value", "averaging_period_s"])
    d = d[d["variable"] == variable]
    # CDIP times are sample starts; use the sample centre. ERA5 is instantaneous.
    t = d["time_utc"] + pd.to_timedelta(d["averaging_period_s"].fillna(0) / 2, unit="s")
    return pd.Series(d["value"].to_numpy(), index=pd.DatetimeIndex(t)).sort_index()


# --------------------------------------------------------------------------- #
# Splits and audit
# --------------------------------------------------------------------------- #


def make_splits(m: Frame, spec: dict[str, Any]) -> dict[str, SplitResult]:
    """The three protocols of the config."""
    out = {}
    for name in ("day", "week"):
        p = spec[name]
        out[name] = grouped_split(m, p["group_col"], p["fractions"], seed=p["seed"])
    y = spec["year"]
    out["year"] = time_block_split(
        m,
        "time_utc",
        val_start=y["val_start"],
        test_start=y["test_start"],
        embargo=y["embargo"],
        group_col=y["group_col"],
    )
    return out


def audit(m: Frame, split: SplitResult, spec: dict[str, Any], extra: list[str]) -> dict[str, Any]:
    """Leakage checks plus the day-adjacency and hash diagnostics described above."""
    a = spec["audit"]
    rep = check_leakage(
        m,
        split,
        group_cols=[*a["group_cols"], *extra],
        time_col=a["time_col"],
        min_gap=a["min_gap"],
    )
    lab = split.labels
    t = _ns(m["time_utc"])
    tr = np.sort(t[(lab == TRAIN).to_numpy()])
    te = t[(lab == TEST).to_numpy()]
    pos = np.searchsorted(tr, te)
    gap = np.minimum(
        np.abs(te - tr[np.clip(pos - 1, 0, tr.size - 1)]),
        np.abs(tr[np.clip(pos, 0, tr.size - 1)] - te),
    )
    part = m.assign(split=lab.to_numpy()).groupby("hst_day")["split"].first()
    days = pd.to_datetime(part.index)
    train_days = set(days[part.eq(TRAIN).to_numpy()])
    test_days = days[part.eq(TEST).to_numpy()]
    adjacent = sum(
        1
        for d in test_days
        if (d - pd.Timedelta(days=1)) in train_days or (d + pd.Timedelta(days=1)) in train_days
    )
    h = m["dhash"].to_numpy(dtype=np.uint64)
    cam = m["camera"].to_numpy()
    near = 0
    for c in CAMERAS:
        hte = h[(lab == TEST).to_numpy() & (cam == c)]
        htr = h[(lab == TRAIN).to_numpy() & (cam == c)]
        if hte.size and htr.size:
            dist = np.bitwise_count(hte[:, None] ^ htr[None, :])
            near += int((dist <= 4).any(axis=1).sum())
    return {
        "ok": rep.ok,
        "shared_groups": {k: len(v) for k, v in rep.shared_groups.items()},
        "temporal_neighbours_within_min_gap": rep.temporal_neighbours,
        "min_train_test_gap_h": round(float(gap.min()) / 3.6e12, 2) if gap.size else None,
        "test_days": len(test_days),
        "test_days_next_to_a_train_day": adjacent,
        "test_images_with_a_train_image_within_4_dhash_bits": near,
        "test_images": int((lab == TEST).sum()),
    }


def hash_diagnostics(m: Frame, seed: int) -> dict[str, Any]:
    """dHash distances between images of one camera on different days (random pairs)."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in CAMERAS:
        g = m[m["camera"] == c]
        i, j = rng.integers(0, len(g), 20000), rng.integers(0, len(g), 20000)
        diff = g["hst_day"].to_numpy()[i] != g["hst_day"].to_numpy()[j]
        h = g["dhash"].to_numpy(dtype=np.uint64)
        d = np.bitwise_count(h[i[diff]] ^ h[j[diff]])
        out[c] = {
            "pairs": int(diff.sum()),
            "dhash_bits_quantiles": {str(q): float(np.quantile(d, q)) for q in (0.05, 0.5, 0.95)},
            "fraction_within_4_bits": round(float((d <= 4).mean()), 3),
        }
    return out


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


def ridge_per_camera(
    X: np.ndarray, y: np.ndarray, cam: np.ndarray, part: np.ndarray, alphas: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Choose alpha on val per camera, refit on train + val, predict every row."""
    pred = np.full(len(y), np.nan)
    chosen = {}
    for c in CAMERAS:
        tr, va, on = (part == TRAIN) & (cam == c), (part == VAL) & (cam == c), cam == c
        if not tr.any():
            continue
        scores = [
            R.rmse(y[va], RidgeBaseline(alpha=a).fit(X[tr], y[tr]).predict(X[va]))
            if va.any()
            else np.nan
            for a in alphas
        ]
        best = float(alphas[int(np.nanargmin(scores))]) if np.isfinite(scores).any() else 1.0
        fit = (tr | va) & np.isfinite(y)
        pred[on] = RidgeBaseline(alpha=best).fit(X[fit], y[fit]).predict(X[on])
        chosen[c] = best
    return pred, chosen


def linear_fit(x: np.ndarray, y: np.ndarray, rows: np.ndarray) -> tuple[float, float]:
    """Least-squares ``y = a + b x`` on the given rows (finite values only)."""
    ok = rows & np.isfinite(x) & np.isfinite(y)
    b, a = np.polyfit(x[ok], y[ok], 1)
    return float(a), float(b)


def baselines(
    m: Frame,
    X: np.ndarray,
    y: np.ndarray,
    part: np.ndarray,
    cdip: pd.Series,
    era5: np.ndarray,
    cfg: dict[str, Any],
    target: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Predictions of every baseline for every row of ``m`` (fitted on train/val only)."""
    fit = np.isin(part, [TRAIN, VAL])
    cam = m["camera"].to_numpy()
    alphas = np.logspace(**cfg["ridge_alphas"])
    info: dict[str, Any] = {}
    p: dict[str, np.ndarray] = {}
    p["mean"] = MeanBaseline().fit(y[fit]).predict(len(y))
    clim = ClimatologyBaseline(["hst_month"]).fit(m[fit], target)
    p["climatology_month"] = clim.predict(m)
    for lag in cfg["persistence_lags_h"]:
        p[f"persistence_{lag}h"] = PersistenceBaseline(pd.Timedelta(hours=lag)).predict(
            m["time_utc"],
            pd.Series(cdip.index),
            cdip.to_numpy(),
            max_age=cfg["persistence_max_age"],
        )
    p["era5_raw"] = era5
    a, b = linear_fit(era5, y, fit)
    p["era5_linear"] = a + b * era5
    info["era5_linear"] = {"intercept": round(a, 4), "slope": round(b, 4)}
    ww = m[["whitewater_fraction"]].to_numpy()
    p["ridge_whitewater"], info["ridge_whitewater_alpha"] = ridge_per_camera(
        ww, y, cam, part, alphas
    )
    p["ridge_blocks"], info["ridge_blocks_alpha"] = ridge_per_camera(X, y, cam, part, alphas)
    # Q-M6: the image models the nowcast's residual. The residual's alpha is chosen
    # with the nowcast fitted on train only, so val stays unseen during selection.
    a_tr, b_tr = linear_fit(era5, y, part == TRAIN)
    resid_sel = y - (a_tr + b_tr * era5)
    _, chosen = ridge_per_camera(X, resid_sel, cam, part, alphas)
    resid = y - p["era5_linear"]
    combo = p["era5_linear"].copy()
    for c, alpha in chosen.items():
        on, rows = cam == c, fit & (cam == c) & np.isfinite(resid)
        combo[on] += RidgeBaseline(alpha=alpha).fit(X[rows], resid[rows]).predict(X[on])
    p["era5_linear+ridge_blocks"] = combo
    info["era5_linear+ridge_blocks_alpha"] = chosen
    return p, info


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score(
    y: np.ndarray, pred: np.ndarray, groups: np.ndarray, boot: dict[str, Any]
) -> dict[str, Any]:
    """Metrics on finite pairs, with day-bootstrap intervals for RMSE and R^2."""
    ok = np.isfinite(y) & np.isfinite(pred)
    rec: dict[str, Any] = {"n": int(ok.sum()), "days": int(pd.unique(groups[ok]).size)}
    for name, fn in METRICS.items():
        rec[name] = fn(y[ok], pred[ok])
    for name in ("rmse", "r2"):
        _, lo, hi = block_bootstrap_ci(
            y[ok],
            pred[ok],
            METRICS[name],
            groups=groups[ok],
            n_boot=boot["n_boot"],
            alpha=boot["alpha"],
            seed=boot["seed"],
        )
        rec[f"{name}_lo"], rec[f"{name}_hi"] = lo, hi
    return rec


def _rmse_gain(y: np.ndarray, both: np.ndarray) -> float:
    """RMSE of column 0 minus RMSE of column 1 (negative: column 0 is better)."""
    return R.rmse(y, both[:, 0]) - R.rmse(y, both[:, 1])


def compare(
    y: np.ndarray, new: np.ndarray, ref: np.ndarray, groups: np.ndarray, boot: dict[str, Any]
) -> dict[str, Any]:
    """Paired RMSE difference on rows where both predict, resampling whole days."""
    both = np.column_stack([new, ref])
    ok = np.isfinite(y) & np.isfinite(both).all(axis=1)
    d, lo, hi = block_bootstrap_ci(
        y[ok],
        both[ok],
        _rmse_gain,
        groups=groups[ok],
        n_boot=boot["n_boot"],
        alpha=boot["alpha"],
        seed=boot["seed"],
    )
    return {"n": int(ok.sum()), "rmse_difference": d, "lo": lo, "hi": hi}


def main() -> int:
    cfg = load_yaml(CONFIG_PATH)
    name = cfg["name"]
    out = data_dir("processed") / "benchmarks" / name
    out.mkdir(parents=True, exist_ok=True)
    samples, X_all = load_samples()

    # --- Manifest --------------------------------------------------------------
    built = build_manifest(
        samples.drop(columns=["dhash"]),
        id_col=cfg["task"]["id_col"],
        target_cols=list(cfg["task"]["labels"].values()),
        filters=cfg["task"]["filters"],
    )
    keep = samples["sample_id"].isin(built.manifest["sample_id"]).to_numpy()
    m = samples.loc[keep].sort_values("time_utc").reset_index()
    X = X_all[m["index"].to_numpy()]
    m = m.drop(columns="index")
    splits = make_splits(m, cfg["split"])
    for proto, sp in splits.items():
        m[f"split_{proto}"] = sp.labels.to_numpy()

    # --- Audit -----------------------------------------------------------------
    daily = m.groupby("hst_day")["Hs_label_m"].mean()
    daily.index = pd.to_datetime(daily.index)
    consecutive = daily[daily.index.to_series().diff().eq(pd.Timedelta(days=1)).to_numpy()]
    prev = daily.reindex(consecutive.index - pd.Timedelta(days=1)).to_numpy()
    summary: dict[str, Any] = {
        "population": {
            "indexed": len(samples),
            "in_manifest": len(m),
            "excluded_by": built.exclusions["excluded_by"].value_counts().to_dict(),
            "days": int(m["hst_day"].nunique()),
            "weeks": int(m["hst_week"].nunique()),
            "by_camera": m["camera"].value_counts().to_dict(),
        },
        "daily_mean_Hs_lag1_correlation": round(
            float(np.corrcoef(prev, consecutive.to_numpy())[0, 1]), 3
        ),
        "splits": {p: sp.summary() for p, sp in splits.items()},
        "split_params": {p: sp.params for p, sp in splits.items()},
        "audit": {
            p: audit(m, sp, cfg["split"], ["hst_week"] if p == "week" else [])
            for p, sp in splits.items()
        },
        "dhash_same_camera_different_days": hash_diagnostics(m, cfg["seed"]),
    }

    # --- Baselines -------------------------------------------------------------
    cdip = load_series(
        str(data_dir("processed") / "cdip" / "106" / "bulk_2009-01-01_2014-01-01.parquet"),
        "wave_significant_height",
    )
    era5_s = load_series(
        str(data_dir("processed") / "era5" / "pacioos_waimea" / "bulk_2009-01_2013-12.parquet"),
        "wave_significant_height",
    )
    era5 = np.interp(
        _ns(m["time_utc"]), _ns(era5_s.index), era5_s.to_numpy(), left=np.nan, right=np.nan
    )
    boot = cfg["evaluation"]["bootstrap"]
    edges = [float(e) for e in cfg["evaluation"]["sea_state_edges"]]
    rows, strata, preds, comps, fits = [], [], [], [], {}
    for proto in splits:
        part = m[f"split_{proto}"].to_numpy()
        test = part == TEST
        g = m["hst_day"].to_numpy()
        for rule, col in cfg["task"]["labels"].items():
            y = m[col].to_numpy(dtype=float)
            p, info = baselines(m, X, y, part, cdip, era5, cfg["baselines"], col)
            fits[f"{proto}/{rule}"] = info
            for new, ref in COMPARISONS:
                comps.append(
                    {
                        "protocol": proto,
                        "label": rule,
                        "candidate": new,
                        "reference": ref,
                        **compare(y[test], p[new][test], p[ref][test], g[test], boot),
                    }
                )
            if rule == "window_mean":
                p["label_rule_difference"] = m[cfg["task"]["labels"]["nearest"]].to_numpy()
            for base, pred in p.items():
                rows.append(
                    {
                        "protocol": proto,
                        "label": rule,
                        "baseline": base,
                        **score(y[test], pred[test], g[test], boot),
                    }
                )
                preds.append(
                    pd.DataFrame(
                        {
                            "sample_id": m.loc[test, "sample_id"].to_numpy(),
                            "protocol": proto,
                            "label": rule,
                            "baseline": base,
                            "y_true": y[test],
                            "y_pred": pred[test],
                        }
                    )
                )
                if proto == "day":
                    bins = pd.cut(y[test], edges, right=False)
                    for b in bins.categories:
                        sel = np.asarray(bins == b)
                        ok = sel & np.isfinite(pred[test])
                        strata.append(
                            {
                                "label": rule,
                                "baseline": base,
                                "Hs_bin_m": str(b),
                                "n": int(ok.sum()),
                                "days": int(pd.unique(g[test][ok]).size),
                                "bias": R.bias(y[test][ok], pred[test][ok]) if ok.any() else np.nan,
                                "rmse": R.rmse(y[test][ok], pred[test][ok]) if ok.any() else np.nan,
                            }
                        )
    results = pd.DataFrame(rows)
    summary["fits"] = fits
    summary["era5_grid_point"] = "22.0N 158.0W, 40.2 km from the cameras"

    # --- Write -----------------------------------------------------------------
    m_out = m.drop(columns=["dhash", "feature_row"])
    built.manifest = m_out
    paths = built.write(out, "manifest")
    paths["predictions"] = out / "predictions.parquet"
    paths["results"] = out / "results.csv"
    paths["strata"] = out / "strata.csv"
    paths["comparisons"] = out / "comparisons.csv"
    paths["summary"] = out / "summary.json"
    pd.concat(preds, ignore_index=True).to_parquet(paths["predictions"], index=False)
    results.round(4).to_csv(paths["results"], index=False)
    pd.DataFrame(strata).round(4).to_csv(paths["strata"], index=False)
    pd.DataFrame(comps).round(4).to_csv(paths["comparisons"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=1, default=str) + "\n")
    days = (
        m.groupby("hst_day")
        .agg(
            hst_week=("hst_week", "first"),
            images=("sample_id", "size"),
            split_day=("split_day", "first"),
            split_week=("split_week", "first"),
            split_year=("split_year", "first"),
        )
        .reset_index()
    )
    ml = data_dir("manifests") / "ml"
    paths["days"] = ml / f"{name}.days.csv"
    days.to_csv(paths["days"], index=False)
    audit_dir = data_dir("interim") / "waimea_audit"
    d_proc = data_dir("processed")
    prov = build_provenance(
        artifact_id=name,
        inputs=[
            CONFIG_PATH,
            d_proc / "pairs" / "waimea_cdip106" / "samples.parquet",
            audit_dir / "images.parquet",
            audit_dir / "features.npz",
            d_proc / "cdip" / "106" / "bulk_2009-01-01_2014-01-01.parquet",
            d_proc / "era5" / "pacioos_waimea" / "bulk_2009-01_2013-12.parquet",
        ],
        outputs=list(paths.values()),
        command="python scripts/waimea_benchmark.py",
        config=cfg,
        notes="P1 of docs/operations/collection_plan.md; results in docs/datasets/pacioos_beachcam.md.",
    )
    write_provenance(prov, out / "provenance.yaml")
    keep_dir = data_dir("manifests") / "processed"
    for src, dst in [
        (out / "provenance.yaml", f"{name}.provenance.yaml"),
        (paths["summary"], f"{name}.summary.json"),
        (paths["results"], f"{name}.results.csv"),
        (paths["strata"], f"{name}.strata.csv"),
        (paths["comparisons"], f"{name}.comparisons.csv"),
    ]:
        shutil.copyfile(src, keep_dir / dst)
    show = results[results["label"] == "window_mean"][
        ["protocol", "baseline", "n", "days", "bias", "rmse", "rmse_lo", "rmse_hi", "r2"]
    ]
    print(
        json.dumps(
            {k: summary[k] for k in ("population", "splits", "audit")}, indent=1, default=str
        )
    )
    print(show.round(3).to_string(index=False))
    print(pd.DataFrame(comps).round(3).to_string(index=False))
    return 0 if all(a["ok"] for a in summary["audit"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
