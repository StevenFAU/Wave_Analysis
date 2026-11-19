#!/usr/bin/env python
"""Bar S2 on new weeks: the gated partition swap against ERA5 + wind on strips archived after the package.

Run from the checkout that holds ``data/`` and ``runs/`` (or set ``WAVE_ANALYSIS_ROOT``)::

    wave-analysis ndbc download --camera-stations --realtime --gzip     # a .spec/.txt snapshot after the weeks
    python scripts/buoycam_era5.py download --start 2026-11 --end 2026-11 --param swh --param shww --param shts
    python scripts/buoycam_era5.py extract
    python scripts/buoycam_pairs.py --start 2026-11-14T16:00Z --out data/processed/pairs/ndbc_buoycam_fresh
    python scripts/buoycam_fusion_confirm.py build
    python scripts/buoycam_fusion_confirm.py score --dry-run    # checks and counts; scores nothing
    python scripts/buoycam_fusion_confirm.py predict            # CPU, about 0.4 s per strip and model
    python scripts/buoycam_fusion_confirm.py score

Protocol fixed on 2026-11-18, before any fusion was fitted (``docs/experiments/
ndbc_buoycam_sea_state.md``, "Stage 2: partition-aware fusion", "Confirmation on new
weeks"); the details below were fixed on 2026-11-19, before any fresh strip was scored.
Nothing is fitted here.

* **Strips:** every strip taken from 2026-11-14 16:00 UTC on (the v2cv package ends at
  15:00, labelled from the realtime snapshot of 15:33) at one of the package's 79 buoys,
  with the package's light rules and labels: the config's filters (paired, six views of a
  known format, sun at or above -6 deg, at least 3 usable views, an H_s label), the
  window mean of the buoy's own ``.spec`` WVHT, WWH and SwH records centred within 30 min
  of the image, and its wind by the same window (``scripts/buoycam_pairs.py --start``).
  The records come from a snapshot saved after the weeks they label; a buoy's strips are
  kept up to 1 h before its latest snapshot, so every label window is complete. The
  caption band is cut and the strip re-encoded as in the package (JPEG quality 95).
* **Camera:** each strip is predicted by the three DINOv3-gentle seed models of its own
  buoy's fold (``runs/buoy-fold<k>-*-seed<s>-cv-dinov3-gentle/best.pt``; the fold is the
  package's ``cv_fold``), so neither its buoy nor any buoy within 300 km was seen in
  training. Seeds are merged as ``scripts/buoycam_oof.py`` does (its own function): the
  mean of the log heights; one log-normal with the seeds' first two moments. Inference
  runs on the CPU in float32; the stage-1 runs predicted in bfloat16 on GPUs, so
  ``predict`` first re-predicts a sample of the package's strips (``--check-per-fold``)
  and compares them with the stage-1 out-of-fold table (``check_camera.json``).
* **ERA5:** its total, wind-sea and swell heights at the buoy (``scripts/buoycam_era5.py``),
  interpolated to the strip's time (no value across a gap over 3 h or past the series'
  end); for the "ERA5 alone" line, each fold's line fitted on its training strips of the
  package, as stage 1 (checked against the stage-1 out-of-fold table).
* **Fusion:** each fold's combiners as fitted on the package
  (``runs/buoycam-cv-fusion-20261119/models.pkl``, ``scripts/buoycam_partition_fusion.py``),
  applied without refitting. ``score`` first re-applies them to the package's strips and
  compares the result with that run's ``fusion_oof.csv``.
* **Bar S2, unchanged:** the gated partition swap must lower the RMSE of H_s **and** of
  the wind-wave height against ERA5 + wind, each 95 % interval of the difference below
  zero (paired bootstrap over whole buoys: 1,000 draws, seed 0). Each target is scored on
  the strips where every input and that target exist. Beside it, as out of fold: every
  other candidate, the bootstrap over station groups, and the gate.
* **When:** the set is large enough when ERA5 covers at least 7 days after the package
  (to 2026-11-21 16:00 UTC); every fresh strip with all inputs up to ERA5's end at the
  time of the run is scored. The doc's estimate of that size (about 6,000 strips at 70 or
  more buoys) is reported beside the counts, not used as a gate: on 2026-11-19 cameras
  were archiving at 70 of the 79 buoys. ``score`` refuses an earlier set. ``score
  --dry-run`` runs the checks and counts the fresh strips with every input but scores
  none, so no interim number can decide when the bar is scored.

Writes: ``build`` -> ``data/processed/packages/ndbc_buoycam_fresh_v2cv/`` (``images/``,
``labels.csv``, ``exclusions.csv``, ``summary.json``, ``provenance.yaml``); ``predict``
-> ``<out>/camera/fold<k>-seed<s>/predictions.csv`` (each model; reused on a rerun),
``<out>/camera.csv`` (seeds merged) and ``check_camera.json``; ``score`` ->
``<out>/fusion.json``, ``report.md``, ``fusion_fresh.csv`` (``--dry-run``:
``dry_run.json``). ``<out>`` defaults to ``runs/buoycam-cv-fusion-confirm``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buoycam_oof import ERA5, fold_runs, merge_fold  # scripts/ is on sys.path
from buoycam_partition_fusion import (
    BAR,
    COMPARATOR,
    TARGETS,
    features,
    inputs_ok,
    predict_fold,
    with_comparator,
)
from compare_papers import HEIGHTS, LABELS
from wave_analysis.config import data_dir, load_yaml, repo_root
from wave_analysis.datasets import build_manifest, build_provenance, write_provenance
from wave_analysis.datasets.buoycam import (
    crop_strip,
    snapshot_stamp,
    usable_view_mask,
    usable_views,
    view_columns,
)
from wave_analysis.datasets.provenance import git_state
from wave_analysis.datasets.sea_state import linear_correction, nowcast_at
from wave_analysis.evaluation.regression import rmse
from wave_analysis.evaluation.uncertainty import paired_block_bootstrap_ci
from wave_analysis.ingest.checksum import sha256_file
from wave_analysis.ingest.durable import atomic_open

#: The code's own checkout: configs come from here, data and runs from ``repo_root()``.
CODE = Path(__file__).resolve().parents[1]
CONFIG = CODE / "configs/experiments/ndbc_buoycam_heights_v2cv.yaml"
#: The first strip time after the package (its strips end at 15:00 UTC).
START = pd.Timestamp("2026-11-14T16:00:00Z")
MIN_ERA5_DAYS = 7
#: The doc's estimate of the set at 7 days (reported, not a gate).
EXPECTED_STRIPS, EXPECTED_BUOYS = 6000, 70
#: A buoy's strips end this long before its latest snapshot (complete 30-min label windows).
SNAPSHOT_MARGIN = pd.Timedelta("1h")
FUSION_RUN = "runs/buoycam-cv-fusion-20261119"
STAGE1_OOF = "runs/buoycam-cv-stage1-20261119/oof_dinov3-gentle.csv"
ARM_GLOB = "buoy-fold*-seed*-cv-dinov3-gentle"


def slug(name: str) -> str:
    """A candidate's column stem in ``fusion_oof.csv`` (as ``buoycam_partition_fusion.py`` writes it)."""
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def read_labels(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"station_id": str, "cv_fold": str}, low_memory=False)


# ----------------------------------------------------------------------------- build


def latest_snapshot(station: str, ndbc: Path) -> pd.Timestamp | None:
    """Time of the station's latest realtime ``.spec`` snapshot (``None`` if it has none)."""
    paths = list((ndbc / station / "spec").glob(f"{station}.spec.*Z.gz"))
    if not paths:
        return None
    return pd.to_datetime(max(snapshot_stamp(p) for p in paths), format="%Y%m%dT%H%M%SZ", utc=True)


def _crop(job: tuple[str, str, int]) -> tuple[str, str, int, int, int]:
    src, dest, quality = job
    sha, size, w, h = crop_strip(src, dest, quality=quality)
    return dest, sha, size, w, h


def build(args: argparse.Namespace) -> int:
    root = repo_root()
    cfg = load_yaml(args.config)
    task, ims, pkg = cfg["task"], cfg["images"], cfg["package"]
    raw = pd.read_parquet(args.pairs)
    raw["time_utc"] = pd.to_datetime(raw["time_utc"], utc=True)
    raw = raw[raw["time_utc"] >= START].reset_index(drop=True)
    ndbc = data_dir("raw") / "ndbc"
    snaps = {st: latest_snapshot(st, ndbc) for st in sorted(raw["station_id"].unique())}
    end = raw["station_id"].map(
        {st: (s - SNAPSHOT_MARGIN if s is not None else START) for st, s in snaps.items()}
    )
    late = (raw["time_utc"] >= end).to_numpy()
    raw["usable_views"] = usable_views(
        raw, min_mean=float(ims["min_view_mean"]), min_std=float(ims["min_view_std"])
    )
    raw["station_day"] = raw["station_id"] + "|" + raw["time_utc"].dt.strftime("%Y-%m-%d")
    plab = read_labels(args.package / "labels.csv")
    buoys = plab.drop_duplicates("station_id").set_index("station_id")
    known = raw["station_id"].isin(buoys.index).to_numpy()
    raw["after_snapshot_margin"] = ~late
    raw["package_buoy"] = known
    rules = [
        {"column": "package_buoy", "op": "eq", "value": True},
        {"column": "after_snapshot_margin", "op": "eq", "value": True},
        *task["filters"],
    ]
    built = build_manifest(raw, id_col=task["id_col"], target_cols=[task["label"]], filters=rules)
    m = built.manifest
    m["station_group"] = m["station_id"].map(buoys["station_group"])
    m["cv_fold"] = m["station_id"].map(buoys["cv_fold"])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    m["image"] = [
        f"images/{s}/{t:%Y}/{t:%m}/{Path(src).name}"
        for s, t, src in zip(m["station_id"], m["time_utc"], m["local_path"], strict=True)
    ]
    previous: dict[str, str] = {}
    if (out / "labels.csv").exists():
        prev = pd.read_csv(out / "labels.csv", usecols=["image", "sha256_package"])
        previous = dict(zip(prev["image"], prev["sha256_package"], strict=True))
    sha = pd.Series("", index=m.index, dtype=object)
    size = pd.Series(0, index=m.index, dtype="int64")
    jobs = []
    for i, (img, src) in enumerate(zip(m["image"], m["local_path"], strict=True)):
        dest = out / img
        if dest.exists() and img in previous and sha256_file(dest) == previous[img]:
            sha[i], size[i] = previous[img], dest.stat().st_size
        else:
            p = Path(src)
            jobs.append(
                (str(p if p.is_absolute() else root / p), str(dest), int(pkg["jpeg_quality"]))
            )
    if jobs:
        log(f"cropping {len(jobs):,} strips with {args.workers} processes")
        where = {str(out / img): i for i, img in enumerate(m["image"])}
        with Pool(args.workers) as pool:
            for dest, h, nbytes, _w, _h in pool.imap_unordered(_crop, jobs, 32):
                sha[where[dest]], size[where[dest]] = h, nbytes
    m["sha256_package"], m["size_bytes_package"] = sha, size
    cols = [
        task["id_col"], "image", "sha256_package", "size_bytes_package", "width", "height",
        "station_id", "camera_code", "station_group", "cv_fold", "latitude", "longitude",
        "time_utc", "station_day", "sun_elevation_deg", "sun_azimuth_deg", "illumination",
        *view_columns("view_mean"), *view_columns("view_std"), "usable_views",
        "Hs_label_m", "Hs_label_spread_m", "n_label_records", "wind_wave_label_m",
        "swell_label_m", "wind_speed_mps", "wind_gust_mps", "wind_dir_deg", "n_wind_records",
        "local_path", "sha256",
    ]  # fmt: skip
    labels = m[cols].rename(columns={"local_path": "source_path", "sha256": "source_sha256"})
    labels["time_utc"] = labels["time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    with atomic_open(out / "labels.csv", "w", newline="") as fh:
        labels.to_csv(fh, index=False, float_format="%.6g")
    with atomic_open(out / "exclusions.csv", "w", newline="") as fh:
        built.exclusions.to_csv(fh, index=False)
    stamps = [s for s in snaps.values() if s is not None]
    summary = {
        "strips": len(m),
        "buoys": int(m["station_id"].nunique()),
        "first": str(m["time_utc"].min()),
        "last": str(m["time_utc"].max()),
        "start_utc": str(START),
        "latest_snapshot": {"earliest_buoy": str(min(stamps)), "latest_buoy": str(max(stamps))}
        if stamps
        else None,
        "candidates": len(raw),
        "excluded_by": built.exclusions["excluded_by"].value_counts().to_dict(),
        "by_fold": m["cv_fold"].value_counts().sort_index().to_dict(),
        "by_illumination": m["illumination"].value_counts().to_dict(),
        "with_wind_wave_label": int(m["wind_wave_label_m"].notna().sum()),
        "with_buoy_wind": int(m["wind_speed_mps"].notna().sum()),
        "by_day": m.groupby(m["time_utc"].dt.strftime("%Y-%m-%d")).size().to_dict(),
    }
    with atomic_open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    write_provenance(
        build_provenance(
            artifact_id="ndbc_buoycam_fresh_v2cv",
            inputs=[args.pairs, args.config, args.package / "labels.csv"],
            outputs=[out / f for f in ("labels.csv", "exclusions.csv", "summary.json")],
            config={
                "start_utc": str(START),
                "snapshot_margin": str(SNAPSHOT_MARGIN),
                "filters": rules,
                "label": task["label"],
                "jpeg_quality": int(pkg["jpeg_quality"]),
            },
            notes=f"{len(m):,} strips archived after the v2cv package, caption band cut",
            git=git_state(CODE),
        ),
        out / "provenance.yaml",
    )
    print(json.dumps(summary, indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------- predict


def load_model(run: Path, mixer: dict[str, Any]) -> tuple[Any, dict[str, Any], Any]:
    """A stage-1 run's ``best.pt`` as an evaluation model on the CPU (no download of weights)."""
    import torch

    from wave_analysis.models.sea_state import TrainSettings
    from wave_analysis.models.wave_heights import MultiViewRegressor

    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if ck.get("aux", "none") != "none":
        raise SystemExit(f"{run}: per-view extra inputs are not supported here")
    st = TrainSettings.from_config(dict(ck["settings"]))
    model = MultiViewRegressor(
        st.backbone, len(ck["targets"]), d_model=int(mixer.get("d_model", 256)),
        heads=int(mixer.get("heads", 4)), layers=int(mixer.get("layers", 1)),
        dropout=float(mixer.get("dropout", 0.1)), pretrained=False,
        distribution=bool(ck.get("distribution")),
    )  # fmt: skip
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck, st


def predict_run(run: Path, table: pd.DataFrame, root: Path, cfg: dict[str, Any],
                args: argparse.Namespace) -> pd.DataFrame:  # fmt: skip
    """One model's heights for ``table``'s strips, in ``predictions.csv`` form."""
    import torch
    from torch.utils.data import DataLoader

    from wave_analysis.models.sea_state import build_transforms, input_size
    from wave_analysis.models.wave_heights import HeightsDataset, predict_heights

    model, ck, st = load_model(run, cfg["model"].get("mixer", {}))
    names = list(cfg["model"]["target_names"])
    if list(ck["targets"]) != list(cfg["model"]["targets"]):
        raise SystemExit(f"{run}: targets {ck['targets']} differ from the config's")
    size = input_size(st, (int(table["height"].iloc[0]), int(table["width"].iloc[0])))
    ims = cfg["images"]
    mask = usable_view_mask(table, min_mean=float(ims["min_view_mean"]),
                            min_std=float(ims["min_view_std"]), n_views=st.n_views)  # fmt: skip
    ds = HeightsDataset(root, table, st, build_transforms(st, size, train=False), mask,
                        np.zeros((len(table), len(names))))  # fmt: skip
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.loader_workers)
    pred, mu, sigma = predict_heights(model, dl, torch.device("cpu"), mean=ck["mean"],
                                      std=ck["std"], amp="none", with_distribution=True)  # fmt: skip
    fr = pd.DataFrame({"sample_id": table["sample_id"], "partition": "fresh",
                       "station_id": table["station_id"]})  # fmt: skip
    for j, (col, nm) in enumerate(zip(cfg["model"]["targets"], names, strict=True)):
        fr[f"true_{nm}_m"] = table[col].to_numpy(dtype=float)
        fr[f"pred_{nm}_m"] = pred[:, j]
        fr[f"mu_{nm}_log"], fr[f"sigma_{nm}_log"] = mu[:, j], sigma[:, j]
    return fr


def predict_folds(table: pd.DataFrame, root: Path, by_fold: dict[str, list[Path]],
                  cfg: dict[str, Any], dest: Path, args: argparse.Namespace) -> pd.DataFrame:  # fmt: skip
    """Every strip of ``table`` by its fold's seeds (one directory per model in ``dest``), merged."""
    parts = []
    for fold, runs in by_fold.items():
        k = fold.removeprefix("fold")
        t = table[table["cv_fold"] == k].sort_values("sample_id").reset_index(drop=True)
        if t.empty:
            continue
        dirs = []
        for i, run in enumerate(runs):
            d = dest / f"{fold}-seed{i}"
            f = d / "predictions.csv"
            if f.exists() and set(pd.read_csv(f, usecols=["sample_id"])["sample_id"]) == set(
                t["sample_id"]
            ):
                log(f"{fold} seed {i}: reusing {f}")
            else:
                t0 = time.time()
                fr = predict_run(run, t, root, cfg, args)
                d.mkdir(parents=True, exist_ok=True)
                fr.to_csv(f, index=False, float_format="%.6f")
                (d / "run.txt").write_text(f"{run}\n", encoding="utf-8")
                dt = time.time() - t0
                log(
                    f"{fold} seed {i}: {len(t):,} strips in {dt / 60:.1f} min ({dt / len(t):.2f} s each) <- {run.name}"
                )
            dirs.append(d)
        merged = merge_fold(dirs)
        merged["cv_fold"] = k
        parts.append(merged)
    return pd.concat(parts, ignore_index=True).sort_values("sample_id").reset_index(drop=True)


def camera_check(merged: pd.DataFrame, oof: pd.DataFrame) -> dict[str, Any]:
    """The CPU merge of package strips against the stage-1 out-of-fold table (log differences)."""
    ref = oof.set_index("sample_id").loc[merged["sample_id"]]
    out: dict[str, Any] = {"n": len(merged)}
    for h in HEIGHTS:
        d = np.log(merged[f"pred_{h}_m"].to_numpy()) - np.log(ref[f"pred_{h}_m"].to_numpy())
        s = merged[f"sigma_{h}_log"].to_numpy() - ref[f"sigma_{h}_log"].to_numpy()
        y = ref[f"true_{h}_m"].to_numpy(dtype=float)
        ok = np.isfinite(y)
        out[h] = {
            "mean_abs_log_diff": float(np.mean(np.abs(d))),
            "max_abs_log_diff": float(np.max(np.abs(d))),
            "mean_log_diff": float(np.mean(d)),
            "max_abs_sigma_diff": float(np.max(np.abs(s))),
            "rmse_m_cpu": rmse(y[ok], merged[f"pred_{h}_m"].to_numpy()[ok]),
            "rmse_m_stage1": rmse(y[ok], ref[f"pred_{h}_m"].to_numpy()[ok]),
        }
    return out


def predict(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(args.threads)
    root = repo_root()
    runs = sorted((root / "runs").glob(ARM_GLOB))
    by_fold, meta = fold_runs(runs)
    if sorted(by_fold) != [f"fold{k}" for k in range(5)] or any(
        len(r) != 3 for r in by_fold.values()
    ):
        raise SystemExit(
            f"expected 5 folds x 3 seeds of {ARM_GLOB}, got { {k: len(v) for k, v in by_fold.items()} }"
        )
    cfg = load_yaml(CODE / "configs/experiments" / f"{meta['config']}.yaml")
    args.out.mkdir(parents=True, exist_ok=True)
    git = git_state(CODE)
    # 1. A sample of the package's strips, against the stage-1 out-of-fold table.
    oof = pd.read_csv(root / STAGE1_OOF, dtype={"station_id": str, "cv_fold": str})
    plab = read_labels(args.package / "labels.csv")
    pick = pd.concat([g.sample(min(args.check_per_fold, len(g)), random_state=0)
                      for _, g in oof.groupby("cv_fold")])["sample_id"]  # fmt: skip
    check_tab = plab[plab["sample_id"].isin(pick)].reset_index(drop=True)
    log(f"check: {len(check_tab)} package strips ({args.check_per_fold} per fold) x 3 seeds")
    chk = camera_check(
        predict_folds(check_tab, args.package, by_fold, cfg, args.out / "camera_check", args), oof
    )
    chk["precision"] = "float32 on the CPU (stage 1: bfloat16 on GPUs)"
    (args.out / "check_camera.json").write_text(json.dumps(chk, indent=1))
    log("check: mean |log difference| " + ", ".join(f"{h} {chk[h]['mean_abs_log_diff']:.4f} (max {chk[h]['max_abs_log_diff']:.4f})" for h in HEIGHTS))  # fmt: skip
    if max(chk[h]["mean_abs_log_diff"] for h in HEIGHTS) > args.check_tolerance:
        raise SystemExit(
            f"the CPU predictions differ from stage 1 by more than {args.check_tolerance} in log on average"
        )
    # 2. The fresh strips.
    fresh = read_labels(args.fresh / "labels.csv")
    log(f"fresh: {len(fresh):,} strips at {fresh['station_id'].nunique()} buoys x 3 seeds")
    cam = predict_folds(fresh, args.fresh, by_fold, cfg, args.out / "camera", args)
    if len(cam) != len(fresh):
        raise SystemExit(f"{len(cam):,} merged strips, {len(fresh):,} fresh")
    cam.to_csv(args.out / "camera.csv", index=False, float_format="%.5f")
    write_provenance(
        build_provenance(
            artifact_id="buoycam_fusion_confirm_camera",
            inputs=[
                args.fresh / "labels.csv",
                *[r / "best.pt" for rs in by_fold.values() for r in rs],
            ],
            outputs=[args.out / "camera.csv", args.out / "check_camera.json"],
            config={
                "runs": {k: [str(r.relative_to(root)) for r in v] for k, v in by_fold.items()},
                "threads": args.threads,
                "precision": "float32",
                "device": "cpu",
            },
            git=git,
        ),
        args.out / "provenance_camera.yaml",
    )
    log(f"done: {args.out / 'camera.csv'}")
    return 0


# ----------------------------------------------------------------------------- score


class _Unpickler(pickle.Unpickler):
    """``models.pkl`` was written by ``buoycam_partition_fusion.py`` run as ``__main__``."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__":
            module = "buoycam_partition_fusion"
        return super().find_class(module, name)


def load_models(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as fh:
        return _Unpickler(fh).load()  # type: ignore[no-any-return]


def era5_raw(table: pd.DataFrame, era: pd.DataFrame) -> dict[str, np.ndarray]:
    """ERA5's three heights interpolated to each strip's time, as ``scripts/buoycam_oof.py``."""
    t = pd.to_datetime(table["time_utc"], utc=True)
    st = table["station_id"].to_numpy()
    raw = {}
    for h, col in ERA5.items():
        x = np.full(len(table), np.nan)
        for sid, gr in era.groupby("station_id"):
            rows = st == str(sid)
            if rows.any():
                x[rows] = nowcast_at(t[rows], gr.set_index("time_utc")[col].dropna(), max_gap="3h")
        raw[h] = x
    return raw


def era5_lines(plab: pd.DataFrame, era: pd.DataFrame) -> dict[str, dict[str, tuple[float, float]]]:
    """Each fold's ERA5 line per height, fitted on its training strips of the package (as stage 1)."""
    raw = era5_raw(plab, era)
    folds = sorted(set(plab["cv_fold"].astype(str)) - {"", "nan"})
    return {
        k: {h: linear_correction(raw[h], plab[LABELS[h]].to_numpy(dtype=float),
                                 (plab[f"split_fold{k}"] == "train").to_numpy()) for h in HEIGHTS}
        for k in folds
    }  # fmt: skip


def with_era5(table: pd.DataFrame, era: pd.DataFrame, lines: dict) -> pd.DataFrame:
    raw = era5_raw(table, era)
    out = table.copy()
    fold = table["cv_fold"].astype(str).to_numpy()
    for h in HEIGHTS:
        v = np.full(len(table), np.nan)
        for k, ab in lines.items():
            rows = fold == k
            v[rows] = ab[h][0] + ab[h][1] * raw[h][rows]
        out[f"era5_{h}_m"], out[f"era5_{h}_raw_m"] = v, raw[h]
    return out


def apply_folds(models: dict, f: pd.DataFrame, ok: np.ndarray) -> tuple[dict, np.ndarray]:
    """Every candidate's heights and the gate, each strip by its own fold's combiners."""
    preds: dict[str, dict[str, np.ndarray]] = {}
    gate = np.full(len(f), np.nan)
    for k in sorted(f["cv_fold"].unique()):
        rows = (f["cv_fold"] == k).to_numpy() & ok
        if not rows.any():
            continue
        m = models[k]
        for name, pr in predict_fold(m, f[rows]).items():
            for h, v in pr.items():
                preds.setdefault(name, {}).setdefault(h, np.full(len(f), np.nan))[rows] = v
        gate[rows] = m["swap_gated"].gate_values(with_comparator(m, f[rows]))
    return preds, gate


def score(args: argparse.Namespace) -> int:
    root = repo_root()
    git = git_state(CODE)
    kw = {"n_boot": args.n_boot, "alpha": 0.05, "seed": 0}
    era = pd.read_parquet(args.era5)
    era["station_id"] = era["station_id"].astype(str)
    plab = read_labels(args.package / "labels.csv")
    oof = pd.read_csv(root / STAGE1_OOF, dtype={"station_id": str, "cv_fold": str})
    models = load_models(root / FUSION_RUN / "models.pkl")
    checks: dict[str, Any] = {}

    # 1. ERA5 lines from the package, against the stage-1 out-of-fold table.
    lines = era5_lines(plab, era)
    chk = with_era5(oof[["sample_id", "station_id", "time_utc", "cv_fold"]], era, lines)
    checks["era5_vs_stage1_max_abs_m"] = {
        h: float(np.nanmax(np.abs(chk[f"era5_{h}_{s}m"].to_numpy() - oof[f"era5_{h}_{s}m"].to_numpy())))
        for h in HEIGHTS for s in ("", "raw_")
    }  # fmt: skip
    checks["era5_lines"] = {
        k: {h: [round(a, 5), round(b, 5)] for h, (a, b) in v.items()} for k, v in lines.items()
    }
    # 2. The combiners re-applied to the package's strips, against fusion_oof.csv.
    fp = features(oof, plab)
    pp, gp = apply_folds(models, fp, inputs_ok(fp))
    ref = pd.read_csv(root / FUSION_RUN / "fusion_oof.csv", dtype={"station_id": str})
    ref = ref.set_index("sample_id").loc[fp["sample_id"]]
    diffs = {
        name: max(
            float(np.nanmax(np.abs(v[h] - ref[f"{slug(name)}__{h}_m"].to_numpy()))) for h in v
        )
        for name, v in pp.items()
    }
    diffs["gate"] = float(np.nanmax(np.abs(gp - ref["gate"].to_numpy())))
    checks["fusion_vs_oof_max_abs_m"] = diffs
    worst_era5 = max(checks["era5_vs_stage1_max_abs_m"].values())
    worst_fusion = max(diffs.values())
    log(
        f"checks: ERA5 vs stage 1 max {worst_era5:.5f} m; combiners vs fusion_oof max {worst_fusion:.5f}"
    )
    if worst_era5 > 1e-3 or worst_fusion > 1e-3:
        raise SystemExit(
            "a check failed: the package's ERA5 or the combiners do not reproduce stage 1/2"
        )

    # 3. The fresh strips.
    fresh = read_labels(args.fresh / "labels.csv")
    cam_path = args.out / "camera.csv"
    if cam_path.exists():
        cam = pd.read_csv(cam_path, dtype={"station_id": str, "cv_fold": str})
        cam = cam.drop(
            columns=[c for c in cam.columns if c.startswith("true_")]
            + ["partition", "station_id", "cv_fold"]
        )
    elif args.dry_run:
        cam = pd.DataFrame({"sample_id": fresh["sample_id"]})
        for h in HEIGHTS:  # placeholders: the dry run counts inputs, it scores nothing
            cam[f"pred_{h}_m"], cam[f"sigma_{h}_log"], cam[f"mu_{h}_log"] = 1.0, 0.1, 0.0
    else:
        raise SystemExit(f"no {cam_path}: run predict first")
    d = fresh[["sample_id", "station_id", "station_group", "cv_fold", "time_utc"]].copy()
    for h in HEIGHTS:
        d[f"true_{h}_m"] = fresh[LABELS[h]].to_numpy(dtype=float)
    d = with_era5(d, era, lines).merge(cam, on="sample_id", how="left", validate="one_to_one")
    f = features(d, fresh)
    ok = inputs_ok(f)
    has_hs = ok & f["true_hs_m"].notna().to_numpy()
    era_end = pd.to_datetime(
        min(era.loc[era[c].notna(), "time_utc"].max() for c in ERA5.values()), utc=True
    )
    days = (era_end - START) / pd.Timedelta("1D")
    n_buoys = int(f.loc[has_hs, "station_id"].nunique())
    size = {
        "fresh_strips": len(f), "fresh_buoys": int(f["station_id"].nunique()),
        "first": str(pd.to_datetime(f["time_utc"]).min()), "last": str(pd.to_datetime(f["time_utc"]).max()),
        "era5_end": str(era_end), "era5_days_after_package": round(float(days), 2),
        "with_every_input": int(ok.sum()), "scored_hs": int(has_hs.sum()), "scored_buoys": n_buoys,
        "scored_wind_wave": int((ok & f["true_wind_wave_m"].notna().to_numpy()).sum()),
        "missing": {c: int((~np.isfinite(f[c].to_numpy(float))).sum())
                    for c in ("era5_hs", "era5_ww", "era5_sw", "cam_hs", "wind", "gust", "view_grey")},
    }  # fmt: skip
    large = days >= MIN_ERA5_DAYS
    size["large_enough"] = bool(large)
    size["expected"] = {"strips": EXPECTED_STRIPS, "buoys": EXPECTED_BUOYS}
    log(f"fresh: {size['scored_hs']:,} strips with every input at {n_buoys} buoys; ERA5 to {era_end} "
        f"({days:.1f} days after the package; the rule needs {MIN_ERA5_DAYS}; the doc expects about "
        f"{EXPECTED_STRIPS:,} strips at {EXPECTED_BUOYS} or more buoys)")  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        (args.out / "dry_run.json").write_text(
            json.dumps({"size": size, "checks": checks}, indent=1, default=str)
        )
        log(f"dry run: nothing scored; {'large enough' if large else 'not large enough yet'}")
        return 0
    if not large:
        raise SystemExit(
            f"ERA5 covers {days:.1f} days after the package, the rule needs {MIN_ERA5_DAYS}; nothing scored"
        )
    if not cam_path.exists() or d[[f"pred_{h}_m" for h in HEIGHTS]].isna().any().any():
        raise SystemExit("some fresh strips have no camera prediction: rerun predict")

    # 4. Bar S2 on the fresh strips.
    preds, gate = apply_folds(models, f, ok)
    oof_res = json.loads((root / FUSION_RUN / "fusion.json").read_text())
    res: dict[str, Any] = {"protocol": "docs/experiments/ndbc_buoycam_sea_state.md, Stage 2, Confirmation on new weeks",
                           "size": size, "checks": checks, "targets": {}, "bootstrap": kw}  # fmt: skip
    for tname, tcol in TARGETS.items():
        y = f[tcol].to_numpy(float)
        sel = ok & np.isfinite(y)
        g, grp = f.loc[sel, "station_id"].to_numpy(), f.loc[sel, "station_group"].to_numpy()
        r: dict[str, Any] = {"n": int(sel.sum()), "buoys": len(set(g)), "groups": len(set(grp)),
                             "rmse_m": {}, "rmse_m_out_of_fold": oof_res["targets"][tname]["rmse_m"],
                             "minus_bar_comparator": {}}  # fmt: skip
        for name, pr in preds.items():
            r["rmse_m"][name] = rmse(y[sel], pr[tname][sel])
        base = preds[COMPARATOR][tname][sel]
        for name, pr in preds.items():
            if name != COMPARATOR:
                r["minus_bar_comparator"][name] = {
                    "by_buoy": list(paired_block_bootstrap_ci(y[sel], pr[tname][sel], base, rmse, groups=g, **kw)),
                    "by_group": list(paired_block_bootstrap_ci(y[sel], pr[tname][sel], base, rmse, groups=grp, **kw)),
                }  # fmt: skip
        r["bar_side_met"] = bool(r["minus_bar_comparator"][BAR]["by_buoy"][2] < 0)
        res["targets"][tname] = r
    res["bar_met"] = all(res["targets"][t]["bar_side_met"] for t in TARGETS)
    sel = ok & np.isfinite(gate)
    gf = f[sel].assign(gate=gate[sel])
    res["gate"] = {
        "mean": float(gf["gate"].mean()),
        "by_illumination": gf.groupby("illumination")["gate"].agg(["mean", "size"]).round(3).to_dict("index"),
        "share_below_0.5": float((gf["gate"] < 0.5).mean()),
    }  # fmt: skip
    (args.out / "fusion.json").write_text(json.dumps(res, indent=1, default=str))
    out = f[["sample_id", "station_id", "station_group", "cv_fold", "time_utc", "true_hs_m",
             "true_wind_wave_m", "illumination"]].copy()  # fmt: skip
    out["gate"] = gate
    for name, pr in preds.items():
        for h, v in pr.items():
            out[f"{slug(name)}__{h}_m"] = v
    out.to_csv(args.out / "fusion_fresh.csv", index=False, float_format="%.4f")

    lines_md = [f"# Bar S2 on new weeks ({size['first'][:16]} to {size['last'][:16]} UTC; ERA5 to {era_end:%Y-%m-%d %H:%M})\n",
                f"Checks: ERA5 lines reproduce stage 1 (max {worst_era5:.5f} m); combiners reproduce the out-of-fold "
                f"fusion (max {worst_fusion:.5f}).\n"]  # fmt: skip
    for tname, r in res["targets"].items():
        lines_md.append(
            f"## {tname} ({r['n']:,} strips, {r['buoys']} buoys, {r['groups']} groups)\n"
        )
        for name, v in sorted(r["rmse_m"].items(), key=lambda kv: kv[1]):
            dd = r["minus_bar_comparator"].get(name)
            o = r["rmse_m_out_of_fold"].get(name)
            lines_md.append(f"- {name}: RMSE {v:.3f} m" + (f" (out of fold {o:.3f})" if o is not None else "") + (
                f"; minus ERA5 + wind {dd['by_buoy'][0]:+.3f} [{dd['by_buoy'][1]:+.3f}, {dd['by_buoy'][2]:+.3f}] "
                f"(by group [{dd['by_group'][1]:+.3f}, {dd['by_group'][2]:+.3f}])" if dd else ""))  # fmt: skip
        lines_md.append("")
    gt = res["gate"]
    lines_md.append(f"Gate: mean {gt['mean']:.2f}; below 0.5 on {gt['share_below_0.5']:.1%} of strips; by light "
                    + ", ".join(f"{k} {v['mean']:.2f}" for k, v in gt["by_illumination"].items()))  # fmt: skip
    lines_md.append(f"\n**Bar S2 on new weeks: {'met' if res['bar_met'] else 'not met'}.**")
    text = "\n".join(lines_md) + "\n"
    (args.out / "report.md").write_text(text, encoding="utf-8")
    write_provenance(
        build_provenance(
            artifact_id="buoycam_fusion_confirm",
            inputs=[
                args.fresh / "labels.csv",
                cam_path,
                args.era5,
                args.package / "labels.csv",
                root / STAGE1_OOF,
                root / FUSION_RUN / "models.pkl",
            ],
            outputs=[args.out / f for f in ("fusion.json", "report.md", "fusion_fresh.csv")],
            config={"start_utc": str(START), "min_era5_days": MIN_ERA5_DAYS, **kw},
            git=git,
        ),
        args.out / "provenance.yaml",
    )
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    fresh_default = data_dir("processed") / "packages/ndbc_buoycam_fresh_v2cv"
    pkg_default = data_dir("processed") / "packages/ndbc_buoycam_heights_v2cv"
    out_default = repo_root() / "runs/buoycam-cv-fusion-confirm"

    b = sub.add_parser(
        "build", help="light rules, labels, folds and cropped images of the fresh strips"
    )
    b.add_argument(
        "--pairs",
        type=Path,
        default=data_dir("processed") / "pairs/ndbc_buoycam_fresh/samples.parquet",
    )
    b.add_argument("--config", type=Path, default=CONFIG)
    b.add_argument("--package", type=Path, default=pkg_default)
    b.add_argument("--out", type=Path, default=fresh_default)
    b.add_argument("--workers", type=int, default=8)
    b.set_defaults(func=build)

    p = sub.add_parser("predict", help="the camera: each buoy's fold's three seeds, merged (CPU)")
    p.add_argument("--fresh", type=Path, default=fresh_default)
    p.add_argument("--package", type=Path, default=pkg_default)
    p.add_argument("--out", type=Path, default=out_default)
    p.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--loader-workers", type=int, default=4)
    p.add_argument("--check-per-fold", type=int, default=40)
    p.add_argument(
        "--check-tolerance",
        type=float,
        default=0.03,
        help="largest mean |log difference| from stage 1 accepted (bfloat16 there, float32 here)",
    )
    p.set_defaults(func=predict)

    s = sub.add_parser("score", help="bar S2 on the fresh strips (checks first)")
    s.add_argument("--fresh", type=Path, default=fresh_default)
    s.add_argument("--package", type=Path, default=pkg_default)
    s.add_argument(
        "--era5", type=Path, default=data_dir("processed") / "era5/buoycam/era5_at_buoys.parquet"
    )
    s.add_argument("--out", type=Path, default=out_default)
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--dry-run", action="store_true", help="checks and counts only; scores nothing")
    s.set_defaults(func=score)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
