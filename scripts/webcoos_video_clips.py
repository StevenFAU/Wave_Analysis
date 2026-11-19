#!/usr/bin/env python
"""WebCOOS video clips for wave models: pair each clip with its CDIP records, and cut a short frame sequence.

    python scripts/webcoos_video_clips.py pair     [cameras...]   # labels -> data/processed/video/clips.parquet
    python scripts/webcoos_video_clips.py frames   [--workers 6]  # 64 frames at 4 fps per clip, 256 px high JPEG
    python scripts/webcoos_video_clips.py windows  [--workers 4]  # every 16-s window of each clip + its pixel spectrum
    python scripts/webcoos_video_clips.py physics  [--workers 4]  # spectrum recipe v2 + breaking features per clip (M6-M7)
    python scripts/webcoos_video_clips.py pack                    # data/processed/packages/<name>/ (clips + frames + SHA256SUMS)
    GPU_HOST=<ssh-alias> python scripts/webcoos_video_clips.py ship  # the windows to the GPU server, through Cloudflare R2

**Pair.** Every clip under ``data/raw/webcoos_video/<camera>/`` (downloaded by
``scripts/webcoos_video_sample.sh``) gets its start (file name), length (``ffprobe``,
cached) and midpoint, and the label of ADR 0010: the mean H_s of the CDIP records whose
sample centre lies within 30 min of the clip's midpoint; peak period and direction from
the nearest record (a peak can jump between swell systems, so it is not averaged). The
CDIP records come from ``scripts/webcoos_pairs.py``'s loader (archived records win
over realtime copies), so stills and clips follow one rule. Cameras and buoys: those of
``webcoos_pairs.PAIRINGS``.

**Frames.** From each clip, 16 s around its midpoint at 4 frames per second (64
frames, one to three wave periods: V-JEPA 2's own pretraining rate and length,
``research_notes/Vision backbones for wave cameras/video_backbones.md`` section 3),
scaled to 256 px high with the camera's aspect kept, as JPEG in
``data/processed/video/frames_4fps/<camera>/<clip>/NN.jpg``. Burned-in overlays
(Currituck's "Hampton Inn" label, ``MASKS``) are blacked out first, so no model can
read the site from text. The same frames feed the single-image model (the middle
frame) and the video model (all 64), so the two see identical pixels. About 1.5 MB
per clip: small enough to send to the GPU server, where the 2592-2688 px originals
(30-90 MB) are not.

**Windows** (the whole-clip bars M3-M5, ``docs/experiments/webcoos_video.md``). Each
clip is cut whole at 4 fps, 256 px high, masks first, and its first ``64 K`` frames are
kept: K = floor(duration / 16 s) non-overlapping 16-s windows from the clip's start
(fewer if the stream holds fewer frames than its stated length). They are stored as
one uncompressed zip per clip, ``data/processed/video/windows_4fps/<camera>/<clip>.zip``
(``0000.jpg`` ...; window w is frames 64 w .. 64 w + 63), the very JPEG files ffmpeg
writes, so the pixels are made the same way as the 64-frame cut's. Beside each zip a
JSON file holds K, the zip's SHA-256 and the clip's pixel spectrum
(``wave_analysis.processing.pixel_spectrum``, computed from the same frames). About
30-40 MB per 10-min clip.

**Pack.** ``data/processed/packages/<name>/``: ``clips.csv`` (the usable clips that
have all their frames), ``frames/`` (hard links, no second copy) and ``SHA256SUMS``;
with windows cut, also ``windows.csv`` (K and checksum per clip) and
``pixel_spectrum.csv`` (the spectrum's features) and ``pixel_spectrum_S.npz`` (the
averaged spectra). The package grows as clips download; ``deploy/dgx/run.sh sync``
sends only new files. The windows themselves travel apart (``ship``).

**Ship.** The windows are about 35 MB a clip, 25 times the 64-frame cut, and the SSH
route to the GPU server carried 1.5-2 Mbit/s (2026-11-17/18), about 16-21 GB a day.
``ship`` sends them through the project's Cloudflare R2 bucket instead (36 Mbit/s up
from the desktop, 20 MB/s down to the server, measured 2026-11-18): new zips go up in
batches to a staging prefix (``--bwlimit``, so the home line keeps room for the
running downloads), the server fetches each through a short-lived signed link (kept
off command lines), checks its SHA-256 and lists it in ``VERIFIED``, and the staged
copies are deleted. R2 credentials never leave the desktop. Server folder:
``<BASE>/video_windows/<name>/<camera>/<clip>.zip`` (``/data/video_windows/...`` in the
containers). ``ship --package`` sends the package the same way, as one tar, and leaves it
where ``deploy/dgx/run.sh sync`` would (so ``run.sh push`` finds it there).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir, portable_path, repo_root
from wave_analysis.ingest.durable import atomic_open
from wave_analysis.processing.pixel_spectrum import RECIPE as SPECTRUM_RECIPE
from wave_analysis.processing.pixel_spectrum import (
    RECIPE_V2,
    block_means,
    physics_v2,
    pixel_spectrum,
)
from wave_analysis.processing.solar import solar_position
from wave_analysis.processing.synchronize import label_from_candidates, pair_window
from wave_analysis.sources.webcoos import ONPREM_POSITIONS, still_time

N_FRAMES, FPS, SPAN_S = 64, 4, 16
HEIGHT = 256  # px; width follows the camera's aspect (even)
OUT = data_dir("processed") / "video"
FRAMES = OUT / "frames_4fps"
WINDOWS = OUT / "windows_4fps"  # whole clips, K windows of 64 frames (bars M3-M5)
R2_STAGING = "r2:wave-analysis-raw/staging/webcoos_video_windows"
# Burned-in text, blacked out in the source frame (ffmpeg drawbox; iw/ih = source size).
MASKS = {"currituck_hampton_inn": "drawbox=x=0:y=ih-170:w=560:h=170:color=black:t=fill"}


def cdip_energy_period(station: str) -> pd.Series:
    """The energy period Tm-1,0 = m_-1 / m_0 of every CDIP record's full spectrum, by record start (UTC).

    From the standardized spectra beside the bulk records (``data/processed/cdip/<station>/
    spectrum_*.parquet``); an archived record wins over its realtime copy, as in the bulk loader.
    """
    d = data_dir("processed") / "cdip" / station
    cols = ["time_utc", "frequency_hz", "bandwidth_hz", "spectral_density_m2_hz"]
    parts = [pd.read_parquet(f, columns=cols).assign(realtime="_realtime_" in f.name)
             for f in sorted(d.glob("spectrum_*.parquet"))]  # fmt: skip
    if not parts:
        return pd.Series(dtype=float, name="Te_s")
    sp = pd.concat(parts, ignore_index=True)
    sp = sp[sp["realtime"] == sp.groupby("time_utc")["realtime"].transform("min")]
    e = sp["spectral_density_m2_hz"] * sp["bandwidth_hz"]
    g = sp.assign(e=e, em1=e / sp["frequency_hz"]).groupby("time_utc")
    return (g["em1"].sum() / g["e"].sum()).rename("Te_s")


def _pairs_module():
    spec = importlib.util.spec_from_file_location(
        "webcoos_pairs", repo_root() / "scripts" / "webcoos_pairs.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["webcoos_pairs"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def probe(path: str) -> dict:
    """Length, frame rate and size of a clip (``ffprobe``)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,duration:format=duration", "-of", "json", path],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    try:
        j = json.loads(out.stdout)
        st = j["streams"][0]
        num, den = (int(x) for x in st["avg_frame_rate"].split("/"))
        dur = float(st.get("duration") or j["format"]["duration"])
        return {"path": path, "duration_s": dur, "fps": num / den if den else np.nan,
                "width": int(st["width"]), "height": int(st["height"]), "readable": True}  # fmt: skip
    except Exception:
        return {
            "path": path,
            "duration_s": np.nan,
            "fps": np.nan,
            "width": 0,
            "height": 0,
            "readable": False,
        }


def pair(args: argparse.Namespace) -> int:
    wp = _pairs_module()
    root = data_dir("raw") / "webcoos_video"
    cams = args.cameras or sorted(c for c in wp.PAIRINGS if (root / c).is_dir())
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / "probe_cache.parquet"
    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=["path"])
    rows = []
    for cam in cams:
        files = sorted(str(p) for p in (root / cam).rglob("*.mp4"))
        new = [f for f in files if f not in set(cache["path"])]
        if new:
            with Pool(args.workers) as pool:
                cache = pd.concat(
                    [cache, pd.DataFrame(pool.map(probe, new, chunksize=8))], ignore_index=True
                )
            cache.to_parquet(cache_path, index=False)
        c = cache[cache["path"].isin(files)].copy()
        c["start_utc"] = [still_time(Path(p).name, cam) for p in c["path"]]
        c["mid_utc"] = c["start_utc"] + pd.to_timedelta(c["duration_s"].fillna(0) / 2, unit="s")
        c["camera"] = cam
        station = str(wp.PAIRINGS[cam]["cdip"])
        lat, lon = ONPREM_POSITIONS[cam]
        rec, _ = wp.cdip_records(station, lat, lon)
        rec = rec.assign(
            time_utc=rec["start_utc"] + pd.to_timedelta(rec["averaging_period_s"] / 2, unit="s")
        )
        rec = rec.join(cdip_energy_period(station), on="start_utc").reset_index(drop=True)
        c = c.reset_index(drop=True)
        tgt = c.rename(columns={"mid_utc": "time_utc"})[["time_utc"]]
        pw = pair_window(tgt, rec, window="60min")
        cand = pd.concat(
            [pd.Series(pw["target_index"].to_numpy(), name="clip"), pw.drop(columns=["target_index", "reference_index"]),
             rec.loc[pw["reference_index"], ["Hs_m", "Tp_s", "Tm02_s", "Te_s", "Dp_deg"]].reset_index(drop=True)],
            axis=1,
        )  # fmt: skip
        win = label_from_candidates(
            cand, "Hs_m", rule="window_mean", target="clip", max_offset="30min"
        ).reindex(c.index)
        c["Hs_label_m"] = win["label"]
        c["n_label_records"] = win["n_records"].fillna(0).astype(int)
        for col, out in (
            ("Tp_s", "Tp_label_s"),
            ("Tm02_s", "Tm02_label_s"),
            ("Te_s", "Te_label_s"),
        ):
            c[out] = label_from_candidates(
                cand, col, rule="nearest", target="clip", max_offset="30min"
            ).reindex(c.index)["label"]
        c["Dp_label_deg"] = label_from_candidates(cand, "Dp_deg", rule="nearest", target="clip", max_offset="30min",
                                                  circular=True).reindex(c.index)["label"]  # fmt: skip
        el, _ = solar_position(c["mid_utc"], np.full(len(c), lat), np.full(len(c), lon))
        c["sun_elevation_deg"] = np.round(el, 2)
        c["cdip_station"] = station
        tz = str(wp.PAIRINGS[cam]["timezone"])
        c["local_day"] = c["mid_utc"].dt.tz_convert(tz).dt.strftime("%Y-%m-%d")
        c["camera_day"] = cam + "|" + c["local_day"]
        c["clip_id"] = [Path(p).stem for p in c["path"]]
        rows.append(c)
        print(f"{cam}: {len(c):,} clips, {int(c['readable'].sum()):,} readable, {int(c['Hs_label_m'].notna().sum()):,} "
              f"labelled (CDIP {station}); median length {c['duration_s'].median():.0f} s", flush=True)  # fmt: skip
    t = pd.concat(rows, ignore_index=True)
    t["exclusion"] = np.select(
        [~t["readable"].astype(bool), t["duration_s"] < SPAN_S + 2, t["Hs_label_m"].isna()],
        ["unreadable", "shorter_than_span", "no_buoy_within_30min"], default="",
    )  # fmt: skip
    t["path"] = [portable_path(p) for p in t["path"]]
    cols = ["clip_id", "camera", "path", "start_utc", "mid_utc", "duration_s", "fps", "width", "height",
            "sun_elevation_deg", "local_day", "camera_day", "cdip_station", "Hs_label_m", "n_label_records",
            "Tp_label_s", "Tm02_label_s", "Te_label_s", "Dp_label_deg", "exclusion"]  # fmt: skip
    t[cols].to_parquet(OUT / "clips.parquet", index=False)
    ok = t[t["exclusion"] == ""]
    summary = {"clips": len(t), "usable": len(ok), "by_camera": ok["camera"].value_counts().to_dict(),
               "camera_days": int(ok["camera_day"].nunique()), "exclusion": t["exclusion"].replace("", "ok").value_counts().to_dict(),
               "Hs_label_m": ok["Hs_label_m"].describe().round(2).to_dict()}  # fmt: skip
    with atomic_open(OUT / "clips_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))
    return 0


def _frames(job: tuple[str, str, float, str]) -> tuple[str, int]:
    src, dest, mid_offset, camera = job
    d = Path(dest)
    if d.is_dir() and len(list(d.glob("*.jpg"))) == N_FRAMES:
        return dest, N_FRAMES
    tmp = d.with_name(d.name + ".part")
    tmp.mkdir(parents=True, exist_ok=True)
    start = max(0.0, mid_offset - SPAN_S / 2)
    vf = ",".join(f for f in (MASKS.get(camera, ""), f"fps={FPS}", f"scale=-2:{HEIGHT}") if f)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-i", src, "-t", str(SPAN_S),
         "-vf", vf, "-frames:v", str(N_FRAMES), "-q:v", "3", str(tmp / "%02d.jpg")],
        capture_output=True, timeout=300,
    )  # fmt: skip
    n = len(list(tmp.glob("*.jpg")))
    if n == N_FRAMES:
        if d.exists():
            for f in d.glob("*"):
                f.unlink()
            d.rmdir()
        tmp.rename(d)
    return dest, n


def frames(args: argparse.Namespace) -> int:
    t = pd.read_parquet(OUT / "clips.parquet")
    t = t[t["exclusion"] == ""]
    root = repo_root()
    jobs = [(str(root / p if not Path(p).is_absolute() else p), str(FRAMES / cam / cid), dur / 2, cam)
            for p, cam, cid, dur in zip(t["path"], t["camera"], t["clip_id"], t["duration_s"], strict=True)]  # fmt: skip
    done, short = 0, []
    with Pool(args.workers) as pool:
        for dest, n in pool.imap_unordered(_frames, jobs, chunksize=4):
            done += 1
            if n != N_FRAMES:
                short.append(dest)
            if done % 200 == 0:
                print(f"  {done:,}/{len(jobs):,}", flush=True)
    print(
        f"{len(jobs) - len(short):,} clips with {N_FRAMES} frames; {len(short)} short: {short[:3]}"
    )
    return 0


def n_windows(duration_s: float) -> int:
    """K = floor(duration / 16 s), the clip's non-overlapping 16-s windows from its start."""
    return int(np.floor(duration_s / SPAN_S + 1e-9))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _windows(job: tuple[str, str, float, str, str]) -> dict:
    """Cut one clip whole (4 fps, 256 px, masks), keep its K windows, zip them, and add its spectrum.

    A zip already there is kept; its spectrum is recomputed when the JSON is missing.
    """
    src, dest, duration, camera, clip_id = job
    zpath, jpath = Path(dest).with_suffix(".zip"), Path(dest).with_suffix(".json")
    if zpath.exists() and jpath.exists():
        return json.loads(jpath.read_text())
    k_design = n_windows(duration)
    row: dict = {"clip_id": clip_id, "camera": camera, "duration_s": duration, "K_design": k_design}
    tmp = Path(
        tempfile.mkdtemp(prefix="wa-win-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None)
    )
    try:
        t0 = time.time()
        if not zpath.exists():
            vf = ",".join(
                f for f in (MASKS.get(camera, ""), f"fps={FPS}", f"scale=-2:{HEIGHT}") if f
            )
            r = subprocess.run(
                ["nice", "-n", "10", "ffmpeg", "-v", "error", "-nostdin", "-threads", "4", "-y", "-i", src,
                 "-vf", vf, "-frames:v", str(N_FRAMES * k_design), "-q:v", "3", str(tmp / "%04d.jpg")],
                capture_output=True, text=True, timeout=3600,
            )  # fmt: skip
            n = len(list(tmp.glob("*.jpg")))
            k = min(k_design, n // N_FRAMES)
            if k < 1:
                return row | {
                    "frames_decoded": n,
                    "K": 0,
                    "error": (r.stderr or "no frames")[-300:],
                }
            zpath.parent.mkdir(parents=True, exist_ok=True)
            part = zpath.with_suffix(".zip.part")
            with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_STORED) as z:
                for i in range(N_FRAMES * k):  # ffmpeg counts from 1, the zip from 0
                    z.write(tmp / f"{i + 1:04d}.jpg", f"{i:04d}.jpg")
            part.rename(zpath)
            row["frames_decoded"] = n
        from PIL import Image

        with zipfile.ZipFile(zpath) as z:
            names = sorted(z.namelist())
            grey = np.stack([np.asarray(Image.open(z.open(m)).convert("L")) for m in names])
        row.setdefault("frames_decoded", len(names))  # a kept zip: at least its frames
        spec = pixel_spectrum(block_means(grey).reshape(len(names), -1))
        row |= {"K": len(names) // N_FRAMES, "cut_s": round(time.time() - t0, 1),
                "zip_bytes": zpath.stat().st_size, "zip_sha256": _sha256(zpath),
                "frame_w": int(grey.shape[2]), "frame_h": int(grey.shape[1]), "spectrum_recipe": SPECTRUM_RECIPE,
                "spectrum": {k2: spec[k2] for k2 in ("n_blocks", "n_live", "n_water")} | spec["features"],
                "spectrum_f": spec["f"], "spectrum_S": spec["S"], "spectrum_background": spec["background"]}  # fmt: skip
        with atomic_open(jpath, "w") as fh:
            json.dump(row, fh)
        return row
    except Exception as e:  # one bad clip must not stop the others
        row["error"] = repr(e)[-300:]
        return row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def windows(args: argparse.Namespace) -> int:
    t = pd.read_parquet(OUT / "clips.parquet")
    t = t[t["exclusion"] == ""]
    if args.limit:
        t = t.head(args.limit)
    root = repo_root()
    jobs = [(str(root / p if not Path(p).is_absolute() else p), str(WINDOWS / cam / cid), float(dur), cam, cid)
            for p, cam, cid, dur in zip(t["path"], t["camera"], t["clip_id"], t["duration_s"], strict=True)]  # fmt: skip
    todo = [j for j in jobs if not Path(j[1]).with_suffix(".json").exists()]
    print(
        f"{len(jobs):,} usable clips, {len(jobs) - len(todo):,} already cut, {len(todo):,} to cut",
        flush=True,
    )
    done, failed, short, t0 = 0, [], [], time.time()
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(_windows, todo, chunksize=1):
            done += 1
            if "error" in row:
                failed.append((row["clip_id"], row["error"][:120]))
            elif row["K"] < row["K_design"]:
                short.append((row["clip_id"], row["K"], row["K_design"]))
            if done % 20 == 0 or done == len(todo):
                print(
                    f"  {done:,}/{len(todo):,} clips, {(time.time() - t0) / done:.1f} s each",
                    flush=True,
                )
    print(f"cut {done - len(failed):,}; failed {len(failed)} {failed[:3]}; fewer frames than their length "
          f"(K below floor(duration / 16)): {len(short)} {short[:5]}")  # fmt: skip
    return 1 if failed else 0


def _physics(job: tuple[str, str, str]) -> dict:
    """Spectrum recipe v2 and the breaking features of one clip's windows zip, in a JSON beside it."""
    zp, cam, cid = job
    dst = Path(zp).with_suffix(".physics_v2.json")
    try:
        from PIL import Image

        with zipfile.ZipFile(zp) as z:
            names = sorted(n for n in z.namelist() if n.endswith(".jpg"))
            grey = np.stack([np.asarray(Image.open(z.open(m)).convert("L")) for m in names])
        r = physics_v2(block_means(grey))
        row = {"clip_id": cid, "camera": cam, "physics_recipe": RECIPE_V2, "frames": len(names),
               **{k: r.get(k) for k in ("n_blocks", "n_live", "n_water", "threshold", "f", "S", "S_weighted", "S_on", "coherence")},
               "features": r["features"]}  # fmt: skip
    except Exception as e:
        return {"clip_id": cid, "error": repr(e)}
    with atomic_open(dst, "w") as fh:
        json.dump(row, fh)
    return {"clip_id": cid}


def physics(args: argparse.Namespace) -> int:
    t = pd.read_parquet(OUT / "clips.parquet")
    t = t[t["exclusion"] == ""]
    jobs = [(str(WINDOWS / c / f"{i}.zip"), c, i) for c, i in zip(t["camera"], t["clip_id"], strict=True)
            if (WINDOWS / c / f"{i}.json").exists() and (WINDOWS / c / f"{i}.zip").exists()]  # fmt: skip
    todo = [j for j in jobs if not Path(j[0]).with_suffix(".physics_v2.json").exists()]
    print(f"{len(jobs):,} clips with windows, {len(todo):,} without physics v2 yet", flush=True)
    failed, t0 = [], time.time()
    with Pool(args.workers) as pool:
        for k, row in enumerate(pool.imap_unordered(_physics, todo, chunksize=1), 1):
            if "error" in row:
                failed.append((row["clip_id"], row["error"][:120]))
            if k % 50 == 0 or k == len(todo):
                print(
                    f"  {k:,}/{len(todo):,} clips, {(time.time() - t0) / k:.1f} s each", flush=True
                )
    print(f"physics v2 of {len(todo) - len(failed):,} clips; failed {len(failed)} {failed[:3]}")
    return 1 if failed else 0


def _physics_table(t: pd.DataFrame) -> pd.DataFrame:
    """physics_v2.csv: the clip's v2 features (prefix ``v2_``) for the clips in ``t`` that have them."""
    rows = []
    for cam, cid in zip(t["camera"], t["clip_id"], strict=True):
        j = WINDOWS / cam / f"{cid}.physics_v2.json"
        if j.exists():
            r = json.loads(j.read_text())
            if r.get("physics_recipe") == RECIPE_V2:
                rows.append(
                    {
                        "clip_id": cid,
                        "v2_n_water": r["n_water"],
                        **{f"v2_{k}": v for k, v in r["features"].items()},
                    }
                )
    return pd.DataFrame(rows)


def _windows_tables(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """windows.csv, pixel_spectrum.csv and the averaged spectra of the clips in ``t`` that have windows."""
    rows, spec, curves = [], [], {}
    for cam, cid in zip(t["camera"], t["clip_id"], strict=True):
        j = WINDOWS / cam / f"{cid}.json"
        if not j.exists() or not (WINDOWS / cam / f"{cid}.zip").exists():
            continue
        r = json.loads(j.read_text())
        if "error" in r or r.get("K", 0) < 1 or r.get("spectrum_recipe") != SPECTRUM_RECIPE:
            continue
        rows.append({k: r[k] for k in ("clip_id", "camera", "duration_s", "K_design", "K", "frames_decoded",
                                       "frame_w", "frame_h", "zip_bytes", "zip_sha256")})  # fmt: skip
        spec.append({"clip_id": cid, **r["spectrum"]})
        curves[cid] = (np.asarray(r["spectrum_f"]), np.asarray(r["spectrum_S"]))
    return pd.DataFrame(rows), pd.DataFrame(spec), curves


def pack(args: argparse.Namespace) -> int:
    t = pd.read_parquet(OUT / "clips.parquet")
    t = t[t["exclusion"] == ""].copy()
    have = [len(list((FRAMES / c / i).glob("*.jpg"))) == N_FRAMES
            for c, i in zip(t["camera"], t["clip_id"], strict=True)]  # fmt: skip
    t = t[np.array(have, dtype=bool)].drop(columns=["path"]).sort_values("clip_id")
    pkg = data_dir("processed") / "packages" / args.name
    (pkg / "frames").mkdir(parents=True, exist_ok=True)
    keep = set()
    for c, i in zip(t["camera"], t["clip_id"], strict=True):
        d = pkg / "frames" / c / i
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted((FRAMES / c / i).glob("*.jpg")):
            keep.add(d / f.name)
            if not (d / f.name).exists():
                (d / f.name).hardlink_to(f)
    for f in (pkg / "frames").rglob("*.jpg"):  # clips dropped since the last pack
        if f not in keep:
            f.unlink()
    with atomic_open(pkg / "clips.csv", "w") as fh:
        t.to_csv(fh, index=False)
    listed = ["clips.csv", "frames"]
    win, spec, curves = _windows_tables(t)
    if len(win):
        with atomic_open(pkg / "windows.csv", "w") as fh:
            win.sort_values("clip_id").to_csv(fh, index=False)
        with atomic_open(pkg / "pixel_spectrum.csv", "w") as fh:
            spec.sort_values("clip_id").to_csv(fh, index=False)
        ids = sorted(curves)
        np.savez_compressed(pkg / "pixel_spectrum_S.npz", clip_id=np.array(ids),
                            **{f"f_{i}": curves[c][0] for i, c in enumerate(ids)},
                            **{f"S_{i}": curves[c][1] for i, c in enumerate(ids)})  # fmt: skip
        listed += ["windows.csv", "pixel_spectrum.csv", "pixel_spectrum_S.npz"]
    phys = _physics_table(t)
    if len(phys):
        with atomic_open(pkg / "physics_v2.csv", "w") as fh:
            phys.sort_values("clip_id").to_csv(fh, index=False)
        listed.append("physics_v2.csv")
    sums = subprocess.run(
        f"find {' '.join(listed)} -type f -print0 | sort -z | xargs -0 sha256sum",
        shell=True, cwd=pkg, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    with atomic_open(pkg / "SHA256SUMS", "w") as fh:
        fh.write(sums)
    size = sum(f.stat().st_size for f in keep) / 1e9
    print(f"{pkg}: {len(t):,} clips ({t['camera'].value_counts().to_dict()}), "
          f"{t['local_day'].nunique()} days, {size:.2f} GB")  # fmt: skip
    if len(win):
        w = t.merge(win[["clip_id", "K"]], on="clip_id")
        print(f"  windows of {len(win):,} clips ({int(win['K'].sum()):,} windows, {win['zip_bytes'].sum() / 1e9:.1f} GB), "
              f"{w['local_day'].nunique()} days; {len(t) - len(win):,} clips without windows yet")  # fmt: skip
    return 0


def _ssh(host: str, cmd: str, stdin: str | None = None) -> str:
    return subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", host, cmd], input=stdin,
                          capture_output=True, text=True, check=True).stdout  # fmt: skip


def _ship_package(host: str, base: str, args: argparse.Namespace) -> int:
    """The package as one tar through R2, left on the server as ``deploy/dgx/run.sh sync`` leaves it
    (``<BASE>/packages/<name>.tar`` and ``<name>.content``, the SHA-256 of its SHA256SUMS)."""
    src = data_dir("processed") / "packages" / args.name
    content = _sha256(src / "SHA256SUMS")
    if (
        _ssh(host, f"cat {base}/packages/{args.name}.content 2>/dev/null || true").strip()
        == content
    ):
        print(f"package {args.name} already on the server (content {content[:12]})")
        return 0
    with tempfile.TemporaryDirectory(dir=data_dir("interim")) as tmp:
        tarf = Path(tmp) / f"{args.name}.tar"
        subprocess.run(["tar", "-C", str(src.parent), "-cf", str(tarf), args.name], check=True)
        sha = _sha256(tarf)
        obj = f"{R2_STAGING}/{args.name}.package.tar"
        t0 = time.time()
        subprocess.run(
            ["rclone", "copyto", "--bwlimit", args.bwlimit, "--s3-no-check-bucket", str(tarf), obj],
            check=True,
        )
        link = subprocess.run(["rclone", "link", "--expire", "12h", obj], capture_output=True, text=True,
                              check=True).stdout.strip()  # fmt: skip
        part = f"{base}/packages/{args.name}.tar.part"
        _ssh(
            host,
            f"umask 077 && cat > {base}/packages/.link-{args.name}",
            stdin=f'url = "{link}"\noutput = "{part}"\n',
        )
        out = _ssh(host, f"""set -e; cd {base}/packages
curl -sS --fail --retry 3 -K .link-{args.name}; rm -f .link-{args.name}
echo "{sha}  {args.name}.tar.part" | sha256sum -c --quiet
chmod 644 {args.name}.tar.part; mv {args.name}.tar.part {args.name}.tar; echo {content} > {args.name}.content; echo ok""")  # fmt: skip
        subprocess.run(["rclone", "deletefile", obj], check=True)
        if out.strip().splitlines()[-1:] != ["ok"]:
            raise SystemExit(f"package transfer failed on the server: {out[-300:]}")
        print(f"package {args.name} ({tarf.stat().st_size / 1e9:.2f} GB) on the server and checked "
              f"(content {content[:12]}), {time.time() - t0:.0f} s")  # fmt: skip
    return 0


def ship(args: argparse.Namespace) -> int:
    """Send the package's windows to the GPU server through R2, in checked batches."""
    host = args.host or os.environ.get("GPU_HOST")
    if not host:
        raise SystemExit("set GPU_HOST (or --host) to the server name in ~/.ssh/config")
    base = args.base or _ssh(host, "echo /raid/scratch/$(id -un)/wave-analysis").strip()
    if args.package:
        return _ship_package(host, base, args)
    dest = f"{base}/video_windows/{args.name}"
    win = pd.read_csv(data_dir("processed") / "packages" / args.name / "windows.csv")
    _ssh(host, f"mkdir -p {dest} && touch {dest}/VERIFIED")
    have = {
        line.split()[1] for line in _ssh(host, f"cat {dest}/VERIFIED").splitlines() if line.strip()
    }
    todo = win[
        [f"{c}/{i}.zip" not in have for c, i in zip(win["camera"], win["clip_id"], strict=True)]
    ]
    print(f"{len(win):,} clips with windows; {len(have):,} on the server; {len(todo):,} to send "
          f"({todo['zip_bytes'].sum() / 1e9:.1f} GB)", flush=True)  # fmt: skip
    t0, sent = time.time(), 0
    for s in range(0, len(todo), args.batch):
        b = todo.iloc[s : s + args.batch]
        rel = [f"{c}/{i}.zip" for c, i in zip(b["camera"], b["clip_id"], strict=True)]
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as lst:
            lst.write("\n".join(rel) + "\n")
            lst.flush()
            subprocess.run(["rclone", "copy", "--files-from", lst.name, "--bwlimit", args.bwlimit,
                            "--s3-no-check-bucket", str(WINDOWS), f"{R2_STAGING}/{args.name}"], check=True)  # fmt: skip
        links = [subprocess.run(["rclone", "link", "--expire", "12h", f"{R2_STAGING}/{args.name}/{r}"],
                                capture_output=True, text=True, check=True).stdout.strip() for r in rel]  # fmt: skip
        # curl reads the signed links from a config file (mode 600), never from its command line
        cfg = "".join(
            f'url = "{u}"\noutput = "{dest}/{r}.part"\n' for u, r in zip(links, rel, strict=True)
        )
        _ssh(host, f"umask 077 && cat > {dest}/.links", stdin=cfg)
        sums = "".join(f"{h}  {r}.part\n" for h, r in zip(b["zip_sha256"], rel, strict=True))
        out = _ssh(host, f"""set -e; cd {dest}; mkdir -p {' '.join(sorted({r.split('/')[0] for r in rel}))}
curl -sS --fail --retry 3 --create-dirs --parallel --parallel-max 4 -K .links; rm -f .links
cat > .sums; sha256sum -c --quiet .sums
while read -r h p; do mv "$p" "${{p%.part}}"; echo "$h  ${{p%.part}}" >> VERIFIED; done < .sums
rm -f .sums; echo ok""", stdin=sums)  # fmt: skip
        if out.strip().splitlines()[-1:] != ["ok"]:
            raise SystemExit(f"batch {s // args.batch} failed on the server: {out[-300:]}")
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as lst:
            lst.write("\n".join(rel) + "\n")
            lst.flush()
            subprocess.run(
                ["rclone", "delete", "--files-from", lst.name, f"{R2_STAGING}/{args.name}"],
                check=True,
            )
        sent += int(b["zip_bytes"].sum())
        print(f"  {s + len(b):,}/{len(todo):,} clips on the server and checked, {sent / 1e9:.1f} GB, "
              f"{sent * 8 / 1e6 / (time.time() - t0):.1f} Mbit/s", flush=True)  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pair")
    p.add_argument("cameras", nargs="*")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=pair)
    f = sub.add_parser("frames")
    f.add_argument("--workers", type=int, default=6)
    f.set_defaults(func=frames)
    w = sub.add_parser("windows")
    w.add_argument("--workers", type=int, default=4)
    w.add_argument("--limit", type=int, help="first N usable clips only (a test)")
    w.set_defaults(func=windows)
    h = sub.add_parser("physics")
    h.add_argument("--workers", type=int, default=4)
    h.set_defaults(func=physics)
    k = sub.add_parser("pack")
    k.add_argument("--name", default="webcoos_video_v1")
    k.set_defaults(func=pack)
    s = sub.add_parser("ship")
    s.add_argument("--name", default="webcoos_video_v1")
    s.add_argument("--host", help="the server in ~/.ssh/config (default: $GPU_HOST)")
    s.add_argument(
        "--base",
        help="the project's folder on the server (default: /raid/scratch/<user>/wave-analysis)",
    )
    s.add_argument("--batch", type=int, default=40, help="clips per batch")
    s.add_argument("--package", action="store_true",
                   help="send the package itself (one tar), as run.sh sync would, instead of the windows")  # fmt: skip
    s.add_argument(
        "--bwlimit",
        default="3M",
        help="upload cap (rclone --bwlimit; the home line does about 4.3M)",
    )
    s.set_defaults(func=ship)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
