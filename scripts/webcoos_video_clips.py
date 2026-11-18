#!/usr/bin/env python
"""WebCOOS video clips for wave models: pair each clip with its CDIP records, and cut a short frame sequence.

    python scripts/webcoos_video_clips.py pair     [cameras...]   # labels -> data/processed/video/clips.parquet
    python scripts/webcoos_video_clips.py frames   [--workers 6]  # 64 frames at 4 fps per clip, 256 px high JPEG
    python scripts/webcoos_video_clips.py pack                    # data/processed/packages/<name>/ (clips + frames + SHA256SUMS)

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

**Pack.** ``data/processed/packages/<name>/``: ``clips.csv`` (the usable clips that
have all their frames), ``frames/`` (hard links, no second copy) and ``SHA256SUMS``.
The package grows as clips download; ``deploy/dgx/run.sh sync`` sends only new files.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir, portable_path, repo_root
from wave_analysis.ingest.durable import atomic_open
from wave_analysis.processing.solar import solar_position
from wave_analysis.processing.synchronize import label_from_candidates, pair_window
from wave_analysis.sources.webcoos import ONPREM_POSITIONS, still_time

N_FRAMES, FPS, SPAN_S = 64, 4, 16
HEIGHT = 256  # px; width follows the camera's aspect (even)
OUT = data_dir("processed") / "video"
FRAMES = OUT / "frames_4fps"
# Burned-in text, blacked out in the source frame (ffmpeg drawbox; iw/ih = source size).
MASKS = {"currituck_hampton_inn": "drawbox=x=0:y=ih-170:w=560:h=170:color=black:t=fill"}


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
        rec = rec.reset_index(drop=True)
        c = c.reset_index(drop=True)
        tgt = c.rename(columns={"mid_utc": "time_utc"})[["time_utc"]]
        pw = pair_window(tgt, rec, window="60min")
        cand = pd.concat(
            [pd.Series(pw["target_index"].to_numpy(), name="clip"), pw.drop(columns=["target_index", "reference_index"]),
             rec.loc[pw["reference_index"], ["Hs_m", "Tp_s", "Tm02_s", "Dp_deg"]].reset_index(drop=True)],
            axis=1,
        )  # fmt: skip
        win = label_from_candidates(
            cand, "Hs_m", rule="window_mean", target="clip", max_offset="30min"
        ).reindex(c.index)
        c["Hs_label_m"] = win["label"]
        c["n_label_records"] = win["n_records"].fillna(0).astype(int)
        for col, out in (("Tp_s", "Tp_label_s"), ("Tm02_s", "Tm02_label_s")):
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
            "Tp_label_s", "Tm02_label_s", "Dp_label_deg", "exclusion"]  # fmt: skip
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
    sums = subprocess.run(
        "find clips.csv frames -type f -print0 | sort -z | xargs -0 sha256sum",
        shell=True, cwd=pkg, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    with atomic_open(pkg / "SHA256SUMS", "w") as fh:
        fh.write(sums)
    size = sum(f.stat().st_size for f in keep) / 1e9
    print(f"{pkg}: {len(t):,} clips ({t['camera'].value_counts().to_dict()}), "
          f"{t['local_day'].nunique()} days, {size:.2f} GB")  # fmt: skip
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
    k = sub.add_parser("pack")
    k.add_argument("--name", default="webcoos_video_v1")
    k.set_defaults(func=pack)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
