#!/usr/bin/env python
"""Test 9 package: still / time-exposure / brightest-pixel triples of the four CDIP-paired
WebCOOS cameras, resized for the frozen DINOv3 probes (desktop, CPU).

Design: ``docs/experiments/sensing_timex_bars.md``; config
``configs/experiments/sensing_timex_v1.yaml``. A triple is a ``timex`` and a ``brt`` image
with the same stamp and the camera's still whose time lies inside their ten minutes (the
one nearest the middle), with a complete CDIP pairing and the sun above the horizon. The
label is the still's ADR 0010 H_s. The split is by date only (a held-out year).

Writes ``data/processed/packages/sensing_timex_v1/``: ``triples.csv``,
``images/<camera>/<still|timex|brt>/<stamp>.jpg`` (256 px high, masks first) and
``SHA256SUMS``. Ship it with ``scripts/webcoos_video_clips.py ship --package --name
sensing_timex_v1`` (R2), then ``deploy/dgx/run.sh push`` finds it on the server.

    python scripts/sensing_timex_package.py [--workers 4] [--limit 200]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from wave_analysis.config import data_dir, load_yaml, repo_root

KINDS = ("still", "timex", "brt")
RAW = {"still": "webcoos", "timex": "webcoos_timex", "brt": "webcoos_brt"}
PREFIX = {"still": "", "timex": "timex.", "brt": "brt."}


def scan(kind: str, camera: str) -> dict[str, str]:
    """``stamp -> path relative to data/raw`` for one product of one camera (file names only)."""
    root = data_dir("raw") / RAW[kind] / camera
    pat = re.compile(
        rf"^{re.escape(PREFIX[kind])}{camera}-(\d{{4}}-\d{{2}}-\d{{2}}-\d{{6}})Z\.jpg$"
    )
    out = {}
    for dp, _, files in os.walk(root):
        for f in files:
            m = pat.match(f)
            if m:
                out[m[1]] = str(Path(dp, f).relative_to(data_dir("raw")))
    return out


def triples(cfg: dict) -> tuple[pd.DataFrame, dict]:
    rows, counts = [], {}
    win = pd.Timedelta(minutes=int(cfg["data"]["window_min"]))
    for cam, station in cfg["cameras"].items():
        t, b = scan("timex", cam), scan("brt", cam)
        stamps = sorted(set(t) & set(b))
        w = pd.DataFrame({"stamp": stamps})
        w["start_utc"] = pd.to_datetime(w["stamp"], format="%Y-%m-%d-%H%M%S", utc=True)
        w["centre"] = w["start_utc"] + win / 2
        pairs = data_dir("processed") / "pairs" / f"webcoos_{cam}_cdip{station}" / "samples.parquet"
        s = pd.read_parquet(pairs, columns=["local_path", "time_utc", "exclusion", "sun_elevation_deg",
                                            "Hs_label_m", "Tp_label_s", "n_label_records"])  # fmt: skip
        s["still_utc"] = pd.to_datetime(s["time_utc"], utc=True)
        s = s.drop(columns="time_utc").sort_values("still_utc")
        m = pd.merge_asof(
            w.sort_values("centre"), s, left_on="centre", right_on="still_utc", direction="nearest"
        )
        inside = (m["still_utc"] >= m["start_utc"]) & (m["still_utc"] < m["start_utc"] + win)
        paired = inside & m["exclusion"].fillna("").eq("")
        day = paired & (m["sun_elevation_deg"] >= float(cfg["data"]["min_sun_elevation_deg"]))
        counts[cam] = {"timex_brt_stamps": len(w), "still_inside": int(inside.sum()),
                       "paired": int(paired.sum()), "daylight": int(day.sum())}  # fmt: skip
        k = m[day].copy()
        k["camera"], k["cdip_station"] = cam, station
        k["timex_path"] = [t[x] for x in k["stamp"]]
        k["brt_path"] = [b[x] for x in k["stamp"]]
        k["still_path"] = [str(Path(p).relative_to("data/raw")) if str(p).startswith("data/raw") else p
                           for p in k["local_path"]]  # fmt: skip
        rows.append(k)
    d = pd.concat(rows, ignore_index=True)
    d["local_day"] = d["still_utc"].dt.tz_convert("America/New_York").dt.date.astype(str)
    return d, counts


def assign_split(d: pd.DataFrame, cfg: dict) -> pd.Series:
    """By date only: train / val (whole ISO weeks of the training year) / test / later."""
    sp = cfg["split"]
    day = pd.to_datetime(d["local_day"])
    year = day.dt.year
    iso = day.dt.isocalendar()
    week = (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)).to_numpy()
    tr_weeks = np.array(sorted(set(week[year.to_numpy() == int(sp["train_year"])])))
    rng = np.random.default_rng(int(sp["seed"]))
    n_val = round(float(sp["val_week_share"]) * len(tr_weeks))
    val_weeks = set(rng.choice(tr_weeks, size=n_val, replace=False))
    split = np.full(len(d), "", dtype=object)
    tr = year.to_numpy() == int(sp["train_year"])
    split[tr] = np.where(np.isin(week[tr], list(val_weeks)), "val", "train")
    te = (year == int(sp["test_year"])) & (day.dt.dayofyear > int(sp["embargo_days"]))
    split[te.to_numpy()] = "test"
    split[(year == int(sp["later_year"])).to_numpy()] = "later"
    return pd.Series(split, index=d.index)


def _resize(job: tuple[str, str, str, int, int, list | None]) -> tuple[str, str | None, str]:
    """Mask in source pixels, resize to ``height`` (aspect kept, even width), JPEG.

    Returns ``(dest, sha256, "")``, or ``(dest, None, error)`` when the source does not decode
    fully (a truncated or damaged file: a data-quality exclusion, made before any label is
    read). An output already written by an earlier run is kept (resume)."""
    src, dest, _, height, quality, mask = job
    if Path(dest).exists():  # written whole (renamed from .part) by an earlier run
        return dest, hashlib.sha256(Path(dest).read_bytes()).hexdigest(), ""
    try:
        im = Image.open(src)
        w0, h0 = im.size
        im.draft("RGB", (int(np.ceil(height * w0 / h0)), height))  # fast decode at a reduced scale
        im.load()  # decode all of it now: a truncated file fails here
        im = im.convert("RGB")
    except (OSError, SyntaxError, ValueError) as e:
        return dest, None, f"{type(e).__name__}: {e}"
    if mask:
        x, from_bottom, mw, mh = mask
        r = im.size[1] / h0
        ImageDraw.Draw(im).rectangle(
            [
                x * r,
                im.size[1] - from_bottom * r,
                (x + mw) * r,
                im.size[1] - (from_bottom - mh) * r,
            ],
            fill=(0, 0, 0),
        )
    w = int(round(height * w0 / h0 / 2) * 2)
    im = im.resize((w, height), Image.BICUBIC)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    tmp = dest + ".part"
    im.save(tmp, "JPEG", quality=quality)
    os.replace(tmp, dest)
    return dest, hashlib.sha256(Path(dest).read_bytes()).hexdigest(), ""


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config", type=Path, default=repo_root() / "configs/experiments/sensing_timex_v1.yaml"
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="first N triples per camera (smoke test)")
    a = p.parse_args()
    cfg = load_yaml(a.config)
    out = data_dir("processed") / "packages" / cfg["name"]
    d, counts = triples(cfg)
    if a.limit:
        d = d.groupby("camera", group_keys=False).head(a.limit).reset_index(drop=True)
    d["split"] = assign_split(d, cfg)
    d = d[d["split"] != ""].sort_values(["camera", "stamp"]).reset_index(drop=True)
    print("triples", counts)
    print(d.groupby(["split", "camera"]).size().unstack(fill_value=0).to_string())
    raw = data_dir("raw")
    jobs = []
    for r in d.itertuples():
        for k in KINDS:
            dest = out / "images" / r.camera / k / f"{r.stamp}.jpg"
            jobs.append((str(raw / getattr(r, f"{k}_path")), str(dest), k, int(cfg["data"]["height_px"]),
                         int(cfg["data"]["jpeg_quality"]), cfg["data"]["masks"].get(r.camera)))  # fmt: skip
    sums, failed = {}, []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, ((dest, sha, err), job) in enumerate(
            zip(ex.map(_resize, jobs, chunksize=64), jobs, strict=True)
        ):
            if sha is None:
                failed.append((job, err))
            else:
                sums[str(Path(dest).relative_to(out))] = sha
            if i % 10000 == 0:
                print(f"{i:,}/{len(jobs):,} images, {len(failed)} unreadable", flush=True)
    # Unreadable sources: their whole triple leaves the package and is listed (path, size, SHA-256)
    skipped = []
    for (src, dest, kind, *_), err in failed:
        f = Path(src)
        skipped.append({"camera": Path(dest).parent.parent.name, "stamp": Path(dest).stem, "kind": kind,
                        "raw_path": str(f.relative_to(raw)), "size_bytes": f.stat().st_size if f.exists() else None,
                        "sha256": hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None,
                        "error": err})  # fmt: skip
    share = len(failed) / max(1, len(jobs))
    print(f"unreadable source images: {len(failed)} of {len(jobs):,} ({share:.3%})")
    if share > 0.01:
        raise SystemExit(
            "more than 1 % of the source images do not decode: a collector problem? stopping"
        )
    bad = {(r["camera"], r["stamp"]) for r in skipped}
    keep = np.array([(c, st) not in bad for c, st in zip(d["camera"], d["stamp"], strict=True)])
    for c, st in bad:
        for k in KINDS:
            rel = f"images/{c}/{k}/{st}.jpg"
            sums.pop(rel, None)
            (out / rel).unlink(missing_ok=True)
    d = d[keep].reset_index(drop=True)
    pd.DataFrame(skipped, columns=["camera", "stamp", "kind", "raw_path", "size_bytes", "sha256", "error"]).to_csv(
        out / "skipped_images.csv", index=False)  # fmt: skip
    sums["skipped_images.csv"] = hashlib.sha256(
        (out / "skipped_images.csv").read_bytes()
    ).hexdigest()
    cols = ["camera", "cdip_station", "stamp", "start_utc", "still_utc", "local_day", "split",
            "Hs_label_m", "Tp_label_s", "n_label_records", "sun_elevation_deg",
            "still_path", "timex_path", "brt_path"]  # fmt: skip
    d[cols].to_csv(out / "triples.csv", index=False)
    sums["triples.csv"] = hashlib.sha256((out / "triples.csv").read_bytes()).hexdigest()
    (out / "SHA256SUMS").write_text("".join(f"{s}  {f}\n" for f, s in sorted(sums.items())))
    print(
        f"{len(d):,} triples ({len(bad)} dropped for unreadable images), {3 * len(d):,} images in {out}"
    )


if __name__ == "__main__":
    main()
