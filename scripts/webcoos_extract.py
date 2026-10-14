#!/usr/bin/env python
"""One pass over a WebCOOS camera's archived stills: integrity and image features.

For every still of the camera that the archive holds, or that a ledger says
was downloaded, this records the ledger rows it has, recomputes its SHA-256
against them, checks that the JPEG is complete (EOI marker) and decodes without
error, and extracts what the checks of the capture time and of the view need:

* the mean luma and the mean colour saturation of the frame. At night the
  cameras switch to a monochrome mode, so saturation separates night frames
  from day frames far better than brightness does;
* per-block statistics on a 16 x 16 grid at 1/4 scale: mean luma and the
  fraction of "white" pixels (HSV value > 0.78 and saturation < 0.18, i.e.
  bright and unsaturated: foam, but also cloud, glare and a wet lens);
* a 96 x 54 grey thumbnail (changes of the view, visual checks).

The stills carry no time stamp in the picture (only a fixed site label in a
corner) and no EXIF: the capture time is the one in the file name.

Nothing in the archive is modified. Outputs go to
``data/interim/webcoos_audit/<camera>/`` (``images.parquet`` and
``features.npz``, rows in the same order).

Usage::

    uv run python scripts/webcoos_extract.py currituck_hampton_inn [--workers 16]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from wave_analysis.config import data_dir, portable_path
from wave_analysis.sources.webcoos import MANIFESTS_DIR, local_path, still_time

GRID = 16
THUMB = (96, 54)  # width, height


def features(path: Path) -> dict[str, Any]:
    """Integrity checks and features for one file."""
    out: dict[str, Any] = {"exists": path.exists()}
    if not out["exists"]:
        return out
    raw = path.read_bytes()
    out["size_bytes"] = len(raw)
    out["sha256"] = hashlib.sha256(raw).hexdigest()
    out["eoi"] = raw.rstrip(b"\x00")[-2:] == b"\xff\xd9"
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception as exc:  # truncated or corrupt
        out["decode_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return out
    out["width"], out["height"], out["mode"] = im.width, im.height, im.mode
    out["has_exif"] = bool(im.getexif())
    small = im.convert("RGB").reduce(4)
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32) / 255.0
    luma = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
    white = (hsv[..., 2] > 0.78) & (hsv[..., 1] < 0.18)
    h, w = luma.shape
    ys = np.linspace(0, h, GRID + 1).astype(int)
    xs = np.linspace(0, w, GRID + 1).astype(int)
    bw = np.empty((GRID, GRID), np.float32)
    bl = np.empty((GRID, GRID), np.float32)
    for i in range(GRID):
        for j in range(GRID):
            bw[i, j] = white[ys[i] : ys[i + 1], xs[j] : xs[j + 1]].mean()
            bl[i, j] = luma[ys[i] : ys[i + 1], xs[j] : xs[j + 1]].mean()
    out["block_white"], out["block_luma"] = bw, bl
    out["thumb"] = np.asarray(small.convert("L").resize(THUMB, Image.Resampling.BOX))
    out["mean_luma"] = float(luma.mean())
    out["mean_saturation"] = float(hsv[..., 1].mean())
    return out


def ledger_rows(root: Path, camera: str) -> pd.DataFrame:
    """One row per file name that a ledger of ``camera`` mentions.

    The hourly collector writes ``<camera>.csv`` and a historical download
    ``<camera>.<host>.csv``; both name a still as WebCOOS does. A file that was
    downloaded keeps its latest checksum row; one that never was keeps its
    latest status (``failed`` or ``not_found``).
    """
    cols = ["url", "sha256", "size_bytes", "retrieved_at", "status"]
    parts = []
    for path in sorted((root / MANIFESTS_DIR).glob(f"{camera}*.csv")):
        if path.name.split(".")[0] != camera:
            continue  # another camera whose name starts the same
        df = pd.read_csv(path, usecols=["product", *cols], dtype=str)
        df = df[df["product"] == "image"].drop(columns="product")
        parts.append(df.assign(ledger=path.name))
    if not parts:
        return pd.DataFrame(columns=["name", *cols, "ledger", "ledgers", "ledger_sha256_differ"])
    rows = pd.concat(parts, ignore_index=True)
    rows["name"] = rows["url"].str.rsplit("/", n=1).str[-1]
    rows["downloaded"] = rows["sha256"].notna()
    rows = rows.sort_values(["name", "downloaded", "retrieved_at"], kind="stable")
    got = rows[rows["downloaded"]].groupby("name")
    extra = pd.DataFrame(
        {
            "ledgers": got["ledger"].agg(lambda s: ",".join(dict.fromkeys(s))),
            "ledger_sha256_differ": got["sha256"].agg(lambda s: len(set(s)) > 1),
        }
    )
    last = rows.groupby("name").tail(1).set_index("name").drop(columns="downloaded")
    return last.join(extra).reset_index()


def camera_table(root: Path, camera: str) -> pd.DataFrame:
    """Every still of ``camera`` on disk or in a ledger, with its path."""
    on_disk = {p.name: p for p in sorted((root / camera).glob("[0-9]*/[0-9]*/[0-9]*/*.jpg"))}
    led = ledger_rows(root, camera).set_index("name")
    names = sorted(set(on_disk) | set(led.index))
    rows = []
    for name in names:
        t = still_time(name, camera)
        if t is None:
            continue  # not a still of this camera
        e = led.loc[name] if name in led.index else None
        path = on_disk.get(name) or local_path(root, camera, name, t)
        differ = (
            e is not None
            and pd.notna(e["ledger_sha256_differ"])
            and bool(e["ledger_sha256_differ"])
        )
        rows.append(
            {
                "camera": camera,
                "name": name,
                "time_utc": t,
                "local_path": portable_path(path),
                "url": None if e is None else e["url"],
                "ledger_status": None if e is None else e["status"],
                "ledger_sha256": None if e is None else e["sha256"],
                "ledger_size": None
                if e is None or pd.isna(e["size_bytes"])
                else int(e["size_bytes"]),
                "ledgers": None if e is None else e["ledgers"],
                "ledger_sha256_differ": differ,
                "path": path,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cameras", nargs="+", help="camera slugs, e.g. currituck_hampton_inn")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, help="first N images per camera (testing)")
    args = ap.parse_args(argv)

    root = data_dir("raw") / "webcoos"
    for cam in args.cameras:
        out_dir = data_dir("interim") / "webcoos_audit" / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        table = camera_table(root, cam)
        if args.limit:
            table = table.head(args.limit)
        print(f"{cam}: {len(table):,} stills on disk or in a ledger", flush=True)

        keys = ("thumb", "block_white", "block_luma")
        arrays: dict[str, list[Any]] = {k: [] for k in keys}
        rec = []
        with ProcessPoolExecutor(args.workers) as pool:
            for i, f in enumerate(pool.map(features, table["path"], chunksize=32)):
                for k in keys:
                    arrays[k].append(f.pop(k, None))
                rec.append(f)
                if (i + 1) % 5000 == 0:
                    print(f"{i + 1:,} done", flush=True)
        feats = pd.DataFrame(rec, index=table.index)
        for col in ("sha256", "decode_error", "mean_luma", "mean_saturation"):
            if col not in feats:
                feats[col] = None
        table = pd.concat([table.drop(columns="path"), feats], axis=1)
        table["sha256_matches_ledger"] = (table["sha256"] == table["ledger_sha256"]).where(
            table["sha256"].notna() & table["ledger_sha256"].notna()
        )
        table["features_ok"] = table["mean_luma"].notna() & table["decode_error"].isna()
        shapes = {"thumb": THUMB[::-1], "block_white": (GRID, GRID), "block_luma": (GRID, GRID)}
        packed = {}
        for k in keys:
            fill = np.zeros(shapes[k], np.uint8 if k == "thumb" else np.float32)
            packed[k] = np.stack([a if a is not None else fill for a in arrays[k]])
        np.savez_compressed(out_dir / "features.npz", **packed)
        table.to_parquet(out_dir / "images.parquet", index=False)
        print(
            f"{cam}: {int(table['exists'].sum()):,} files, "
            f"{int(table['features_ok'].sum()):,} decoded; wrote {out_dir}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
