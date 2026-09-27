#!/usr/bin/env python
"""One pass over the Waimea Bay beach-camera archive: integrity and image features.

For every image in the PacIOOS indexes of ``beachcam_003`` and ``beachcam_004``
this records whether the archive holds it, recomputes its SHA-256 against the
ledger, checks that the JPEG is complete (EOI marker) and decodes without
error, and extracts what the timing checks need:

* the caption strip (grey, full resolution), which carries the capture time;
* a 1/16-scale grey thumbnail (camera-view changes, visual checks);
* per-block statistics on a 16 x 16 grid at 1/4 scale: mean luma and the
  fraction of "white" pixels (HSV value > 0.78 and saturation < 0.18, i.e.
  bright and unsaturated: foam, but also cloud and glare);
* the mean luma of the scene (caption excluded).

Nothing in the archive is modified. Outputs go to
``data/interim/waimea_audit/`` (``images.parquet`` and ``features.npz``).

Usage::

    uv run python scripts/waimea_extract.py [--workers 16]
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

from wave_analysis.config import data_dir
from wave_analysis.ingest.manifest import read_manifest
from wave_analysis.sources.pacioos import WAIMEA, ledger_path, local_path, parse_index

#: Caption strip: rows 1238-1252 hold the 10-px text; 41 cells of 7 px from x = 7.
CAPTION_ROWS = (1238, 1253)
CAPTION_COLS = (0, 300)
SCENE_ROWS = 1232  # rows above the caption band
GRID = 16
THUMB_FACTOR = 16


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
    rgb = im.convert("RGB")
    grey = np.asarray(rgb.convert("L"))
    if grey.shape != (1254, 1624):
        out["decode_error"] = f"unexpected size {grey.shape}"
        return out
    out["caption"] = grey[slice(*CAPTION_ROWS), slice(*CAPTION_COLS)].copy()
    scene = rgb.crop((0, 0, 1624, SCENE_ROWS))
    out["thumb"] = np.asarray(scene.convert("L").reduce(THUMB_FACTOR))
    small = scene.reduce(4)  # 406 x 308
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
    out["mean_luma"] = float(luma.mean())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, help="first N images per camera (testing)")
    args = ap.parse_args(argv)

    root = data_dir("raw") / "pacioos" / "beachcam"
    out_dir = data_dir("interim") / "waimea_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cam in WAIMEA:
        snaps = sorted((root / "_index").glob(f"{cam}_*.csv"))
        index = parse_index(snaps[-1], cam)
        ledger = {e.url: e for e in read_manifest(ledger_path(root, cam))}
        if args.limit:
            index = index.head(args.limit)
        for r in index.itertuples(index=False):
            e = ledger.get(r.url)
            rows.append(
                {
                    "camera": cam,
                    "name": r.name,
                    "url": r.url,
                    "time_utc": r.time_utc,
                    "index_size": r.size_bytes,
                    "ledger_status": e.status.value if e else None,
                    "ledger_sha256": e.sha256 if e else None,
                    "ledger_size": e.size_bytes if e else None,
                    "path": local_path(root, r.url),
                }
            )
    table = pd.DataFrame(rows)
    print(f"{len(table):,} indexed images", flush=True)

    keys = ("caption", "thumb", "block_white", "block_luma")
    arrays: dict[str, list[Any]] = {k: [] for k in keys}
    rec = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, f in enumerate(pool.map(features, table["path"], chunksize=32)):
            for k in keys:
                arrays[k].append(f.pop(k, None))
            rec.append(f)
            if (i + 1) % 2000 == 0:
                print(f"{i + 1:,} done", flush=True)
    table = pd.concat([table.drop(columns="path"), pd.DataFrame(rec)], axis=1)
    ok = (
        table["sha256"].notna() & table["decode_error"].isna()
        if "decode_error" in table
        else table["sha256"].notna()
    )
    table["features_ok"] = ok
    shapes = {
        "caption": (15, 300),
        "thumb": (77, 102),
        "block_white": (GRID, GRID),
        "block_luma": (GRID, GRID),
    }
    packed = {}
    for k in keys:
        fill = np.zeros(shapes[k], np.uint8 if k in ("caption", "thumb") else np.float32)
        packed[k] = np.stack([a if a is not None else fill for a in arrays[k]])
    np.savez_compressed(out_dir / "features.npz", **packed)
    table.to_parquet(out_dir / "images.parquet", index=False)
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
