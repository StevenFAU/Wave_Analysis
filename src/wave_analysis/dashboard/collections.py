"""Totals of the data collections held on the collector host, for the dashboard.

One entry per collection: how many items (images, files, months) are held,
their size, the period of data they cover, when the collection last changed,
and a check of the files on disk against the per-request ledger. Built from
the ledgers and a directory scan; nothing is fetched and no image is read.

The output is published, so it holds counts, dates and upstream identifiers
only: no local paths, host names or user names, and no images (the WebCOOS
usage guidelines do not address redistribution).

Collections (each skipped when its directory or ledger is absent):

=====================  ==============================================  ==========
id                     Ledger                                          Items
=====================  ==============================================  ==========
``pacioos_waimea``     ``raw/pacioos/beachcam/_manifests/*.csv``        images
``webcoos``            ``raw/webcoos/_manifests/*.csv``                 images
``era5_waves``         ``manifests/raw/era5.csv``                       months
``cdip``               ``manifests/raw/cdip.csv``                       files
``ndbc_history``       ``manifests/raw/ndbc.csv``                       files
``external``           ``manifests/raw/external.csv``                   files
=====================  ==============================================  ==========

The NDBC buoy-camera archive and the realtime-spectra snapshots are summarised
by :mod:`wave_analysis.dashboard.status`, which adds them to this list.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from wave_analysis.sources.era5 import raw_path

_ARCHIVED = {"verified", "unchanged", "changed"}
_ERA5_STATE = re.compile(r";\s*(\w+) \(was \w+\)\s*$")
_LEDGER_COLS = ("product", "station_id", "period", "url", "size_bytes", "status", "retrieved_at")
#: NDBC yearly archive file, e.g. ``41008h2023.txt.gz`` or ``41013jb2005.txt.gz``
#: (``b``: second grid segment); realtime snapshots carry a time stamp instead.
_NDBC_HISTORY_NAME = re.compile(r"^[A-Za-z0-9]+?[hwdijka]b?(\d{4})\.txt\.gz$")


@dataclass(frozen=True)
class CollectionInputs:
    """Where the collections live (``data/raw`` and ``data/manifests``)."""

    raw_root: Path
    manifests_root: Path
    external_root: Path | None = None  # data/external (default: next to raw_root)

    @property
    def external(self) -> Path:
        """Root of the published research datasets."""
        return self.external_root or self.raw_root.parent / "external"


def _iso(ts: Any) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def read_ledgers(paths: Iterable[Path], extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Ledger rows from CSV files, skipping a partially written last line."""
    cols = (*_LEDGER_COLS, "note", *extra)
    rows: list[dict[str, str]] = []
    for p in sorted(paths):
        with p.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if None in row or any(row.get(c) is None for c in _LEDGER_COLS):
                    continue  # truncated line (file being appended to)
                rows.append({c: row.get(c) or "" for c in cols})
    df = pd.DataFrame(rows, columns=list(cols))
    df["retrieved_at"] = pd.to_datetime(
        df["retrieved_at"], utc=True, errors="coerce", format="ISO8601"
    )
    df["size_bytes"] = pd.to_numeric(df["size_bytes"], errors="coerce")
    return df.dropna(subset=["retrieved_at"])


def scan_files(root: Path, suffix: str) -> pd.DataFrame:
    """Files under ``root`` with ``suffix``: top-level directory, name, size.

    Directories starting with ``_`` or ``.`` (listings, ledgers, indexes) are
    skipped.
    """
    rows: list[tuple[str, str, int]] = []

    def walk(d: str, top: str) -> None:
        with os.scandir(d) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    walk(e.path, top)
                elif e.name.endswith(suffix):
                    rows.append((top, e.name, e.stat().st_size))

    if root.exists():
        for top in os.scandir(root):
            if top.is_dir() and not top.name.startswith(("_", ".")):
                walk(top.path, top.name)
    return pd.DataFrame(rows, columns=["part", "name", "size_bytes"])


def disk_check(ledgered: set[tuple[str, str]], on_disk: set[tuple[str, str]]) -> dict[str, int]:
    """Agreement of verified ledger rows and files, both as ``(part, file name)``."""
    return {
        "on_disk": len(on_disk),
        "ledger": len(ledgered),
        "missing_files": len(ledgered - on_disk),
        "unledgered_files": len(on_disk - ledgered),
    }


def _latest(ledger: pd.DataFrame) -> pd.DataFrame:
    """The last row per URL (a URL may be retried, re-fetched or found later)."""
    return ledger.sort_values("retrieved_at").drop_duplicates("url", keep="last")


def _image_collection(ledger: pd.DataFrame, files: pd.DataFrame) -> dict[str, Any]:
    """Counts, period and ledger/disk agreement for an image archive."""
    img = _latest(ledger[ledger["product"] == "image"])
    img = img.assign(name=img["url"].str.rsplit("/", n=1).str[-1])
    held = img[img["status"].isin(_ARCHIVED)]
    t = pd.to_datetime(held["period"], utc=True, errors="coerce", format="ISO8601")
    on_disk = set(zip(files["part"], files["name"], strict=True))
    ledgered = set(zip(held["station_id"], held["name"], strict=True))
    parts = []
    for pid, g in held.assign(t=t).groupby("station_id"):
        f = files[files["part"] == pid]
        parts.append(
            {
                "id": str(pid),
                "count": len(f),
                "bytes": int(f["size_bytes"].sum()),
                "first": _iso(g["t"].min()),
                "last": _iso(g["t"].max()),
                "not_found": int(
                    ((img["station_id"] == pid) & (img["status"] == "not_found")).sum()
                ),
            }
        )
    return {
        "unit": "images",
        "count": len(files),
        "bytes": int(files["size_bytes"].sum()),
        "first": _iso(t.min()),
        "last": _iso(t.max()),
        "updated": _iso(ledger["retrieved_at"].max()),
        "not_found": int((img["status"] == "not_found").sum()),
        "failed": int((img["status"] == "failed").sum()),
        "check": disk_check(ledgered, on_disk),
        "parts": parts,
    }


def pacioos_waimea(inputs: CollectionInputs) -> dict[str, Any] | None:
    """The Waimea Bay beach-camera archive (complete), with its pairing summary."""
    root = inputs.raw_root / "pacioos" / "beachcam"
    ledgers = sorted((root / "_manifests").glob("*.csv"))
    if not ledgers:
        return None
    out: dict[str, Any] = {
        "id": "pacioos_waimea",
        "dataset_id": "pacioos_beachcam",
        "name": "PacIOOS beach cameras, Waimea Bay",
        "provider": "PacIOOS",
        "mode": "complete",
        **_image_collection(read_ledgers(ledgers), scan_files(root, ".jpg")),
    }
    summary = inputs.manifests_root / "processed" / "waimea_cdip106.summary.json"
    if summary.exists():
        s = json.loads(summary.read_text(encoding="utf-8"))
        exc = s["images"]["exclusion"]
        out["pairing"] = {
            "reference": "CDIP 106",
            "paired": int(exc.get("paired", 0)),
            "paired_days": int(s["images"]["paired_days"]),
            "trusted_time": int(exc.get("paired", 0)) + int(exc.get("no_buoy_within_30min", 0)),
            "exclusion": {k: int(v) for k, v in exc.items() if k != "paired"},
        }
    return out


def webcoos(inputs: CollectionInputs) -> dict[str, Any] | None:
    """WebCOOS one-minute stills thinned to a time grid (hourly collector)."""
    root = inputs.raw_root / "webcoos"
    ledgers = sorted((root / "_manifests").glob("*.csv"))
    if not ledgers:
        return None
    return {
        "id": "webcoos",
        "dataset_id": "webcoos",
        "name": "WebCOOS coastal cameras",
        "provider": "WebCOOS / SECOORA",
        "mode": "hourly",
        **_image_collection(read_ledgers(ledgers), scan_files(root, ".jpg")),
    }


def era5_waves(inputs: CollectionInputs) -> dict[str, Any] | None:
    """ERA5 wave parameters: one file per site and month, final or preliminary."""
    ledger_path = inputs.manifests_root / "raw" / "era5.csv"
    if not ledger_path.exists():
        return None
    ledger = read_ledgers([ledger_path])
    held = ledger[ledger["status"].isin(_ARCHIVED)].sort_values("retrieved_at")
    held = held.drop_duplicates(["station_id", "period"], keep="last")
    held = held.assign(state=held["note"].str.extract(_ERA5_STATE)[0].fillna("unknown"))
    files = scan_files(inputs.raw_root / "era5", ".nc")
    names = [
        raw_path(Path(), s, pd.Period(p, "M")).name
        for s, p in zip(held["station_id"], held["period"], strict=True)
    ]
    on_disk = set(zip(files["part"], files["name"], strict=True))
    ledgered = set(zip(held["station_id"], names, strict=True))
    parts = []
    for site, g in held.groupby("station_id"):
        f = files[files["part"] == site]
        parts.append(
            {
                "id": str(site),
                "count": len(g),
                "bytes": int(f["size_bytes"].sum()),
                "first": str(g["period"].min()),
                "last": str(g["period"].max()),
                "states": g["state"].value_counts().sort_index().to_dict(),
            }
        )
    return {
        "id": "era5_waves",
        "dataset_id": "era5_waves",
        "name": "ERA5 wave parameters at camera sites",
        "provider": "ECMWF / Copernicus CDS",
        "mode": "on_request",
        "unit": "site-months",
        "count": len(held),
        "bytes": int(files["size_bytes"].sum()),
        "first": str(held["period"].min()) if len(held) else None,
        "last": str(held["period"].max()) if len(held) else None,
        "updated": _iso(ledger["retrieved_at"].max()),
        "failed": int(
            (
                ledger.sort_values("retrieved_at").drop_duplicates(
                    ["station_id", "period"], keep="last"
                )["status"]
                == "failed"
            ).sum()
        ),
        "states": held["state"].value_counts().sort_index().to_dict(),
        "check": disk_check(ledgered, on_disk),
        "parts": parts,
    }


def cdip(inputs: CollectionInputs) -> dict[str, Any] | None:
    """CDIP buoy files (whole netCDF files as served), with each file's time coverage."""
    ledger_path = inputs.manifests_root / "raw" / "cdip.csv"
    if not ledger_path.exists():
        return None
    ledger = read_ledgers([ledger_path])
    held = _latest(ledger)
    held = held[held["status"].isin(_ARCHIVED)]
    held = held.assign(name=held["url"].str.rsplit("/", n=1).str[-1])
    files = scan_files(inputs.raw_root / "cdip", ".nc")
    on_disk = set(zip(files["part"], files["name"], strict=True))
    ledgered = set(zip(held["station_id"], held["name"], strict=True))
    parts = []
    for stn, g in held.groupby("station_id"):
        cover = [_nc_coverage(inputs.raw_root / "cdip" / str(stn) / n) for n in g["name"]]
        starts = [c[0] for c in cover if c[0]]
        ends = [c[1] for c in cover if c[1]]
        parts.append(
            {
                "id": str(stn),
                "count": len(g),
                "bytes": int(files.loc[files["part"] == stn, "size_bytes"].sum()),
                "first": min(starts) if starts else None,
                "last": max(ends) if ends else None,
                "products": sorted(g["product"].unique().tolist()),
            }
        )
    firsts = [str(p["first"]) for p in parts if p["first"]]
    lasts = [str(p["last"]) for p in parts if p["last"]]
    return {
        "id": "cdip",
        "dataset_id": "cdip",
        "name": "CDIP Waverider buoys",
        "provider": "CDIP (Scripps)",
        "mode": "on_request",
        "unit": "files",
        "count": len(held),
        "bytes": int(files["size_bytes"].sum()),
        "first": min(firsts) if firsts else None,
        "last": max(lasts) if lasts else None,
        "updated": _iso(ledger["retrieved_at"].max()),
        "check": disk_check(ledgered, on_disk),
        "parts": parts,
    }


def ndbc_history(inputs: CollectionInputs) -> dict[str, Any] | None:
    """NDBC yearly archive files (``data/historical/<product>/``), by product."""
    ledger_path = inputs.manifests_root / "raw" / "ndbc.csv"
    if not ledger_path.exists():
        return None
    ledger = read_ledgers([ledger_path])
    held = _latest(ledger)
    held = held[held["status"].isin(_ARCHIVED)]
    held = held.assign(name=held["url"].str.rsplit("/", n=1).str[-1])
    files: list[tuple[str, str, str, int]] = []  # station, product, name, size
    root = inputs.raw_root / "ndbc"
    if root.exists():
        for sdir in os.scandir(root):
            if not sdir.is_dir() or sdir.name.startswith(("_", ".")) or sdir.name == "buoycam":
                continue
            for pdir in os.scandir(sdir.path):
                if not pdir.is_dir():
                    continue
                for entry in os.scandir(pdir.path):
                    if _NDBC_HISTORY_NAME.match(entry.name):
                        files.append((sdir.name, pdir.name, entry.name, entry.stat().st_size))
    df = pd.DataFrame(files, columns=["station_id", "product", "name", "size_bytes"])
    on_disk = set(zip(df["station_id"], df["name"], strict=True))
    ledgered = set(zip(held["station_id"], held["name"], strict=True))
    parts = []
    for product, g in held.groupby("product"):
        f = df[df["product"] == product]
        parts.append(
            {
                "id": str(product),
                "count": len(g),
                "bytes": int(f["size_bytes"].sum()),
                "first": str(g["period"].min()),
                "last": str(g["period"].max()),
                "stations": int(g["station_id"].nunique()),
            }
        )
    return {
        "id": "ndbc_history",
        "dataset_id": "ndbc",
        "name": "NDBC historical files",
        "provider": "NOAA NDBC",
        "mode": "on_request",
        "unit": "files",
        "count": len(held),
        "bytes": int(df["size_bytes"].sum()),
        "first": str(held["period"].min()) if len(held) else None,
        "last": str(held["period"].max()) if len(held) else None,
        "updated": _iso(ledger["retrieved_at"].max()),
        "check": disk_check(ledgered, on_disk),
        "parts_count": int(held["station_id"].nunique()),
        "parts": parts,
    }


def external(inputs: CollectionInputs) -> dict[str, Any] | None:
    """Published research datasets (``data/external/<source id>/``), by dataset.

    The ``SOURCE.yaml`` copy of each config entry is written by the downloader
    and is not a downloaded file, so it is not counted.
    """
    ledger_path = inputs.manifests_root / "raw" / "external.csv"
    if not ledger_path.exists():
        return None
    ledger = read_ledgers([ledger_path], extra=("source_id",))
    held = ledger.sort_values("retrieved_at").drop_duplicates(["source_id", "product"], keep="last")
    held = held[held["status"].isin(_ARCHIVED)]
    files = scan_files(inputs.external, "")
    files = files[files["name"] != "SOURCE.yaml"]
    on_disk = set(zip(files["part"], files["name"], strict=True))
    ledgered = set(zip(held["source_id"], held["product"], strict=True))
    parts = [
        {
            "id": str(sid),
            "count": len(g),
            "bytes": int(files.loc[files["part"] == sid, "size_bytes"].sum()),
            "first": None,
            "last": None,
        }
        for sid, g in held.groupby("source_id")
    ]
    return {
        "id": "external",
        "dataset_id": None,  # several registry entries: see parts
        "name": "Published research datasets",
        "provider": "Dataset authors (UW ResearchWorks, AADC)",
        "mode": "complete",
        "unit": "files",
        "count": len(held),
        "bytes": int(files["size_bytes"].sum()),
        "first": None,
        "last": None,
        "updated": _iso(ledger["retrieved_at"].max()),
        "check": disk_check(ledgered, on_disk),
        "parts": parts,
    }


def _nc_coverage(path: Path) -> tuple[str | None, str | None]:
    """``time_coverage_start``/``end`` from a netCDF header (``None`` if unreadable)."""
    try:
        import xarray as xr

        with xr.open_dataset(path, decode_times=False) as ds:
            a = ds.attrs
            return (
                _iso(pd.Timestamp(a["time_coverage_start"])),
                _iso(pd.Timestamp(a["time_coverage_end"])),
            )
    except Exception:  # missing optional backend, missing file or attribute
        return None, None


def build_collections(inputs: CollectionInputs) -> list[dict[str, Any]]:
    """Every collection present on this host (see module docstring)."""
    out = []
    for build in (pacioos_waimea, webcoos, era5_waves, cdip, ndbc_history, external):
        entry = build(inputs)
        if entry is not None:
            out.append(entry)
    return out
