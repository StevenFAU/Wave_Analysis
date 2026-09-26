"""PacIOOS beach-camera archive: Waimea Bay and Waikiki, Oahu, 2009-2013.

The Pacific Islands Ocean Observing System (PacIOOS) publishes four retired
beach cameras on its ERDDAP server as tabledap datasets: one row per image,
with the image served from ERDDAP's ``files/`` tree (verified 2026-09-26)::

    https://pae-paha.pacioos.hawaii.edu/erddap/tabledap/beachcam_003.csv?time,url,name,size
    https://pae-paha.pacioos.hawaii.edu/erddap/files/beachcam_003/2012/01/05/beachcam_003.20120105T070000.jpg

Facts from the dataset metadata (``erddap/info/beachcam_003``)
-------------------------------------------------------------
* ``time`` is UTC. File names carry **Hawaii Standard Time** (UTC-10), and the
  ``files/`` directories are HST dates: ``...20120105T170000.jpg`` is
  2012-01-06 03:00 UTC. :func:`parse_index` checks this offset on every row.
* "Minutes and seconds always rounded to 00:00 in filename. **Actual time of
  photo may occur anytime within the specified hour.** See timestamp on photo
  image for more precise time." The index time is therefore the start of a
  one-hour window, not the capture time; pairing with wave records must allow
  for that (or read the burned-in timestamp).
* Licence: "The data may be used and redistributed for free but is not
  intended for legal use"; acknowledge PacIOOS. Citation form: "PacIOOS.
  [year-of-data-download], [Title], [Data access URL], accessed [date]."
* The index row gives each file's ``size`` in bytes, so every download is
  checked against it.

Archive layout
--------------
``<root>/<dataset>/<YYYY>/<MM>/<DD>/<file>.jpg`` mirrors ERDDAP's ``files/``
tree (HST dates). ``<root>/_index/<dataset>_<stamp>.csv`` holds the index as
served, and ``<root>/_manifests/<dataset>.csv`` is the per-request ledger
(kept with the archive, like the buoy-camera ledger; ADR 0008). Runs resume:
images already archived with a matching size are not requested again.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, read_manifest

DATASET_ID = "pacioos_beachcam"
ERDDAP = "https://pae-paha.pacioos.hawaii.edu/erddap"
FILES_PREFIX = f"{ERDDAP}/files/"

#: ERDDAP dataset -> view (registry ``pacioos_beachcam``).
BEACHCAMS: dict[str, str] = {
    "beachcam_001": "Waikiki (oblique)",
    "beachcam_002": "Waikiki (nadir)",
    "beachcam_003": "Waimea Bay (beach)",
    "beachcam_004": "Waimea Bay (offshore)",
}
#: The Waimea Bay pair, 6.3 km from CDIP 106.
WAIMEA = ("beachcam_003", "beachcam_004")

#: Hours from HST (file names) to UTC.
HST_TO_UTC_HOURS = 10

INDEX_VARIABLES = (
    "time",
    "latitude",
    "longitude",
    "altitude",
    "aboveGround",
    "url",
    "name",
    "size",
)

INDEX_DIR = "_index"
MANIFESTS_DIR = "_manifests"

#: Receiver for manifest entries as they are produced.
Sink = Callable[[list[ManifestEntry]], object]


def _check_dataset(dataset: str) -> str:
    if dataset not in BEACHCAMS:
        raise ValueError(
            f"unknown PacIOOS beach camera {dataset!r}; expected one of {list(BEACHCAMS)}"
        )
    return dataset


def index_url(
    dataset: str,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> str:
    """ERDDAP tabledap CSV query for a camera's image index, optionally for ``[start, end)``."""
    _check_dataset(dataset)
    query = ",".join(INDEX_VARIABLES)
    for op, t in ((">=", start), ("<", end)):
        if t is not None:
            ts = pd.Timestamp(t)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            query += "&" + quote(f"time{op}{ts:%Y-%m-%dT%H:%M:%SZ}", safe="=:-T")
    return f"{ERDDAP}/tabledap/{dataset}.csv?{query}"


def name_time_hst(name: str) -> pd.Timestamp:
    """Local (HST) hour in a file name like ``beachcam_003.20120105T070000.jpg``."""
    stem = name.rsplit(".", 1)[0]
    return pd.Timestamp(pd.to_datetime(stem.rsplit(".", 1)[-1], format="%Y%m%dT%H%M%S"))


def parse_index(source: str | Path, dataset: str) -> pd.DataFrame:
    """Parse a tabledap CSV index (second row: units) into a checked table.

    Columns: ``dataset, time_utc, name, url, size_bytes, latitude, longitude,
    altitude_m, above_ground_m``. Raises if a URL is outside the dataset's
    ``files/`` tree, a name does not match its URL, a file name's HST hour is
    not ``time_utc - 10 h``, or a time is duplicated.
    """
    _check_dataset(dataset)
    raw = pd.read_csv(source, skiprows=[1], dtype={"url": str, "name": str})
    missing = set(INDEX_VARIABLES) - set(raw.columns)
    if missing:
        raise ValueError(f"index lacks columns {sorted(missing)}")
    out = pd.DataFrame(
        {
            "dataset": dataset,
            "time_utc": pd.to_datetime(raw["time"], utc=True),
            "name": raw["name"],
            "url": raw["url"],
            "size_bytes": raw["size"].astype("int64"),
            "latitude": raw["latitude"].astype("float64"),
            "longitude": raw["longitude"].astype("float64"),
            "altitude_m": raw["altitude"].astype("float64"),
            "above_ground_m": raw["aboveGround"].astype("float64"),
        }
    )
    problems = []
    prefix = f"{FILES_PREFIX}{dataset}/"
    bad_url = ~out["url"].str.startswith(prefix)
    if bad_url.any():
        problems.append(f"{int(bad_url.sum())} URLs outside {prefix}")
    bad_name = out["url"].str.rsplit("/", n=1).str[-1] != out["name"]
    if bad_name.any():
        problems.append(f"{int(bad_name.sum())} names differ from their URL")
    hst = pd.to_datetime([name_time_hst(n) for n in out["name"]]).tz_localize("UTC")
    offset = (out["time_utc"] - hst) / pd.Timedelta(hours=1)
    bad_tz = offset != HST_TO_UTC_HOURS
    if bad_tz.any():
        problems.append(
            f"{int(bad_tz.sum())} file names are not UTC-{HST_TO_UTC_HOURS} h "
            f"(first: {out.loc[bad_tz, 'name'].iloc[0]})"
        )
    dup = out["time_utc"].duplicated()
    if dup.any():
        problems.append(f"{int(dup.sum())} duplicate times")
    if problems:
        raise ValueError(f"{dataset} index: " + "; ".join(problems))
    return out.sort_values("time_utc").reset_index(drop=True)


def local_path(dest_root: Path, url: str) -> Path:
    """Archive path mirroring ERDDAP's ``files/`` tree."""
    if not url.startswith(FILES_PREFIX):
        raise ValueError(f"not an ERDDAP files/ URL: {url}")
    rel = url.removeprefix(FILES_PREFIX)
    if ".." in rel.split("/"):
        raise ValueError(f"unsafe path in URL: {url}")
    return dest_root / rel


def ledger_path(dest_root: Path, dataset: str) -> Path:
    """Per-camera request ledger."""
    return dest_root / MANIFESTS_DIR / f"{_check_dataset(dataset)}.csv"


def fetch_index(
    downloader: Downloader,
    dataset: str,
    dest_root: Path,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> tuple[ManifestEntry, pd.DataFrame | None]:
    """Download a camera's index as served, store it, and parse it."""
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    dest = dest_root / INDEX_DIR / f"{dataset}_{stamp}.csv"
    entry = downloader.fetch(
        index_url(dataset, start, end),
        dest,
        source_id=DATASET_ID,
        product="index",
        station_id=dataset,
        period=_period(start, end),
    )
    if entry.status != EntryStatus.VERIFIED:
        return entry, None
    return entry, parse_index(dest, dataset)


def archived(dest_root: Path, dataset: str) -> tuple[set[str], set[str]]:
    """``(archived_urls, not_found_urls)`` from the ledger.

    A URL counts as archived only if its file exists with the size the ledger
    recorded, so a deleted or truncated file is fetched again.
    """
    done: set[str] = set()
    missing: set[str] = set()
    ledger = ledger_path(dest_root, dataset)
    if not ledger.exists():
        return done, missing
    for e in read_manifest(ledger):
        if e.product != "image":
            continue
        if e.status == EntryStatus.NOT_FOUND:
            missing.add(e.url)
        elif e.sha256 is not None and e.size_bytes is not None:
            p = local_path(dest_root, e.url)
            if p.exists() and p.stat().st_size == e.size_bytes:
                done.add(e.url)
    return done, missing - done


@dataclass
class ArchiveSummary:
    """Outcome of one :func:`archive_images` run."""

    dataset: str
    indexed: int = 0
    already_archived: int = 0
    known_missing: int = 0
    downloaded: int = 0
    not_found: int = 0
    failed: int = 0
    size_mismatch: int = 0
    bytes: int = 0
    seconds: float = 0.0
    failures: list[str] = field(default_factory=list)

    def line(self) -> str:
        """One-line report."""
        return (
            f"{self.dataset}: {self.indexed:,} indexed, {self.already_archived:,} already archived, "
            f"{self.downloaded:,} downloaded ({self.bytes / 1e6:,.1f} MB), "
            f"{self.not_found} not found, {self.failed} failed, "
            f"{self.size_mismatch} size mismatches, {self.seconds / 60:.1f} min"
        )


def archive_images(
    downloader: Downloader,
    index: pd.DataFrame,
    dest_root: Path,
    *,
    sink: Sink,
    limit: int | None = None,
    progress: Callable[[ArchiveSummary], object] | None = None,
    progress_every: int = 500,
) -> ArchiveSummary:
    """Download the images in ``index`` that are not yet archived.

    Every request is passed to ``sink`` as soon as it completes, so an
    interrupted run keeps its ledger. A download whose size differs from the
    index is kept but noted in the ledger and counted in ``size_mismatch``.
    """
    datasets = index["dataset"].unique()
    if len(datasets) != 1:
        raise ValueError("index must hold exactly one dataset")
    dataset = str(datasets[0])
    summary = ArchiveSummary(dataset=dataset, indexed=len(index))
    done, missing = archived(dest_root, dataset)
    t0 = time.monotonic()
    todo = index[~index["url"].isin(done | missing)]
    summary.already_archived = int(index["url"].isin(done).sum())
    summary.known_missing = int(index["url"].isin(missing).sum())
    if limit is not None:
        todo = todo.head(limit)
    rows = zip(
        todo["url"].astype(str).tolist(),
        pd.DatetimeIndex(todo["time_utc"]).strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
        todo["size_bytes"].astype(int).tolist(),
        strict=True,
    )
    for i, (url, stamp, expected_size) in enumerate(rows, start=1):
        entry = downloader.fetch(
            url,
            local_path(dest_root, url),
            source_id=DATASET_ID,
            product="image",
            station_id=dataset,
            period=stamp,
        )
        if entry.status == EntryStatus.VERIFIED:
            summary.downloaded += 1
            summary.bytes += entry.size_bytes or 0
            if entry.size_bytes != expected_size:
                summary.size_mismatch += 1
                entry = entry.model_copy(
                    update={"note": f"size differs from index ({expected_size} bytes)"}
                )
        elif entry.status == EntryStatus.NOT_FOUND:
            summary.not_found += 1
        else:
            summary.failed += 1
            summary.failures.append(f"{url}: {entry.http_status or entry.note}")
        sink([entry])
        summary.seconds = time.monotonic() - t0
        if progress is not None and i % progress_every == 0:
            progress(summary)
    summary.seconds = time.monotonic() - t0
    return summary


def index_summary(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Images, bytes and time span per camera."""
    rows = []
    for f in frames:
        rows.append(
            {
                "dataset": f["dataset"].iloc[0] if len(f) else "",
                "images": len(f),
                "gigabytes": float(np.sum(f["size_bytes"])) / 1e9,
                "first_utc": f["time_utc"].min(),
                "last_utc": f["time_utc"].max(),
            }
        )
    return pd.DataFrame(rows)


def _period(start: pd.Timestamp | str | None, end: pd.Timestamp | str | None) -> str:
    if start is None and end is None:
        return "all"
    return f"{start or ''}/{end or ''}"
