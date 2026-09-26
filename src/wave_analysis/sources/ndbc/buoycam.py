"""Prospective archiver for NDBC buoy-camera images.

Why this exists
---------------
The 2026-09-26 inventory found 91 NDBC stations with a buoy camera, all with
historical directional wave spectra and 87 publishing directional spectra in
realtime. No public *historical* archive of the camera images was found: the
``buoycams.php`` listing exposes only the latest image per station. The largest
openly buildable image + directional-spectrum dataset is therefore one that is
**collected prospectively**, by capturing each new image as it appears. Running
this collector hourly (see ``docs/datasets/ndbc.md#buoy-camera-archiving``)
starts that dataset.

Image format (observed on 79 images, 2026-09-26 00:10 UTC)
----------------------------------------------------------
* JPEG strip of **six camera views** side by side, facing different
  headings (a "90°" label marks the first). 78 of 79 stations: 2880 x 300 px
  (6 x 480 px views); one (44008): 2304 x 246 px. **Read the geometry from
  each image**, never from the listing alone.
* File name ``{CAM}_{YYYY}_{MM}_{DD}_{HHMM}.jpg`` (e.g. ``W04A_2026_09_26_0010.jpg``);
  the time matches the caption burned into the image ("09/26/2026 0010 UTC").
  Images are stamped at minute 10 of the hour (occasionally minute 0).
* **A caption is burned into the bottom of every image** (station ID, UTC
  time, heading label). Models can read it, which leaks station identity and
  invalidates cross-station evaluation. On all 47 daylight 2880 x 300 images
  it occupies exactly the bottom **30 rows** (:data:`CAPTION_BAND_PX`). For
  other heights :func:`caption_band_rows` scales proportionally; that fallback
  is *unverified* until a daylight image of that format is measured.
* Visible per view: independent horizon tilt (buoy roll/pitch), sun glint,
  distant land, hull/tower structure at the edges, lens spots. Quality
  annotation should therefore run **per view** (:func:`split_views`).

Cadence and retention (measured 2026-09-26)
-------------------------------------------
* One image per camera per hour, **day and night**, stamped at minute 10
  (99.7 %) or occasionally minute 0. The minute is **not** fixed per camera:
  46089, 46050 and 42056 switch between :10 and :00 from hour to hour.
  Images appear on the server about 20 minutes after the stamp
  (``Last-Modified`` 00:31 for a 00:10 image).
* Superseded images stay downloadable at their own URL for about **72 hours**
  (oldest available image = now - 72 h; older requests return 404). The
  directory itself is not listable (403), so older images are recovered by
  constructing their file names: :func:`backfill_cameras`. An archiver that
  is down for less than ~3 days therefore loses nothing.
* Some hours are simply absent (404 for every candidate minute inside the
  retention window); those gaps are recorded in the manifest as
  ``not_found`` and not re-requested.

Archive layout
--------------
``<root>/<station>/<YYYY>/<MM>/<file>.jpg`` for images,
``<root>/_listings/buoycams_<stamp>.json`` for listing snapshots, and
``<root>/_manifests/<YYYY-MM>.csv`` for the per-request ledger (monthly
partitions, kept with the archive rather than in Git because it grows by
~2,000 rows a day; see ADR 0008).
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, utcnow
from wave_analysis.sources.ndbc.inventory import parse_buoycams
from wave_analysis.sources.ndbc.products import BUOYCAM_IMAGE_URL, BUOYCAM_URL

#: Bottom rows of a 300-px-high strip containing the burned-in caption
#: (measured: exactly 30 rows on 47/47 daylight images, 2026-09-26).
CAPTION_BAND_PX = 30
_REFERENCE_HEIGHT_PX = 300

#: Number of camera views per strip.
N_VIEWS = 6

#: Hours a superseded image stays downloadable (measured 2026-09-26).
RETENTION_HOURS = 72

#: Default backfill window: inside the retention window with a safety margin.
DEFAULT_BACKFILL_HOURS = 70

#: Minute stamps tried for each missing hour. Observed 2026-09-23..26: 5,313
#: images at :10, 156 at :00, 2 at :50. The minute is not fixed per camera, so
#: the latest listed image's minute is tried first, then these.
CANDIDATE_MINUTES = (10, 0)

#: Receiver for manifest entries as they are produced.
Sink = Callable[[list[ManifestEntry]], object]

LISTINGS_DIR = "_listings"
MANIFESTS_DIR = "_manifests"

_NAME = re.compile(
    r"^(?P<cam>[A-Z0-9]+)_(?P<y>\d{4})_(?P<m>\d{2})_(?P<d>\d{2})_(?P<hh>\d{2})(?P<mm>\d{2})\.jpg$"
)


def parse_buoycam_filename(name: str) -> tuple[str, pd.Timestamp]:
    """Return ``(camera_code, capture_time_utc)`` from a buoycam image file name."""
    m = _NAME.match(name)
    if m is None:
        raise ValueError(f"unrecognised buoycam file name: {name!r}")
    ts = pd.Timestamp(
        year=int(m["y"]),
        month=int(m["m"]),
        day=int(m["d"]),
        hour=int(m["hh"]),
        minute=int(m["mm"]),
        tz="UTC",
    )
    return m["cam"], ts


def image_path(dest_root: Path, station_id: str, file_name: str) -> Path:
    """Archive path of an image: ``<root>/<station>/<YYYY>/<MM>/<file>``."""
    _, ts = parse_buoycam_filename(file_name)
    return dest_root / station_id / f"{ts:%Y}" / f"{ts:%m}" / file_name


def image_file_name(camera_code: str, time_utc: pd.Timestamp) -> str:
    """File name NDBC uses for an image of ``camera_code`` stamped ``time_utc``."""
    return f"{camera_code}_{time_utc:%Y_%m_%d_%H%M}.jpg"


def manifest_path(dest_root: Path, when: pd.Timestamp | None = None) -> Path:
    """Monthly manifest partition for requests made at ``when`` (default: now)."""
    when = pd.Timestamp(utcnow()) if when is None else when
    return dest_root / MANIFESTS_DIR / f"{when:%Y-%m}.csv"


def known_missing_urls(dest_root: Path) -> set[str]:
    """Image URLs already recorded as ``not_found`` in the archive manifests.

    A 404 for an hour *earlier* than a camera's latest listed image means that
    image was never published, so it is not requested again.
    """
    urls: set[str] = set()
    for p in sorted((dest_root / MANIFESTS_DIR).glob("*.csv"))[-3:]:
        with p.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["product"] == "buoycam:image" and row["status"] == "not_found":
                    urls.add(row["url"])
    return urls


def _fetch_image(
    downloader: Downloader, dest_root: Path, station_id: str, file_name: str
) -> ManifestEntry:
    _, captured = parse_buoycam_filename(file_name)
    return downloader.fetch(
        f"{BUOYCAM_IMAGE_URL}/{file_name}",
        image_path(dest_root, station_id, file_name),
        source_id="ndbc",
        product="buoycam:image",
        station_id=station_id,
        period=captured.strftime("%Y-%m-%dT%H:%MZ"),
    )


def snapshot_cameras(
    downloader: Downloader, dest_root: Path
) -> tuple[list[ManifestEntry], pd.DataFrame]:
    """Download every *new* current buoy-camera image and the listing itself.

    Images already on disk (same file name) are skipped without a request, so
    repeated runs build a de-duplicated archive. The listing JSON is saved on
    every run, preserving the station-to-camera mapping over time.

    Returns
    -------
    entries, cameras
        Manifest entries for every request made, and the parsed listing
        (empty if the listing could not be fetched).
    """
    entries: list[ManifestEntry] = []
    stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
    listing_path = dest_root / LISTINGS_DIR / f"buoycams_{stamp}.json"
    listing = downloader.fetch(
        BUOYCAM_URL, listing_path, source_id="ndbc", product="buoycam:listing", period="snapshot"
    )
    entries.append(listing)
    if listing.local_path is None:
        return entries, pd.DataFrame(columns=["station_id", "latest_image"])
    cams = parse_buoycams(Path(listing.local_path).read_text(), pd.Timestamp(listing.retrieved_at))
    for row in cams.itertuples(index=False):
        name = str(row.latest_image or "")
        if not _NAME.match(name):
            continue
        if image_path(dest_root, str(row.station_id), name).exists():
            continue
        entries.append(_fetch_image(downloader, dest_root, str(row.station_id), name))
    return entries, cams


def backfill_candidates(
    cameras: pd.DataFrame,
    dest_root: Path,
    *,
    now: pd.Timestamp,
    hours: int = DEFAULT_BACKFILL_HOURS,
    skip_urls: Iterable[str] = (),
) -> list[tuple[str, tuple[str, ...]]]:
    """Hours to recover: ``(station_id, candidate_file_names)`` per missing hour.

    For each camera in the listing, every hour *before* its latest image and
    newer than ``now - hours`` is considered. The minute stamp is not fixed
    per camera (mostly :10, occasionally :00), so each hour gets one
    candidate name per minute in :data:`CANDIDATE_MINUTES`, with the latest
    image's minute first. An hour is skipped if any candidate is already on
    disk; candidates already known to be missing are dropped. ``hours`` must
    stay below :data:`RETENTION_HOURS`.
    """
    if hours >= RETENTION_HOURS:
        raise ValueError(f"backfill window must be < {RETENTION_HOURS} h retention, got {hours}")
    skip = set(skip_urls)
    oldest = now - pd.Timedelta(hours=hours)
    out: list[tuple[str, tuple[str, ...]]] = []
    for row in cameras.itertuples(index=False):
        name = str(row.latest_image or "")
        if not _NAME.match(name):
            continue
        sid = str(row.station_id)
        cam, latest = parse_buoycam_filename(name)
        minutes = tuple(dict.fromkeys((latest.minute, *CANDIDATE_MINUTES)))
        hour = latest.floor("h") - pd.Timedelta(hours=1)
        while hour + pd.Timedelta(minutes=max(minutes)) >= oldest:
            names = [image_file_name(cam, hour + pd.Timedelta(minutes=m)) for m in minutes]
            if not any(image_path(dest_root, sid, n).exists() for n in names):
                todo = tuple(n for n in names if f"{BUOYCAM_IMAGE_URL}/{n}" not in skip)
                if todo:
                    out.append((sid, todo))
            hour -= pd.Timedelta(hours=1)
    return out


def backfill_cameras(
    downloader: Downloader,
    dest_root: Path,
    cameras: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    hours: int = DEFAULT_BACKFILL_HOURS,
    sink: Sink | None = None,
    batch_size: int = 100,
) -> list[ManifestEntry]:
    """Recover images from the last ``hours`` that the archive does not hold.

    NDBC keeps superseded images for ~72 h (see module notes), so this closes
    gaps left by downtime or missed runs. For each missing hour the candidate
    minute stamps are tried in order until one exists. Every 404 is recorded
    and not requested again (:func:`known_missing_urls`). If ``sink`` is given
    it receives entries in batches as they complete, so a long backfill that
    is interrupted still leaves a manifest row for every file written.
    """
    now = pd.Timestamp(utcnow()) if now is None else now
    todo = backfill_candidates(
        cameras, dest_root, now=now, hours=hours, skip_urls=known_missing_urls(dest_root)
    )
    entries: list[ManifestEntry] = []
    pending: list[ManifestEntry] = []
    for sid, names in todo:
        for fname in names:
            e = _fetch_image(downloader, dest_root, sid, fname)
            entries.append(e)
            pending.append(e)
            if e.status != EntryStatus.NOT_FOUND:
                break  # found it (or a transport failure: move on to the next hour)
        if sink is not None and len(pending) >= batch_size:
            sink(pending)
            pending = []
    if sink is not None and pending:
        sink(pending)
    return entries


def archive_cameras(
    downloader: Downloader,
    dest_root: Path,
    *,
    backfill_hours: int | None = DEFAULT_BACKFILL_HOURS,
    now: pd.Timestamp | None = None,
    sink: Sink | None = None,
) -> list[ManifestEntry]:
    """One archiver run: listing + new images, then backfill (unless disabled).

    With ``sink`` (e.g. a manifest writer), entries are handed over as each
    stage completes instead of only at the end.
    """
    entries, cams = snapshot_cameras(downloader, dest_root)
    if sink is not None:
        sink(entries)
    if backfill_hours and not cams.empty:
        entries += backfill_cameras(
            downloader, dest_root, cams, now=now, hours=backfill_hours, sink=sink
        )
    return entries


def archive_index(dest_root: Path) -> pd.DataFrame:
    """Index archived images: station, camera code, capture time, path.

    This table is the *target* side for synchronising images with wave spectra
    (:func:`wave_analysis.processing.synchronize.pair_nearest`).
    """
    rows = []
    for p in sorted(dest_root.rglob("*.jpg")):
        rel = p.relative_to(dest_root)
        if rel.parts[0].startswith("_"):
            continue
        try:
            cam, ts = parse_buoycam_filename(p.name)
        except ValueError:
            continue
        rows.append(
            {"station_id": rel.parts[0], "camera_code": cam, "time_utc": ts, "path": str(p)}
        )
    return pd.DataFrame(rows, columns=["station_id", "camera_code", "time_utc", "path"])


def summarize(entries: list[ManifestEntry]) -> dict[str, int]:
    """Count entries by status (for CLI output)."""
    out: dict[str, int] = {}
    for e in entries:
        out[e.status.value] = out.get(e.status.value, 0) + 1
    out["new_images"] = sum(
        1 for e in entries if e.product == "buoycam:image" and e.status == EntryStatus.VERIFIED
    )
    return out


def caption_band_rows(image_height: int) -> int:
    """Rows to mask at the bottom of a strip of the given height (see module notes)."""
    if image_height == _REFERENCE_HEIGHT_PX:
        return CAPTION_BAND_PX
    return round(CAPTION_BAND_PX * image_height / _REFERENCE_HEIGHT_PX)


def split_views(image: NDArray[np.generic], n_views: int = N_VIEWS) -> list[NDArray[np.generic]]:
    """Crop the caption band and split a buoycam strip into its camera views.

    Parameters
    ----------
    image
        ``H x W`` or ``H x W x C`` array of the full strip.
    n_views
        Views per strip (6 for all formats observed so far). ``W`` must be
        divisible by ``n_views``.
    """
    h, w = image.shape[:2]
    if w % n_views:
        raise ValueError(f"strip width {w} is not divisible into {n_views} views")
    body = image[: h - caption_band_rows(h)]
    step = w // n_views
    return [body[:, i * step : (i + 1) * step] for i in range(n_views)]
