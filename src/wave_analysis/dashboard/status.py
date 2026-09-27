"""Live archive status for the dashboard, built on the collector host.

Inputs are the buoy-camera archive (``<root>/<station>/<YYYY>/<MM>/*.jpg``),
its ledger (``<root>/_manifests/*.csv``), its listing snapshots
(``<root>/_listings/``), the realtime-spectra snapshots
(``data/raw/ndbc/<station>/<product>/<file>.<stamp>[.gz]``), and the offsite
sync stamp. Nothing is fetched from the network. The output contains no local
paths, host names, or user names, because it is published to a public branch.

Per-station hourly coverage is encoded as one character per hour (oldest
first) over the status window:

====== ==========================================================
``0-5`` image archived, stamped at minute ``digit * 10``
``x``   not published: every candidate URL returned 404 (in ledger)
``?``   gap: due, not archived, no 404 recorded (collector missed it)
``.``   not expected: station not yet listed, or hour still pending
====== ==========================================================

An hour is *due* once ``DUE_AFTER`` has passed since its start, as of the
newest listing snapshot (the last collector run). A parallel string gives the
illumination at minute 10 of each hour (``d`` day, ``l`` low sun, ``t``
twilight, ``n`` night) from :mod:`wave_analysis.processing.solar`.

``collections`` lists every data collection on the host with its totals and a
check of files against ledger (:mod:`wave_analysis.dashboard.collections`),
starting with this archive and the realtime-spectra snapshots.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wave_analysis import __version__
from wave_analysis.dashboard.collections import CollectionInputs, build_collections, disk_check
from wave_analysis.processing.solar import illumination_category, solar_position
from wave_analysis.sources.ndbc.buoycam import LISTINGS_DIR, MANIFESTS_DIR, parse_buoycam_filename
from wave_analysis.sources.ndbc.inventory import parse_buoycams

SCHEMA = "wave-analysis/dashboard-status@1"
WINDOW_HOURS = 168
# NDBC posts an image ~20-30 min after its :10 stamp; the collector runs at :40.
DUE_AFTER = pd.Timedelta(minutes=50)
OFFSITE_STAMP = ".offsite_last_sync"
_ILLUM_CODE = {
    "day": "d",
    "low_sun": "l",
    "civil_twilight": "t",
    "nautical_twilight": "t",
    "night": "n",
}
_LISTING_NAME = re.compile(r"^buoycams_(\d{8}T\d{6}Z)\.json$")
_RT_STAMP = re.compile(r"\.(\d{8}T\d{6}Z)(?:\.gz)?$")


def _float_or_none(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _iso(ts: pd.Timestamp | dt.datetime | None) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def scan_images(dest_root: Path) -> pd.DataFrame:
    """Every archived image on disk: station, camera code, time, bytes.

    Uses ``os.scandir`` (one ``stat`` per file) rather than ``rglob`` so that a
    year of images (~700k files) scans in seconds.
    """
    cols = ["station_id", "camera", "time_utc", "size_bytes"]
    rows: list[tuple[str, str, pd.Timestamp, int]] = []

    def walk(d: Path) -> Iterator[os.DirEntry[str]]:
        with os.scandir(d) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    yield from walk(Path(e.path))
                elif e.name.endswith(".jpg"):
                    yield e

    if not dest_root.exists():
        return pd.DataFrame(columns=cols)
    for top in os.scandir(dest_root):
        if not top.is_dir() or top.name.startswith(("_", ".")):
            continue
        for e in walk(Path(top.path)):
            try:
                cam, ts = parse_buoycam_filename(e.name)
            except ValueError:
                continue
            rows.append((top.name, cam, ts, e.stat().st_size))
    df = pd.DataFrame(rows, columns=cols)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    return df


def read_ledger(dest_root: Path) -> pd.DataFrame:
    """All ledger rows needed for status, tolerant of a partially written last line."""
    cols = ["product", "station_id", "period", "url", "status", "retrieved_at"]
    rows: list[dict[str, str]] = []
    for p in sorted((dest_root / MANIFESTS_DIR).glob("*.csv")):
        with p.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if None in row or any(row.get(c) is None for c in cols):
                    continue  # truncated line (file being appended to)
                rows.append({c: row[c] for c in cols})
    df = pd.DataFrame(rows, columns=cols)
    df["retrieved_at"] = pd.to_datetime(df["retrieved_at"], utc=True, errors="coerce")
    df["period_utc"] = pd.to_datetime(
        df["period"], format="%Y-%m-%dT%H:%MZ", utc=True, errors="coerce"
    )
    return df.dropna(subset=["retrieved_at"])


def latest_listing(dest_root: Path) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    """Time and content of the newest ``buoycams.php`` snapshot."""
    snaps = sorted(
        p for p in (dest_root / LISTINGS_DIR).glob("buoycams_*.json") if _LISTING_NAME.match(p.name)
    )
    for p in reversed(snaps):
        m = _LISTING_NAME.match(p.name)
        assert m is not None
        when = pd.Timestamp(dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ"), tz="UTC")
        try:
            return when, parse_buoycams(p.read_text(encoding="utf-8"), when)
        except ValueError:
            continue  # partially written snapshot; fall back to the previous one
    return None, pd.DataFrame()


# --------------------------------------------------------------------------- #
# Derived tables
# --------------------------------------------------------------------------- #


def coverage_string(
    hours: pd.DatetimeIndex,
    image_times: pd.Series,
    not_found_hours: set[pd.Timestamp],
    *,
    active_from: pd.Timestamp | None,
    active_until: pd.Timestamp | None,
    as_of: pd.Timestamp,
) -> str:
    """Encode one station's hourly coverage (see module docstring)."""
    minute_by_hour: dict[pd.Timestamp, int] = {}
    for t in sorted(image_times):
        minute_by_hour.setdefault(t.floor("h"), t.minute)
    out = []
    for h in hours:
        if h in minute_by_hour:
            out.append(str(min(minute_by_hour[h] // 10, 5)))
        elif h in not_found_hours:
            out.append("x")
        elif (
            active_from is not None
            and h >= active_from
            and (active_until is None or h <= active_until)
            and h + DUE_AFTER <= as_of
        ):
            out.append("?")
        else:
            out.append(".")
    return "".join(out)


def illumination_string(hours: pd.DatetimeIndex, lat: float | None, lon: float | None) -> str:
    """Illumination code per hour at minute 10 (``-`` when position unknown)."""
    if lat is None or lon is None or not np.isfinite(lat) or not np.isfinite(lon):
        return "-" * len(hours)
    times = hours + pd.Timedelta(minutes=10)
    elev, _ = solar_position(times, np.full(len(times), lat), np.full(len(times), lon))
    return "".join(_ILLUM_CODE[c] for c in illumination_category(elev))


def collector_runs(ledger: pd.DataFrame, since: pd.Timestamp) -> list[dict[str, Any]]:
    """One record per archiver run (a listing request starts each run)."""
    lst = ledger[ledger["product"] == "buoycam:listing"].sort_values("retrieved_at")
    if lst.empty:
        return []
    starts = lst["retrieved_at"].to_numpy()
    img = ledger[ledger["product"] == "buoycam:image"]
    run_idx = np.searchsorted(starts, img["retrieved_at"].to_numpy(), side="right") - 1
    img = img.assign(run=run_idx)
    runs = []
    for i, start in enumerate(lst["retrieved_at"]):
        if start < since:
            continue
        g = img[img["run"] == i]
        runs.append(
            {
                "at": _iso(start),
                "new_images": int((g["status"] == "verified").sum()),
                "not_found": int((g["status"] == "not_found").sum()),
                "failed": int((g["status"] == "failed").sum()),
                "duration_s": (
                    round((g["retrieved_at"].max() - start).total_seconds()) if not g.empty else 0
                ),
            }
        )
    return runs


def realtime_snapshots(ndbc_root: Path) -> list[dict[str, Any]]:
    """Realtime-spectra snapshots on disk, grouped by snapshot date."""
    groups: dict[str, dict[str, Any]] = {}
    if not ndbc_root.exists():
        return []
    for sdir in os.scandir(ndbc_root):
        if (
            not sdir.is_dir()
            or sdir.name.startswith(("_", "."))
            or sdir.name
            in {
                "buoycam",
                "index",
            }
        ):
            continue
        for pdir in os.scandir(sdir.path):
            if not pdir.is_dir():
                continue
            for f in os.scandir(pdir.path):
                m = _RT_STAMP.search(f.name)
                if m is None:
                    continue
                day = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}"
                g = groups.setdefault(
                    day, {"date": day, "stations": set(), "products": set(), "files": 0, "bytes": 0}
                )
                g["stations"].add(sdir.name)
                g["products"].add(pdir.name)
                g["files"] += 1
                g["bytes"] += f.stat().st_size
    return [
        {**g, "stations": len(g["stations"]), "products": sorted(g["products"])}
        for _, g in sorted(groups.items())
    ]


def offsite_status(raw_root: Path) -> dict[str, Any]:
    """Offsite copy state from the stamp ``scripts/sync_offsite.sh`` writes."""
    stamp = raw_root / OFFSITE_STAMP
    if not stamp.exists():
        return {"last_sync": None}
    text = stamp.read_text(encoding="utf-8").strip()
    try:
        return {"last_sync": _iso(pd.Timestamp(text))}
    except ValueError:
        return {"last_sync": None}


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StatusInputs:
    """Locations the status is built from."""

    buoycam_root: Path
    ndbc_root: Path
    raw_root: Path
    manifests_root: Path | None = None  # data/manifests: ERA5 and CDIP ledgers


def build_status(
    inputs: StatusInputs,
    *,
    now: pd.Timestamp | None = None,
    window_hours: int = WINDOW_HOURS,
    latest_obs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble ``status.json`` (see module docstring for the coverage encoding)."""
    now = now or pd.Timestamp.now(tz="UTC")
    images = scan_images(inputs.buoycam_root)
    ledger = read_ledger(inputs.buoycam_root)
    listed_at, listing = latest_listing(inputs.buoycam_root)
    as_of = listed_at or now

    end = as_of.floor("h")
    hours = pd.date_range(end - pd.Timedelta(hours=window_hours - 1), end, freq="h")
    window_start = hours[0]

    nf = ledger[(ledger["product"] == "buoycam:image") & (ledger["status"] == "not_found")]
    nf_hours = nf.dropna(subset=["period_utc"]).assign(hour=lambda d: d["period_utc"].dt.floor("h"))
    have_hours = images.assign(hour=images["time_utc"].dt.floor("h"))
    # A 404 at one minute stamp does not mean the hour is unpublished if another stamp exists.
    nf_only = nf_hours.merge(
        have_hours[["station_id", "hour"]].drop_duplicates(),
        on=["station_id", "hour"],
        how="left",
        indicator=True,
    )
    nf_only = nf_only[nf_only["_merge"] == "left_only"][["station_id", "hour"]].drop_duplicates()

    listing_by_id = (
        {str(r.station_id): r for r in listing.itertuples(index=False)} if not listing.empty else {}
    )
    station_ids = sorted(set(images["station_id"]) | set(listing_by_id))

    dates = (
        pd.date_range(images["time_utc"].min().floor("D"), as_of.floor("D"), freq="D")
        if not images.empty
        else pd.DatetimeIndex([], tz="UTC")
    )
    stations: list[dict[str, Any]] = []
    coverage_totals = {"archived": 0, "not_published": 0, "gap": 0}
    for sid in station_ids:
        g = images[images["station_id"] == sid]
        lst = listing_by_id.get(sid)
        lat = _float_or_none(lst.latitude) if lst is not None else None
        lon = _float_or_none(lst.longitude) if lst is not None else None
        camera = None
        if lst is not None and lst.latest_image:
            try:
                camera, _ = parse_buoycam_filename(str(lst.latest_image))
            except ValueError:
                camera = None
        if camera is None and not g.empty:
            camera = str(g.sort_values("time_utc")["camera"].iloc[-1])
        first = g["time_utc"].min() if not g.empty else None
        last = g["time_utc"].max() if not g.empty else None
        in_window = g[g["time_utc"] >= window_start]
        cov = coverage_string(
            hours,
            in_window["time_utc"],
            set(nf_only.loc[nf_only["station_id"] == sid, "hour"]),
            active_from=first.floor("h") if first is not None else None,
            active_until=None if lst is not None and lst.latest_image else last,
            as_of=as_of,
        )
        coverage_totals["archived"] += sum(c.isdigit() for c in cov)
        coverage_totals["not_published"] += cov.count("x")
        coverage_totals["gap"] += cov.count("?")
        per_day = g.groupby(g["time_utc"].dt.floor("D")).size().reindex(dates, fill_value=0)
        rec: dict[str, Any] = {
            "station_id": sid,
            "name": str(lst.camera_name) if lst is not None and lst.camera_name else None,
            "lat": lat,
            "lon": lon,
            "camera": camera,
            "listed": lst is not None,
            "latest_listed_image": (str(lst.latest_image) or None) if lst is not None else None,
            "images": len(g),
            "bytes": int(g["size_bytes"].sum()),
            "first_image": _iso(first),
            "last_image": _iso(last),
            "coverage": cov,
            "illumination": illumination_string(hours, lat, lon),
            "daily": per_day.astype(int).tolist(),
        }
        if latest_obs and sid in latest_obs:
            rec["latest_obs"] = latest_obs[sid]
        stations.append(rec)

    illum_counts: dict[str, int] = {}
    minute_counts: dict[str, int] = {}
    for s in stations:
        for c, i in zip(s["coverage"], s["illumination"], strict=True):
            if c.isdigit():
                illum_counts[i] = illum_counts.get(i, 0) + 1
                minute_counts[str(int(c) * 10)] = minute_counts.get(str(int(c) * 10), 0) + 1

    img_rows = ledger[ledger["product"] == "buoycam:image"]
    verified_urls = int((img_rows["status"] == "verified").sum())
    by_day_bytes = images.groupby(images["time_utc"].dt.floor("D"))["size_bytes"].sum()
    realtime = realtime_snapshots(inputs.ndbc_root)
    collections = _ndbc_collections(images, img_rows, nf_only, listed_at, realtime)
    if inputs.manifests_root is not None:
        collections += build_collections(CollectionInputs(inputs.raw_root, inputs.manifests_root))
    return {
        "schema": SCHEMA,
        "generated_at": _iso(now),
        "software_version": __version__,
        "window": {"start": _iso(window_start), "hours": window_hours, "as_of": _iso(as_of)},
        "collector": {
            "last_run": _iso(listed_at),
            "runs": collector_runs(ledger, window_start),
            "cadence_minutes": 60,
        },
        "archive": {
            "images": len(images),
            "bytes": int(images["size_bytes"].sum()),
            "stations_with_images": int(images["station_id"].nunique()),
            "stations_listed": len(listing_by_id),
            "first_image": _iso(images["time_utc"].min()) if not images.empty else None,
            "last_image": _iso(images["time_utc"].max()) if not images.empty else None,
            "window_coverage": coverage_totals,
            "window_illumination": illum_counts,
            "window_minute_stamps": minute_counts,
            "ledger_rows": len(ledger),
            "ledger_verified_images": verified_urls,
        },
        "daily": {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "images": [sum(s["daily"][i] for s in stations) for i in range(len(dates))],
            "bytes": by_day_bytes.reindex(dates, fill_value=0).astype(int).tolist(),
        },
        "stations": stations,
        "realtime": realtime,
        "collections": collections,
        "offsite": offsite_status(inputs.raw_root),
    }


def _ndbc_collections(
    images: pd.DataFrame,
    img_rows: pd.DataFrame,
    never_published: pd.DataFrame,
    listed_at: pd.Timestamp | None,
    realtime: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collection entries for the buoy-camera archive and the realtime snapshots."""
    verified = img_rows[img_rows["status"] == "verified"]
    ledgered = set(
        zip(verified["station_id"], verified["url"].str.rsplit("/", n=1).str[-1], strict=True)
    )
    names = [
        f"{cam}_{t:%Y_%m_%d_%H%M}.jpg"
        for cam, t in zip(images["camera"], images["time_utc"], strict=True)
    ]
    on_disk = set(zip(images["station_id"], names, strict=True))
    out: list[dict[str, Any]] = [
        {
            "id": "ndbc_buoycam",
            "dataset_id": "ndbc",
            "name": "NDBC buoy cameras",
            "provider": "NOAA NDBC",
            "mode": "hourly",
            "unit": "images",
            "count": len(images),
            "bytes": int(images["size_bytes"].sum()),
            "first": _iso(images["time_utc"].min()) if not images.empty else None,
            "last": _iso(images["time_utc"].max()) if not images.empty else None,
            "updated": _iso(listed_at),
            "not_found": len(never_published),
            "failed": int((img_rows["status"] == "failed").sum()),
            "check": disk_check(ledgered, on_disk),
            "parts_count": int(images["station_id"].nunique()),
        }
    ]
    if realtime:
        out.append(
            {
                "id": "ndbc_realtime",
                "dataset_id": "ndbc",
                "name": "NDBC realtime wave files (snapshots)",
                "provider": "NOAA NDBC",
                "mode": "twice_monthly",
                "unit": "files",
                "count": sum(r["files"] for r in realtime),
                "bytes": sum(r["bytes"] for r in realtime),
                "first": realtime[0]["date"],
                "last": realtime[-1]["date"],
                "updated": realtime[-1]["date"],
                "parts_count": max(r["stations"] for r in realtime),
            }
        )
    return out
