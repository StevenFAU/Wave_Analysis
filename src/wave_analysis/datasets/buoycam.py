"""Building blocks of the NDBC buoy-camera sea-state dataset.

The camera and the wave sensor are on the same buoy, so the label is measured
where the image is taken (no shore-to-buoy distance, unlike a beach camera).
The label is the buoy's significant wave height (``WVHT``) from the realtime
spectral summary (``.spec``), computed on board from the hull-motion record.
Method and choices: ``docs/experiments/ndbc_buoycam_sea_state.md``.

* :func:`spec_records` merges the twice-monthly realtime ``.spec`` snapshots of
  a station into one record per time stamp (the latest snapshot wins; values
  that differ between snapshots are counted).
* :func:`station_positions` takes each station's position from the saved
  ``buoycams.php`` listings.
* :func:`strip_stats` reads a strip and measures each view (brightness and
  contrast), for daylight and quality rules.
* :func:`crop_strip` writes a strip without its burned-in caption, so no
  packaged image carries the station ID or the time.
* :func:`spatial_groups` joins buoys closer than a distance into one group
  (single linkage), so neighbouring buoys never straddle a station split.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from wave_analysis.processing.geospatial import haversine_m
from wave_analysis.sources.ndbc.buoycam import CAPTION_BAND_PX, N_VIEWS, caption_band_rows
from wave_analysis.sources.ndbc.parsers import parse_spectral_summary

#: Nominal length of an NDBC wave record (20-min acquisition, NDBC acq.shtml).
RECORD_S = 1200

#: ``.spec`` columns kept, with the names used in the pairing table.
SPEC_COLUMNS = {
    "WVHT": "Hs_m",
    "SwH": "swell_height_m",
    "SwP": "swell_period_s",
    "WWH": "wind_wave_height_m",
    "WWP": "wind_wave_period_s",
    "APD": "mean_period_s",
    "MWD": "mean_direction_deg",
    "STEEPNESS": "steepness",
}


def snapshot_stamp(path: Path) -> str:
    """The download stamp of a realtime snapshot (``41004.spec.20261101T082107Z.gz``)."""
    parts = path.name.split(".")
    if len(parts) < 4 or not parts[-2].endswith("Z"):
        raise ValueError(f"not a stamped realtime snapshot: {path.name}")
    return parts[-2]


def spec_records(paths: Iterable[Path], station_id: str) -> pd.DataFrame:
    """One row per ``.spec`` time stamp of a station, merged over snapshots.

    Each realtime file covers the last 45 days, so snapshots overlap. For a
    stamp in several snapshots the latest snapshot's values are kept, and
    ``n_snapshots`` / ``snapshots_disagree`` (``Hs_m`` differs by more than
    0.05 m) say how often it was seen and whether NDBC revised it. The stamp
    is taken as the record's nominal centre: NDBC rounds wave stamps to the
    half hour, and the ``.spec`` stamp lies at the centre or up to 10 min
    before it depending on the payload (``docs/datasets/ndbc.md``).
    """
    frames = []
    for p in sorted(paths, key=snapshot_stamp):
        d = parse_spectral_summary(p).data
        d = d[[c for c in SPEC_COLUMNS if c in d.columns]].rename(columns=SPEC_COLUMNS)
        frames.append(d.assign(snapshot=snapshot_stamp(p)).reset_index())
    cols = ["station_id", "time_utc", *SPEC_COLUMNS.values(), "snapshot", "n_snapshots"]
    if not frames:
        return pd.DataFrame(columns=[*cols, "snapshots_disagree", "start_utc", "end_utc"])
    allrec = pd.concat(frames, ignore_index=True)
    g = allrec.groupby("time_utc")["Hs_m"]
    spread = (g.max() - g.min()).fillna(0.0)
    rec = allrec.sort_values(["time_utc", "snapshot"]).groupby("time_utc").last()
    rec["n_snapshots"] = allrec.groupby("time_utc").size()
    rec["snapshots_disagree"] = spread > 0.05
    rec = rec.reset_index()
    rec.insert(0, "station_id", station_id)
    half = pd.Timedelta(seconds=RECORD_S / 2)
    rec["start_utc"] = rec["time_utc"] - half
    rec["end_utc"] = rec["time_utc"] + half
    for c in SPEC_COLUMNS.values():
        if c not in rec:
            rec[c] = np.nan
    return rec[[*cols, "snapshots_disagree", "start_utc", "end_utc"]]


def station_positions(listings: Iterable[Path]) -> pd.DataFrame:
    """Median position of every station over the saved listings, and how far it moved.

    ``moved_km`` is the largest distance of any listed position from the
    median: a moored buoy swings on its watch circle (a few km at most); a
    larger value means a redeployment or a listing error.
    """
    rows = []
    for p in listings:
        try:
            items = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            continue
        for it in items:
            if it.get("lat") is None or it.get("lng") is None:
                continue
            rows.append((str(it.get("id", "")).upper(), float(it["lat"]), float(it["lng"])))
    d = pd.DataFrame(rows, columns=["station_id", "latitude", "longitude"])
    med = d.groupby("station_id")[["latitude", "longitude"]].median()
    far = d.join(med, on="station_id", rsuffix="_med")
    far["km"] = (
        haversine_m(far["latitude"], far["longitude"], far["latitude_med"], far["longitude_med"])
        / 1000
    )
    med["moved_km"] = far.groupby("station_id")["km"].max()
    med["n_listings"] = d.groupby("station_id").size()
    return med.reset_index()


def strip_stats(path: str | Path, n_views: int = N_VIEWS) -> dict[str, Any]:
    """Geometry, checksum and per-view brightness/contrast of a strip (caption excluded).

    ``view_mean_i`` / ``view_std_i`` are the grey-level mean and standard
    deviation (0-255) of view ``i``. ``readable`` is False when the file does
    not decode or its width is not divisible into ``n_views`` views.
    """
    out: dict[str, Any] = {"readable": False, "width": 0, "height": 0, "sha256": ""}
    try:
        raw = Path(path).read_bytes()
        out["sha256"] = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            g = np.asarray(im.convert("L"), dtype=np.float32)
    except (OSError, ValueError, Image.DecompressionBombError):
        return out
    h, w = g.shape
    out["width"], out["height"] = w, h
    if w % n_views or h <= 2 * CAPTION_BAND_PX:
        return out
    body = g[: h - caption_band_rows(h)]
    step = w // n_views
    for i in range(n_views):
        v = body[:, i * step : (i + 1) * step]
        out[f"view_mean_{i}"] = round(float(v.mean()), 2)
        out[f"view_std_{i}"] = round(float(v.std()), 2)
    out["strip_mean"] = round(float(body.mean()), 2)
    out["readable"] = True
    return out


def crop_strip(src: str | Path, dest: str | Path, *, quality: int) -> tuple[str, int, int, int]:
    """Write ``src`` without its caption band as JPEG; return ``(sha256, bytes, width, height)``."""
    with Image.open(src) as im:
        rgb = im.convert("RGB")
    w, h = rgb.size
    body = rgb.crop((0, 0, w, h - caption_band_rows(h)))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    body.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return hashlib.sha256(data).hexdigest(), len(data), body.size[0], body.size[1]


def spatial_groups(
    station_ids: ArrayLike, latitude: ArrayLike, longitude: ArrayLike, *, km: float
) -> dict[str, str]:
    """Group label of each station: stations within ``km`` of one another (chained) share one.

    Single linkage on great-circle distance: two buoys closer than ``km``
    always share a group, so a station split by group puts every buoy at
    least ``km`` from every buoy of another partition. Labels are
    ``g<lowest station id>`` so they do not depend on input order.
    """
    ids = np.asarray(station_ids).astype(str)
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    if ids.size == 1:
        return {str(ids[0]): f"g{ids[0]}"}
    d = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) / 1000
    d = np.asarray(d, dtype=np.float64)
    np.fill_diagonal(d, 0.0)
    labels = fcluster(linkage(squareform(d, checks=False), "single"), km, "distance")
    first: dict[int, str] = {}
    for i in np.argsort(ids):
        first.setdefault(int(labels[i]), str(ids[i]))
    return {str(s): f"g{first[int(c)]}" for s, c in zip(ids, labels, strict=True)}


def view_columns(prefix: str, n_views: int = N_VIEWS) -> list[str]:
    """``[prefix_0, ..., prefix_{n-1}]``."""
    return [f"{prefix}_{i}" for i in range(n_views)]


def usable_view_mask(
    stats: pd.DataFrame, *, min_mean: float, min_std: float, n_views: int = N_VIEWS
) -> NDArray[np.bool_]:
    """``(strips, n_views)``: which views have mean grey >= ``min_mean`` and std >= ``min_std``."""
    m = stats[view_columns("view_mean", n_views)].to_numpy(dtype=np.float64)
    s = stats[view_columns("view_std", n_views)].to_numpy(dtype=np.float64)
    ok = np.nan_to_num(m, nan=-1) >= min_mean
    ok &= np.nan_to_num(s, nan=-1) >= min_std
    return np.asarray(ok, dtype=np.bool_)


def usable_views(
    stats: pd.DataFrame, *, min_mean: float, min_std: float, n_views: int = N_VIEWS
) -> NDArray[np.int64]:
    """Number of usable views per strip (:func:`usable_view_mask`)."""
    mask = usable_view_mask(stats, min_mean=min_mean, min_std=min_std, n_views=n_views)
    return np.asarray(mask.sum(axis=1), dtype=np.int64)


def check_filters(spec: Mapping[str, Any]) -> None:
    """Fail early on a config whose daylight/quality rules are missing."""
    for key in ("min_sun_elevation_deg", "min_view_mean", "min_view_std", "min_usable_views"):
        if key not in spec:
            raise KeyError(f"images.{key} is required")


__all__ = [
    "RECORD_S",
    "SPEC_COLUMNS",
    "check_filters",
    "crop_strip",
    "snapshot_stamp",
    "spatial_groups",
    "spec_records",
    "station_positions",
    "strip_stats",
    "usable_view_mask",
    "usable_views",
    "view_columns",
]
