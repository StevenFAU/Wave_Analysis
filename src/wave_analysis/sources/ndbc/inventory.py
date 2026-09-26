"""NDBC station / product / period availability inventory.

This is the project's *first coding target* (repository specification, "Recommended
First Coding Target"): before any model is chosen, determine which
multimodal datasets can realistically be built.

Rule: **availability belongs to station + product + period**, never to the
provider as a whole. The inventory is therefore a long table with one row per
(station, product, year) for the historical archive and one row per
(station, product) for the 45-day realtime window, plus a separate snapshot of
which stations currently expose a buoy camera.

Sources (all public, no authentication):

* historical directory indexes ``/data/historical/{dir}/`` (file name, last
  modified, size) - one request per product directory;
* realtime directory index ``/data/realtime2/``;
* station metadata ``activestations.xml`` and ``station_table.txt``;
* buoy-camera listing ``buoycams.php`` (JSON). NDBC does not publish a
  historical camera image archive; camera availability is therefore recorded
  as a dated *snapshot* and must be re-inventoried over time to build history.

Every index page fetched is saved byte-for-byte and recorded in the manifest,
so an inventory can be reproduced and diffed against a later one.
"""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import ManifestEntry
from wave_analysis.sources.ndbc.products import (
    BUOYCAM_IMAGE_URL,
    BUOYCAM_URL,
    HISTORICAL_URL,
    PRODUCTS,
    REALTIME_URL,
    SPECTRAL_SET,
    STATION_TABLE_URL,
    STATION_XML_URL,
    Category,
)

# Apache autoindex row: <a href="41010h2023.txt.gz">...</a></td><td align="right">2024-02-06 16:35  </td><td align="right">219K</td>
_LISTING_ROW = re.compile(
    r'<a href="(?P<name>[^"?/]+)">[^<]*</a>\s*</td>\s*<td[^>]*>\s*(?P<modified>\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s*</td>\s*<td[^>]*>\s*(?P<size>[^<]*?)\s*</td>'
)
# ``{station}{code}[b]{YYYY}.txt.gz``. The optional ``b`` marks a *second segment* for the
# same station-year, typically after a payload change with a different frequency grid
# (e.g. swr1/41013j2005 on the 0.03-Hz grid vs 41013jb2005 on the 47-band grid from Aug 2005).
HISTORICAL_FILENAME = re.compile(
    r"^(?P<station>[0-9A-Za-z]+?)(?P<code>[a-z])(?P<variant>b?)(?P<year>\d{4})\.txt\.gz$"
)
_RT_NAME = re.compile(r"^(?P<station>[0-9A-Za-z]+)\.(?P<ext>[a-z_0-9]+)$")


def _size_to_bytes(s: str) -> float:
    s = s.strip()
    if not s or s == "-":
        return np.nan
    mult = {"K": 1024, "M": 1024**2, "G": 1024**3}
    if s[-1] in mult:
        return float(s[:-1]) * mult[s[-1]]
    return float(s)


def parse_directory_listing(page: str) -> pd.DataFrame:
    """Parse an Apache autoindex page into ``name, last_modified_utc, size_bytes``.

    NDBC's server reports listing times without a zone; they are recorded as
    naive strings converted to UTC by assumption (documented in the output
    metadata) - they are used only for change detection, not for science.
    """
    rows = [
        (m["name"], m["modified"], _size_to_bytes(html.unescape(m["size"])))
        for m in _LISTING_ROW.finditer(page)
    ]
    df = pd.DataFrame(rows, columns=["name", "last_modified", "size_bytes"])
    df["last_modified"] = pd.to_datetime(df["last_modified"], format="%Y-%m-%d %H:%M", utc=True)
    return df


def historical_listing_to_inventory(listing: pd.DataFrame, directory: str) -> pd.DataFrame:
    """Convert one historical directory listing into inventory rows."""
    product = next(p for p in PRODUCTS.values() if p.historical_dir == directory)
    recs = []
    unmatched = []
    for name, modified, size in listing.itertuples(index=False):
        m = HISTORICAL_FILENAME.match(name)
        if not m or m["code"] != product.historical_code:
            unmatched.append(name)
            continue
        recs.append(
            {
                "station_id": m["station"].upper(),
                "product": product.key,
                "category": product.category.value,
                "period": m["year"] + m["variant"],
                "segment": m["variant"] or "a",
                "year": int(m["year"]),
                "archive": "historical",
                "filename": name,
                "url": f"{HISTORICAL_URL}/{directory}/{name}",
                "last_modified": modified,
                "size_bytes": size,
            }
        )
    df = pd.DataFrame.from_records(recs)
    df.attrs["unmatched"] = unmatched
    return df


def realtime_listing_to_inventory(listing: pd.DataFrame) -> pd.DataFrame:
    """Convert the ``realtime2`` directory listing into inventory rows."""
    ext_to_product = {p.realtime_ext: p for p in PRODUCTS.values() if p.realtime_ext}
    recs = []
    for name, modified, size in listing.itertuples(index=False):
        m = _RT_NAME.match(name)
        if not m or m["ext"] not in ext_to_product:
            continue
        p = ext_to_product[m["ext"]]
        recs.append(
            {
                "station_id": m["station"].upper(),
                "product": p.key,
                "category": p.category.value,
                "period": "realtime",
                "year": pd.NA,
                "archive": "realtime",
                "filename": name,
                "url": f"{REALTIME_URL}/{name}",
                "last_modified": modified,
                "size_bytes": size,
            }
        )
    return pd.DataFrame.from_records(recs)


def parse_active_stations(xml_text: str) -> pd.DataFrame:
    """Parse ``activestations.xml`` into a station table."""
    root = ET.fromstring(xml_text)
    rows = []
    for st in root.iter("station"):
        a = st.attrib
        rows.append(
            {
                "station_id": a.get("id", "").upper(),
                "name": a.get("name"),
                "latitude": float(a["lat"]) if a.get("lat") else np.nan,
                "longitude": float(a["lon"]) if a.get("lon") else np.nan,
                "elevation_m": float(a["elev"]) if a.get("elev") else np.nan,
                "owner": a.get("owner"),
                "program": a.get("pgm"),
                "station_type": a.get("type"),
                "has_met": a.get("met") == "y",
                "has_currents": a.get("currents") == "y",
                "has_water_quality": a.get("waterquality") == "y",
                "is_dart": a.get("dart") == "y",
            }
        )
    return pd.DataFrame(rows)


def parse_station_table(text: str) -> pd.DataFrame:
    """Parse ``station_table.txt`` (pipe-delimited) for hull and payload metadata."""
    rows = []
    for ln in text.splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        parts = [html.unescape(p.strip()) for p in ln.split("|")]
        parts += [""] * (10 - len(parts))
        rows.append(
            {
                "station_id": parts[0].upper(),
                "owner_code": parts[1] or None,
                "platform_type": parts[2] or None,
                "hull": parts[3] or None,
                "table_name": parts[4] or None,
                "payload": parts[5] or None,
                "location_text": parts[6] or None,
                "timezone_code": parts[7] or None,
            }
        )
    return pd.DataFrame(rows).drop_duplicates("station_id")


def parse_buoycams(json_text: str, snapshot_utc: pd.Timestamp) -> pd.DataFrame:
    """Parse the ``buoycams.php`` JSON listing into a camera snapshot table."""
    rows = []
    for item in json.loads(json_text):
        img = item.get("img") or ""
        rows.append(
            {
                "station_id": str(item.get("id", "")).upper(),
                "camera_name": item.get("name"),
                "latitude": item.get("lat"),
                "longitude": item.get("lng"),
                "latest_image": img,
                "latest_image_url": f"{BUOYCAM_IMAGE_URL}/{img}" if img else None,
                "image_width": item.get("width"),
                "image_height": item.get("height"),
                "snapshot_utc": snapshot_utc,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass
class InventoryResult:
    """Outputs of a full inventory run."""

    files: pd.DataFrame  # long: station, product, period, url, size, modified
    stations: pd.DataFrame  # station metadata (merged XML + table)
    cameras: pd.DataFrame  # camera snapshot
    manifest: list[ManifestEntry]
    snapshot_utc: pd.Timestamp

    def availability_matrix(self) -> pd.DataFrame:
        """Station x product matrix of the number of historical years available."""
        hist = self.files[self.files["archive"] == "historical"]
        return (
            hist.groupby(["station_id", "product"])["year"]
            .nunique()
            .unstack(fill_value=0)
            .sort_index()
        )

    def station_summary(self) -> pd.DataFrame:
        """One row per station with capability flags and year coverage per category."""
        f = self.files
        hist = f[f["archive"] == "historical"]
        rt = f[f["archive"] == "realtime"]
        years = hist.groupby(["station_id", "product"])["year"].agg(["min", "max", "nunique"])
        out = pd.DataFrame(index=pd.Index(sorted(set(f["station_id"])), name="station_id"))
        for prod in ("stdmet", *SPECTRAL_SET, "adcp", "adcp2", "ocean"):
            if prod in years.index.get_level_values("product"):
                y = years.xs(prod, level="product")
                out[f"{prod}_years"] = y["nunique"].reindex(out.index).fillna(0).astype(int)
                out[f"{prod}_first"] = y["min"].reindex(out.index)
                out[f"{prod}_last"] = y["max"].reindex(out.index)
            else:
                out[f"{prod}_years"] = 0
        # Directional-spectrum years: all five spectral products present in the same year.
        spec = hist[hist["product"].isin(SPECTRAL_SET)]
        full = spec.groupby(["station_id", "year"])["product"].nunique()
        full = full[full == len(SPECTRAL_SET)].groupby("station_id").size()
        out["directional_spectrum_years"] = full.reindex(out.index).fillna(0).astype(int)
        rt_products = rt.groupby("station_id")["product"].apply(lambda s: ",".join(sorted(set(s))))
        out["realtime_products"] = rt_products.reindex(out.index)
        out["has_camera_now"] = out.index.isin(set(self.cameras["station_id"]))
        out = out.join(self.stations.set_index("station_id"), how="left")
        return out


def run_inventory(
    downloader: Downloader,
    raw_root: Path,
    *,
    historical_dirs: Iterable[str] | None = None,
    include_realtime: bool = True,
) -> InventoryResult:
    """Fetch all index pages, save them under ``raw_root``, and build the inventory.

    Parameters
    ----------
    downloader
        Configured :class:`~wave_analysis.ingest.downloader.Downloader`.
    raw_root
        Directory for the raw index snapshots (e.g. ``data/raw/ndbc/index/<date>``).
    historical_dirs
        Subset of historical directories to scan; default is every product with
        a historical archive.
    """
    snapshot = pd.Timestamp.now(tz="UTC").floor("s")
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest: list[ManifestEntry] = []
    frames: list[pd.DataFrame] = []

    def get(url: str, name: str, product: str) -> str | None:
        entry = downloader.fetch(
            url,
            raw_root / name,
            source_id="ndbc",
            product=product,
            period=f"snapshot-{snapshot:%Y%m%d}",
        )
        manifest.append(entry)
        if entry.local_path is None:
            return None
        return Path(entry.local_path).read_text(encoding="utf-8", errors="replace")

    dirs = (
        list(historical_dirs)
        if historical_dirs is not None
        else [p.historical_dir for p in PRODUCTS.values() if p.historical_dir]
    )
    for d in dirs:
        page = get(f"{HISTORICAL_URL}/{d}/", f"historical_{d}.html", f"index:{d}")
        if page is not None:
            frames.append(historical_listing_to_inventory(parse_directory_listing(page), d))
    if include_realtime:
        page = get(f"{REALTIME_URL}/", "realtime2.html", "index:realtime2")
        if page is not None:
            frames.append(realtime_listing_to_inventory(parse_directory_listing(page)))

    xml_text = get(STATION_XML_URL, "activestations.xml", "stations:xml")
    table_text = get(STATION_TABLE_URL, "station_table.txt", "stations:table")
    cams_text = get(BUOYCAM_URL, "buoycams.json", "cameras:snapshot")

    stations = parse_active_stations(xml_text) if xml_text else pd.DataFrame(columns=["station_id"])
    if table_text:
        stations = stations.merge(parse_station_table(table_text), on="station_id", how="outer")
    cameras = (
        parse_buoycams(cams_text, snapshot) if cams_text else pd.DataFrame(columns=["station_id"])
    )
    files = (
        pd.concat([f for f in frames if not f.empty], ignore_index=True)
        if frames
        else pd.DataFrame()
    )
    if not files.empty:
        files["year"] = files["year"].astype("Int64")
        files["snapshot_utc"] = snapshot
    return InventoryResult(files, stations, cameras, manifest, snapshot)


def category_of(product: str) -> Category:
    """Category of a product key."""
    return PRODUCTS[product].category
