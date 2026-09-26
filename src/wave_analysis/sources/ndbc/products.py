"""Catalogue of NDBC data products and their upstream locations.

Verified 2026-09-25 against
https://www.ndbc.noaa.gov/faq/rt_data_access.shtml (realtime extensions) and
the directory index at https://www.ndbc.noaa.gov/data/historical/ (historical
directories and one-letter file codes, inferred from file names present in
each directory).

URL patterns
------------
Realtime (rolling last 45 days)::

    https://www.ndbc.noaa.gov/data/realtime2/{STATION}.{ext}

Historical (annual, gzipped)::

    https://www.ndbc.noaa.gov/data/historical/{dir}/{station}{code}{YYYY}.txt.gz

Historical file names are lower-case for most stations but some identifiers
contain upper-case letters (e.g. ``4cONFh2020.txt.gz``); always prefer the exact
file name recorded by the inventory over reconstructing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

BASE_URL = "https://www.ndbc.noaa.gov"
REALTIME_URL = f"{BASE_URL}/data/realtime2"
HISTORICAL_URL = f"{BASE_URL}/data/historical"
STATION_XML_URL = f"{BASE_URL}/activestations.xml"
STATION_TABLE_URL = f"{BASE_URL}/data/stations/station_table.txt"
BUOYCAM_URL = f"{BASE_URL}/buoycams.php"
BUOYCAM_IMAGE_URL = f"{BASE_URL}/images/buoycam"


class Category(StrEnum):
    """Broad product category, used to group availability in the inventory."""

    METEOROLOGICAL = "meteorological"
    WAVE_SUMMARY = "wave_summary"
    WAVE_SPECTRAL = "wave_spectral"
    WAVE_DIRECTIONAL = "wave_directional"
    CURRENT = "current"
    OCEANOGRAPHIC = "oceanographic"
    WATER_LEVEL = "water_level"
    OTHER = "other"
    IMAGERY = "imagery"


@dataclass(frozen=True)
class NDBCProduct:
    """One NDBC product type."""

    key: str
    description: str
    category: Category
    realtime_ext: str | None
    historical_dir: str | None
    historical_code: str | None
    parser: str | None = None  # name of parser function in ndbc.parsers
    note: str = ""


PRODUCTS: dict[str, NDBCProduct] = {
    p.key: p
    for p in (
        NDBCProduct(
            "stdmet",
            "Standard meteorological data (wind, pressure, temperature, bulk waves)",
            Category.METEOROLOGICAL,
            "txt",
            "stdmet",
            "h",
            "parse_stdmet",
        ),
        NDBCProduct(
            "spec",
            "Spectral wave summary (swell/wind-sea partition, steepness)",
            Category.WAVE_SUMMARY,
            "spec",
            None,
            None,
            "parse_spectral_summary",
            "Realtime only; no historical archive directory.",
        ),
        NDBCProduct(
            "swden",
            "Spectral wave density S(f)",
            Category.WAVE_SPECTRAL,
            "data_spec",
            "swden",
            "w",
            "parse_spectral",
        ),
        NDBCProduct(
            "swdir",
            "Directional: mean direction alpha1(f)",
            Category.WAVE_DIRECTIONAL,
            "swdir",
            "swdir",
            "d",
            "parse_spectral",
        ),
        NDBCProduct(
            "swdir2",
            "Directional: principal direction alpha2(f)",
            Category.WAVE_DIRECTIONAL,
            "swdir2",
            "swdir2",
            "i",
            "parse_spectral",
        ),
        NDBCProduct(
            "swr1",
            "Directional: first normalised polar Fourier coefficient r1(f)",
            Category.WAVE_DIRECTIONAL,
            "swr1",
            "swr1",
            "j",
            "parse_spectral",
            "Historical files are scaled by 100 (NDBC measdes); un-scaled on parse.",
        ),
        NDBCProduct(
            "swr2",
            "Directional: second normalised polar Fourier coefficient r2(f)",
            Category.WAVE_DIRECTIONAL,
            "swr2",
            "swr2",
            "k",
            "parse_spectral",
            "Historical files are scaled by 100 (NDBC measdes); un-scaled on parse.",
        ),
        NDBCProduct(
            "adcp",
            "Acoustic Doppler Current Profiler (current profile)",
            Category.CURRENT,
            "adcp",
            "adcp",
            "a",
            "parse_adcp",
        ),
        NDBCProduct(
            "adcp2",
            "ADCP extended format (per-bin diagnostics, echo intensity)",
            Category.CURRENT,
            None,
            "adcp2",
            "b",
            None,
            "Raw-only in v0.1; multi-line record format not yet parsed.",
        ),
        NDBCProduct(
            "cwind",
            "Continuous winds (10-min)",
            Category.METEOROLOGICAL,
            "cwind",
            "cwind",
            "c",
            "parse_stdmet",
        ),
        NDBCProduct(
            "ocean",
            "Oceanographic (T, S, O2, turbidity, ...)",
            Category.OCEANOGRAPHIC,
            "ocean",
            "ocean",
            "o",
            "parse_stdmet",
        ),
        NDBCProduct("srad", "Solar radiation", Category.OTHER, None, "srad", "r", "parse_stdmet"),
        NDBCProduct(
            "supl",
            "Supplemental measurements (hourly min pressure, max wind)",
            Category.METEOROLOGICAL,
            None,
            "supl",
            "s",
            "parse_stdmet",
        ),
        NDBCProduct(
            "tide", "Water level (realtime)", Category.WATER_LEVEL, "tide", None, None, None
        ),
        NDBCProduct(
            "wlevel", "Water level (historical)", Category.WATER_LEVEL, None, "wlevel", "l", None
        ),
        NDBCProduct(
            "dart", "DART tsunameter water-column height", Category.OTHER, None, "dart", "t", None
        ),
        NDBCProduct(
            "mmbcur", "Marsh-McBirney current meter", Category.CURRENT, None, "mmbcur", "m", None
        ),
        NDBCProduct("rain", "Hourly rain", Category.METEOROLOGICAL, None, "rain", "y", None),
        NDBCProduct("rain10", "10-minute rain", Category.METEOROLOGICAL, None, "rain10", "z", None),
        NDBCProduct("rain24", "24-hour rain", Category.METEOROLOGICAL, None, "rain24", "x", None),
        NDBCProduct(
            "drift",
            "Drifting-buoy meteorology",
            Category.METEOROLOGICAL,
            "drift",
            "drift",
            "e",
            None,
        ),
    )
}

#: Products that together form the full directional spectrum (NDBC 5-file set).
SPECTRAL_SET: tuple[str, ...] = ("swden", "swdir", "swdir2", "swr1", "swr2")

#: Products whose values are r1/r2 and are scaled x100 in historical files.
SCALED_R_PRODUCTS: frozenset[str] = frozenset({"swr1", "swr2"})


def realtime_url(station_id: str, product: str) -> str:
    """URL of a realtime (last 45 days) file."""
    p = PRODUCTS[product]
    if p.realtime_ext is None:
        raise ValueError(f"product {product!r} has no realtime file")
    return f"{REALTIME_URL}/{station_id}.{p.realtime_ext}"


def historical_filename(station_id: str, product: str, year: int) -> str:
    """Conventional historical file name (lower-cased station id)."""
    p = PRODUCTS[product]
    if p.historical_code is None:
        raise ValueError(f"product {product!r} has no historical archive")
    return f"{station_id.lower()}{p.historical_code}{year:04d}.txt.gz"


def historical_url(station_id: str, product: str, year: int, *, filename: str | None = None) -> str:
    """URL of an annual historical file; pass ``filename`` from the inventory when known."""
    p = PRODUCTS[product]
    if p.historical_dir is None:
        raise ValueError(f"product {product!r} has no historical archive")
    fname = filename or historical_filename(station_id, product, year)
    return f"{HISTORICAL_URL}/{p.historical_dir}/{fname}"


def product_for_historical_dir(directory: str) -> NDBCProduct:
    """Reverse lookup from historical directory name."""
    for p in PRODUCTS.values():
        if p.historical_dir == directory:
            return p
    raise KeyError(directory)
