"""Recent sea state for the dashboard, read from the head of NDBC realtime files.

NDBC realtime2 text files (``<station>.txt``, ``<station>.spec``) list the newest
rows first and the server honours HTTP range requests. One small request per
station therefore returns its latest observations: about 6 KB per station per
hour in steady state, instead of the full 45-day file (~600 KB). Rows are merged
into a rolling local cache (``KEEP_DAYS``), so a missed hour is filled on the
next run, and an outage of up to about a week is refilled by requesting up to
``MAX_BYTES``.

These values are **provisional realtime data for display only**. They are not
quality-controlled and are not the project's research record: labels come from
the archived realtime snapshots and NDBC historical files (ADR 0003, 0008).
NDBC sends no CORS headers, so the browser cannot fetch these files itself;
that is why the collector host prepares them.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from wave_analysis.ingest.downloader import USER_AGENT
from wave_analysis.sources.ndbc.parsers import parse_spectral_summary, parse_stdmet

REALTIME_BASE = "https://www.ndbc.noaa.gov/data/realtime2"
KEEP_DAYS = 8
PUBLISH_HOURS = 84  # the 72 h image window plus half a day of context
MIN_BYTES = 6 * 1024
MAX_BYTES = 96 * 1024


@dataclass(frozen=True)
class Product:
    """A realtime file whose head is cached."""

    name: str
    ext: str
    rows_per_hour: float  # upper bound on the reporting rate
    bytes_per_row: int
    variables: tuple[str, ...]
    decimals: Mapping[str, int] = field(default_factory=dict)


PRODUCTS: tuple[Product, ...] = (
    Product(
        "stdmet",
        "txt",
        6.0,
        95,
        ("WVHT", "DPD", "APD", "MWD", "WSPD", "GST", "WDIR", "PRES", "ATMP", "WTMP"),
        {
            "WVHT": 2,
            "DPD": 1,
            "APD": 1,
            "MWD": 0,
            "WSPD": 1,
            "GST": 1,
            "WDIR": 0,
            "PRES": 1,
            "ATMP": 1,
            "WTMP": 1,
        },
    ),
    Product(
        "spec",
        "spec",
        2.0,
        75,
        ("SwH", "SwP", "SwD_deg", "WWH", "WWP", "WWD_deg", "STEEPNESS"),
        {"SwH": 2, "SwP": 1, "SwD_deg": 1, "WWH": 2, "WWP": 1, "WWD_deg": 1},
    ),
)

UNITS = {
    "WVHT": "m", "DPD": "s", "APD": "s", "MWD": "degT", "WSPD": "m/s", "GST": "m/s",
    "WDIR": "degT", "PRES": "hPa", "ATMP": "degC", "WTMP": "degC", "SwH": "m", "SwP": "s",
    "SwD_deg": "degT", "WWH": "m", "WWP": "s", "WWD_deg": "degT", "STEEPNESS": "category",
}  # fmt: skip

# (url, n_bytes) -> body, or None when the file does not exist (HTTP 404).
Fetcher = Callable[[str, int], bytes | None]


class RangeFetcher:
    """HTTP range fetcher with the project User-Agent and a politeness interval."""

    def __init__(self, *, min_interval_s: float = 1.0, timeout_s: float = 30.0) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=timeout_s, follow_redirects=True
        )
        self._min_interval = min_interval_s
        self._last = 0.0

    def __call__(self, url: str, n_bytes: int) -> bytes | None:
        """First ``n_bytes`` of ``url``; ``None`` if the file does not exist."""
        wait = self._min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        # NDBC ignores Range on gzip-encoded responses and sends the whole file,
        # so ask for identity encoding.
        r = self._client.get(
            url, headers={"Range": f"bytes=0-{n_bytes - 1}", "Accept-Encoding": "identity"}
        )
        if r.status_code == 404:
            return None
        if r.status_code not in (200, 206):
            r.raise_for_status()
            raise httpx.HTTPStatusError(
                f"unexpected status {r.status_code}", request=r.request, response=r
            )
        return r.content

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


def complete_lines(data: bytes) -> str:
    """Decode a range response and drop the (usually cut) final partial line."""
    text = data.decode("utf-8", errors="replace")
    cut = text.rfind("\n")
    return text[: cut + 1] if cut >= 0 else ""


def parse_head(text: str, product: Product) -> pd.DataFrame:
    """Parse the head of a realtime file into ``station``-less rows indexed by time."""
    if not any(ln and not ln.startswith("#") for ln in text.splitlines()):
        return pd.DataFrame(columns=list(product.variables))
    data = text.encode("utf-8")  # the parsers read a str argument as a file path
    table = (
        parse_stdmet(data, realtime=True) if product.ext == "txt" else parse_spectral_summary(data)
    )
    df = table.data.reindex(columns=list(product.variables))
    return df[~df.index.duplicated(keep="first")].sort_index()


def bytes_needed(last: pd.Timestamp | None, now: pd.Timestamp, product: Product) -> int:
    """Range size covering the rows published since ``last`` (with margin)."""
    if last is None or pd.isna(last):
        return MAX_BYTES
    hours = max((now - last).total_seconds() / 3600.0, 0.0) + 2.0
    n = int(hours * product.rows_per_hour * product.bytes_per_row) + 400
    return int(min(max(n, MIN_BYTES), MAX_BYTES))


def cache_path(cache_dir: Path, product: Product) -> Path:
    """Parquet cache file for one product."""
    return cache_dir / f"{product.name}.parquet"


def load_cache(cache_dir: Path, product: Product) -> pd.DataFrame:
    """Cached rows for one product: columns ``station_id``, ``time_utc``, variables."""
    p = cache_path(cache_dir, product)
    if not p.exists():
        return pd.DataFrame(columns=["station_id", "time_utc", *product.variables])
    return pd.read_parquet(p)


def _merge(old: pd.DataFrame, new: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    frames = [f for f in (old, new) if not f.empty]
    if not frames:
        return old
    both = pd.concat(frames, ignore_index=True)
    both["time_utc"] = pd.to_datetime(both["time_utc"], utc=True)
    both = both.drop_duplicates(["station_id", "time_utc"], keep="last")
    both = both[both["time_utc"] >= now - pd.Timedelta(days=KEEP_DAYS)]
    return both.sort_values(["station_id", "time_utc"]).reset_index(drop=True)


@dataclass
class RefreshSummary:
    """Outcome of one :func:`refresh` run (published in ``status.json``)."""

    requests: int = 0
    rows: int = 0
    missing_files: int = 0
    bytes_received: int = 0
    oversized: int = 0  # responses larger than requested (range ignored upstream)
    failures: list[str] = field(default_factory=list)


def refresh(
    station_ids: Iterable[str],
    cache_dir: Path,
    fetch: Fetcher,
    *,
    now: pd.Timestamp | None = None,
    products: Iterable[Product] = PRODUCTS,
) -> RefreshSummary:
    """Fetch new rows for every station and product and merge them into the cache.

    A failure for one station is recorded and does not stop the others.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary = RefreshSummary()
    ids = sorted({s.upper() for s in station_ids})
    for product in products:
        cache = load_cache(cache_dir, product)
        last_by_station = (
            pd.to_datetime(cache["time_utc"], utc=True).groupby(cache["station_id"]).max()
            if not cache.empty
            else pd.Series(dtype="datetime64[ns, UTC]")
        )
        new_frames = []
        for sid in ids:
            url = f"{REALTIME_BASE}/{sid}.{product.ext}"
            n = bytes_needed(last_by_station.get(sid), now, product)
            summary.requests += 1
            try:
                body = fetch(url, n)
            except (httpx.HTTPError, OSError) as exc:
                summary.failures.append(f"{sid}.{product.ext}: {exc}")
                continue
            if body is None:
                summary.missing_files += 1
                continue
            summary.bytes_received += len(body)
            if len(body) > n:
                summary.oversized += 1
            try:
                rows = parse_head(complete_lines(body), product)
            except (ValueError, IndexError) as exc:
                summary.failures.append(f"{sid}.{product.ext}: parse error: {exc}")
                continue
            if rows.empty:
                continue
            rows = rows.rename_axis("time_utc").reset_index().assign(station_id=sid)
            new_frames.append(rows)
            summary.rows += len(rows)
        new = pd.concat(new_frames, ignore_index=True) if new_frames else cache.iloc[0:0]
        merged = _merge(cache, new, now)
        tmp = cache_path(cache_dir, product).with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(cache_path(cache_dir, product))
    return summary


def _series(values: pd.Series, decimals: int | None) -> list[Any]:
    if not pd.api.types.is_numeric_dtype(values):
        return [None if pd.isna(v) else str(v) for v in values]
    arr = values.to_numpy(dtype="float64")
    if decimals is not None:
        arr = np.round(arr, decimals)
    return [None if not np.isfinite(v) else (int(v) if decimals == 0 else float(v)) for v in arr]


def station_payload(
    station_id: str,
    caches: Mapping[str, pd.DataFrame],
    *,
    now: pd.Timestamp,
    hours: int = PUBLISH_HOURS,
) -> dict[str, Any] | None:
    """Recent rows for one station as column arrays (epoch seconds for time)."""
    since = now - pd.Timedelta(hours=hours)
    out: dict[str, Any] = {"station_id": station_id, "products": {}}
    for product in PRODUCTS:
        df = caches.get(product.name)
        if df is None or df.empty:
            continue
        g = df[
            (df["station_id"] == station_id) & (pd.to_datetime(df["time_utc"], utc=True) >= since)
        ]
        g = g.dropna(how="all", subset=[v for v in product.variables if v in g])
        if g.empty:
            continue
        t = pd.to_datetime(g["time_utc"], utc=True)
        block: dict[str, Any] = {"t": (t.astype("int64") // 10**9).astype(int).tolist()}
        for v in product.variables:
            if v in g:
                block[v] = _series(g[v], product.decimals.get(v))
        out["products"][product.name] = block
    if not out["products"]:
        return None
    out["units"] = {v: UNITS[v] for p in PRODUCTS for v in p.variables}
    return out


# Groups reported together: (product, key variable, variables taken from that row).
_LATEST_GROUPS = (
    ("stdmet", "WVHT", ("WVHT", "DPD", "APD", "MWD"), "wave_time"),
    ("stdmet", "WSPD", ("WSPD", "GST", "WDIR"), "wind_time"),
    ("stdmet", "WTMP", ("WTMP", "ATMP"), "temp_time"),
    ("spec", "SwH", ("SwH", "SwP", "SwD_deg", "WWH", "WWP", "WWD_deg", "STEEPNESS"), "spec_time"),
)


def latest_values(
    caches: Mapping[str, pd.DataFrame], *, now: pd.Timestamp, max_age_hours: float = 3.0
) -> dict[str, dict[str, Any]]:
    """Latest observation per station, per group of co-reported variables.

    Wave, wind, temperature and spectral-summary values are each taken from the
    newest row in which their key variable is present (NDBC reports them at
    different minutes), and only if that row is younger than ``max_age_hours``.
    """
    since = now - pd.Timedelta(hours=max_age_hours)
    decimals = {v: d for p in PRODUCTS for v, d in p.decimals.items()}
    out: dict[str, dict[str, Any]] = {}
    for product_name, key, variables, time_key in _LATEST_GROUPS:
        df = caches.get(product_name)
        if df is None or df.empty or key not in df:
            continue
        df = df[(pd.to_datetime(df["time_utc"], utc=True) >= since) & df[key].notna()]
        if df.empty:
            continue
        latest = df.sort_values("time_utc").groupby("station_id").tail(1)
        for row in latest.itertuples(index=False):
            rec = out.setdefault(str(row.station_id), {})
            rec[time_key] = (
                pd.Timestamp(str(row.time_utc)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            for v in variables:
                val = getattr(row, v, None)
                if val is None or (not isinstance(val, str) and pd.isna(val)):
                    continue
                rec[v] = val if isinstance(val, str) else round(float(val), decimals.get(v, 2))
    return out
