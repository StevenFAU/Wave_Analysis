"""WebCOOS (Webcam Coastal Observation System; SECOORA / NOAA IOOS) adapter.

WebCOOS runs coastal webcams with archived one-minute stills and video. Five
of its North Carolina cameras (``currituck_hampton_inn``, ``jennette_north``,
``jennette_south``, ``oakisland_east``, ``oakisland_west``) are the cameras of
the Yin et al. (2025) wave-breaking dataset, the held-out test set of Kamagata
et al. (2026). See ``docs/datasets/webcoos.md``.

API (verified 2026-09-26)
-------------------------
Base ``https://app.webcoos.org/webcoos/api/v1``; every request needs the header
``Authorization: Token <key>`` (HTTP 403 without it)::

    /assets/                                  cameras (paginated): feeds -> products -> services
    /services/<service>/inventory/            time bins with image counts and bytes (slow)
    /elements/?service=<service>&starting_after=<t>&starting_before=<t>
                                              image files (paginated), with capture time and size

An element's ``data.extents.temporal.min`` is the capture time to the second
(e.g. ``oakisland_west-2026-09-25-150048Z.jpg`` at 15:00:48 UTC), and
``data.properties.url`` points to a public S3 bucket. Image files are therefore
downloaded **without** the token; the client only sends it to the API host.

The token is read from ``$WEBCOOS_API_TOKEN`` or from
``~/.config/wave-analysis/webcoos_token`` (which must not be readable by
others). It is never logged or written to a ledger.

Usage guidelines (https://webcoos.org/usage-guidelines, read 2026-09-26)
------------------------------------------------------------------------
* "Download only a reasonable amount of data. We may ban access if your use is
  unreasonable or creates a strain on system resources."
* "If you would like to download a large amount of data that is greater than
  90 days old, please contact webcoos@secoora.org first." :func:`check_age`
  enforces this: more than :data:`HISTORICAL_SAMPLE_LIMIT` images older than
  90 days need an explicit ``historical_approved`` flag.
* Research publications acknowledge SECOORA (:data:`ACKNOWLEDGEMENT`) and cite
  "WebCOOS [year(s) of imagery]".
"""

from __future__ import annotations

import os
import stat
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np
import pandas as pd

from wave_analysis.ingest import archive
from wave_analysis.ingest.archive import ArchiveItem, ArchiveSummary, Sink
from wave_analysis.ingest.downloader import USER_AGENT, Downloader

DATASET_ID = "webcoos"
API_BASE = "https://app.webcoos.org/webcoos/api/v1"
API_HOST = "app.webcoos.org"
USAGE_GUIDELINES_URL = "https://webcoos.org/usage-guidelines"
CONTACT = "webcoos@secoora.org"

TOKEN_ENV = "WEBCOOS_API_TOKEN"
TOKEN_FILE = Path("~/.config/wave-analysis/webcoos_token")

#: Age beyond which WebCOOS asks to be contacted before large downloads.
HISTORICAL_AGE_DAYS = 90
#: Images older than that which may be fetched without approval (a sample).
HISTORICAL_SAMPLE_LIMIT = 500

#: Elements per page requested (``page_size``; verified to work, default 100).
ELEMENTS_PAGE_SIZE = 1000

#: Product whose service holds the one-minute stills.
STILLS_PRODUCT = "one-minute-stills"

#: The Yin et al. (2025) cameras.
YIN2025_CAMERAS = (
    "currituck_hampton_inn",
    "jennette_north",
    "jennette_south",
    "oakisland_east",
    "oakisland_west",
)

ACKNOWLEDGEMENT = (
    '"SECOORA is acknowledged for leading the WebCOOS program, which collected '
    '[images of interest] in the WebCOOS dataset in this study," citing WebCOOS as '
    '"WebCOOS [year(s) of imagery]"'
)

LISTINGS_DIR = "_listings"
MANIFESTS_DIR = "_manifests"


class TokenError(RuntimeError):
    """The API token is missing or stored insecurely."""


class HistoricalAccessError(RuntimeError):
    """A request exceeds what the usage guidelines allow without contacting WebCOOS."""


def load_token(env: Mapping[str, str] | None = None, path: Path | None = None) -> str:
    """The API token from ``$WEBCOOS_API_TOKEN`` or the token file.

    Raises :class:`TokenError` if neither is set, or if the file can be read by
    group or others (as ssh does for private keys). Messages never contain the
    token.
    """
    env = os.environ if env is None else env
    value = env.get(TOKEN_ENV, "").strip()
    if value:
        return value
    p = (path or TOKEN_FILE).expanduser()
    if not p.exists():
        raise TokenError(
            f"no WebCOOS token: set ${TOKEN_ENV} or save it to {p} (see docs/datasets/webcoos.md)"
        )
    if p.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise TokenError(f"{p} is readable by others; run: chmod 600 {p}")
    value = p.read_text(encoding="utf-8").strip()
    if not value:
        raise TokenError(f"{p} is empty")
    return value


class WebCOOSClient:
    """Minimal WebCOOS API client: throttled, retrying, and host-locked.

    The token is sent only to :data:`API_HOST`; pagination links pointing
    anywhere else raise instead of being followed.
    """

    def __init__(
        self,
        token: str,
        *,
        min_interval_s: float = 1.0,
        timeout_s: float = 600.0,
        max_retries: int = 3,
        backoff_s: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._last = 0.0
        self._client = httpx.Client(
            headers={
                "Authorization": f"Token {token}",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            timeout=timeout_s,
            follow_redirects=False,
            transport=transport,
        )

    def close(self) -> None:
        """Close the connection pool."""
        self._client.close()

    def __enter__(self) -> WebCOOSClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def get_json(self, url: str, params: Mapping[str, str] | None = None) -> Any:
        """GET an API URL (absolute, or relative to :data:`API_BASE`) and decode JSON."""
        if not url.startswith("https://"):
            url = f"{API_BASE}/{url.lstrip('/')}"
        if urlsplit(url).hostname != API_HOST:
            raise ValueError(f"refusing to send the WebCOOS token to {urlsplit(url).hostname}")
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self._client.get(url, params=params)
            except httpx.TransportError as exc:
                last = exc
            else:
                if resp.status_code in (401, 403):
                    raise TokenError(f"WebCOOS rejected the token (HTTP {resp.status_code})")
                if resp.status_code not in (429, 500, 502, 503, 504):
                    resp.raise_for_status()
                    return resp.json()
                last = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            if attempt < self.max_retries:
                time.sleep(self.backoff_s * 2**attempt)
        assert last is not None
        raise last

    def paginate(self, path: str, params: Mapping[str, str] | None = None) -> Iterator[Any]:
        """Yield ``results`` items across all pages."""
        data = self.get_json(path, params)
        while True:
            yield from data["results"]
            nxt = (data.get("pagination") or {}).get("next")
            if not nxt:
                return
            data = self.get_json(nxt)

    def assets(self) -> list[dict[str, Any]]:
        """All camera assets."""
        return list(self.paginate("assets/"))

    def inventory(self, service: str) -> dict[str, Any]:
        """A service's inventory (time bins with counts and bytes). Can take minutes."""
        return dict(self.get_json(f"services/{service}/inventory/"))

    def elements(
        self, service: str, start: pd.Timestamp | str, end: pd.Timestamp | str
    ) -> list[dict[str, Any]]:
        """Image elements of a service with capture times in ``(start, end)``."""
        params = {
            "service": service,
            "starting_after": _iso(start),
            "starting_before": _iso(end),
            # 1000 per page (the default is 100): a tenth of the requests.
            "page_size": str(ELEMENTS_PAGE_SIZE),
        }
        return list(self.paginate("elements/", params))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def camera_table(assets: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per camera: slug, label, position, region, status and its services.

    ``stills_service`` is the service of the one-minute stills product, if any.
    """
    rows = []
    for a in assets:
        d = a["data"]
        c, p = d["common"], d.get("properties") or {}
        services: dict[str, str] = {}
        for feed in a.get("feeds") or []:
            for prod in feed.get("products") or []:
                slug = prod["data"]["common"]["slug"]
                for svc in prod.get("services") or []:
                    services.setdefault(slug, svc["data"]["common"]["slug"])
        coords = ((p.get("location") or {}).get("coordinates") or [np.nan, np.nan])[:2]
        rows.append(
            {
                "camera": c["slug"],
                "label": c.get("label", ""),
                "latitude": float(coords[1]),
                "longitude": float(coords[0]),
                "state": p.get("state_or_territory"),
                "region": p.get("region"),
                "group": p.get("group"),
                "timezone": p.get("timezone"),
                "access_level": c.get("access_level"),
                "status": (a.get("disposition") or {}).get("slug"),
                "stills_service": services.get(STILLS_PRODUCT),
                "products": ",".join(sorted(services)),
            }
        )
    return pd.DataFrame(rows).sort_values("camera").reset_index(drop=True)


def element_table(elements: list[dict[str, Any]], camera: str) -> pd.DataFrame:
    """Image elements as a table: capture time (UTC), file name, URL, size, id.

    Raises if a file name is not a plain name, a URL is not HTTPS, or a time is
    missing, since those would make the archive path or the pairing unsafe.
    """
    rows = []
    for e in elements:
        d = e["data"]
        name = d["common"]["label"]
        url = d["properties"]["url"]
        if "/" in name or name.startswith(".") or not name:
            raise ValueError(f"unsafe element file name {name!r}")
        if urlsplit(url).scheme != "https":
            raise ValueError(f"element URL is not HTTPS: {url}")
        size = d["properties"].get("size")
        rows.append(
            {
                "camera": camera,
                "time_utc": pd.Timestamp(d["extents"]["temporal"]["min"]).tz_convert("UTC"),
                "name": name,
                "url": url,
                "size_bytes": int(size) if size not in (None, "", "None") else pd.NA,
                "element_id": e.get("uuid") or d.get("uuid"),
            }
        )
    out = pd.DataFrame(
        rows, columns=["camera", "time_utc", "name", "url", "size_bytes", "element_id"]
    )
    out["time_utc"] = pd.to_datetime(out["time_utc"], utc=True)
    out["size_bytes"] = out["size_bytes"].astype("Int64")
    return out.sort_values("time_utc").drop_duplicates("url").reset_index(drop=True)


def select_on_grid(
    elements: pd.DataFrame,
    *,
    every: str | pd.Timedelta = "30min",
    tolerance: str | pd.Timedelta = "5min",
    offset: str | pd.Timedelta = "0min",
) -> pd.DataFrame:
    """The image nearest each grid time (``every``, shifted by ``offset``).

    Grid times without an image within ``tolerance`` are skipped; an image is
    used at most once. Adds ``grid_time`` and ``offset_s`` (capture time minus
    grid time). Pairing with wave records uses the exact capture time; the grid
    only thins one-minute stills to the cadence of the labels.
    """
    if elements.empty:
        return elements.assign(grid_time=pd.Series(dtype="datetime64[ns, UTC]"), offset_s=0.0)
    step, tol, off = pd.Timedelta(every), pd.Timedelta(tolerance), pd.Timedelta(offset)
    t = pd.DatetimeIndex(elements["time_utc"])
    first = (t.min() - off).floor(step) + off
    grid = pd.date_range(first, t.max() + tol, freq=step)
    tv, gv = _ns(t), _ns(grid)
    tol_ns = int(tol / pd.Timedelta(1, "ns"))
    pos = np.searchsorted(tv, gv)
    chosen: dict[int, pd.Timestamp] = {}
    for k in range(len(gv)):
        best, best_d = -1, tol_ns + 1
        for j in (int(pos[k]) - 1, int(pos[k])):
            if 0 <= j < len(tv) and abs(int(tv[j]) - int(gv[k])) < best_d:
                best, best_d = j, abs(int(tv[j]) - int(gv[k]))
        if best >= 0 and best not in chosen:
            chosen[best] = grid[k]
    idx = sorted(chosen)
    out = elements.iloc[idx].copy()
    out["grid_time"] = pd.DatetimeIndex([chosen[i] for i in idx])
    out["offset_s"] = (out["time_utc"] - out["grid_time"]).dt.total_seconds()
    return out.reset_index(drop=True)


def check_age(
    times: pd.Series,
    *,
    now: pd.Timestamp | None = None,
    historical_approved: bool = False,
    sample_limit: int = HISTORICAL_SAMPLE_LIMIT,
) -> int:
    """Enforce the 90-day guideline; return the number of images older than 90 days.

    Raises :class:`HistoricalAccessError` if more than ``sample_limit`` images
    are older than :data:`HISTORICAL_AGE_DAYS` and WebCOOS has not approved the
    download (``historical_approved``).
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    old = int(
        (pd.to_datetime(times, utc=True) < now - pd.Timedelta(days=HISTORICAL_AGE_DAYS)).sum()
    )
    if old > sample_limit and not historical_approved:
        raise HistoricalAccessError(
            f"{old:,} images are older than {HISTORICAL_AGE_DAYS} days. WebCOOS asks to be "
            f"contacted first ({CONTACT}; {USAGE_GUIDELINES_URL}). Once they agree, rerun "
            f"with --historical-approved. Up to {sample_limit} such images may be fetched "
            "as a sample without it."
        )
    return old


# --------------------------------------------------------------------------- #
# Archive
# --------------------------------------------------------------------------- #


def local_path(dest_root: Path, camera: str, name: str, time_utc: pd.Timestamp) -> Path:
    """``<root>/<camera>/<YYYY>/<MM>/<DD>/<name>`` by UTC capture date."""
    if "/" in camera or "/" in name or camera.startswith(".") or name.startswith("."):
        raise ValueError(f"unsafe camera or file name: {camera!r}, {name!r}")
    t = pd.Timestamp(time_utc).tz_convert("UTC")
    return dest_root / camera / f"{t:%Y}" / f"{t:%m}" / f"{t:%d}" / name


def ledger_path(dest_root: Path, camera: str) -> Path:
    """Per-camera request ledger."""
    if "/" in camera or camera.startswith("."):
        raise ValueError(f"unsafe camera name {camera!r}")
    return dest_root / MANIFESTS_DIR / f"{camera}.csv"


def archive_stills(
    downloader: Downloader,
    selected: pd.DataFrame,
    dest_root: Path,
    *,
    sink: Sink,
    limit: int | None = None,
    progress: Any = None,
    progress_every: int = 200,
) -> ArchiveSummary:
    """Download selected stills of one camera that are not yet archived.

    ``downloader`` must not carry the API token: the files are public.
    """
    cameras = selected["camera"].unique()
    if len(cameras) != 1:
        raise ValueError("selection must hold exactly one camera")
    camera = str(cameras[0])
    paths = {
        url: local_path(dest_root, camera, name, t)
        for url, name, t in zip(
            selected["url"], selected["name"], selected["time_utc"], strict=True
        )
    }
    items = [
        ArchiveItem(
            url,
            paths[url],
            pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ"),
            None if pd.isna(size) else int(size),
        )
        for url, t, size in zip(
            selected["url"], selected["time_utc"], selected["size_bytes"], strict=True
        )
    ]
    # Only the selected URLs need a path: ledger rows for other URLs cannot match.
    unknown = dest_root / camera / ".not-selected"
    return archive.archive_files(
        downloader,
        items,
        ledger=ledger_path(dest_root, camera),
        path_for=lambda url: paths.get(url, unknown),
        source_id=DATASET_ID,
        collection=camera,
        sink=sink,
        limit=limit,
        progress=progress,
        progress_every=progress_every,
    )


def write_listing(selected: pd.DataFrame, dest_root: Path, camera: str) -> Path:
    """Save the selected elements (the run's index) under ``_listings/``."""
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    path = dest_root / LISTINGS_DIR / f"{camera}_{stamp}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(path, index=False)
    return path


def _ns(t: pd.DatetimeIndex) -> np.ndarray[Any, np.dtype[np.int64]]:
    """UTC nanoseconds since the epoch, whatever the index's resolution."""
    naive = t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t
    return np.asarray(naive.as_unit("ns").to_numpy(dtype="datetime64[ns]").astype(np.int64))


def _iso(t: pd.Timestamp | str) -> str:
    ts = pd.Timestamp(t)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return f"{ts:%Y-%m-%dT%H:%M:%SZ}"
