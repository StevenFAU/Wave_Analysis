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

On-premise archive (email from WebCOOS, 2026-10-09)
---------------------------------------------------
For bulk historical downloads WebCOOS asked us to use their on-premise archive
(:data:`ONPREM_BASE`) instead of the API and S3, at most at the pace of
``wget --wait=1s --limit-rate=20m``, and to tell them when a transfer starts.
It is an nginx directory listing,
``<product>/<group>/<camera>/<YYYY>/<MM>/<DD>/*.jpg``. The ``jpg`` stills have
the same file names and bytes as S3 (checked 2026-10-09). ``timex`` and ``brt``
(from 2024) are the mean and the brightest value of each pixel over the
10-minute video clip starting at their time stamp, named
``timex.<camera>-<stamp>Z.jpg``. :class:`OnPremArchive` walks it; no token is
needed.
"""

from __future__ import annotations

import os
import re
import stat
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

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

#: Length of each listing query (see :meth:`WebCOOSClient.elements`).
ELEMENTS_WINDOW = "1D"

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

#: Beachfront cameras within 25 km of a buoy that publishes directional spectra,
#: chosen on 2026-10-08 from the 86-camera list and one midday still of each
#: (``docs/literature/data_source_scan_2026-10.md``). Views of harbours, river
#: mouths and the inside of Grays Harbor (``kahului_*``, ``ferrybeach_north``,
#: ``campellis_south``, ``oceanshores_*``) were left out: their water is sheltered
#: from the sea state the buoy measures.
NEAR_BUOY_CAMERAS = (
    "crescent_city",  # NDBC 46027, 19.8 km
    "currituck_sailfish",  # NDBC 44056 (Duck FRF), 17.8 km
    "masonboro_inlet",  # CDIP 150 (41110), 10.2 km; ocean between the inlet jetties
    "oceancity",  # CDIP 263 (44084), 19.7 km
    "pgtalabama_south",  # NDBC 42012, 23.3 km; burned-in time overlay
    "pgtalabama_west",  # NDBC 42012, 23.3 km; burned-in time overlay
    "san_elijo",  # CDIP 262 (46274), 5.8 km
    "stinson",  # NDBC 46026, 23.8 km
    "vabeach_hamptonos",  # CDIP 147 (44099), 23.8 km
    "westport_north",  # CDIP 036 (46211), 9.5 km
    "westport_south",  # CDIP 036 (46211), 9.5 km
)

#: Named camera sets, used on the command line as ``@name``.
CAMERA_SETS: dict[str, tuple[str, ...]] = {
    "yin2025": YIN2025_CAMERAS,
    "near_buoy": NEAR_BUOY_CAMERAS,
}

ACKNOWLEDGEMENT = (
    '"SECOORA is acknowledged for leading the WebCOOS program, which collected '
    '[images of interest] in the WebCOOS dataset in this study," citing WebCOOS as '
    '"WebCOOS [year(s) of imagery]"'
)

LISTINGS_DIR = "_listings"
MANIFESTS_DIR = "_manifests"

#: WebCOOS's on-premise archive, for bulk historical downloads.
ONPREM_BASE = "https://stage-ams.srv.axds.co/archive"

#: Products of the on-premise archive: name -> (directory, file-name prefix).
#: ``stills`` are the one-minute stills (frames of the video). ``timex`` (the
#: time average) and ``brt`` (the brightest value of each pixel) are made from
#: the 10-minute video clip that starts at their time stamp (see
#: ``docs/datasets/webcoos.md``).
ONPREM_PRODUCTS: dict[str, tuple[str, str]] = {
    "stills": ("jpg", ""),
    "timex": ("timex", "timex."),
    "brt": ("brt", "brt."),
}

#: Group directory of each camera in the on-premise archive (as given by WebCOOS).
ONPREM_GROUPS: dict[str, str] = {
    "currituck_hampton_inn": "noaa",
    "currituck_sailfish": "noaa",
    "jennette_north": "noaa",
    "jennette_south": "noaa",
    "oakisland_east": "uncw",
    "oakisland_west": "uncw",
    "masonboro_inlet": "uncw",
    "cocoabeach": "uncw",
}

#: Pace WebCOOS asked for: ``wget --wait=1s --limit-rate=20m`` (wget's m is MiB).
ONPREM_WAIT_S = 1.0
ONPREM_MAX_BYTES_PER_S = 20 * 2**20

#: Month indexes of the on-premise archive, under ``_listings/``.
ONPREM_INDEX_DIR = "onprem"


def expand_cameras(names: Iterable[str]) -> list[str]:
    """Camera slugs with ``@set`` names (:data:`CAMERA_SETS`) expanded, in order, once each."""
    out: list[str] = []
    for name in names:
        if name.startswith("@"):
            if name[1:] not in CAMERA_SETS:
                raise KeyError(f"unknown camera set {name!r}; choose from {sorted(CAMERA_SETS)}")
            out.extend(CAMERA_SETS[name[1:]])
        else:
            out.append(name)
    return list(dict.fromkeys(out))


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
        self,
        service: str,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        *,
        window: str | pd.Timedelta = ELEMENTS_WINDOW,
        progress: Callable[[pd.Timestamp, int], object] | None = None,
    ) -> list[dict[str, Any]]:
        """Image elements of a service with capture times in ``(start, end)``.

        The range is listed in ``window``-long pieces: one long query is slow
        on the server (an 85-day listing had not finished after 10 minutes),
        while a day comes back in seconds. Pieces overlap by one second so
        that no element on a boundary is missed; :func:`element_table` drops
        the duplicates. ``progress(window_end, n_so_far)`` is called per piece.
        """
        t0, t1 = _utc(start), _utc(end)
        step = pd.Timedelta(window)
        out: list[dict[str, Any]] = []
        a = t0
        while a < t1:
            b = min(a + step, t1)
            params = {
                "service": service,
                "starting_after": _iso(a - pd.Timedelta(seconds=1) if a > t0 else a),
                "starting_before": _iso(b + pd.Timedelta(seconds=1) if b < t1 else b),
                # 1000 per page (the default is 100): a tenth of the requests.
                "page_size": str(ELEMENTS_PAGE_SIZE),
            }
            out.extend(self.paginate("elements/", params))
            if progress is not None:
                progress(b, len(out))
            a = b
        return out


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
    out["grid_time"] = pd.DatetimeIndex([chosen[i] for i in idx], dtype=grid.dtype)
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


def ledger_path(dest_root: Path, camera: str, host: str | None = None) -> Path:
    """Per-camera request ledger; with ``host``, that host's own (``<camera>.<host>.csv``).

    The hourly collector writes ``<camera>.csv`` on one host. Other writers,
    such as a historical download, use a ledger named for their host, so that
    every ledger has a single writer when hosts exchange the archive.
    """
    if "/" in camera or camera.startswith("."):
        raise ValueError(f"unsafe camera name {camera!r}")
    if host is not None and not re.fullmatch(r"[a-z0-9][a-z0-9-]*", host):
        raise ValueError(f"unsafe host name {host!r}")
    return dest_root / MANIFESTS_DIR / (f"{camera}.{host}.csv" if host else f"{camera}.csv")


def archive_stills(
    downloader: Downloader,
    selected: pd.DataFrame,
    dest_root: Path,
    *,
    sink: Sink,
    ledger: Path | None = None,
    limit: int | None = None,
    progress: Any = None,
    progress_every: int = 200,
) -> ArchiveSummary:
    """Download selected stills of one camera that are not yet archived.

    ``downloader`` must not carry the API token: the files are public.
    ``ledger`` (default :func:`ledger_path`) must be the file ``sink`` writes.
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
        ledger=ledger or ledger_path(dest_root, camera),
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


def on_disk(selected: pd.DataFrame, dest_root: Path) -> pd.Series:
    """Whether each selected still is already archived: same file name and size.

    Stills fetched from S3 by the hourly collector, or by another host and
    exchanged, are the same files as in the on-premise archive.
    """
    have = []
    for cam, name, t, size in zip(
        selected["camera"],
        selected["name"],
        selected["time_utc"],
        selected["size_bytes"],
        strict=True,
    ):
        p = local_path(dest_root, str(cam), str(name), t)
        have.append(p.exists() and (pd.isna(size) or p.stat().st_size == int(size)))
    return pd.Series(have, index=selected.index, dtype=bool)


# --------------------------------------------------------------------------- #
# On-premise archive
# --------------------------------------------------------------------------- #

_LISTING_ROW = re.compile(
    r'<a href="(?P<href>[^"]+)">[^<]*</a>\s+'
    r"(?P<modified>\d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2})\s+(?P<size>\d+|-)"
)
_STILL_NAME = re.compile(r"^(?P<camera>.+)-(?P<stamp>\d{4}-\d{2}-\d{2}-\d{6})Z\.jpg$")


def parse_listing(html: str) -> pd.DataFrame:
    """Entries of an nginx directory listing: ``name``, ``is_dir`` and ``size_bytes``.

    nginx lists exact sizes in bytes, and none for directories. The parent
    link and any name that is not a plain file or directory name are dropped.
    """
    rows = []
    for m in _LISTING_ROW.finditer(html):
        href = unquote(m["href"])
        name = href.removesuffix("/")
        if not name or "/" in name or name.startswith("."):
            continue
        size = None if m["size"] == "-" else int(m["size"])
        rows.append({"name": name, "is_dir": href.endswith("/"), "size_bytes": size})
    out = pd.DataFrame(rows, columns=["name", "is_dir", "size_bytes"])
    out["is_dir"] = out["is_dir"].astype(bool)
    out["size_bytes"] = out["size_bytes"].astype("Int64")
    return out


def still_time(name: str, camera: str, prefix: str = "") -> pd.Timestamp | None:
    """Capture time (UTC) in the name of one of ``camera``'s images, else ``None``.

    The name carries the capture time to the second: it equalled the API's
    capture time for all 43,999 stills listed by the collector up to 2026-10-09.
    ``prefix`` is the product's (e.g. ``timex.``; see :data:`ONPREM_PRODUCTS`).
    """
    if not name.startswith(prefix):
        return None
    m = _STILL_NAME.match(name[len(prefix) :])
    if m is None or m["camera"] != camera:
        return None
    return pd.Timestamp(pd.to_datetime(m["stamp"], format="%Y-%m-%d-%H%M%S", utc=True))


def onprem_stills(
    listing: pd.DataFrame, camera: str, day_url: str, prefix: str = ""
) -> pd.DataFrame:
    """The images in a day directory's listing, as an :func:`element_table`.

    Other files (e.g. ``<camera>.jpg``, editing leftovers) are ignored.
    """
    rows = []
    for name, is_dir, size in zip(
        listing["name"], listing["is_dir"], listing["size_bytes"], strict=True
    ):
        t = None if is_dir else still_time(str(name), camera, prefix)
        if t is not None:
            rows.append(
                {
                    "camera": camera,
                    "time_utc": t,
                    "name": name,
                    "url": f"{day_url}{name}",
                    "size_bytes": size,
                    "element_id": None,
                }
            )
    out = pd.DataFrame(
        rows, columns=["camera", "time_utc", "name", "url", "size_bytes", "element_id"]
    )
    out["time_utc"] = pd.to_datetime(out["time_utc"], utc=True)
    out["size_bytes"] = out["size_bytes"].astype("Int64")
    return out.sort_values("time_utc").reset_index(drop=True)


class OnPremArchive:
    """Walks one product of WebCOOS's on-premise archive (no token), paced by ``downloader``.

    Use one :class:`~wave_analysis.ingest.downloader.Downloader` for listings
    and files, so that its wait applies to every request.
    """

    def __init__(
        self, downloader: Downloader, product: str = "stills", base: str = ONPREM_BASE
    ) -> None:
        if product not in ONPREM_PRODUCTS:
            raise ValueError(f"unknown product {product!r}; known: {', '.join(ONPREM_PRODUCTS)}")
        self.downloader = downloader
        self.base = base.rstrip("/")
        self.directory, self.prefix = ONPREM_PRODUCTS[product]

    def camera_url(self, camera: str) -> str:
        """Directory of a camera."""
        group = ONPREM_GROUPS.get(camera)
        if group is None:
            raise ValueError(
                f"no on-premise path known for {camera!r}; known: {', '.join(ONPREM_GROUPS)}"
            )
        return f"{self.base}/{self.directory}/{group}/{camera}/"

    def listing(self, url: str) -> pd.DataFrame:
        """A directory's entries; none if it does not exist (HTTP 404)."""
        if not url.startswith(f"{self.base}/"):
            raise ValueError(f"not in the on-premise archive: {url}")
        resp = self.downloader.get(url)
        if resp.status_code == 404:
            return parse_listing("")
        resp.raise_for_status()
        return parse_listing(resp.text)

    def _subdirs(self, url: str, digits: int) -> list[str]:
        df = self.listing(url)
        names = df.loc[df["is_dir"], "name"]
        return sorted(n for n in names if n.isdigit() and len(n) == digits)

    def months(
        self, camera: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
    ) -> list[pd.Period]:
        """Months with a directory for ``camera`` that overlap ``[start, end)``."""
        lo = None if start is None else _utc(start).tz_localize(None)
        hi = None if end is None else _utc(end).tz_localize(None)
        root = self.camera_url(camera)
        out = []
        for y in self._subdirs(root, 4):
            if (lo is not None and int(y) < lo.year) or (hi is not None and int(y) > hi.year):
                continue
            for m in self._subdirs(f"{root}{y}/", 2):
                p = pd.Period(f"{y}-{m}", "M")
                if (lo is None or p.end_time >= lo) and (hi is None or p.start_time < hi):
                    out.append(p)
        return out

    def month_stills(
        self,
        camera: str,
        month: pd.Period,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        progress: Callable[[str], object] | None = None,
    ) -> pd.DataFrame:
        """Stills of ``camera`` in ``month``, listing each day that overlaps ``[start, end)``."""
        url = f"{self.camera_url(camera)}{month.strftime('%Y/%m')}/"
        lo = None if start is None else _utc(start).tz_localize(None)
        hi = None if end is None else _utc(end).tz_localize(None)
        frames = []
        for d in self._subdirs(url, 2):
            day = pd.Timestamp(f"{month.strftime('%Y-%m')}-{d}")
            if (lo is not None and day + pd.Timedelta("1D") <= lo) or (
                hi is not None and day >= hi
            ):
                continue
            day_url = f"{url}{d}/"
            frames.append(onprem_stills(self.listing(day_url), camera, day_url, self.prefix))
            if progress is not None:
                progress(d)
        if not frames:
            return onprem_stills(parse_listing(""), camera, url, self.prefix)
        return pd.concat(frames, ignore_index=True).sort_values("time_utc", ignore_index=True)


def month_index_path(dest_root: Path, camera: str, month: pd.Period) -> Path:
    """Saved listing of a camera's month in the on-premise archive."""
    if "/" in camera or camera.startswith("."):
        raise ValueError(f"unsafe camera name {camera!r}")
    return dest_root / LISTINGS_DIR / ONPREM_INDEX_DIR / camera / f"{month.strftime('%Y-%m')}.csv"


def read_month_index(path: Path) -> pd.DataFrame:
    """A saved month listing (see :func:`write_month_index`)."""
    df = pd.read_csv(path, dtype={"name": str, "url": str, "camera": str})
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="ISO8601")
    df["size_bytes"] = df["size_bytes"].astype("Int64")
    df["element_id"] = None
    return df


def write_month_index(stills: pd.DataFrame, path: Path, listed_at: pd.Timestamp) -> None:
    """Save a closed month's listing, written once (atomically) and never changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".partial-{path.name}")
    stills.drop(columns="element_id").assign(listed_at=_iso(listed_at)).to_csv(tmp, index=False)
    os.replace(tmp, path)


def select_between(
    elements: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    **grid: Any,
) -> pd.DataFrame:
    """:func:`select_on_grid` keeping the grid times in ``[start, end)``.

    Selecting a month at a time this way gives each grid time one still, as
    one selection over the whole range would. Only at a boundary can the still
    differ from the nearest one (the nearest may lie in the other piece).
    """
    sel = select_on_grid(elements, **grid)
    keep = (sel["grid_time"] >= _utc(start)) & (sel["grid_time"] < _utc(end))
    return sel[keep].reset_index(drop=True)


def _ns(t: pd.DatetimeIndex) -> np.ndarray[Any, np.dtype[np.int64]]:
    """UTC nanoseconds since the epoch, whatever the index's resolution."""
    naive = t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t
    return np.asarray(naive.as_unit("ns").to_numpy(dtype="datetime64[ns]").astype(np.int64))


def _utc(t: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _iso(t: pd.Timestamp | str) -> str:
    return f"{_utc(t):%Y-%m-%dT%H:%M:%SZ}"
