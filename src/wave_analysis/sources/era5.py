"""ECMWF ERA5 reanalysis ocean-wave parameters from the Copernicus Climate Data Store.

Access (verified 2026-09-26): the CDS API, with a personal access token in
``~/.cdsapirc`` (or ``$CDSAPI_URL`` / ``$CDSAPI_KEY``), read by the ``cdsapi``
client. The dataset's licence (CC-BY 4.0) must first be accepted once on the
dataset page. Dataset ``reanalysis-era5-single-levels``, DOI
10.24381/cds.adbb2d47.

What the files contain, from the returned netCDF and ECMWF's documentation
------------------------------------------------------------------------
* The wave model runs on a 0.36 deg reduced grid; the CDS serves it
  interpolated to a regular 0.5 deg grid. Grid points over land are NaN.
* Every wave parameter is *instantaneous* (``GRIB_stepType = instant``) at
  ``valid_time``, hourly, UTC.
* ``expver`` (a coordinate along ``valid_time``) is ``0001`` for final ERA5
  and ``0005`` for the preliminary ERA5T of the last ~3 months, which ERA5
  later replaces. A request past the latest available hour returns only the
  hours that exist.
* Definitions (Bidlot, *Ocean wave model output parameters*, ECMWF 2020):
  ``swh`` = 4 sqrt(m0); ``mwp`` = m-1/m0, the energy period (not Tm01);
  ``mp1`` = m0/m1; ``mp2`` = sqrt(m0/m2); ``pp1d`` = reciprocal of the peak
  frequency from a parabolic fit around the discrete spectral maximum;
  ``mwd`` = mean direction, degrees true, coming from. The wind-sea and
  total-swell parameters (``shww`` ... ``mdts``) apply the same definitions to
  ERA5's wind-sea/swell split of the spectrum.

Requests are one calendar month for a small box around a site; the raw layer
keeps the whole box and :meth:`ERA5WaveSource.normalize` takes the nearest
grid point with sea data. See ``docs/datasets/era5.md``.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from wave_analysis import __version__
from wave_analysis.ingest.checksum import sha256_file
from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, utcnow
from wave_analysis.processing.geospatial import haversine_m
from wave_analysis.qc import qartod
from wave_analysis.qc.flags import QCTestResult
from wave_analysis.qc.waves import steepness_test
from wave_analysis.schemas.observation import (
    ProcessingLevel,
    TimeReference,
    coerce_observation_frame,
    make_observation_id,
)
from wave_analysis.schemas.variables import VARIABLES
from wave_analysis.sources.base import DataSource

DATASET_ID = "era5_waves"
CDS_DATASET = "reanalysis-era5-single-levels"
CDS_HOST = "cds.climate.copernicus.eu"
CDS_API = f"https://{CDS_HOST}/api"
#: The dataset's retrieve endpoint; recorded as the manifest URL of every request.
RETRIEVE_URL = f"{CDS_API}/retrieve/v1/processes/{CDS_DATASET}"
DATASET_PAGE = f"https://{CDS_HOST}/datasets/{CDS_DATASET}"
CATALOGUE_URL = f"{CDS_API}/catalogue/v1/collections/{CDS_DATASET}"
DOI = "10.24381/cds.adbb2d47"
CITATION = (
    "Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horanyi, A., Munoz Sabater, J., "
    "Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., "
    "Dee, D., Thepaut, J-N. (2023): ERA5 hourly data on single levels from 1940 to present. "
    "Copernicus Climate Change Service (C3S) Climate Data Store (CDS), "
    f"DOI: {DOI} (accessed <date>)"
)
CONFIG_FILE = Path.home() / ".cdsapirc"

#: Record interval [s].
RECORD_INTERVAL_S = 3600.0
#: Half-width of the requested box around a site [deg]; the wave grid is 0.5 deg.
DEFAULT_BOX_DEG = 1.0
GRID_DEG = 0.5
#: Per-request field limit shown on the CDS download form (2026-09-26).
MAX_FIELDS = 121_000

#: Age after which a file that can still change upstream is requested again.
RECHECK_AFTER = {"incomplete": pd.Timedelta(days=1), "preliminary": pd.Timedelta(days=30)}

#: ``expver`` -> ``source_qc_flag``.
EXPVER = {"0001": "era5", "0005": "era5t"}
_UNITS = {"m": "m", "s": "s", "Degree true": "degree", "degrees": "degree"}


@dataclass(frozen=True)
class WaveParameter:
    """One ERA5 wave parameter and its canonical variable."""

    request: str  # CDS request name
    short: str  # netCDF variable (GRIB shortName)
    param_id: int  # GRIB paramId, checked on every parsed file
    variable: str  # canonical variable (schemas/variables.py)
    method: str  # estimator, recorded in the observation 'method'


PARAMETERS: tuple[WaveParameter, ...] = (
    WaveParameter(
        "significant_height_of_combined_wind_waves_and_swell",
        "swh",
        140229,
        "wave_significant_height",
        "spectral_hm0",
    ),
    WaveParameter(
        "peak_wave_period", "pp1d", 140231, "wave_peak_period", "spectral_tp_parabolic_fit"
    ),
    WaveParameter("mean_wave_period", "mwp", 140232, "wave_energy_period", "spectral_tm_10"),
    WaveParameter(
        "mean_wave_period_based_on_first_moment",
        "mp1",
        140220,
        "wave_mean_period_tm01",
        "spectral_tm01",
    ),
    WaveParameter(
        "mean_zero_crossing_wave_period", "mp2", 140221, "wave_mean_period_tm02", "spectral_tm02"
    ),
    WaveParameter(
        "mean_wave_direction", "mwd", 140230, "wave_mean_from_direction", "spectral_mean_direction"
    ),
    WaveParameter(
        "significant_height_of_wind_waves",
        "shww",
        140234,
        "wind_wave_significant_height",
        "spectral_hm0",
    ),
    WaveParameter(
        "mean_period_of_wind_waves", "mpww", 140236, "wind_wave_energy_period", "spectral_tm_10"
    ),
    WaveParameter(
        "mean_direction_of_wind_waves",
        "mdww",
        140235,
        "wind_wave_from_direction",
        "spectral_mean_direction",
    ),
    WaveParameter(
        "significant_height_of_total_swell",
        "shts",
        140237,
        "swell_significant_height",
        "spectral_hm0",
    ),
    WaveParameter(
        "mean_period_of_total_swell", "mpts", 140239, "swell_energy_period", "spectral_tm_10"
    ),
    WaveParameter(
        "mean_direction_of_total_swell",
        "mdts",
        140238,
        "swell_from_direction",
        "spectral_mean_direction",
    ),
)


class CDSConfigError(RuntimeError):
    """The CDS client is missing or not configured."""


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def months(start: str | pd.Period, end: str | pd.Period) -> list[pd.Period]:
    """Calendar months from ``start`` to ``end`` inclusive (``"YYYY-MM"``)."""
    first, last = pd.Period(start, freq="M"), pd.Period(end, freq="M")
    if last < first:
        raise ValueError(f"end {last} is before start {first}")
    return list(pd.period_range(first, last, freq="M"))


def area_around(latitude: float, longitude: float, box_deg: float = DEFAULT_BOX_DEG) -> list[float]:
    """CDS ``area`` ``[north, west, south, east]``: the box snapped outward to the grid."""
    if box_deg <= 0:
        raise ValueError("box_deg must be positive")

    def snap(x: float, up: bool) -> float:
        f = math.ceil if up else math.floor
        return float(f(round(x / GRID_DEG, 9)) * GRID_DEG)

    north = min(snap(latitude + box_deg, True), 90.0)
    south = max(snap(latitude - box_deg, False), -90.0)
    west, east = snap(longitude - box_deg, False), snap(longitude + box_deg, True)
    if west < -180.0 or east > 180.0:
        raise ValueError("boxes across the antimeridian are not supported")
    return [north, west, south, east]


def build_request(
    period: pd.Period, area: Sequence[float], parameters: Sequence[WaveParameter] = PARAMETERS
) -> dict[str, Any]:
    """The CDS request for every hour of one month."""
    days = calendar.monthrange(period.year, period.month)[1]
    return {
        "product_type": ["reanalysis"],
        "variable": [p.request for p in parameters],
        "year": [f"{period.year:04d}"],
        "month": [f"{period.month:02d}"],
        "day": [f"{d:02d}" for d in range(1, days + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [float(x) for x in area],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def request_fields(request: Mapping[str, Any]) -> int:
    """Fields a request asks for (the unit of the CDS size limit); area does not count."""
    n = 1
    for key in ("variable", "year", "month", "day", "time"):
        n *= len(request[key])
    return n


def request_digest(request: Mapping[str, Any]) -> str:
    """SHA-256 of the request's canonical JSON (identifies what a raw file was cut from)."""
    blob = json.dumps({"dataset": CDS_DATASET, **request}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def raw_path(dest_root: Path, site_id: str, period: pd.Period) -> Path:
    """``<root>/<site>/era5_waves_<site>_<YYYY-MM>.nc``."""
    if not site_id or "/" in site_id or site_id.startswith("."):
        raise ValueError(f"unsafe site id {site_id!r}")
    return dest_root / site_id / f"{DATASET_ID}_{site_id}_{period}.nc"


def month_status(path: Path, period: pd.Period) -> str:
    """State of a raw month file.

    ``missing``; ``incomplete`` (hours of the month absent, e.g. the month is
    not over or ERA5T had not reached its end); ``preliminary`` (all hours,
    some ERA5T); ``final`` (all hours, all ERA5); or ``unreadable``.
    """
    if not path.exists():
        return "missing"
    try:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            times = pd.DatetimeIndex(ds["valid_time"].values)
            expver = _expver(ds, len(times))
    except Exception:
        return "unreadable"
    hours = pd.date_range(period.start_time, period.end_time.floor("h"), freq="h")
    if not hours.isin(times).all():
        return "incomplete"
    if any(e != "0001" for e in expver):
        return "preliminary"
    return "final"


def needs_request(
    path: Path, state: str, *, now: pd.Timestamp | None = None, refresh: bool = False
) -> bool:
    """Whether to request a month again, given its file's state and age.

    Missing and unreadable files are always requested, final ones only with
    ``refresh``. Files that can still change upstream are requested again once
    they are older than :data:`RECHECK_AFTER` (new hours appear daily; ERA5
    replaces ERA5T about 2-3 months after real time).
    """
    if refresh or state in ("missing", "unreadable"):
        return True
    if state == "final":
        return False
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    age = now - pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    return bool(age >= RECHECK_AFTER[state])


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
def check_config(env: Mapping[str, str] | None = None, path: Path = CONFIG_FILE) -> str:
    """Where the CDS key will come from; raises :class:`CDSConfigError` if unusable.

    Never reads the key: only checks that the file exists, is not empty, and
    is not readable by other users.
    """
    env = os.environ if env is None else env
    if env.get("CDSAPI_URL") and env.get("CDSAPI_KEY"):
        return "$CDSAPI_URL/$CDSAPI_KEY"
    if not path.exists():
        raise CDSConfigError(
            f"no CDS key: create {path} (see docs/datasets/era5.md) or set $CDSAPI_URL/$CDSAPI_KEY"
        )
    st = path.stat()
    if st.st_size == 0:
        raise CDSConfigError(f"{path} is empty")
    if st.st_mode & 0o077:
        raise CDSConfigError(f"{path} is readable by other users; run: chmod 600 {path}")
    return str(path)


def make_client(*, log: Callable[[str], object] | None = None) -> Any:
    """A ``cdsapi`` client that only talks to the CDS and never logs the key.

    The client's debug output (which includes the key) is discarded; its
    info and warning messages go to ``log``.
    """
    check_config()
    try:
        import cdsapi
    except ImportError as exc:
        raise CDSConfigError("cdsapi is not installed; run: uv sync --extra cds") from exc

    def relay(*args: Any, **kwargs: Any) -> None:
        if log is not None and args:
            msg = str(args[0])
            with_args = args[1:]
            log(msg % with_args if with_args else msg)

    client = cdsapi.Client(
        quiet=True,
        progress=False,
        debug=False,
        debug_callback=lambda *a, **k: None,
        info_callback=relay,
        warning_callback=relay,
        error_callback=relay,
    )
    host = urlparse(str(client.url)).hostname
    if host != CDS_HOST:
        raise CDSConfigError(
            f"CDS url points to {host!r}, not {CDS_HOST}; refusing to send the key"
        )
    return client


def available_until(get_json: Callable[[str], Any] | None = None) -> pd.Timestamp | None:
    """End of the dataset's temporal extent from the public catalogue (no key needed)."""
    if get_json is None:
        import httpx

        def get_json(url: str) -> Any:
            r = httpx.get(url, timeout=60)
            r.raise_for_status()
            return r.json()

    meta = get_json(CATALOGUE_URL)
    try:
        end = meta["extent"]["temporal"]["interval"][0][1]
    except (KeyError, IndexError, TypeError):
        return None
    return pd.Timestamp(end).tz_convert("UTC") if end else None


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class ERA5WaveSource(DataSource):
    """ERA5 ocean-wave parameters around camera sites.

    Parameters
    ----------
    client
        Object with ``retrieve(dataset, request, target)`` (a ``cdsapi`` client
        from :func:`make_client`); created on first use if omitted.
    log
        Receives progress messages.
    """

    dataset_id = DATASET_ID

    def __init__(self, client: Any = None, *, log: Callable[[str], object] | None = None) -> None:
        self._client = client
        self._log = log
        self._last_request_id: str | None = None

    @property
    def client(self) -> Any:
        """The CDS client, created from ``~/.cdsapirc`` on first use."""
        if self._client is None:
            self._client = make_client(log=self._note)
        return self._client

    def _note(self, msg: str) -> None:
        if msg.startswith("Request ID is "):
            self._last_request_id = msg.removeprefix("Request ID is ").strip()
        if self._log is not None:
            self._log(msg)

    # ------------------------------------------------------------------ #
    def discover(
        self,
        sites: Mapping[str, tuple[float, float]],
        periods: Iterable[pd.Period],
        *,
        dest_root: Path,
        box_deg: float = DEFAULT_BOX_DEG,
        refresh: bool = False,
        now: pd.Timestamp | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """The request plan: one row per site and month.

        Gives each month's file state and whether :meth:`fetch` would request
        it (:func:`needs_request`).
        """
        rows = []
        periods = list(periods)
        for site, (lat, lon) in sites.items():
            area = area_around(lat, lon, box_deg)
            for p in periods:
                req = build_request(p, area)
                path = raw_path(dest_root, site, p)
                state = month_status(path, p)
                rows.append(
                    {
                        "site_id": site,
                        "period": str(p),
                        "area": area,
                        "fields": request_fields(req),
                        "status": state,
                        "request": needs_request(path, state, now=now, refresh=refresh),
                        "path": path,
                    }
                )
        return pd.DataFrame(
            rows, columns=["site_id", "period", "area", "fields", "status", "request", "path"]
        )

    # ------------------------------------------------------------------ #
    def fetch(
        self,
        site_id: str,
        latitude: float,
        longitude: float,
        periods: Iterable[pd.Period],
        *,
        dest_root: Path,
        box_deg: float = DEFAULT_BOX_DEG,
        refresh: bool = False,
        now: pd.Timestamp | None = None,
        sink: Callable[[list[ManifestEntry]], object] | None = None,
    ) -> list[ManifestEntry]:
        """Retrieve monthly files for one site, as decided by :func:`needs_request`.

        Final months are skipped; ``incomplete`` and ``preliminary`` months are
        requested again once their files are old enough to have changed
        upstream; ``refresh`` requests every month. Each file is written atomically with its request beside it
        (``.request.json``). Every attempt is returned and, if given, passed to
        ``sink`` as it completes.
        """
        area = area_around(latitude, longitude, box_deg)
        entries: list[ManifestEntry] = []
        for period in periods:
            path = raw_path(dest_root, site_id, period)
            state = month_status(path, period)
            if not needs_request(path, state, now=now, refresh=refresh):
                continue
            request = build_request(period, area)
            if request_fields(request) > MAX_FIELDS:
                raise ValueError(f"request for {period} exceeds {MAX_FIELDS} fields")
            entry = self._retrieve(site_id, period, request, path, state)
            entries.append(entry)
            if sink is not None:
                sink([entry])
        return entries

    def _retrieve(
        self, site_id: str, period: pd.Period, request: dict[str, Any], path: Path, state: str
    ) -> ManifestEntry:
        digest = request_digest(request)
        previous = sha256_file(path) if path.exists() else None
        path.parent.mkdir(parents=True, exist_ok=True)
        for stale in path.parent.glob(f".{path.name}.*.part"):  # left by an interrupted run
            stale.unlink(missing_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        self._last_request_id = None
        base = {
            "source_id": DATASET_ID,
            "product": "wave_single_levels",
            "station_id": site_id,
            "period": str(period),
            "url": RETRIEVE_URL,
            "software_version": __version__,
        }
        try:
            self.client.retrieve(CDS_DATASET, request, str(tmp))
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            msg = _redact(f"{type(exc).__name__}: {exc}", getattr(self._client, "key", None))
            return ManifestEntry(
                **base,
                retrieved_at=utcnow(),
                status=EntryStatus.FAILED,
                note=f"request {digest[:16]}; {msg[:500]}",
            )
        sha = sha256_file(tmp)
        size = tmp.stat().st_size
        os.replace(tmp, path)
        path.with_suffix(".request.json").write_text(
            json.dumps({"dataset": CDS_DATASET, "request": request, "sha256": digest}, indent=1)
            + "\n"
        )
        new_state = month_status(path, period)
        if previous is None:
            status = EntryStatus.VERIFIED
        else:
            status = EntryStatus.UNCHANGED if previous == sha else EntryStatus.CHANGED
        job = f"; CDS request {self._last_request_id}" if self._last_request_id else ""
        return ManifestEntry(
            **base,
            local_path=str(path),
            sha256=sha,
            size_bytes=size,
            retrieved_at=utcnow(),
            status=status,
            note=f"request {digest[:16]}{job}; {new_state} (was {state})",
        )

    # ------------------------------------------------------------------ #
    def parse(self, path: Path, **kwargs: Any) -> Any:
        """Open a monthly file as an ``xarray.Dataset`` (loaded), checking parameter ids.

        Raises ``ValueError`` if a variable's ``GRIB_paramId`` differs from
        :data:`PARAMETERS` (a renamed or substituted parameter).
        """
        import xarray as xr

        with xr.open_dataset(path) as ds:
            for p in PARAMETERS:
                if p.short in ds:
                    pid = ds[p.short].attrs.get("GRIB_paramId")
                    if pid is not None and int(pid) != p.param_id:
                        raise ValueError(
                            f"{path.name}: {p.short} has GRIB_paramId {pid}, expected {p.param_id}"
                        )
            out = ds.load()
        return out

    # ------------------------------------------------------------------ #
    def normalize(
        self,
        parsed: Any,
        *,
        site_id: str,
        latitude: float,
        longitude: float,
        source_file: str,
        source_url: str | None = RETRIEVE_URL,
        source_checksum: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Wave parameters at the grid point with sea data nearest the site.

        ``station_id`` is the site; ``latitude``/``longitude`` are the grid
        point's, and ``platform_id`` names it. Values are instantaneous
        (``time_reference = center``, ``averaging_period_s = 0``).
        ``source_qc_flag`` is ``era5`` (final) or ``era5t`` (preliminary).
        """
        ds = parsed
        point = nearest_sea_point(ds, latitude, longitude)
        times = pd.DatetimeIndex(ds["valid_time"].values).tz_localize("UTC")
        expver = _expver(ds, len(times))
        flag = [EXPVER.get(e, f"expver_{e}") for e in expver]
        frames = []
        for p in PARAMETERS:
            if p.short not in ds:
                continue
            raw_units = str(ds[p.short].attrs.get("units", ""))
            units = _UNITS.get(raw_units)
            spec = VARIABLES[p.variable]
            if units != spec.units:
                raise ValueError(f"{p.short}: units {raw_units!r} do not map to {spec.units!r}")
            values = np.asarray(
                ds[p.short].isel(latitude=point.ilat, longitude=point.ilon).values,
                dtype=np.float64,
            )
            if spec.is_angle:
                values = np.mod(values, 360.0)
            frame = pd.DataFrame(
                {
                    "time_utc": times,
                    "variable": p.variable,
                    "value": values,
                    "units": units,
                    "method": p.method,
                    "source_qc_flag": flag,
                }
            )
            frames.append(frame[frame["value"].notna()])
        if not frames:
            return coerce_observation_frame(pd.DataFrame(columns=["time_utc"]))
        long = pd.concat(frames, ignore_index=True)
        long["dataset_id"] = DATASET_ID
        long["station_id"] = site_id
        long["platform_id"] = point.platform_id
        long["latitude"] = point.latitude
        long["longitude"] = point.longitude
        long["time_reference"] = TimeReference.CENTER.value
        long["averaging_period_s"] = 0.0
        long["source_file"] = source_file
        long["source_url"] = source_url
        long["source_checksum"] = source_checksum
        long["processing_level"] = ProcessingLevel.STANDARDIZED.value
        long["processing_version"] = __version__
        long["observation_id"] = [
            make_observation_id(DATASET_ID, site_id, "wave", v, t)
            for v, t in zip(long["variable"], long["time_utc"], strict=True)
        ]
        return coerce_observation_frame(long)

    # ------------------------------------------------------------------ #
    def validate(
        self, observations: pd.DataFrame, *, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[QCTestResult]:
        """QARTOD gap, range and rate-of-change tests plus steepness, on the Hs series.

        Model output needs no instrument checks; these catch a broken request
        or file (missing hours, units, a land point) rather than bad values.
        """
        cfg = config or {}
        rng = cfg.get("gross_range", {})
        hs = observations[observations["variable"] == "wave_significant_height"].sort_values(
            "time_utc"
        )
        tp = observations[observations["variable"] == "wave_peak_period"].set_index("time_utc")[
            "value"
        ]
        t = hs["time_utc"].to_numpy()
        v = hs["value"].to_numpy(dtype=np.float64)
        hs_rng = rng.get("wave_significant_height", {"fail": [0.0, 25.0], "suspect": [0.05, 15.0]})
        results = [
            qartod.gap_test(t, cfg.get("expected_interval_s", RECORD_INTERVAL_S)),
            qartod.gross_range_test(
                v, tuple(hs_rng["fail"]), tuple(hs_rng["suspect"]), name="gross_range_hs"
            ),
            qartod.rate_of_change_test(v, t, **cfg.get("rate_of_change", {"max_change": 2.0})),
        ]
        tp_aligned = tp.reindex(pd.DatetimeIndex(hs["time_utc"])).to_numpy(dtype=np.float64)
        results.append(steepness_test(v, tp_aligned, **cfg.get("steepness", {})))
        return results


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridPoint:
    """The grid point used for a site."""

    ilat: int
    ilon: int
    latitude: float
    longitude: float
    distance_km: float

    @property
    def platform_id(self) -> str:
        """Identifier of the grid point, e.g. ``era5_grid_21.50_-158.50``."""
        return f"era5_grid_{self.latitude:.2f}_{self.longitude:.2f}"


def nearest_sea_point(ds: Any, latitude: float, longitude: float) -> GridPoint:
    """The grid point nearest ``(latitude, longitude)`` with significant wave height data.

    Land points are NaN at every time, so a point counts as sea if it has a
    value at any time in the file.
    """
    lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    swh = ds["swh"].transpose("valid_time", "latitude", "longitude").to_numpy()
    sea = np.isfinite(swh).any(axis=0)
    glat, glon = np.meshgrid(lat, lon, indexing="ij")
    dist = np.asarray(haversine_m(latitude, longitude, glat, glon), dtype=np.float64)
    dist[~sea] = np.inf
    if not np.isfinite(dist).any():
        raise ValueError("no grid point with sea data in the requested area; use a larger box")
    i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
    return GridPoint(int(i), int(j), float(lat[i]), float(lon[j]), float(dist[i, j]) / 1000.0)


def _expver(ds: Any, n: int) -> list[str]:
    if "expver" not in ds.variables:
        return ["unknown"] * n
    values = np.atleast_1d(np.asarray(ds["expver"].values)).astype(str)
    return [str(x) for x in np.broadcast_to(values, (n,))]


def _redact(message: str, secret: str | None) -> str:
    return message.replace(secret, "<key>") if secret else message
