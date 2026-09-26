"""Coastal Data Information Program (CDIP, Scripps) adapter.

Access (verified 2026-09-26): THREDDS, no authentication. Each station has an
aggregate *historic* file with every publicly released record, per-deployment
files that also carry the raw heave/north/west displacement, and a *realtime*
file for the current deployment::

    https://thredds.cdip.ucsd.edu/thredds/fileServer/cdip/archive/{stn}p1/{stn}p1_historic.nc
    https://thredds.cdip.ucsd.edu/thredds/fileServer/cdip/archive/{stn}p1/{stn}p1_d{NN}.nc
    https://thredds.cdip.ucsd.edu/thredds/fileServer/cdip/realtime/{stn}p1_rt.nc

(``dodsC`` in place of ``fileServer`` gives OPeNDAP access to the same files.)
Whole files are downloaded (``fileServer``) so the raw layer holds the bytes
CDIP serves, with a checksum, rather than a server-side subset.

Conventions, from the file attributes (106p1_historic.nc, 2026-09-26)
---------------------------------------------------------------------
* ``waveTime`` is the **UTC start** of the sample; ``waveTimeBounds`` gives
  start and end. Spectra are computed on 1600 s samples, reported every 30 min.
* ``waveFlagPrimary`` uses the UNESCO/IOC (2013) scheme ``1 good,
  2 not_evaluated, 3 questionable, 4 bad, 9 missing`` - the same values as the
  project's QARTOD flags. ``waveFlagSecondary`` gives the reason (0-19).
  Historic files contain *publicly released data only, excluding all records
  flagged bad*.
* Directions are degrees true, coming-from; CDIP adds the magnetic declination.
* ``waveA1Value`` ... ``waveB2Value`` are relative to true north, so
  ``alpha1 = atan2(b1, a1)`` (:func:`~wave_analysis.processing.spectra.ndbc_from_fourier`).
* ``waveBandwidth`` is published and is used for integration.
* ``waveTa`` is the first-moment mean period (T_m01) and ``waveTz`` the
  second-moment period (T_m02), per their CF standard names.
* Licence (global attribute): "These data may be redistributed and used
  without restriction." Acknowledgement: CDIP is primarily supported by
  USACE; station partners are named per file (``acknowledgment``).

See ``docs/datasets/cdip.md``.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from wave_analysis import __version__
from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import ManifestEntry
from wave_analysis.processing.spectra import FrequencySpectrum, bulk_parameters, ndbc_from_fourier
from wave_analysis.qc import qartod
from wave_analysis.qc.flags import QCFlag, QCTestResult
from wave_analysis.qc.waves import steepness_test
from wave_analysis.schemas.observation import (
    ProcessingLevel,
    TimeReference,
    coerce_observation_frame,
    make_observation_id,
)
from wave_analysis.schemas.variables import VARIABLES
from wave_analysis.sources.base import DataSource

DATASET_ID = "cdip"
THREDDS = "https://thredds.cdip.ucsd.edu/thredds"

#: Products this adapter downloads.
PRODUCTS = ("historic", "realtime", "deployment")

#: Nominal record interval [s] (``time_coverage_resolution`` PT30M).
RECORD_INTERVAL_S = 1800.0

#: Wave-record variables kept by :meth:`CDIPSource.parse`.
WAVE_VARIABLES = (
    "waveTimeBounds",
    "waveFlagPrimary",
    "waveFlagSecondary",
    "waveHs",
    "waveTp",
    "waveTa",
    "waveDp",
    "waveTz",
    "wavePeakPSD",
    "waveSourceIndex",
    "waveFrequencyBounds",
    "waveFrequencyFlagPrimary",
    "waveBandwidth",
    "waveEnergyDensity",
    "waveMeanDirection",
    "waveA1Value",
    "waveB1Value",
    "waveA2Value",
    "waveB2Value",
)
#: Station and deployment metadata kept by :meth:`CDIPSource.parse`.
META_VARIABLES = (
    "sourceFilename",
    "metaDeployLatitude",
    "metaDeployLongitude",
    "metaWaterDepth",
    "metaDeclination",
    "metaStationName",
    "metaStationLatitude",
    "metaStationLongitude",
)

#: CDIP bulk variable -> canonical variable. Units are converted by name below.
BULK_MAP: dict[str, str] = {
    "waveHs": "wave_significant_height",
    "waveTp": "wave_peak_period",
    "waveTa": "wave_mean_period_tm01",
    "waveTz": "wave_mean_period_tm02",
    "waveDp": "wave_from_direction_at_peak",
}
_CDIP_UNITS = {"meter": "m", "second": "s", "degreeT": "degree"}

#: Derived bulk parameter -> (canonical variable, method tag); as for NDBC.
DERIVED_MAP: dict[str, tuple[str, str]] = {
    "Hm0": ("wave_significant_height", "spectral_hm0"),
    "Tp_smooth": ("wave_peak_period", "spectral_tp_young1995"),
    "Tm01": ("wave_mean_period_tm01", "spectral_tm01"),
    "Tm02": ("wave_mean_period_tm02", "spectral_tm02"),
    "Dm": ("wave_mean_from_direction", "spectral_dm_kuik1988"),
}

_STATION = re.compile(r"^\d{3}$")
_DEPLOYMENT_FILE = re.compile(r"^(?P<stn>\d{3})p1_d(?P<n>\d{2})\.nc$")
_SIZE_UNITS = {"bytes": 1.0, "Kbytes": 1e3, "Mbytes": 1e6, "Gbytes": 1e9, "Tbytes": 1e12}
_THREDDS_NS = {"t": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}


def station_code(station: str | int) -> str:
    """CDIP's three-digit station code (``106``, ``067``) from an int or string."""
    code = str(station).strip().lower().removesuffix("p1").zfill(3)
    if not _STATION.match(code):
        raise ValueError(f"not a CDIP station code: {station!r}")
    return code


def file_path(station: str | int, product: str, deployment: int | None = None) -> str:
    """Path of a CDIP file below ``thredds/<service>/cdip/``."""
    stn = station_code(station)
    if product == "historic":
        return f"archive/{stn}p1/{stn}p1_historic.nc"
    if product == "realtime":
        return f"realtime/{stn}p1_rt.nc"
    if product == "deployment":
        if deployment is None or not 1 <= deployment <= 99:
            raise ValueError("a deployment number 1-99 is required")
        return f"archive/{stn}p1/{stn}p1_d{deployment:02d}.nc"
    raise ValueError(f"unknown CDIP product {product!r}; expected one of {PRODUCTS}")


def file_url(path: str, service: str = "fileServer") -> str:
    """THREDDS URL of a CDIP file for a service (``fileServer`` or ``dodsC``)."""
    return f"{THREDDS}/{service}/cdip/{path}"


def catalog_url(station: str | int) -> str:
    """THREDDS catalog listing a station's archive files."""
    return f"{THREDDS}/catalog/cdip/archive/{station_code(station)}p1/catalog.xml"


def parse_catalog(xml_text: str, station: str | int) -> pd.DataFrame:
    """Availability table from a station's THREDDS archive catalog.

    One row per file: product (``historic`` or ``deployment``), deployment
    number, file name, download and OPeNDAP URLs, the size THREDDS reports
    (approximate: it is rounded, in decimal units) and its modification time.
    """
    stn = station_code(station)
    root = ET.fromstring(xml_text)
    rows: list[dict[str, Any]] = []
    for ds in root.iter(f"{{{_THREDDS_NS['t']}}}dataset"):
        path = ds.get("urlPath")
        if not path:
            continue
        name = path.rsplit("/", 1)[-1]
        size_el = ds.find("t:dataSize", _THREDDS_NS)
        date_el = ds.find("t:date[@type='modified']", _THREDDS_NS)
        size = (
            float(size_el.text) * _SIZE_UNITS.get(size_el.get("units", ""), np.nan)
            if size_el is not None and size_el.text
            else np.nan
        )
        m = _DEPLOYMENT_FILE.match(name)
        deployment: int | None = None
        if name == f"{stn}p1_historic.nc":
            product = "historic"
        elif m is not None and m["stn"] == stn:
            product, deployment = "deployment", int(m["n"])
        else:
            continue
        rel = path.removeprefix("cdip/")
        rows.append(
            {
                "station_id": stn,
                "product": product,
                "deployment": deployment,
                "file_name": name,
                "url": file_url(rel),
                "opendap_url": file_url(rel, "dodsC"),
                "approx_size_bytes": size,
                "modified_utc": pd.Timestamp(date_el.text)
                if date_el is not None and date_el.text
                else pd.NaT,
            }
        )
    out = pd.DataFrame(
        rows,
        columns=[
            "station_id",
            "product",
            "deployment",
            "file_name",
            "url",
            "opendap_url",
            "approx_size_bytes",
            "modified_utc",
        ],
    )
    out["deployment"] = out["deployment"].astype("Int64")
    out["modified_utc"] = pd.to_datetime(out["modified_utc"], utc=True)
    order = out["product"].map({"historic": 0, "deployment": 1})
    out = out.assign(_order=order).sort_values(["_order", "deployment"])
    return out.drop(columns="_order").reset_index(drop=True)


def deployment_index(ds: Any) -> NDArray[np.intp]:
    """Deployment (0-based row of ``metaDeploy*`` and ``sourceFilename``) of each wave record.

    ``waveSourceIndex`` is **1-based**. Verified on 106p1_historic.nc
    (2026-09-26): index 27 holds 18,356 records, the record count of
    ``106p1_d27.nc``, whereas 0-based it would point to ``106p1_d28.nc``, which
    has 754; index 1 starts on 2001-12-16, the start of ``106p1_d01.nc``.
    Raises if an index falls outside ``1..n``.
    """
    idx = np.asarray(ds["waveSourceIndex"].values, dtype=np.float64)
    n = int(ds.sizes["metaDeployCount"])
    if np.isnan(idx).any():
        raise ValueError("waveSourceIndex has missing values")
    if len(idx) and (idx.min() < 1 or idx.max() > n):
        raise ValueError(f"waveSourceIndex {idx.min():.0f}..{idx.max():.0f} outside 1..{n}")
    return (idx - 1).astype(np.intp)


def source_filenames(ds: Any) -> list[str]:
    """``sourceFilename`` as text (netCDF strings may arrive as bytes)."""
    return [
        v.decode() if isinstance(v, bytes) else str(v)
        for v in np.asarray(ds["sourceFilename"].values).ravel()
    ]


class CDIPSource(DataSource):
    """Coastal Data Information Program adapter (Datawell Waverider buoys).

    Parameters
    ----------
    downloader
        Shared :class:`Downloader`; one is created with default politeness
        settings if omitted.
    """

    dataset_id = DATASET_ID

    def __init__(self, downloader: Downloader | None = None) -> None:
        self.downloader = downloader or Downloader()

    # ------------------------------------------------------------------ #
    def discover(self, stations: Iterable[str | int], **kwargs: Any) -> pd.DataFrame:
        """Archive files per station from the THREDDS catalogs (one request per station)."""
        frames = []
        for st in stations:
            resp = self.downloader.get(catalog_url(st))
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            frames.append(parse_catalog(resp.text, st))
        if not frames:
            return parse_catalog("<catalog/>", "000")
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    def fetch(
        self,
        station_id: str | int,
        products: Iterable[str] = ("historic",),
        *,
        dest_root: Path,
        deployments: Iterable[int] = (),
        previous: dict[str, str] | None = None,
    ) -> Sequence[ManifestEntry]:
        """Download whole files for one station into ``dest_root/<stn>/``.

        ``historic`` and ``deployment`` files keep their upstream names and are
        replaced on re-fetch (the manifest records whether the bytes changed).
        ``realtime`` files are snapshots of a rolling window, stamped with the
        retrieval time.
        """
        stn = station_code(station_id)
        previous = previous or {}
        entries: list[ManifestEntry] = []
        for product in products:
            if product not in PRODUCTS:
                raise ValueError(f"unknown CDIP product {product!r}; expected one of {PRODUCTS}")
            numbers: list[int | None] = list(deployments) if product == "deployment" else [None]
            if product == "deployment" and not numbers:
                raise ValueError("product 'deployment' needs deployment numbers")
            for n in numbers:
                url = file_url(file_path(stn, product, n))
                name = url.rsplit("/", 1)[-1]
                if product == "realtime":
                    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
                    name = f"{name.removesuffix('.nc')}.{stamp}.nc"
                entries.append(
                    self.downloader.fetch(
                        url,
                        dest_root / stn / name,
                        source_id=DATASET_ID,
                        product=product,
                        station_id=stn,
                        period="realtime"
                        if product == "realtime"
                        else (f"d{n:02d}" if n is not None else "historic"),
                        previous_sha256=previous.get(url),
                    )
                )
        return entries

    # ------------------------------------------------------------------ #
    def parse(
        self,
        path: Path,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Open a CDIP netCDF file as an ``xarray.Dataset`` of wave records.

        Keeps :data:`WAVE_VARIABLES` and :data:`META_VARIABLES` (the SST, GPS
        and buoy-health groups are left in the raw file), restricted to wave
        records whose start time lies in ``[start, end)``, and loads them into
        memory. Fill values become NaN; times are decoded to UTC.
        """
        import xarray as xr

        with xr.open_dataset(path, mask_and_scale=True, decode_times=True) as full:
            keep = [v for v in (*WAVE_VARIABLES, *META_VARIABLES) if v in full.variables]
            ds = full[keep]
            t = pd.DatetimeIndex(ds["waveTime"].values)
            sel = np.ones(len(t), dtype=bool)
            if start is not None:
                sel &= t >= _naive_utc(start)
            if end is not None:
                sel &= t < _naive_utc(end)
            ds = ds.isel(waveTime=np.flatnonzero(sel)).load()
        ds.attrs = dict(full.attrs)
        return ds

    # ------------------------------------------------------------------ #
    def normalize(
        self,
        parsed: Any,
        *,
        station_id: str | int,
        source_file: str,
        source_url: str | None = None,
        source_checksum: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Bulk wave parameters as provider-reported observations.

        ``time_utc`` is the sample start (``time_reference = start``) and
        ``averaging_period_s`` the sample length from ``waveTimeBounds``.
        Position and ``deployment_id`` come from the record's deployment.
        ``source_qc_flag`` is ``"<primary>/<secondary>"`` (CDIP's flags,
        unchanged); ``project_qc_flag`` is left for :meth:`validate`.
        """
        ds = parsed
        stn = station_code(station_id)
        times = pd.to_datetime(ds["waveTime"].values).tz_localize("UTC")
        n = len(times)
        bounds = np.asarray(ds["waveTimeBounds"].values)
        averaging = (bounds[:, 1] - bounds[:, 0]) / np.timedelta64(1, "s")
        dep = deployment_index(ds) if n else np.zeros(0, dtype=np.intp)
        lat = np.asarray(ds["metaDeployLatitude"].values, dtype=np.float64)[dep]
        lon = np.asarray(ds["metaDeployLongitude"].values, dtype=np.float64)[dep]
        names = np.asarray(source_filenames(ds), dtype=object)
        deployment = [Path(s).stem for s in names[dep]]
        primary = _flag_strings(ds["waveFlagPrimary"].values)
        secondary = _flag_strings(ds["waveFlagSecondary"].values)
        source_flag = [f"{p}/{s}" for p, s in zip(primary, secondary, strict=True)]

        frames = []
        for cdip_var, var in BULK_MAP.items():
            if cdip_var not in ds:
                continue
            units = _CDIP_UNITS.get(str(ds[cdip_var].attrs.get("units", "")))
            if units != VARIABLES[var].units:
                raise ValueError(
                    f"{cdip_var}: units {ds[cdip_var].attrs.get('units')!r} do not map to "
                    f"{VARIABLES[var].units!r}"
                )
            values = np.asarray(ds[cdip_var].values, dtype=np.float64)
            if VARIABLES[var].is_angle:
                values = np.mod(values, 360.0)
            frame = pd.DataFrame(
                {
                    "time_utc": times,
                    "variable": var,
                    "value": values,
                    "units": units,
                    "averaging_period_s": averaging,
                    "latitude": lat,
                    "longitude": lon,
                    "deployment_id": deployment,
                    "source_qc_flag": source_flag,
                }
            )
            frames.append(frame[frame["value"].notna()])
        if not frames:
            return coerce_observation_frame(pd.DataFrame(columns=["time_utc"]))
        long = pd.concat(frames, ignore_index=True)
        long["dataset_id"] = DATASET_ID
        long["station_id"] = stn
        long["platform_id"] = f"{stn}p1"
        long["method"] = "provider_reported"
        long["time_reference"] = TimeReference.START.value
        long["source_file"] = source_file
        long["source_url"] = source_url
        long["source_checksum"] = source_checksum
        long["processing_level"] = ProcessingLevel.STANDARDIZED.value
        long["processing_version"] = __version__
        long["observation_id"] = [
            make_observation_id(DATASET_ID, stn, "wave", v, t)
            for v, t in zip(long["variable"], long["time_utc"], strict=True)
        ]
        return coerce_observation_frame(long)

    # ------------------------------------------------------------------ #
    def spectrum(self, parsed: Any, *, station_id: str | int) -> FrequencySpectrum:
        """Directional frequency spectra of the parsed records.

        Density and bandwidth are CDIP's own; the Fourier coefficients are
        converted to the project's ``alpha1, alpha2, r1, r2``. Timestamps are
        sample starts (``meta["time_reference"] = "start"``).
        """
        ds = parsed
        bounds = np.asarray(ds["waveTimeBounds"].values)
        lengths = np.unique((bounds[:, 1] - bounds[:, 0]) / np.timedelta64(1, "s"))
        if len(lengths) > 1:
            raise ValueError(f"mixed sample lengths {lengths.tolist()} s; split the time range")
        a1, b1, a2, b2 = (
            np.asarray(ds[v].values, dtype=np.float64)
            for v in ("waveA1Value", "waveB1Value", "waveA2Value", "waveB2Value")
        )
        alpha1, alpha2, r1, r2 = ndbc_from_fourier(a1, b1, a2, b2)
        return FrequencySpectrum(
            time=np.asarray(ds["waveTime"].values, dtype="datetime64[ns]"),
            frequency=np.asarray(ds["waveFrequency"].values, dtype=np.float64),
            density=np.asarray(ds["waveEnergyDensity"].values, dtype=np.float64),
            bandwidth=np.asarray(ds["waveBandwidth"].values, dtype=np.float64),
            alpha1=alpha1,
            alpha2=alpha2,
            r1=r1,
            r2=r2,
            meta={
                "source": DATASET_ID,
                "station_id": station_code(station_id),
                "time_reference": TimeReference.START.value,
                "averaging_period_s": float(lengths[0]) if len(lengths) else None,
                "file_id": ds.attrs.get("id"),
            },
        )

    def normalize_spectrum(
        self,
        spectrum: FrequencySpectrum,
        *,
        station_id: str | int,
        source_file: str,
        source_url: str | None = None,
        source_checksum: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(spectral_long_table, derived_bulk_observations)`` as for NDBC.

        The derived rows take their averaging period from the spectrum's
        ``meta["averaging_period_s"]`` (the sample length in ``waveTimeBounds``).
        """
        stn = station_code(station_id)
        spec_long = spectrum.to_long_frame()
        spec_long.insert(0, "station_id", stn)
        spec_long.insert(0, "dataset_id", DATASET_ID)
        spec_long["source_file"] = source_file
        spec_long["source_checksum"] = source_checksum
        spec_long["processing_version"] = __version__

        bulk = bulk_parameters(spectrum)
        frames = []
        for col, (var, method) in DERIVED_MAP.items():
            if col not in bulk:
                continue
            f = pd.DataFrame(
                {
                    "time_utc": bulk["time_utc"],
                    "variable": var,
                    "value": bulk[col],
                    "method": method,
                }
            )
            frames.append(f[f["value"].notna()])
        long = pd.concat(frames, ignore_index=True)
        long["dataset_id"] = DATASET_ID
        long["station_id"] = stn
        long["platform_id"] = f"{stn}p1"
        long["units"] = long["variable"].map(lambda v: VARIABLES[v].units)
        long["averaging_period_s"] = spectrum.meta.get("averaging_period_s")
        long["time_reference"] = TimeReference.START.value
        long["source_file"] = source_file
        long["source_url"] = source_url
        long["source_checksum"] = source_checksum
        long["processing_level"] = ProcessingLevel.DERIVED.value
        long["processing_version"] = __version__
        long["observation_id"] = [
            make_observation_id(DATASET_ID, stn, "spectrum_derived", v, t)
            for v, t in zip(long["variable"], long["time_utc"], strict=True)
        ]
        return spec_long, coerce_observation_frame(long)

    # ------------------------------------------------------------------ #
    def validate(
        self, observations: pd.DataFrame, *, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[QCTestResult]:
        """CDIP's own primary flag plus the project's QARTOD and physical checks.

        Results are aligned with the provider-reported significant wave height
        series (sorted by time). ``config`` follows ``configs/qc/wave_qc.yaml``;
        the gap test defaults to CDIP's 30-minute interval.
        """
        cfg = config or {}
        rng = cfg.get("gross_range", {})
        obs = observations[observations["method"] == "provider_reported"]
        hs = obs[obs["variable"] == "wave_significant_height"].sort_values("time_utc")
        tp = obs[obs["variable"] == "wave_peak_period"].set_index("time_utc")["value"]
        t = hs["time_utc"].to_numpy()
        v = hs["value"].to_numpy(dtype=np.float64)
        hs_rng = rng.get("wave_significant_height", {"fail": [0.0, 25.0], "suspect": [0.05, 15.0]})
        results = [
            provider_flag_test(hs["source_qc_flag"]),
            qartod.gap_test(t, cfg.get("expected_interval_s", RECORD_INTERVAL_S)),
            qartod.gross_range_test(
                v, tuple(hs_rng["fail"]), tuple(hs_rng["suspect"]), name="gross_range_hs"
            ),
            qartod.flat_line_test(v, **cfg.get("flat_line", {})),
            qartod.rate_of_change_test(v, t, **cfg.get("rate_of_change", {"max_change": 2.0})),
            qartod.mean_std_test(v, t, **cfg.get("mean_std", {})),
        ]
        tp_aligned = tp.reindex(pd.DatetimeIndex(hs["time_utc"])).to_numpy(dtype=np.float64)
        results.append(steepness_test(v, tp_aligned, **cfg.get("steepness", {})))
        return results


def provider_flag_test(source_flags: Iterable[Any]) -> QCTestResult:
    """CDIP's primary flag as a QC result (its values are the QARTOD values).

    Reads the primary part of ``source_qc_flag`` (``"<primary>/<secondary>"``);
    anything unreadable or outside ``{1, 2, 3, 4, 9}`` is NOT_EVALUATED.
    """
    valid = {int(f) for f in QCFlag}
    flags = []
    for s in source_flags:
        head = str(s).split("/", 1)[0] if s is not None and not pd.isna(s) else ""
        flags.append(int(head) if head.isdigit() and int(head) in valid else 2)
    return QCTestResult(
        "cdip_primary_flag",
        "provider",
        np.asarray(flags, dtype=np.int8),
        {"scheme": "UNESCO/IOC 2013 (1 good, 2 not_evaluated, 3 questionable, 4 bad, 9 missing)"},
        "CDIP waveFlagPrimary; https://cdip.ucsd.edu/documentation",
    )


def _flag_strings(values: Any) -> list[str]:
    arr = np.asarray(values, dtype=np.float64)
    return ["" if np.isnan(x) else str(int(x)) for x in arr]


def _naive_utc(t: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts
