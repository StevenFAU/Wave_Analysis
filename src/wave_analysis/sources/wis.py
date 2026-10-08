"""USACE Wave Information Study (WIS) hindcast at buoy locations.

Access (verified 2026-10-08): THREDDS, no authentication
(``chlthredds.erdc.dren.mil`` redirects to ``chldata.erdc.dren.mil``). Besides
its fixed output stations (``ST6xxxx``, ``ST8xxxx``, ...), WIS writes output at
the positions of the buoys it is evaluated against, named after the buoy::

    https://chldata.erdc.dren.mil/thredds/fileServer/wis/WIS_buoy_sites/{basin}/ST{stn}/ST{stn}.nc4
    https://chldata.erdc.dren.mil/thredds/fileServer/wis/WIS_buoy_sites/{basin}/ST{stn}/{yyyy}/WIS-ocean_waves_ST{stn}_{yyyymm}.nc

The ``.nc4`` file holds the bulk parameters for the whole record; each monthly
file holds the same bulk parameters plus the 2-D spectrum. ``{stn}`` is the
NDBC/WMO id (CDIP 106 is ``51201``). A buoy appears in a basin only for the
years it was on WIS's evaluation list (51201: 1980-2025; 46274: 2022-2025),
and a station can appear in two basins (51003: Pacific and Alaska).

Conventions, from the file attributes (ST51201, 2026-10-08)
-----------------------------------------------------------
* Hourly, instantaneous model output (WAVEWATCH III 4.16 in the Atlantic and
  Pacific, WAM 4.5.1C in the Gulf of Mexico; global attribute ``model``).
  ``time`` is seconds since 1970-01-01 UTC. A monthly file runs from 01:00 on
  the first to 00:00 on the first of the next month.
* ``waveHs`` = 4 sqrt(m0). ``waveTm`` is the energy period m-1/m0 (CF name
  "...inverse_frequency_moment"), ``waveTm1`` = m0/m1, ``waveTm2`` =
  sqrt(m0/m2). ``waveTpPeak`` is the period of the spectral peak band and
  ``waveTp`` a parabolic fit around it. The ``Windsea`` and ``Swell``
  variables apply the same definitions to WIS's wind-sea/swell split.
* Bulk directions (``waveMeanDirection*``, ``windDirection``) are degrees
  true, coming-from (``convention = "meteorologic"``). The 2-D spectrum
  ``directionalWaveEnergyDensity`` [m2 s rad-1] is the opposite: directions
  are going-**toward** (``direction_convention = "toward"``), 72 bins of 5 deg.
  Frequencies are 29 bins, 0.035-0.505 Hz, each 1.1 times the last.
* Quirks: in some months (all of 1988-2017 at 51201) the per-record
  ``longitude`` is stored 360 deg too far west (-518.117 for -158.117); the
  scalar ``lon`` of the ``.nc4`` file is NaN. Positions are wrapped to
  [-180, 180). The README beside the files says the netCDF files are "under
  evaluation" and to use them "at your own risk".
* Winds: OWI-adapted NCEP/NCAR Reanalysis-1 with scatterometer adjustments
  before 2020 and ERA5 from 2020; WIS uses buoys for evaluation, not
  assimilation (WIS presentation, ``wis.erdc.dren.mil/files/RD26_Presentation_USACE_WIS.pdf``).
* Licence (global attribute ``license``): "These data may be redistributed
  and used without restriction", for scholarly use with acknowledgement of the
  USACE Wave Information Study; commercial use needs written permission.

See ``docs/datasets/wis.md``.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wave_analysis import __version__
from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import ManifestEntry
from wave_analysis.processing.spectra import FrequencySpectrum, bulk_parameters
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

DATASET_ID = "usace_wis"
THREDDS = "https://chldata.erdc.dren.mil/thredds"
ACKNOWLEDGEMENT = "USACE Coastal and Hydraulics Laboratory, WIS team"

#: Basins of ``WIS_buoy_sites``, in the order a station found in two is assigned.
BASINS = ("Pacific", "Atlantic", "GulfOfMexico", "GreatLakes", "Alaska")
#: Products this adapter downloads.
PRODUCTS = ("bulk", "month")
#: Record interval [s].
RECORD_INTERVAL_S = 3600.0

#: WIS variable -> (canonical variable, method tag).
BULK_MAP: dict[str, tuple[str, str]] = {
    "waveHs": ("wave_significant_height", "spectral_hm0"),
    "waveTpPeak": ("wave_peak_period", "spectral_tp_band"),
    "waveTm": ("wave_energy_period", "spectral_tm_10"),
    "waveTm1": ("wave_mean_period_tm01", "spectral_tm01"),
    "waveTm2": ("wave_mean_period_tm02", "spectral_tm02"),
    "waveMeanDirection": ("wave_mean_from_direction", "spectral_mean_direction"),
    "waveHsWindsea": ("wind_wave_significant_height", "spectral_hm0"),
    "waveTmWindsea": ("wind_wave_energy_period", "spectral_tm_10"),
    "waveMeanDirectionWindsea": ("wind_wave_from_direction", "spectral_mean_direction"),
    "waveHsSwell": ("swell_significant_height", "spectral_hm0"),
    "waveTmSwell": ("swell_energy_period", "spectral_tm_10"),
    "waveMeanDirectionSwell": ("swell_from_direction", "spectral_mean_direction"),
    "windSpeed": ("wind_speed", "model_10m"),
    "windDirection": ("wind_from_direction", "model_10m"),
}
#: Derived bulk parameter (from the 2-D spectrum) -> (canonical variable, method tag).
DERIVED_MAP: dict[str, tuple[str, str]] = {
    "Hm0": ("wave_significant_height", "spectral_hm0"),
    "Tp_smooth": ("wave_peak_period", "spectral_tp_young1995"),
    "Tm01": ("wave_mean_period_tm01", "spectral_tm01"),
    "Tm02": ("wave_mean_period_tm02", "spectral_tm02"),
    "Dm": ("wave_mean_from_direction", "spectral_dm_kuik1988"),
}
_UNITS = {"m": "m", "s": "s", "degree": "degree", "m s-1": "m s-1"}
_SPECTRUM = "directionalWaveEnergyDensity"

_STATION = re.compile(r"^[A-Z0-9]{5}$")
_MONTH_FILE = re.compile(r"^WIS-ocean_waves_ST(?P<stn>[A-Z0-9]+)_(?P<ym>\d{6})\.nc$")
_SIZE_UNITS = {"bytes": 1.0, "Kbytes": 1e3, "Mbytes": 1e6, "Gbytes": 1e9, "Tbytes": 1e12}
_THREDDS_NS = {"t": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}
_XLINK = "{http://www.w3.org/1999/xlink}"


def station_code(station: str | int) -> str:
    """WIS buoy-site code: the five-character NDBC/WMO id (``51201``), ``ST`` prefix removed."""
    code = str(station).strip().upper().removeprefix("ST")
    if not _STATION.match(code):
        raise ValueError(f"not an NDBC/WMO station id: {station!r}")
    return code


def bulk_path(basin: str, station: str | int) -> str:
    """Path of a buoy site's whole-record bulk file below ``thredds/<service>/``."""
    stn = station_code(station)
    return f"wis/WIS_buoy_sites/{_basin(basin)}/ST{stn}/ST{stn}.nc4"


def month_path(basin: str, station: str | int, period: str | pd.Period) -> str:
    """Path of a buoy site's monthly file (bulk + 2-D spectrum) below ``thredds/<service>/``."""
    stn = station_code(station)
    p = pd.Period(period, freq="M")
    return (
        f"wis/WIS_buoy_sites/{_basin(basin)}/ST{stn}/{p.year}/"
        f"WIS-ocean_waves_ST{stn}_{p.year}{p.month:02d}.nc"
    )


def file_url(path: str, service: str = "fileServer") -> str:
    """THREDDS URL of a WIS file for a service (``fileServer`` or ``dodsC``)."""
    return f"{THREDDS}/{service}/{path}"


def catalog_url(basin: str, station: str | int | None = None, year: int | None = None) -> str:
    """THREDDS catalog of a basin's buoy sites, one site, or one site-year."""
    url = f"{THREDDS}/catalog/wis/WIS_buoy_sites/{_basin(basin)}"
    if station is not None:
        url += f"/ST{station_code(station)}"
        if year is not None:
            url += f"/{year}"
    return url + "/catalog.xml"


def raw_path(dest_root: Path, station: str | int, product: str, period: str | None = None) -> Path:
    """Where a downloaded file is kept: ``<dest_root>/<stn>/`` with the upstream name."""
    stn = station_code(station)
    if product == "bulk":
        return dest_root / stn / f"ST{stn}.nc4"
    if product == "month":
        if period is None:
            raise ValueError("product 'month' needs a period")
        name = month_path(BASINS[0], stn, period).rsplit("/", 1)[-1]
        return dest_root / stn / str(pd.Period(period, freq="M").year) / name
    raise ValueError(f"unknown WIS product {product!r}; expected one of {PRODUCTS}")


def parse_basin_catalog(xml_text: str) -> list[str]:
    """Station codes listed in a basin's ``WIS_buoy_sites`` catalog."""
    root = ET.fromstring(xml_text)
    out = []
    for ref in root.iter(f"{{{_THREDDS_NS['t']}}}catalogRef"):
        title = ref.get(f"{_XLINK}title") or ""
        if title.startswith("ST"):
            try:
                out.append(station_code(title))
            except ValueError:
                continue
    return out


def parse_station_catalog(xml_text: str) -> dict[str, Any]:
    """Bulk-file size [bytes, approximate] and the years with monthly files, from a site catalog."""
    root = ET.fromstring(xml_text)
    size = np.nan
    for ds in root.iter(f"{{{_THREDDS_NS['t']}}}dataset"):
        if (ds.get("urlPath") or "").endswith(".nc4"):
            el = ds.find("t:dataSize", _THREDDS_NS)
            if el is not None and el.text:
                size = float(el.text) * _SIZE_UNITS.get(el.get("units", ""), np.nan)
    years = sorted(
        int(t)
        for ref in root.iter(f"{{{_THREDDS_NS['t']}}}catalogRef")
        if (t := ref.get(f"{_XLINK}title") or "").isdigit()
    )
    return {"approx_size_bytes": size, "years": years}


def _basin(basin: str) -> str:
    if basin not in BASINS:
        raise ValueError(f"unknown WIS basin {basin!r}; expected one of {BASINS}")
    return basin


def wrap_longitude(lon: Any) -> np.ndarray:
    """Longitudes in [-180, 180) (WIS stores some as ``lon - 360``)."""
    return np.mod(np.asarray(lon, dtype=np.float64) + 180.0, 360.0) - 180.0


class WISSource(DataSource):
    """USACE Wave Information Study hindcast output at buoy sites.

    Model output: a context variable and a non-vision baseline, never ground
    truth. ``discover`` finds each station's basin; ``fetch`` downloads whole
    files as served (the hindcast is re-run and replaced from time to time, so
    the manifest records whether a re-fetched file changed).
    """

    dataset_id = DATASET_ID

    def __init__(self, downloader: Downloader | None = None) -> None:
        self.downloader = downloader or Downloader()
        self._basins: dict[str, list[str]] | None = None

    # ------------------------------------------------------------------ #
    def basins(self) -> dict[str, list[str]]:
        """Station codes per basin (one catalog request per basin, cached)."""
        if self._basins is None:
            out: dict[str, list[str]] = {}
            for b in BASINS:
                resp = self.downloader.get(catalog_url(b))
                resp.raise_for_status()
                out[b] = parse_basin_catalog(resp.text)
            self._basins = out
        return self._basins

    def basin_of(self, station: str | int) -> str | None:
        """The basin a station is fetched from (first of :data:`BASINS` that lists it)."""
        stn = station_code(station)
        return next((b for b, ids in self.basins().items() if stn in ids), None)

    def discover(self, stations: Iterable[str | int], **kwargs: Any) -> pd.DataFrame:
        """One row per station: basin, other basins listing it, bulk-file size, years."""
        rows = []
        listing = self.basins()
        for st in stations:
            stn = station_code(st)
            found = [b for b, ids in listing.items() if stn in ids]
            row: dict[str, Any] = {
                "station_id": stn,
                "basin": found[0] if found else None,
                "other_basins": ",".join(found[1:]),
                "approx_size_bytes": np.nan,
                "first_year": pd.NA,
                "last_year": pd.NA,
                "url": None,
            }
            if found:
                resp = self.downloader.get(catalog_url(found[0], stn))
                resp.raise_for_status()
                info = parse_station_catalog(resp.text)
                row["approx_size_bytes"] = info["approx_size_bytes"]
                if info["years"]:
                    row["first_year"], row["last_year"] = info["years"][0], info["years"][-1]
                row["url"] = file_url(bulk_path(found[0], stn))
            rows.append(row)
        out = pd.DataFrame(rows, columns=list(rows[0]) if rows else ["station_id"])
        for c in ("first_year", "last_year"):
            if c in out:
                out[c] = out[c].astype("Int64")
        return out

    # ------------------------------------------------------------------ #
    def fetch(
        self,
        station_id: str | int,
        basin: str,
        *,
        dest_root: Path,
        bulk: bool = True,
        months: Iterable[str | pd.Period] = (),
        previous: dict[str, str] | None = None,
    ) -> Sequence[ManifestEntry]:
        """Download a station's bulk file and/or monthly files into ``dest_root/<stn>/``.

        Files keep their upstream names and are replaced on re-fetch; the
        manifest's ``previous`` checksums mark them ``unchanged`` or ``changed``.
        """
        stn = station_code(station_id)
        previous = previous or {}
        jobs: list[tuple[str, str, Path, str]] = []
        if bulk:
            jobs.append(("bulk", "all", raw_path(dest_root, stn, "bulk"), bulk_path(basin, stn)))
        for m in months:
            p = str(pd.Period(m, freq="M"))
            jobs.append(
                ("month", p, raw_path(dest_root, stn, "month", p), month_path(basin, stn, p))
            )
        entries = []
        for product, period, dest, path in jobs:
            url = file_url(path)
            entries.append(
                self.downloader.fetch(
                    url,
                    dest,
                    source_id=DATASET_ID,
                    product=product,
                    station_id=stn,
                    period=period,
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
        """Open a WIS file as an ``xarray.Dataset`` restricted to times in ``[start, end)``.

        Fill values (-999) become NaN and times are decoded to UTC; the file's
        global attributes are kept. Raises ``ValueError`` for a file without
        ``waveHs`` or a ``model`` attribute (not a WIS station file).
        """
        import xarray as xr

        with xr.open_dataset(path, mask_and_scale=True, decode_times=True) as full:
            if "waveHs" not in full.variables or "model" not in full.attrs:
                raise ValueError(f"{Path(path).name}: not a WIS station file")
            t = pd.DatetimeIndex(full["time"].values)
            sel = np.ones(len(t), dtype=bool)
            if start is not None:
                sel &= t >= _naive_utc(start)
            if end is not None:
                sel &= t < _naive_utc(end)
            ds = full.isel(time=np.flatnonzero(sel)).load()
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
        """Bulk parameters (:data:`BULK_MAP`) as model observations at the buoy site.

        Values are instantaneous (``time_reference = center``,
        ``averaging_period_s = 0``). ``platform_id`` is ``wis_ST<stn>`` and the
        position is the output point's, wrapped to [-180, 180).
        ``source_qc_flag`` is the model named in the file.
        """
        ds = parsed
        stn = station_code(station_id)
        times = pd.DatetimeIndex(ds["time"].values).tz_localize("UTC")
        lat = np.broadcast_to(np.asarray(ds["latitude"].values, dtype=np.float64), (len(times),))
        lon = np.broadcast_to(wrap_longitude(ds["longitude"].values), (len(times),))
        frames = []
        for wis_var, (var, method) in BULK_MAP.items():
            if wis_var not in ds:
                continue
            raw_units = str(ds[wis_var].attrs.get("units", ""))
            units = _UNITS.get(raw_units)
            spec = VARIABLES[var]
            if units != spec.units:
                raise ValueError(f"{wis_var}: units {raw_units!r} do not map to {spec.units!r}")
            values = np.asarray(ds[wis_var].values, dtype=np.float64)
            if spec.is_angle:
                values = np.mod(values, 360.0)
            frame = pd.DataFrame(
                {
                    "time_utc": times,
                    "variable": var,
                    "value": values,
                    "units": units,
                    "method": method,
                    "latitude": lat,
                    "longitude": lon,
                }
            )
            frames.append(frame[frame["value"].notna()])
        if not frames:
            return coerce_observation_frame(pd.DataFrame(columns=["time_utc"]))
        long = pd.concat(frames, ignore_index=True)
        long["dataset_id"] = DATASET_ID
        long["station_id"] = stn
        long["platform_id"] = f"wis_ST{stn}"
        long["time_reference"] = TimeReference.CENTER.value
        long["averaging_period_s"] = 0.0
        long["source_qc_flag"] = str(ds.attrs.get("model", ""))
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
        r"""The monthly files' 2-D spectra as the project's directional frequency spectra.

        :math:`S(f) = \sum_\theta E(f,\theta)\,\Delta\theta`; the first and
        second directional moments are taken with each bin's direction turned
        from going-toward to coming-from (+180 deg), so ``alpha1``/``alpha2``
        are nautical coming-from like every other source. Bandwidths are
        WAVEWATCH III's, :math:`\Delta f_i = f_i (r - 1/r)/2` for the grid's
        ratio :math:`r`; they reproduce ``waveHs`` to about 0.2 % (the model adds a
        high-frequency tail beyond the last bin).
        """
        ds = parsed
        if _SPECTRUM not in ds:
            raise ValueError("no 2-D spectrum: use a monthly file")
        conv = str(ds[_SPECTRUM].attrs.get("direction_convention", ""))
        if conv != "toward":
            raise ValueError(f"unexpected direction convention {conv!r}")
        units = str(ds[_SPECTRUM].attrs.get("units", ""))
        if units != "m2 s rad-1":
            raise ValueError(f"unexpected spectral units {units!r}")
        e = ds[_SPECTRUM].transpose("time", "waveFrequency", "waveDirectionBins").to_numpy()
        f = np.asarray(ds["waveFrequency"].values, dtype=np.float64)
        theta_to = np.asarray(ds["waveDirectionBins"].values, dtype=np.float64)
        dtheta = np.deg2rad(360.0 / theta_to.size)
        th = np.deg2rad(np.mod(theta_to + 180.0, 360.0))  # coming-from, nautical
        density = np.nansum(e, axis=2) * dtheta
        with np.errstate(invalid="ignore", divide="ignore"):
            m = {
                k: np.nansum(e * fn(n * th), axis=2) * dtheta / density
                for k, fn, n in (
                    ("a1", np.cos, 1),
                    ("b1", np.sin, 1),
                    ("a2", np.cos, 2),
                    ("b2", np.sin, 2),
                )
            }
        alpha1 = np.mod(np.rad2deg(np.arctan2(m["b1"], m["a1"])), 360.0)
        alpha2 = np.mod(0.5 * np.rad2deg(np.arctan2(m["b2"], m["a2"])), 360.0)
        ratio = float(np.median(f[1:] / f[:-1]))
        return FrequencySpectrum(
            time=np.asarray(ds["time"].values, dtype="datetime64[ns]"),
            frequency=f,
            density=density,
            bandwidth=0.5 * f * (ratio - 1.0 / ratio),
            alpha1=alpha1,
            alpha2=alpha2,
            r1=np.hypot(m["a1"], m["b1"]),
            r2=np.hypot(m["a2"], m["b2"]),
            meta={
                "source": DATASET_ID,
                "station_id": station_code(station_id),
                "time_reference": TimeReference.CENTER.value,
                "averaging_period_s": 0.0,
                "model": ds.attrs.get("model"),
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
        """Return ``(spectral_long_table, derived_bulk_observations)`` as for CDIP."""
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
        long["platform_id"] = f"wis_ST{stn}"
        long["units"] = long["variable"].map(lambda v: VARIABLES[v].units)
        long["averaging_period_s"] = 0.0
        long["time_reference"] = TimeReference.CENTER.value
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
        """QARTOD gap, range and rate-of-change tests plus steepness on the H_s series.

        As for ERA5: model output needs no instrument checks; these catch a
        broken file (missing hours, units, a wrong variable).
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


def month_of_file(name: str) -> str | None:
    """``YYYY-MM`` of a monthly file name, or ``None`` for another file."""
    m = _MONTH_FILE.match(name)
    return f"{m['ym'][:4]}-{m['ym'][4:]}" if m else None


def _naive_utc(t: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts
