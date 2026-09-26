"""NDBC adapter implementing the common :class:`~wave_analysis.sources.base.DataSource` interface."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
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
from wave_analysis.sources.ndbc import parsers
from wave_analysis.sources.ndbc.inventory import InventoryResult, run_inventory
from wave_analysis.sources.ndbc.products import PRODUCTS, historical_url, realtime_url

DATASET_ID = "ndbc"

# --------------------------------------------------------------------------- #
# Unit conversions (provider unit string -> canonical)
# --------------------------------------------------------------------------- #

_UNIT_CONVERSIONS: dict[tuple[str, str], Callable[[pd.Series], pd.Series]] = {
    ("m/s", "m s-1"): lambda s: s,
    ("m", "m"): lambda s: s,
    ("sec", "s"): lambda s: s,
    ("degT", "degree"): lambda s: s % 360.0,
    ("hPa", "hPa"): lambda s: s,
    ("degC", "degree_Celsius"): lambda s: s,
    ("nmi", "m"): lambda s: s * 1852.0,
    ("mi", "m"): lambda s: s * 1609.344,
    ("ft", "m"): lambda s: s * 0.3048,
}

#: NDBC stdmet column -> (canonical variable, averaging period [s] or None).
STDMET_MAP: dict[str, tuple[str, float | None]] = {
    "WDIR": ("wind_from_direction", 480.0),
    "WSPD": ("wind_speed", 480.0),
    "GST": ("wind_speed_of_gust", None),
    "WVHT": ("wave_significant_height", 1200.0),
    "DPD": ("wave_peak_period", 1200.0),
    "APD": ("wave_mean_period", 1200.0),
    "MWD": ("wave_from_direction_at_peak", 1200.0),
    "PRES": ("air_pressure_at_sea_level", None),
    "PTDY": ("air_pressure_tendency", None),
    "ATMP": ("air_temperature", None),
    "WTMP": ("sea_surface_temperature", None),
    "DEWP": ("dew_point_temperature", None),
    "VIS": ("visibility", None),
    "TIDE": ("water_level", None),
}

#: Realtime .spec column -> canonical variable (with the column holding numeric values).
SPEC_MAP: dict[str, tuple[str, str]] = {
    "SwH": ("swell_significant_height", "SwH"),
    "SwP": ("swell_period", "SwP"),
    "WWH": ("wind_wave_significant_height", "WWH"),
    "WWP": ("wind_wave_period", "WWP"),
    "SwD": ("swell_from_direction", "SwD_deg"),
    "WWD": ("wind_wave_from_direction", "WWD_deg"),
}
_SPEC_UNITS = {"SwH": "m", "SwP": "sec", "WWH": "m", "WWP": "sec", "SwD": "degT", "WWD": "degT"}

#: Derived bulk parameter -> (canonical variable, method tag).
DERIVED_MAP: dict[str, tuple[str, str]] = {
    "Hm0": ("wave_significant_height", "spectral_hm0"),
    "Tp_smooth": ("wave_peak_period", "spectral_tp_young1995"),
    "Tm01": ("wave_mean_period_tm01", "spectral_tm01"),
    "Tm02": ("wave_mean_period_tm02", "spectral_tm02"),
    "Dm": ("wave_mean_from_direction", "spectral_dm_kuik1988"),
}


def _convert(series: pd.Series, from_units: str, to_units: str) -> pd.Series:
    try:
        fn = _UNIT_CONVERSIONS[(from_units, to_units)]
    except KeyError as exc:
        raise ValueError(f"no conversion registered for {from_units!r} -> {to_units!r}") from exc
    return fn(series)


def _long_frame(
    wide: pd.DataFrame,
    *,
    station_id: str,
    product: str,
    source_file: str,
    source_url: str | None,
    source_checksum: str | None,
    level: ProcessingLevel,
    method_by_var: dict[str, str],
    averaging_by_var: dict[str, float | None],
) -> pd.DataFrame:
    """Melt a (time x canonical-variable) frame into observation rows."""
    long = wide.reset_index().melt(id_vars="time_utc", var_name="variable", value_name="value")
    long = long.dropna(subset=["value"]).reset_index(drop=True)
    if long.empty:
        return coerce_observation_frame(long)
    long["dataset_id"] = DATASET_ID
    long["station_id"] = station_id
    long["platform_id"] = station_id
    long["units"] = long["variable"].map(lambda v: VARIABLES[v].units)
    long["method"] = long["variable"].map(method_by_var)
    long["averaging_period_s"] = long["variable"].map(averaging_by_var).astype("float64")
    long["time_reference"] = TimeReference.NOMINAL.value
    long["source_file"] = source_file
    long["source_url"] = source_url
    long["source_checksum"] = source_checksum
    long["processing_level"] = level.value
    long["processing_version"] = __version__
    long["observation_id"] = [
        make_observation_id(DATASET_ID, station_id, product, v, t)
        for v, t in zip(long["variable"], long["time_utc"], strict=True)
    ]
    return coerce_observation_frame(long)


class NDBCSource(DataSource):
    """NOAA National Data Buoy Center adapter.

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
    def discover(self, *, raw_root: Path, **kwargs: Any) -> pd.DataFrame:
        """Run the full availability inventory; returns the long file table."""
        return self.inventory(raw_root=raw_root, **kwargs).files

    def inventory(self, *, raw_root: Path, **kwargs: Any) -> InventoryResult:
        """Run the inventory and return all of its outputs."""
        return run_inventory(self.downloader, raw_root, **kwargs)

    # ------------------------------------------------------------------ #
    def fetch(
        self,
        station_id: str,
        products: Iterable[str],
        *,
        dest_root: Path,
        years: Iterable[int] | None = None,
        realtime: bool = False,
        filenames: dict[tuple[str, int], str] | None = None,
        previous: dict[str, str] | None = None,
        compress_realtime: bool = False,
    ) -> Sequence[ManifestEntry]:
        """Download historical years and/or realtime files for one station.

        Files are stored under ``dest_root/<station>/<product>/`` with their
        upstream file names. ``previous`` maps URL -> earlier SHA-256 to detect
        upstream revisions. Realtime files are snapshots of a rolling 45-day
        window, stamped with the retrieval time; ``compress_realtime`` stores
        them gzip-compressed (about 6x smaller), which suits scheduled
        collection. Parsers read either form.
        """
        entries: list[ManifestEntry] = []
        previous = previous or {}
        sid = station_id.upper()
        for prod in products:
            p = PRODUCTS[prod]
            if years is not None and p.historical_dir is not None:
                for y in years:
                    fname = (filenames or {}).get((prod, y))
                    url = historical_url(station_id, prod, y, filename=fname)
                    dest = dest_root / sid / prod / url.rsplit("/", 1)[-1]
                    entries.append(
                        self.downloader.fetch(
                            url,
                            dest,
                            source_id=DATASET_ID,
                            product=prod,
                            station_id=sid,
                            period=str(y),
                            previous_sha256=previous.get(url),
                        )
                    )
            if realtime and p.realtime_ext is not None:
                url = realtime_url(sid, prod)
                stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
                suffix = ".gz" if compress_realtime else ""
                dest = dest_root / sid / prod / f"{sid}.{p.realtime_ext}.{stamp}{suffix}"
                entries.append(
                    self.downloader.fetch(
                        url,
                        dest,
                        source_id=DATASET_ID,
                        product=prod,
                        station_id=sid,
                        period="realtime",
                        previous_sha256=previous.get(url),
                        compress=compress_realtime,
                    )
                )
        return entries

    # ------------------------------------------------------------------ #
    def parse(self, path: Path, product: str, **kwargs: Any) -> Any:
        """Dispatch to the product parser. Returns a ParsedTable, SpectralBlocks, or DataFrame."""
        p = PRODUCTS[product]
        realtime = kwargs.pop("realtime", None)
        if realtime is None:
            realtime = ".txt.gz" not in path.name
        if p.parser == "parse_stdmet":
            return parsers.parse_stdmet(path, product=product, realtime=realtime)
        if p.parser == "parse_spectral_summary":
            return parsers.parse_spectral_summary(path)
        if p.parser == "parse_spectral":
            return parsers.parse_spectral(path, product, realtime=realtime)
        if p.parser == "parse_adcp":
            return parsers.parse_adcp(path)
        raise NotImplementedError(
            f"no parser for NDBC product {product!r} (raw-only in this version)"
        )

    # ------------------------------------------------------------------ #
    def normalize(
        self,
        parsed: Any,
        *,
        station_id: str,
        source_file: str,
        source_url: str | None = None,
        source_checksum: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Map a parsed stdmet or spec table to the common observation schema."""
        if not isinstance(parsed, parsers.ParsedTable):
            raise TypeError(
                "normalize() expects a ParsedTable; use normalize_spectrum() for spectra"
            )
        if parsed.product == "stdmet":
            mapping = {c: STDMET_MAP[c] for c in parsed.data.columns if c in STDMET_MAP}
            wide = pd.DataFrame(index=parsed.data.index)
            averaging: dict[str, float | None] = {}
            for col, (var, avg) in mapping.items():
                wide[var] = _convert(
                    parsed.data[col], parsed.units.get(col, "unknown"), VARIABLES[var].units
                )
                averaging[var] = avg
        elif parsed.product == "spec":
            wide = pd.DataFrame(index=parsed.data.index)
            averaging = {}
            for col, (var, value_col) in SPEC_MAP.items():
                if value_col in parsed.data:
                    wide[var] = _convert(
                        parsed.data[value_col], _SPEC_UNITS[col], VARIABLES[var].units
                    )
                    averaging[var] = 1200.0
        else:
            raise NotImplementedError(
                f"normalisation for product {parsed.product!r} not implemented"
            )
        return _long_frame(
            wide,
            station_id=station_id.upper(),
            product=parsed.product,
            source_file=source_file,
            source_url=source_url,
            source_checksum=source_checksum,
            level=ProcessingLevel.STANDARDIZED,
            method_by_var=dict.fromkeys(wide.columns, "provider_reported"),
            averaging_by_var=averaging,
        )

    def normalize_spectrum(
        self,
        spectrum: FrequencySpectrum,
        *,
        station_id: str,
        source_file: str,
        source_url: str | None = None,
        source_checksum: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(spectral_long_table, derived_bulk_observations)``.

        The spectral table preserves S(f) and all directional coefficients (the
        highest-information form); bulk parameters are derived observations
        tagged with their estimator in ``method``.
        """
        spec_long = spectrum.to_long_frame()
        spec_long.insert(0, "station_id", station_id.upper())
        spec_long.insert(0, "dataset_id", DATASET_ID)
        spec_long["source_file"] = source_file
        spec_long["source_checksum"] = source_checksum
        spec_long["processing_version"] = __version__

        bulk = bulk_parameters(spectrum).set_index("time_utc")
        wide = pd.DataFrame(index=bulk.index)
        methods = {}
        for col, (var, method) in DERIVED_MAP.items():
            if col in bulk:
                wide[var] = bulk[col]
                methods[var] = method
        obs = _long_frame(
            wide,
            station_id=station_id.upper(),
            product="spectrum_derived",
            source_file=source_file,
            source_url=source_url,
            source_checksum=source_checksum,
            level=ProcessingLevel.DERIVED,
            method_by_var=methods,
            averaging_by_var=dict.fromkeys(wide.columns, 1200.0),
        )
        return spec_long, obs

    # ------------------------------------------------------------------ #
    def validate(
        self, observations: pd.DataFrame, *, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[QCTestResult]:
        """Run QARTOD bulk-wave tests and physical checks on provider-reported waves.

        Returns per-test results aligned with the *significant wave height*
        series of the input (sorted by time). ``config`` follows
        ``configs/qc/wave_qc.yaml``.
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
            qartod.gap_test(t, cfg.get("expected_interval_s", 3600.0)),
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
