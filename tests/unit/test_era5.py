"""ERA5 adapter: requests, resumable monthly retrieval, parsing and normalization.

Files are CDS-shaped netCDF written here (dims valid_time/latitude/longitude,
an ``expver`` coordinate, GRIB_paramId attributes, NaN over land), as in the
files returned by the CDS on 2026-09-26. No network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

# netCDF4's write path triggers NumPy 2.5's shape deprecation (as in test_cdip).
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning"
)

from wave_analysis.cli import build_parser, main  # noqa: E402
from wave_analysis.ingest.manifest import EntryStatus  # noqa: E402
from wave_analysis.schemas.observation import validate_observation_frame  # noqa: E402
from wave_analysis.schemas.variables import VARIABLES  # noqa: E402
from wave_analysis.sources.era5 import (  # noqa: E402
    MAX_FIELDS,
    PARAMETERS,
    CDSConfigError,
    ERA5WaveSource,
    area_around,
    available_until,
    build_request,
    check_config,
    month_status,
    months,
    nearest_sea_point,
    raw_path,
    request_digest,
    request_fields,
)

LATS = (22.5, 22.0, 21.5, 21.0)
LONS = (-158.5, -158.0, -157.5)
SITE = (21.643578, -158.06339)  # PacIOOS Waimea camera
LAND = ((2, 1),)  # (21.5, -158.0): nearest to the site, over Oahu
KEY = "00000000-test-key-0000-000000000000"
DAY = pd.Timedelta(days=1)


def cds_file(
    path: Path,
    hours: pd.DatetimeIndex,
    *,
    expver: str | list[str] = "0001",
    land: tuple[tuple[int, int], ...] = LAND,
    param_ids: dict[str, int] | None = None,
) -> None:
    """Write a netCDF file shaped like a CDS reanalysis-era5-single-levels response."""
    shape = (len(hours), len(LATS), len(LONS))
    ramp = np.linspace(0.0, 0.5, len(hours), dtype=np.float32)[:, None, None]
    data_vars = {}
    for p in PARAMETERS:
        spec = VARIABLES[p.variable]
        if spec.is_angle:
            value, units = 360.0, ("Degree true" if p.short == "mwd" else "degrees")
        elif spec.units == "m":
            value, units = 1.5, "m"
        else:
            value, units = 9.0, "s"
        v = np.full(shape, value, dtype=np.float32) + (0 if spec.is_angle else ramp)
        for i, j in land:
            v[:, i, j] = np.nan
        pid = (param_ids or {}).get(p.short, p.param_id)
        data_vars[p.short] = (
            ("valid_time", "latitude", "longitude"),
            v,
            {"units": units, "GRIB_paramId": pid, "GRIB_stepType": "instant"},
        )
    ev = [expver] * len(hours) if isinstance(expver, str) else expver
    ds = xr.Dataset(
        data_vars,
        coords={
            "valid_time": hours.values,
            "latitude": list(LATS),
            "longitude": list(LONS),
            "expver": ("valid_time", np.array(ev)),
            "number": 0,
        },
        attrs={
            "Conventions": "CF-1.7",
            "institution": "European Centre for Medium-Range Weather Forecasts",
        },
    )
    ds.to_netcdf(path)


def month_hours(period: pd.Period, days: int | None = None) -> pd.DatetimeIndex:
    hours = pd.date_range(period.start_time, period.end_time.floor("h"), freq="h")
    return hours[hours.day <= days] if days else hours


class FakeClient:
    """Stands in for cdsapi.Client: writes a file per request, per a script."""

    key = KEY

    def __init__(self, script: dict[str, dict[str, Any]] | None = None) -> None:
        self.script = script or {}
        self.requests: list[dict[str, Any]] = []

    def retrieve(self, dataset: str, request: dict[str, Any], target: str) -> str:
        assert dataset == "reanalysis-era5-single-levels"
        self.requests.append(request)
        period = pd.Period(f"{request['year'][0]}-{request['month'][0]}", freq="M")
        how = self.script.get(str(period), {})
        if how.get("fail"):
            raise RuntimeError(f"401 Client Error: not authorised for key {KEY}")
        cds_file(
            Path(target),
            month_hours(period, how.get("days")),
            expver=how.get("expver", "0001"),
        )
        return target


# --------------------------------------------------------------------------- #
def test_parameters_map_to_registered_variables():
    assert len({p.param_id for p in PARAMETERS}) == len(PARAMETERS)
    for p in PARAMETERS:
        assert p.variable in VARIABLES
    by_short = {p.short: p.variable for p in PARAMETERS}
    # ECMWF 'mean wave period' is m-1/m0, not Tm01 (Bidlot 2020, eq. 5).
    assert by_short["mwp"] == "wave_energy_period"
    assert by_short["mp1"] == "wave_mean_period_tm01"


def test_months_area_and_request():
    assert [str(p) for p in months("2024-11", "2025-02")] == [
        "2024-11",
        "2024-12",
        "2025-01",
        "2025-02",
    ]
    with pytest.raises(ValueError):
        months("2025-02", "2024-11")
    assert area_around(*SITE) == [23.0, -159.5, 20.5, -157.0]
    assert area_around(21.5, -158.0, 0.5) == [22.0, -158.5, 21.0, -157.5]
    with pytest.raises(ValueError, match="antimeridian"):
        area_around(0.0, 179.8)
    req = build_request(pd.Period("2024-02", freq="M"), area_around(*SITE))
    assert len(req["day"]) == 29 and len(req["time"]) == 24
    assert request_fields(req) == 12 * 29 * 24 <= MAX_FIELDS
    assert request_digest(req) == request_digest(dict(req))
    other = build_request(pd.Period("2024-02", freq="M"), area_around(*SITE, 2.0))
    assert request_digest(other) != request_digest(req)


def test_check_config_never_needs_the_key(tmp_path: Path):
    assert check_config({"CDSAPI_URL": "u", "CDSAPI_KEY": "k"}, tmp_path / "none")
    with pytest.raises(CDSConfigError, match="no CDS key"):
        check_config({}, tmp_path / "none")
    rc = tmp_path / ".cdsapirc"
    rc.write_text("url: https://cds.climate.copernicus.eu/api\nkey: x\n")
    rc.chmod(0o644)
    with pytest.raises(CDSConfigError, match="chmod 600"):
        check_config({}, rc)
    rc.chmod(0o600)
    assert check_config({}, rc) == str(rc)


def test_available_until_reads_the_catalogue_extent():
    meta = {
        "extent": {
            "temporal": {"interval": [["1940-01-01T00:00:00+00:00", "2026-09-20T00:00:00+00:00"]]}
        }
    }
    assert available_until(lambda url: meta) == pd.Timestamp("2026-09-20", tz="UTC")
    assert available_until(lambda url: {}) is None


# --------------------------------------------------------------------------- #
def test_month_status(tmp_path: Path):
    p = pd.Period("2024-02", freq="M")
    f = tmp_path / "x.nc"
    assert month_status(f, p) == "missing"
    cds_file(f, month_hours(p, days=10))
    assert month_status(f, p) == "incomplete"
    cds_file(f, month_hours(p), expver="0005")
    assert month_status(f, p) == "preliminary"
    cds_file(f, month_hours(p))
    assert month_status(f, p) == "final"
    f.write_bytes(b"not netcdf")
    assert month_status(f, p) == "unreadable"


def test_fetch_resumes_and_requests_unfinished_months_again(tmp_path: Path):
    periods = months("2026-06", "2026-08")
    client = FakeClient({"2026-07": {"expver": "0005"}, "2026-08": {"days": 20, "expver": "0005"}})
    src = ERA5WaveSource(client)
    sunk: list[Any] = []
    entries = src.fetch("pacioos_waimea", *SITE, periods, dest_root=tmp_path, sink=sunk.extend)
    assert [e.status for e in entries] == [EntryStatus.VERIFIED] * 3
    assert sunk == entries
    assert [e.note.split("; ")[-1] for e in entries] == [
        "final (was missing)",
        "preliminary (was missing)",
        "incomplete (was missing)",
    ]
    june = raw_path(tmp_path, "pacioos_waimea", periods[0])
    assert june.name == "era5_waves_pacioos_waimea_2026-06.nc"
    saved = json.loads(june.with_suffix(".request.json").read_text())
    assert saved["request"]["area"] == area_around(*SITE)
    assert saved["sha256"].startswith(entries[0].note.split()[1].rstrip(";"))
    assert entries[0].url.endswith("/retrieve/v1/processes/reanalysis-era5-single-levels")
    assert not list(tmp_path.rglob("*.part"))

    # June is final. The unfinished months wait until they could have changed:
    # August (incomplete) after a day, July (ERA5T) after 30 days.
    client.script = {}
    now = pd.Timestamp.now(tz="UTC")
    assert src.fetch("pacioos_waimea", *SITE, periods, dest_root=tmp_path) == []
    plan = src.discover({"pacioos_waimea": SITE}, periods, dest_root=tmp_path, now=now + DAY)
    assert plan["request"].tolist() == [False, False, True]
    again = src.fetch("pacioos_waimea", *SITE, periods, dest_root=tmp_path, now=now + 31 * DAY)
    assert [e.period for e in again] == ["2026-07", "2026-08"]
    assert {e.status for e in again} <= {EntryStatus.CHANGED, EntryStatus.UNCHANGED}
    assert len(client.requests) == 5
    assert src.fetch("pacioos_waimea", *SITE, periods, dest_root=tmp_path, now=now + 99 * DAY) == []
    assert len(src.fetch("pacioos_waimea", *SITE, periods, dest_root=tmp_path, refresh=True)) == 3


def test_failed_request_keeps_old_file_and_redacts_key(tmp_path: Path):
    p = pd.Period("2024-01", freq="M")
    src = ERA5WaveSource(FakeClient({"2024-01": {"expver": "0005"}}))
    src.fetch("s", *SITE, [p], dest_root=tmp_path)
    before = raw_path(tmp_path, "s", p).read_bytes()
    stale = raw_path(tmp_path, "s", p).parent / f".{raw_path(tmp_path, 's', p).name}.x1.part"
    stale.write_bytes(b"")  # as left by a killed run
    src._client = FakeClient({"2024-01": {"fail": True}})
    (failed,) = src.fetch("s", *SITE, [p], dest_root=tmp_path, refresh=True)
    assert failed.status == EntryStatus.FAILED and failed.sha256 is None
    assert KEY not in (failed.note or "") and "<key>" in (failed.note or "")
    assert raw_path(tmp_path, "s", p).read_bytes() == before
    assert not list(tmp_path.rglob("*.part"))
    with pytest.raises(ValueError, match="unsafe"):
        raw_path(tmp_path, "../s", p)


# --------------------------------------------------------------------------- #
def test_parse_checks_parameter_ids(tmp_path: Path):
    p = pd.Period("2024-01", freq="M")
    f = tmp_path / "x.nc"
    cds_file(f, month_hours(p, days=1), param_ids={"mwp": 140220})
    with pytest.raises(ValueError, match="GRIB_paramId"):
        ERA5WaveSource(FakeClient()).parse(f)


def test_nearest_sea_point_skips_land(tmp_path: Path):
    f = tmp_path / "x.nc"
    cds_file(f, month_hours(pd.Period("2024-01", freq="M"), days=1))
    ds = ERA5WaveSource(FakeClient()).parse(f)
    pt = nearest_sea_point(ds, *SITE)
    # (21.5, -158.0) is 17.3 km away but land; the nearest sea point is 22.0, -158.0.
    assert (pt.latitude, pt.longitude) == (22.0, -158.0)
    assert pt.distance_km == pytest.approx(40.17, abs=0.01)
    assert pt.platform_id == "era5_grid_22.00_-158.00"
    all_land = tuple((i, j) for i in range(len(LATS)) for j in range(len(LONS)))
    cds_file(f, month_hours(pd.Period("2024-01", freq="M"), days=1), land=all_land)
    with pytest.raises(ValueError, match="no grid point with sea data"):
        nearest_sea_point(ERA5WaveSource(FakeClient()).parse(f), *SITE)


def test_normalize_and_validate(tmp_path: Path):
    p = pd.Period("2026-09", freq="M")
    hours = month_hours(p, days=2)
    f = tmp_path / "x.nc"
    cds_file(f, hours, expver=["0001"] * 24 + ["0005"] * 24)
    hours = hours.tz_localize("UTC")
    src = ERA5WaveSource(FakeClient())
    obs = src.normalize(
        src.parse(f),
        site_id="pacioos_waimea",
        latitude=SITE[0],
        longitude=SITE[1],
        source_file=f.name,
        source_checksum="0" * 64,
    )
    validate_observation_frame(obs)
    assert len(obs) == 48 * len(PARAMETERS)
    assert obs["observation_id"].is_unique
    assert set(obs["station_id"]) == {"pacioos_waimea"}
    assert set(zip(obs["latitude"], obs["longitude"], strict=True)) == {(22.0, -158.0)}
    assert set(obs["time_reference"]) == {"center"} and set(obs["averaging_period_s"]) == {0.0}
    first = obs[obs["time_utc"] == hours[0]]
    last = obs[obs["time_utc"] == hours[-1]]
    assert set(first["source_qc_flag"]) == {"era5"} and set(last["source_qc_flag"]) == {"era5t"}
    by_var = first.set_index("variable")
    assert by_var.loc["wave_mean_from_direction", "value"] == 0.0  # 360 -> 0
    assert by_var.loc["wave_mean_from_direction", "units"] == "degree"
    assert by_var.loc["wave_energy_period", "method"] == "spectral_tm_10"
    assert by_var.loc["wave_significant_height", "value"] == pytest.approx(1.5)

    results = src.validate(obs.drop(obs[obs["time_utc"] == hours[5]].index))
    assert [r.test for r in results] == ["gap", "gross_range_hs", "rate_of_change", "steepness"]
    gap = results[0].flags
    assert len(gap) == 47 and gap[5] == 3 and (np.delete(gap, 5) == 1).all()


# --------------------------------------------------------------------------- #
def test_cli_plan_needs_no_key(tmp_path: Path, capsys):
    args = build_parser().parse_args(["era5", "download", "pacioos_waimea", "--start", "2009-02"])
    assert (args.box, args.refresh, args.list_only) == (1.0, False, False)
    rc = main(
        [
            "era5",
            "download",
            "pacioos_waimea",
            "--start",
            "2009-02",
            "--end",
            "2009-04",
            "--list-only",
            "--out",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "3 site-months, 3 to request (25,632 fields" in out
    with pytest.raises(SystemExit, match="unknown sites"):
        main(["era5", "download", "nowhere", "--start", "2009-02", "--list-only"])


# --------------------------------------------------------------------------- #
FIXTURE = Path(__file__).parents[1] / "fixtures/era5/era5_waves_oak_island_20260615_20260915.nc"
OAK_ISLAND = (33.9126, -78.2243)  # WebCOOS oakisland_east/west


def test_real_cds_file():
    """A file returned by the CDS: parameter ids, units, expver and the land mask."""
    src = ERA5WaveSource(FakeClient())
    ds = src.parse(FIXTURE)
    assert set(ds.data_vars) == {p.short for p in PARAMETERS}
    assert [str(e) for e in ds["expver"].values] == ["0001", "0001", "0005", "0005"]
    assert int(ds["swh"].isel(valid_time=0).notnull().sum()) == 12
    pt = nearest_sea_point(ds, *OAK_ISLAND)
    assert (pt.latitude, pt.longitude) == (34.0, -78.0)
    obs = src.normalize(
        ds,
        site_id="webcoos_oak_island",
        latitude=OAK_ISLAND[0],
        longitude=OAK_ISLAND[1],
        source_file=FIXTURE.name,
    )
    validate_observation_frame(obs)
    assert len(obs) == 4 * len(PARAMETERS)
    assert list(obs.groupby("time_utc")["source_qc_flag"].first()) == [
        "era5",
        "era5",
        "era5t",
        "era5t",
    ]
    hs = obs[obs["variable"] == "wave_significant_height"]["value"].to_numpy()
    assert hs[0] == pytest.approx(1.30, abs=0.005)
    wide = obs.pivot(index="time_utc", columns="variable", values="value")
    # The wind-sea/swell split partitions the energy, so sqrt(Hs_ww^2 + Hs_sw^2) ~ Hs;
    # in this file the sum is 0-2.3 % below the total, never above it.
    parts = np.hypot(wide["wind_wave_significant_height"], wide["swell_significant_height"])
    ratio = parts / wide["wave_significant_height"]
    assert ((ratio > 0.97) & (ratio <= 1.0 + 1e-6)).all()
    # Tm-10 >= Tm01 >= Tm02 for any spectrum.
    assert (wide["wave_energy_period"] >= wide["wave_mean_period_tm01"]).all()
    assert (wide["wave_mean_period_tm01"] >= wide["wave_mean_period_tm02"]).all()
