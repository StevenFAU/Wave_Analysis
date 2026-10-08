"""WIS adapter: catalogs, paths, download, parsing and normalisation on WIS-shaped files."""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from wave_analysis.cli import _wis_months, _wis_stations
from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import EntryStatus
from wave_analysis.processing.spectra import bulk_parameters
from wave_analysis.schemas.observation import validate_observation_frame
from wave_analysis.sources.wis import (
    BULK_MAP,
    WISSource,
    bulk_path,
    catalog_url,
    file_url,
    month_of_file,
    month_path,
    parse_basin_catalog,
    parse_station_catalog,
    raw_path,
    station_code,
    wrap_longitude,
)

nc4 = pytest.importorskip("netCDF4")
pytest.importorskip("xarray")

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning"
)

T0 = pd.Timestamp("2012-01-01T01:00:00")
FREQ = 0.035 * 1.1 ** np.arange(6)
DIRS = np.arange(87.5, -270.0, -5.0) % 360.0  # 72 bins, WIS order (87.5, 82.5, ...)
FILL = -999.0
UNITS = {
    "waveHs": "m",
    "waveTpPeak": "s",
    "waveTp": "s",
    "waveTm": "s",
    "waveTm1": "s",
    "waveTm2": "s",
    "waveMeanDirection": "degree",
    "waveHsWindsea": "m",
    "waveTmWindsea": "s",
    "waveMeanDirectionWindsea": "degree",
    "waveHsSwell": "m",
    "waveTmSwell": "s",
    "waveMeanDirectionSwell": "degree",
    "windSpeed": "m s-1",
    "windDirection": "degree",
}


def write_wis_file(path: Path, *, n: int = 4, spectrum: bool = False) -> Path:
    """A small file with WIS's variable names, fill value, attributes and longitude quirk."""
    with nc4.Dataset(path, "w") as nc:
        nc.createDimension("time", n)
        nc.setncattr("title", "WIS Pacific Hindcast")
        nc.setncattr("model", "WaveWatch III version 4.16")
        nc.setncattr("station", "51201")
        t = nc.createVariable("time", "f8", ("time",), fill_value=FILL)
        t.units = "seconds since 1970-01-01 00:00:00 UTC"
        t.calendar = "standard"
        t[:] = T0.value // 10**9 + 3600 * np.arange(n)
        lat = nc.createVariable("latitude", "f8", ("time",))
        lat[:] = np.full(n, 21.6705)
        lon = nc.createVariable("longitude", "f8", ("time",))
        lon[:] = np.full(n, -518.117126)  # stored 360 deg too far west
        for name, units in UNITS.items():
            v = nc.createVariable(name, "f8", ("time",), fill_value=FILL)
            v.units = units
            if "Direction" in name:
                v.convention = "meteorologic"
            values = np.full(n, {"m": 1.5, "s": 12.0, "degree": 350.0, "m s-1": 7.0}[units])
            if name == "waveHs":
                values[1] = FILL
            v[:] = values
        if spectrum:
            nc.createDimension("waveFrequency", len(FREQ))
            nc.createDimension("waveDirectionBins", len(DIRS))
            f = nc.createVariable("waveFrequency", "f8", ("waveFrequency",))
            f.units = "s-1"
            f[:] = FREQ
            d = nc.createVariable("waveDirectionBins", "f8", ("waveDirectionBins",))
            d.units = "degree"
            d[:] = DIRS
            e = nc.createVariable(
                "directionalWaveEnergyDensity",
                "f8",
                ("time", "waveFrequency", "waveDirectionBins"),
                fill_value=FILL,
            )
            e.units = "m2 s rad-1"
            e.direction_convention = "toward"
            # All energy in the bin going toward 267.5 deg (west): coming from 87.5 deg.
            grid = np.zeros((n, len(FREQ), len(DIRS)))
            grid[:, 2, np.flatnonzero(np.isclose(DIRS, 267.5))[0]] = 10.0
            e[:] = grid
    return path


def test_station_codes_and_paths(tmp_path):
    assert station_code(51201) == station_code("ST51201") == "51201"
    with pytest.raises(ValueError, match="not an NDBC"):
        station_code("106")
    assert bulk_path("Pacific", "51201") == "wis/WIS_buoy_sites/Pacific/ST51201/ST51201.nc4"
    assert month_path("Atlantic", "41113", "2009-03") == (
        "wis/WIS_buoy_sites/Atlantic/ST41113/2009/WIS-ocean_waves_ST41113_200903.nc"
    )
    assert file_url(bulk_path("Pacific", 51201)).startswith(
        "https://chldata.erdc.dren.mil/thredds/fileServer/wis/"
    )
    assert catalog_url("Pacific", "51201", 2012).endswith("/Pacific/ST51201/2012/catalog.xml")
    with pytest.raises(ValueError, match="unknown WIS basin"):
        bulk_path("Indian", "51201")
    assert raw_path(tmp_path, "51201", "month", "2012-01") == (
        tmp_path / "51201" / "2012" / "WIS-ocean_waves_ST51201_201201.nc"
    )
    assert month_of_file("WIS-ocean_waves_ST51201_201201.nc") == "2012-01"
    assert month_of_file("ST51201.nc4") is None
    assert wrap_longitude(-518.117126) == pytest.approx(-158.117126)


BASIN_XML = """<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink"><dataset name="Pacific/">
 <catalogRef xlink:href="ST51201/catalog.xml" xlink:title="ST51201" name=""/>
 <catalogRef xlink:href="ST46274/catalog.xml" xlink:title="ST46274" name=""/>
 <catalogRef xlink:href="Japan/catalog.xml" xlink:title="Japan" name=""/>
</dataset></catalog>"""

STATION_XML = """<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink"><dataset name="ST51201/">
 <dataset name="ST51201.nc4" urlPath="wis/WIS_buoy_sites/Pacific/ST51201/ST51201.nc4">
  <dataSize units="Mbytes">23.00</dataSize></dataset>
 <catalogRef xlink:href="2013/catalog.xml" xlink:title="2013" name=""/>
 <catalogRef xlink:href="2012/catalog.xml" xlink:title="2012" name=""/>
</dataset></catalog>"""


def test_parse_catalogs():
    assert parse_basin_catalog(BASIN_XML) == ["51201", "46274"]
    info = parse_station_catalog(STATION_XML)
    assert info == {"approx_size_bytes": 23e6, "years": [2012, 2013]}


def _mock(handler) -> Downloader:
    return Downloader(min_interval_s=0, transport=httpx.MockTransport(handler))


def test_discover_and_fetch(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if url.endswith("/Pacific/catalog.xml"):
            return httpx.Response(200, text=BASIN_XML)
        if url.endswith("/ST51201/catalog.xml"):
            return httpx.Response(200, text=STATION_XML)
        if url.endswith("catalog.xml"):
            return httpx.Response(200, text=BASIN_XML.replace("ST51201", "ST41113"))
        return httpx.Response(200, content=b"CDF" + url.encode()[-6:])

    src = WISSource(_mock(handler))
    table = src.discover(["51201", "99999"])
    row = table.set_index("station_id").loc["51201"]
    assert (row["basin"], row["first_year"], row["last_year"]) == ("Pacific", 2012, 2013)
    assert pd.isna(table.set_index("station_id").loc["99999", "basin"])
    assert src.basin_of("41113") == "Atlantic"  # first basin after Pacific that lists it

    entries = src.fetch("51201", "Pacific", dest_root=tmp_path, months=["2012-01"])
    assert [(e.product, e.period) for e in entries] == [("bulk", "all"), ("month", "2012-01")]
    assert all(e.status == EntryStatus.VERIFIED and e.source_id == "usace_wis" for e in entries)
    assert (tmp_path / "51201" / "ST51201.nc4").exists()
    assert (tmp_path / "51201" / "2012" / "WIS-ocean_waves_ST51201_201201.nc").exists()
    again = src.fetch(
        "51201", "Pacific", dest_root=tmp_path, previous={entries[0].url: entries[0].sha256}
    )
    assert again[0].status == EntryStatus.UNCHANGED


def test_parse_and_normalize(tmp_path):
    path = write_wis_file(tmp_path / "ST51201.nc4")
    src = WISSource(_mock(lambda r: httpx.Response(404)))
    ds = src.parse(path, start="2012-01-01T02:00", end="2012-01-01T04:00")
    assert ds.sizes["time"] == 2
    obs = src.normalize(ds, station_id="51201", source_file=path.name)
    validate_observation_frame(obs)
    assert set(obs["variable"]) == {v for v, _ in BULK_MAP.values()}
    hs = obs[obs["variable"] == "wave_significant_height"]
    assert len(hs) == 1  # the fill value became NaN and was dropped
    assert hs["time_utc"].iloc[0] == pd.Timestamp("2012-01-01T03:00Z")
    assert set(obs["longitude"].round(6)) == {-158.117126}
    assert set(obs["time_reference"]) == {"center"} and set(obs["averaging_period_s"]) == {0.0}
    assert set(obs["platform_id"]) == {"wis_ST51201"}
    assert set(obs["source_qc_flag"]) == {"WaveWatch III version 4.16"}
    assert obs.loc[obs["variable"] == "wave_energy_period", "method"].iloc[0] == "spectral_tm_10"
    results = {r.test: r for r in src.validate(obs)}
    assert "gross_range_hs" in results

    bad = tmp_path / "other.nc"
    with nc4.Dataset(bad, "w") as nc:
        nc.createDimension("time", 1)
        nc.createVariable("time", "f8", ("time",))[:] = [0.0]
    with pytest.raises(ValueError, match="not a WIS station file"):
        src.parse(bad)


def test_spectrum_turns_toward_into_coming_from(tmp_path):
    path = write_wis_file(tmp_path / "WIS-ocean_waves_ST51201_201201.nc", spectrum=True)
    src = WISSource(_mock(lambda r: httpx.Response(404)))
    ds = src.parse(path)
    sp = src.spectrum(ds, station_id="51201")
    dtheta = np.deg2rad(5.0)
    assert sp.density[0, 2] == pytest.approx(10.0 * dtheta)
    assert sp.alpha1[0, 2] == pytest.approx(87.5)  # going toward 267.5 = coming from 87.5
    assert sp.r1[0, 2] == pytest.approx(1.0)
    bulk = bulk_parameters(sp)
    df = 0.5 * FREQ[2] * (1.1 - 1 / 1.1)
    assert bulk["Hm0"].iloc[0] == pytest.approx(4 * np.sqrt(10.0 * dtheta * df), rel=1e-6)
    assert bulk["Dm"].iloc[0] == pytest.approx(87.5)
    spec_long, derived = src.normalize_spectrum(sp, station_id="51201", source_file=path.name)
    validate_observation_frame(derived)
    assert set(spec_long["dataset_id"]) == {"usace_wis"}
    with pytest.raises(ValueError, match="no 2-D spectrum"):
        src.spectrum(src.parse(write_wis_file(tmp_path / "bulk.nc4")), station_id="51201")


def test_cli_station_and_month_expansion():
    assert _wis_months("2012-11:2013-02") == ["2012-11", "2012-12", "2013-01", "2013-02"]
    assert _wis_months("2012-11") == ["2012-11"]
    assert _wis_months(None) == []
    stations = _wis_stations(["@sites", "46042"])
    assert stations["51201"] == "2009-01:2013-12"  # Waimea spectra configured
    assert stations["46042"] is None and "41113" in stations
