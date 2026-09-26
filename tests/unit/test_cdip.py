"""CDIP adapter: catalog, download paths, parsing and normalisation on a CDIP-shaped file."""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import EntryStatus
from wave_analysis.schemas.observation import validate_observation_frame
from wave_analysis.sources.cdip import (
    CDIPSource,
    file_path,
    file_url,
    parse_catalog,
    provider_flag_test,
    station_code,
)

nc4 = pytest.importorskip("netCDF4")
pytest.importorskip("xarray")

# netCDF4 1.7's write path sets ndarray.shape, deprecated in NumPy 2.5; only the
# fixture writer triggers it.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning"
)

T0 = pd.Timestamp("2012-01-05T17:00:00Z")
FREQ = np.array([0.05, 0.10, 0.15, 0.20])
BW = np.array([0.05, 0.05, 0.05, 0.05])
DENSITY = np.array([1.0, 4.0, 2.0, 0.5])  # m0 = 7.5 * 0.05 = 0.375 -> Hm0 = 2.449 m
FILL = -999.99


def write_cdip_file(path: Path, *, source_index: tuple[int, ...] = (1, 1, 2, 2)) -> Path:
    """A small file with CDIP's variable names, types, fill values and attributes."""
    n = len(source_index)
    start = (T0.value // 10**9) + 1800 * np.arange(n)
    with nc4.Dataset(path, "w") as nc:
        nc.createDimension("waveTime", n)
        nc.createDimension("waveFrequency", len(FREQ))
        nc.createDimension("metaBoundsCount", 2)
        nc.createDimension("metaDeployCount", 2)
        nc.createDimension("sourceCount", 2)
        nc.setncattr("id", "CDIP_106p1_test_historic")
        nc.setncattr("license", "These data may be redistributed and used without restriction.")

        def var(name, dtype, dims, data, **attrs):
            fill = attrs.pop("_FillValue", None)
            v = nc.createVariable(name, dtype, dims, fill_value=fill)
            for k, a in attrs.items():
                v.setncattr(k, a)
            v[:] = data
            return v

        units = "seconds since 1970-01-01 00:00:00 UTC"
        var("waveTime", "i4", ("waveTime",), start, units=units, calendar="standard")
        var(
            "waveTimeBounds",
            "i4",
            ("waveTime", "metaBoundsCount"),
            np.c_[start, start + 1600],
            units=units,
            calendar="standard",
        )
        var("waveFlagPrimary", "i1", ("waveTime",), [1, 1, 3, 1][:n], _FillValue=np.int8(-127))
        var("waveFlagSecondary", "i1", ("waveTime",), [0, 0, 6, 0][:n], _FillValue=np.int8(-127))
        hs = np.full(n, 4 * np.sqrt(np.sum(DENSITY * BW)))
        hs[1] = FILL
        var("waveHs", "f4", ("waveTime",), hs, _FillValue=np.float32(FILL), units="meter")
        var(
            "waveTp",
            "f4",
            ("waveTime",),
            np.full(n, 10.0),
            _FillValue=np.float32(FILL),
            units="second",
        )
        var(
            "waveTa",
            "f4",
            ("waveTime",),
            np.full(n, 7.0),
            _FillValue=np.float32(FILL),
            units="second",
        )
        var(
            "waveTz",
            "f4",
            ("waveTime",),
            np.full(n, 6.0),
            _FillValue=np.float32(FILL),
            units="second",
        )
        var(
            "waveDp",
            "f4",
            ("waveTime",),
            [360.0, 45.0, 90.0, 180.0][:n],
            _FillValue=np.float32(FILL),
            units="degreeT",
        )
        var("waveSourceIndex", "i4", ("waveTime",), source_index, _FillValue=np.int32(-99999))
        var("waveFrequency", "f4", ("waveFrequency",), FREQ, units="hertz")
        var(
            "waveBandwidth",
            "f4",
            ("waveFrequency",),
            BW,
            _FillValue=np.float32(FILL),
            units="hertz",
        )
        var(
            "waveEnergyDensity",
            "f4",
            ("waveTime", "waveFrequency"),
            np.tile(DENSITY, (n, 1)),
            _FillValue=np.float32(FILL),
            units="meter^2 second",
        )
        # Waves from the east (90 deg): a1 = r1 cos(90) = 0, b1 = r1 sin(90) = 0.5.
        for name, value in (
            ("waveA1Value", 0.0),
            ("waveB1Value", 0.5),
            ("waveA2Value", -0.3),
            ("waveB2Value", 0.0),
        ):
            var(
                name,
                "f4",
                ("waveTime", "waveFrequency"),
                np.full((n, len(FREQ)), value),
                _FillValue=np.float32(FILL),
            )
        var(
            "waveMeanDirection",
            "f4",
            ("waveTime", "waveFrequency"),
            np.full((n, len(FREQ)), 90.0),
            _FillValue=np.float32(FILL),
        )
        names = nc.createVariable("sourceFilename", str, ("sourceCount",))
        names[0] = "106p1_d01.nc"
        names[1] = "106p1_d02.nc"
        var("metaDeployLatitude", "f4", ("metaDeployCount",), [21.6727, 21.6705])
        var("metaDeployLongitude", "f4", ("metaDeployCount",), [-158.1158, -158.1171])
        var("metaWaterDepth", "f4", ("metaDeployCount",), [198.0, 200.0])
    return path


@pytest.fixture
def cdip_file(tmp_path: Path) -> Path:
    return write_cdip_file(tmp_path / "106p1_historic.nc")


# --------------------------------------------------------------------------- #
def test_station_codes_and_urls():
    assert station_code(106) == "106"
    assert station_code("67") == "067"
    assert station_code("106p1") == "106"
    with pytest.raises(ValueError):
        station_code("abc")
    assert file_path(106, "historic") == "archive/106p1/106p1_historic.nc"
    assert file_path("106", "deployment", 9) == "archive/106p1/106p1_d09.nc"
    assert file_path(106, "realtime") == "realtime/106p1_rt.nc"
    with pytest.raises(ValueError):
        file_path(106, "deployment")
    assert file_url("realtime/106p1_rt.nc").endswith(
        "/thredds/fileServer/cdip/realtime/106p1_rt.nc"
    )
    assert "/dodsC/" in file_url("realtime/106p1_rt.nc", "dodsC")


CATALOG = """<?xml version="1.0" encoding="UTF-8"?>
<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0" version="1.0.1">
  <dataset name="106p1/" ID="cdip/archive/106p1">
    <dataset name="106p1_d02.nc" ID="x2" urlPath="cdip/archive/106p1/106p1_d02.nc">
      <dataSize units="Mbytes">484.7</dataSize>
      <date type="modified">2026-05-04T20:18:36.332Z</date>
    </dataset>
    <dataset name="106p1_d01.nc" ID="x1" urlPath="cdip/archive/106p1/106p1_d01.nc">
      <dataSize units="Gbytes">1.2</dataSize>
      <date type="modified">2026-05-04T20:18:17.601Z</date>
    </dataset>
    <dataset name="106p1_historic.nc" ID="xh" urlPath="cdip/archive/106p1/106p1_historic.nc">
      <dataSize units="Gbytes">1.048</dataSize>
      <date type="modified">2026-06-09T21:13:59.069Z</date>
    </dataset>
    <dataset name="106p1_notes.txt" ID="xn" urlPath="cdip/archive/106p1/106p1_notes.txt"/>
  </dataset>
</catalog>"""


def test_parse_catalog_orders_and_sizes():
    files = parse_catalog(CATALOG, 106)
    assert files["file_name"].tolist() == ["106p1_historic.nc", "106p1_d01.nc", "106p1_d02.nc"]
    assert files["product"].tolist() == ["historic", "deployment", "deployment"]
    assert files["deployment"].tolist()[1:] == [1, 2] and pd.isna(files["deployment"].iloc[0])
    assert files["approx_size_bytes"].tolist() == pytest.approx([1.048e9, 1.2e9, 484.7e6])
    assert files["url"].iloc[0] == file_url("archive/106p1/106p1_historic.nc")
    assert str(files["modified_utc"].dt.tz) == "UTC"


def test_fetch_paths_and_manifest(tmp_path: Path):
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        return httpx.Response(200, content=b"CDF")

    dl = Downloader(min_interval_s=0, transport=httpx.MockTransport(handler))
    src = CDIPSource(dl)
    entries = src.fetch(
        106, ["historic", "deployment", "realtime"], dest_root=tmp_path, deployments=[9]
    )
    assert [e.period for e in entries] == ["historic", "d09", "realtime"]
    assert all(e.status == EntryStatus.VERIFIED and e.source_id == "cdip" for e in entries)
    assert (tmp_path / "106" / "106p1_historic.nc").read_bytes() == b"CDF"
    assert (tmp_path / "106" / "106p1_d09.nc").exists()
    assert len(list((tmp_path / "106").glob("106p1_rt.*.nc"))) == 1
    assert seen[0] == file_url("archive/106p1/106p1_historic.nc")
    with pytest.raises(ValueError, match="deployment numbers"):
        src.fetch(106, ["deployment"], dest_root=tmp_path)
    with pytest.raises(ValueError, match="unknown CDIP product"):
        src.fetch(106, ["spectra"], dest_root=tmp_path)


# --------------------------------------------------------------------------- #
def test_parse_decodes_fill_values_and_filters_time(cdip_file: Path):
    src = CDIPSource()
    ds = src.parse(cdip_file)
    assert int(ds.sizes["waveTime"]) == 4
    assert np.isnan(ds["waveHs"].values[1])
    assert pd.Timestamp(ds["waveTime"].values[0]) == T0.tz_localize(None)
    assert ds.attrs["id"] == "CDIP_106p1_test_historic"
    sub = src.parse(cdip_file, start=T0 + pd.Timedelta("30min"), end=T0 + pd.Timedelta("90min"))
    assert int(sub.sizes["waveTime"]) == 2


def test_normalize_bulk(cdip_file: Path):
    src = CDIPSource()
    obs = src.normalize(src.parse(cdip_file), station_id=106, source_file="106p1_historic.nc")
    validate_observation_frame(obs)
    # 5 variables x 4 records, less the one filled-in Hs.
    assert len(obs) == 19
    assert set(obs["variable"]) == {
        "wave_significant_height",
        "wave_peak_period",
        "wave_mean_period_tm01",
        "wave_mean_period_tm02",
        "wave_from_direction_at_peak",
    }
    assert (obs["time_reference"] == "start").all()
    assert (obs["averaging_period_s"] == 1600.0).all()
    assert (obs["method"] == "provider_reported").all()
    dp = obs[obs["variable"] == "wave_from_direction_at_peak"].sort_values("time_utc")
    assert dp["value"].tolist() == [0.0, 45.0, 90.0, 180.0]  # 360 wraps to 0
    first = obs[obs["time_utc"] == T0].iloc[0]
    assert first["deployment_id"] == "106p1_d01"
    assert first["latitude"] == pytest.approx(21.6727, abs=1e-4)
    last = obs[obs["time_utc"] == T0 + pd.Timedelta("90min")].iloc[0]
    assert last["deployment_id"] == "106p1_d02"
    assert last["longitude"] == pytest.approx(-158.1171, abs=1e-4)
    third = obs[obs["time_utc"] == T0 + pd.Timedelta("60min")]
    assert set(third["source_qc_flag"]) == {"3/6"}
    assert obs["observation_id"].is_unique
    assert obs["observation_id"].iloc[0].startswith("cdip:106:wave:")


@pytest.mark.parametrize("index", [(0, 1, 1, 2), (1, 1, 2, 3)])
def test_source_index_outside_one_based_range_raises(tmp_path: Path, index: tuple[int, ...]):
    path = write_cdip_file(tmp_path / "bad.nc", source_index=index)
    src = CDIPSource()
    with pytest.raises(ValueError, match=r"outside 1\.\.2"):
        src.normalize(src.parse(path), station_id=106, source_file="bad.nc")


def test_spectrum_directions_and_derived_hm0(cdip_file: Path):
    src = CDIPSource()
    ds = src.parse(cdip_file)
    spec = src.spectrum(ds, station_id=106)
    assert spec.is_directional
    np.testing.assert_allclose(spec.alpha1, 90.0, atol=1e-4)
    np.testing.assert_allclose(spec.r1, 0.5, atol=1e-6)
    np.testing.assert_allclose(spec.alpha2, 90.0, atol=1e-4)  # 0.5 * atan2(0, -0.3)
    np.testing.assert_allclose(spec.bandwidth, BW)
    assert spec.meta["time_reference"] == "start"
    assert spec.meta["averaging_period_s"] == 1600.0
    spec_long, derived = src.normalize_spectrum(spec, station_id=106, source_file="x.nc")
    validate_observation_frame(derived)
    assert (derived["averaging_period_s"] == 1600.0).all()
    assert len(spec_long) == 4 * len(FREQ)
    hm0 = derived[derived["variable"] == "wave_significant_height"]["value"]
    np.testing.assert_allclose(hm0, 4 * np.sqrt(np.sum(DENSITY * BW)), rtol=1e-6)
    assert (derived["method"] == "spectral_hm0").sum() == 4


def test_validate_includes_provider_flags(cdip_file: Path):
    src = CDIPSource()
    obs = src.normalize(src.parse(cdip_file), station_id=106, source_file="x.nc")
    results = {r.test: r for r in src.validate(obs)}
    assert results["cdip_primary_flag"].flags.tolist() == [1, 3, 1]
    assert results["gap"].params["expected_interval_s"] == 1800.0


def test_provider_flag_test_tolerates_unknown_values():
    r = provider_flag_test(["1/0", "4/2", "9/0", "", None, "7/0", "x"])
    assert r.flags.tolist() == [1, 4, 9, 2, 2, 2, 2]
    assert r.layer == "provider"
