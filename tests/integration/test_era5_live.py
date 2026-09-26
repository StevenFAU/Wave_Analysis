"""Live ERA5 check (``pytest -m network``); skipped when no CDS key is configured."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("cdsapi")

from wave_analysis.sources.era5 import (
    PARAMETERS,
    CDSConfigError,
    ERA5WaveSource,
    available_until,
    check_config,
    make_client,
)

pytestmark = pytest.mark.network


def test_catalogue_extent_is_recent():
    end = available_until()
    assert end is not None and end > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)


def test_one_field_request(tmp_path: Path):
    try:
        check_config()
    except CDSConfigError as exc:
        pytest.skip(str(exc))
    client = make_client()
    target = tmp_path / "one.nc"
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": [PARAMETERS[0].request],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "area": [22.5, -158.5, 21.0, -157.5],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    ds = ERA5WaveSource(client).parse(target)
    assert int(ds["swh"].attrs["GRIB_paramId"]) == 140229
    assert ds.sizes["latitude"] == 4 and ds.sizes["longitude"] == 3
