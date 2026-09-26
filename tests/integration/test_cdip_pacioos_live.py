"""Live checks against CDIP and PacIOOS (deselected by default; run with ``pytest -m network``).

They fail if an upstream layout the adapters rely on changes: CDIP's catalog
and variable names, or the PacIOOS index columns and its UTC/HST naming.
"""

from __future__ import annotations

import io

import pytest

from wave_analysis.ingest.downloader import Downloader
from wave_analysis.sources.cdip import WAVE_VARIABLES, CDIPSource, file_path, file_url
from wave_analysis.sources.pacioos import index_url, parse_index

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def dl():
    with Downloader(min_interval_s=1.0) as d:
        yield d


def test_cdip_catalog_and_variables(dl):
    files = CDIPSource(dl).discover([106])
    assert "historic" in set(files["product"]) and (files["product"] == "deployment").sum() > 10
    dds = dl.get(file_url(file_path(106, "historic"), "dodsC") + ".dds").text
    missing = [v for v in WAVE_VARIABLES if f" {v}[" not in dds]
    assert not missing, missing


def test_pacioos_index_columns_and_time_zone(dl):
    resp = dl.get(index_url("beachcam_003", "2012-01-05T15:00:00Z", "2012-01-06T05:00:00Z"))
    resp.raise_for_status()
    idx = parse_index(io.StringIO(resp.text), "beachcam_003")
    assert len(idx) >= 10  # 06-17 HST hourly
