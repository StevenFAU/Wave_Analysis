"""Live checks against NDBC (deselected by default; run with ``pytest -m network``).

These detect *upstream* format changes: if NDBC alters a header, a sentinel,
or its directory index layout, these tests fail while the offline fixture
tests keep passing - which is exactly the signal needed to update the parsers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import EntryStatus
from wave_analysis.sources.ndbc import NDBCSource, inventory
from wave_analysis.sources.ndbc.products import HISTORICAL_URL

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def dl():
    with Downloader(min_interval_s=1.0) as d:
        yield d


def test_realtime_stdmet_and_spectral_parse(dl, tmp_path: Path):
    src = NDBCSource(dl)
    entries = src.fetch("41010", ["stdmet", "swden", "swr1"], dest_root=tmp_path, realtime=True)
    assert all(e.status == EntryStatus.VERIFIED for e in entries), entries
    for e in entries:
        parsed = src.parse(Path(e.local_path), e.product, realtime=True)
        assert parsed is not None


def test_historical_index_layout_unchanged(dl):
    page = dl.get(f"{HISTORICAL_URL}/swr1/").text
    df = inventory.historical_listing_to_inventory(inventory.parse_directory_listing(page), "swr1")
    assert len(df) > 1000
    assert df.attrs["unmatched"] == []
