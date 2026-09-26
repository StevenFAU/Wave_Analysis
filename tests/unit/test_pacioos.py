"""PacIOOS beach-camera index parsing and the resumable image archiver."""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import pandas as pd
import pytest

from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import EntryStatus, read_manifest, write_manifest
from wave_analysis.sources.pacioos import (
    FILES_PREFIX,
    archive_images,
    archived,
    index_url,
    ledger_path,
    local_path,
    name_time_hst,
    parse_index,
)

HEADER = "time,latitude,longitude,altitude,aboveGround,url,name,size\nUTC,degrees_north,degrees_east,m,m,,,bytes\n"


def row(utc: str, hst: str, size: int = 5, dataset: str = "beachcam_003") -> str:
    day = f"{hst[:4]}/{hst[4:6]}/{hst[6:8]}"
    name = f"{dataset}.{hst}.jpg"
    url = f"{FILES_PREFIX}{dataset}/{day}/{name}"
    return f"{utc},21.643578,-158.06339,31.1,20.1,{url},{name},{size}.0\n"


# The UTC day boundary falls inside the HST day: 14:00 HST = 00:00 UTC next day.
GOOD = (
    HEADER
    + row("2012-01-05T23:00:00Z", "20120105T130000", size=5)
    + row("2012-01-06T00:00:00Z", "20120105T140000", size=6)
    + row("2012-01-06T01:00:00Z", "20120105T150000", size=7)
)


def test_index_url_encodes_constraints():
    url = index_url("beachcam_003", "2012-01-05", "2012-01-06")
    assert url.startswith(
        "https://pae-paha.pacioos.hawaii.edu/erddap/tabledap/beachcam_003.csv?time,"
    )
    assert "time%3E=2012-01-05T00:00:00Z" in url
    assert "time%3C2012-01-06T00:00:00Z" in url
    assert "<" not in url and ">" not in url
    with pytest.raises(ValueError, match="unknown PacIOOS"):
        index_url("beachcam_009")


def test_parse_index_checks_and_types():
    idx = parse_index(io.StringIO(GOOD), "beachcam_003")
    assert len(idx) == 3
    assert str(idx["time_utc"].dt.tz) == "UTC"
    assert idx["size_bytes"].tolist() == [5, 6, 7]
    assert idx["dataset"].unique().tolist() == ["beachcam_003"]
    assert name_time_hst(idx["name"].iloc[1]) == pd.Timestamp("2012-01-05T14:00:00")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (HEADER + row("2012-01-05T13:00:00Z", "20120105T130000"), "not UTC-10 h"),
        (HEADER + row("2012-01-05T23:00:00Z", "20120105T130000") * 2, "duplicate times"),
        (
            HEADER
            + row("2012-01-05T23:00:00Z", "20120105T130000").replace("beachcam_003/", "other/"),
            "outside",
        ),
    ],
)
def test_parse_index_rejects_bad_rows(text: str, message: str):
    with pytest.raises(ValueError, match=message):
        parse_index(io.StringIO(text), "beachcam_003")


def test_local_path_mirrors_files_tree(tmp_path: Path):
    url = f"{FILES_PREFIX}beachcam_003/2012/01/05/beachcam_003.20120105T130000.jpg"
    assert (
        local_path(tmp_path, url)
        == tmp_path / "beachcam_003/2012/01/05/beachcam_003.20120105T130000.jpg"
    )
    with pytest.raises(ValueError):
        local_path(tmp_path, "https://example.org/x.jpg")
    with pytest.raises(ValueError):
        local_path(tmp_path, f"{FILES_PREFIX}beachcam_003/../../etc/passwd")


def _server(sizes: dict[str, int], missing: set[str]) -> tuple[httpx.MockTransport, list[str]]:
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        name = str(req.url).rsplit("/", 1)[-1]
        calls.append(name)
        if name in missing:
            return httpx.Response(404)
        return httpx.Response(200, content=b"x" * sizes[name])

    return httpx.MockTransport(handler), calls


def test_archive_resumes_and_checks_sizes(tmp_path: Path):
    idx = parse_index(io.StringIO(GOOD), "beachcam_003")
    names = idx["name"].tolist()
    # The server returns a different size for the second image and 404 for the third.
    transport, calls = _server({names[0]: 5, names[1]: 9}, {names[2]})
    ledger = ledger_path(tmp_path, "beachcam_003")
    with Downloader(min_interval_s=0, transport=transport) as dl:
        s = archive_images(dl, idx, tmp_path, sink=lambda b: write_manifest(b, ledger))
    assert (s.downloaded, s.not_found, s.failed, s.size_mismatch) == (2, 1, 0, 1)
    rows = list(read_manifest(ledger))
    assert [r.status for r in rows] == [
        EntryStatus.VERIFIED,
        EntryStatus.VERIFIED,
        EntryStatus.NOT_FOUND,
    ]
    assert rows[1].note == "size differs from index (6 bytes)"
    assert rows[0].period == "2012-01-05T23:00:00Z" and rows[0].station_id == "beachcam_003"

    done, missing = archived(tmp_path, "beachcam_003")
    assert len(done) == 2 and len(missing) == 1

    # A second run requests nothing.
    calls.clear()
    with Downloader(min_interval_s=0, transport=transport) as dl:
        s2 = archive_images(dl, idx, tmp_path, sink=lambda b: write_manifest(b, ledger))
    assert calls == [] and (s2.already_archived, s2.known_missing) == (2, 1)

    # A deleted file is fetched again.
    local_path(tmp_path, idx["url"].iloc[0]).unlink()
    with Downloader(min_interval_s=0, transport=transport) as dl:
        s3 = archive_images(dl, idx, tmp_path, sink=lambda b: write_manifest(b, ledger))
    assert calls == [names[0]] and s3.downloaded == 1


def test_archive_limit_and_single_dataset(tmp_path: Path):
    idx = parse_index(io.StringIO(GOOD), "beachcam_003")
    transport, calls = _server(dict.fromkeys(idx["name"], 5), set())
    with Downloader(min_interval_s=0, transport=transport) as dl:
        s = archive_images(dl, idx, tmp_path, sink=lambda b: None, limit=1)
        assert s.downloaded == 1 and len(calls) == 1
        mixed = pd.concat([idx, idx.assign(dataset="beachcam_004")])
        with pytest.raises(ValueError, match="exactly one dataset"):
            archive_images(dl, mixed, tmp_path, sink=lambda b: None)
