"""WebCOOS client (token handling, host lock, pagination), tables, grid thinning and archive."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pandas as pd
import pytest

from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import read_manifest, write_manifest
from wave_analysis.sources.webcoos import (
    API_BASE,
    HISTORICAL_SAMPLE_LIMIT,
    HistoricalAccessError,
    TokenError,
    WebCOOSClient,
    archive_stills,
    camera_table,
    check_age,
    element_table,
    ledger_path,
    load_token,
    local_path,
    select_on_grid,
)

TOKEN = "test-token-0123456789"
S3 = "https://s3.us-west-2.amazonaws.com/webcoos/media/x"


def element(cam: str, t: str, size: int = 4) -> dict:
    stamp = pd.Timestamp(t).strftime("%Y-%m-%d-%H%M%SZ")
    name = f"{cam}-{stamp}.jpg"
    return {
        "uuid": f"u-{stamp}",
        "data": {
            "common": {"label": name},
            "extents": {"temporal": {"min": pd.Timestamp(t).isoformat()}},
            "properties": {"url": f"{S3}/{cam}/{name}", "size": str(size)},
        },
    }


ASSET = {
    "data": {
        "common": {
            "slug": "oakisland_west",
            "label": "Beachfront (west view)",
            "access_level": "public",
        },
        "properties": {
            "location": {"type": "Point", "coordinates": [-78.2243, 33.9126]},
            "state_or_territory": "North Carolina",
            "region": "SECOORA",
            "group": "uncw",
            "timezone": "America/New_York",
        },
    },
    "disposition": {"slug": "up"},
    "feeds": [
        {
            "products": [
                {
                    "data": {"common": {"slug": "one-minute-stills"}},
                    "services": [
                        {"data": {"common": {"slug": "oakisland_west-one-minute-stills-s3"}}}
                    ],
                },
                {
                    "data": {"common": {"slug": "video-archive"}},
                    "services": [{"data": {"common": {"slug": "oakisland_west-video-archive-s3"}}}],
                },
            ]
        }
    ],
}


# --------------------------------------------------------------------------- #
def test_load_token_from_env_and_file(tmp_path: Path):
    assert load_token({"WEBCOOS_API_TOKEN": f" {TOKEN}\n"}, tmp_path / "none") == TOKEN
    f = tmp_path / "tok"
    f.write_text(TOKEN + "\n")
    f.chmod(0o600)
    assert load_token({}, f) == TOKEN
    f.chmod(0o644)
    with pytest.raises(TokenError, match="chmod 600") as exc:
        load_token({}, f)
    assert TOKEN not in str(exc.value)
    with pytest.raises(TokenError, match="no WebCOOS token"):
        load_token({}, tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.write_text("\n")
    os.chmod(empty, 0o600)
    with pytest.raises(TokenError, match="empty"):
        load_token({}, empty)


def test_client_sends_token_only_to_api_host_and_paginates():
    seen: list[tuple[str, str | None]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.host, req.headers.get("authorization")))
        if req.url.params.get("page") == "2":
            return httpx.Response(200, json={"results": [3], "pagination": {"next": None}})
        return httpx.Response(
            200, json={"results": [1, 2], "pagination": {"next": f"{API_BASE}/assets/?page=2"}}
        )

    with WebCOOSClient(TOKEN, min_interval_s=0, transport=httpx.MockTransport(handler)) as c:
        assert list(c.paginate("assets/")) == [1, 2, 3]
        with pytest.raises(ValueError, match="refusing"):
            c.get_json("https://evil.example.org/steal")
    assert seen == [("app.webcoos.org", f"Token {TOKEN}")] * 2


def test_client_reports_rejected_token_without_leaking_it():
    transport = httpx.MockTransport(lambda r: httpx.Response(403, json={"detail": "no"}))
    with (
        WebCOOSClient(TOKEN, min_interval_s=0, transport=transport) as c,
        pytest.raises(TokenError, match="HTTP 403") as exc,
    ):
        c.get_json("assets/")
    assert TOKEN not in str(exc.value)


def test_elements_request_uses_large_pages():
    params: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        params.update(req.url.params)
        return httpx.Response(200, json={"results": [], "pagination": {}})

    with WebCOOSClient(TOKEN, min_interval_s=0, transport=httpx.MockTransport(handler)) as c:
        c.elements("svc", "2026-09-25", "2026-09-26T00:00:00Z")
    assert params == {
        "service": "svc",
        "starting_after": "2026-09-25T00:00:00Z",
        "starting_before": "2026-09-26T00:00:00Z",
        "page_size": "1000",
    }


# --------------------------------------------------------------------------- #
def test_camera_table():
    cams = camera_table([ASSET])
    row = cams.iloc[0]
    assert row["camera"] == "oakisland_west"
    assert (row["latitude"], row["longitude"]) == (33.9126, -78.2243)
    assert row["stills_service"] == "oakisland_west-one-minute-stills-s3"
    assert row["products"] == "one-minute-stills,video-archive"
    assert row["status"] == "up"


def test_element_table_sorts_dedupes_and_rejects_unsafe_names():
    els = [element("cam", "2026-09-25T15:01:10Z"), element("cam", "2026-09-25T15:00:48Z")]
    df = element_table([*els, els[0]], "cam")
    assert df["time_utc"].tolist() == [
        pd.Timestamp("2026-09-25T15:00:48Z"),
        pd.Timestamp("2026-09-25T15:01:10Z"),
    ]
    assert df["size_bytes"].tolist() == [4, 4]
    bad = element("cam", "2026-09-25T15:00:00Z")
    bad["data"]["common"]["label"] = "../x.jpg"
    with pytest.raises(ValueError, match="unsafe"):
        element_table([bad], "cam")
    http = element("cam", "2026-09-25T15:00:00Z")
    http["data"]["properties"]["url"] = "http://example.org/x.jpg"
    with pytest.raises(ValueError, match="HTTPS"):
        element_table([http], "cam")


def test_select_on_grid_picks_nearest_within_tolerance():
    times = ["14:58:30", "15:00:48", "15:03:00", "15:29:00", "16:20:00"]
    df = element_table([element("cam", f"2026-09-25T{t}Z") for t in times], "cam")
    sel = select_on_grid(df, every="30min", tolerance="5min")
    assert sel["time_utc"].dt.strftime("%H:%M:%S").tolist() == ["15:00:48", "15:29:00"]
    assert sel["grid_time"].dt.strftime("%H:%M").tolist() == ["15:00", "15:30"]
    assert sel["offset_s"].tolist() == [48.0, -60.0]
    # On an hourly grid at :20, only 16:20 has an image within 5 min (15:29 is 9 min off).
    shifted = select_on_grid(df, every="1h", tolerance="5min", offset="20min")
    assert shifted["time_utc"].dt.strftime("%H:%M:%S").tolist() == ["16:20:00"]
    assert select_on_grid(df.iloc[:0]).empty


def test_select_on_grid_is_independent_of_time_resolution():
    df = element_table([element("cam", "2026-09-25T15:00:48Z")], "cam")
    for unit in ("s", "ms", "us", "ns"):
        d = df.assign(time_utc=df["time_utc"].dt.as_unit(unit))
        assert len(select_on_grid(d, every="30min", tolerance="5min")) == 1


def test_check_age_enforces_the_90_day_guideline():
    now = pd.Timestamp("2026-09-26T00:00:00Z")
    recent = pd.Series(pd.date_range("2026-09-01", periods=2000, freq="h", tz="UTC"))
    assert check_age(recent, now=now) == 0
    old = pd.Series(
        pd.date_range("2025-01-01", periods=HISTORICAL_SAMPLE_LIMIT + 1, freq="h", tz="UTC")
    )
    with pytest.raises(HistoricalAccessError, match=r"webcoos@secoora\.org"):
        check_age(old, now=now)
    assert check_age(old, now=now, historical_approved=True) == HISTORICAL_SAMPLE_LIMIT + 1
    assert check_age(old.iloc[:HISTORICAL_SAMPLE_LIMIT], now=now) == HISTORICAL_SAMPLE_LIMIT


# --------------------------------------------------------------------------- #
def test_archive_stills_downloads_resumes_and_uses_utc_dates(tmp_path: Path):
    els = [element("cam", "2026-09-25T23:59:40Z", 4), element("cam", "2026-09-26T00:30:05Z", 4)]
    sel = select_on_grid(element_table(els, "cam"), every="30min", tolerance="5min")
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.headers.get("authorization") or "")
        return httpx.Response(200, content=b"jpeg")

    ledger = ledger_path(tmp_path, "cam")
    with Downloader(min_interval_s=0, transport=httpx.MockTransport(handler)) as dl:
        s = archive_stills(dl, sel, tmp_path, sink=lambda b: write_manifest(b, ledger))
        assert s.downloaded == 2 and s.size_mismatch == 0
        assert calls == ["", ""]  # image requests carry no token
        s2 = archive_stills(dl, sel, tmp_path, sink=lambda b: write_manifest(b, ledger))
        assert s2.already_archived == 2 and s2.downloaded == 0 and len(calls) == 2
    first = local_path(tmp_path, "cam", sel["name"].iloc[0], sel["time_utc"].iloc[0])
    assert first == tmp_path / "cam/2026/09/25/cam-2026-09-25-235940Z.jpg"
    assert first.read_bytes() == b"jpeg"
    rows = list(read_manifest(ledger))
    assert {r.source_id for r in rows} == {"webcoos"} and rows[0].station_id == "cam"
    with pytest.raises(ValueError):
        local_path(tmp_path, "../cam", "x.jpg", pd.Timestamp("2026-01-01", tz="UTC"))
