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
    ONPREM_BASE,
    HistoricalAccessError,
    OnPremArchive,
    TokenError,
    WebCOOSClient,
    archive_stills,
    camera_table,
    check_age,
    element_table,
    ledger_path,
    load_token,
    local_path,
    month_index_path,
    onprem_stills,
    parse_listing,
    select_between,
    select_on_grid,
    still_time,
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


def test_elements_are_listed_in_overlapping_daily_windows():
    windows: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        p = req.url.params
        windows.append((p["starting_after"], p["starting_before"]))
        # The same boundary element is returned by both windows that touch it.
        return httpx.Response(
            200,
            json={"results": [element("cam", "2026-09-25T00:00:00Z")], "pagination": {}},
        )

    seen: list[int] = []
    with WebCOOSClient(TOKEN, min_interval_s=0, transport=httpx.MockTransport(handler)) as c:
        els = c.elements(
            "svc",
            "2026-09-24T00:00:00Z",
            "2026-09-26T12:00:00Z",
            progress=lambda t, n: seen.append(n),
        )
    assert windows == [
        ("2026-09-24T00:00:00Z", "2026-09-25T00:00:01Z"),
        ("2026-09-24T23:59:59Z", "2026-09-26T00:00:01Z"),
        ("2026-09-25T23:59:59Z", "2026-09-26T12:00:00Z"),
    ]
    assert seen == [1, 2, 3]
    assert len(element_table(els, "cam")) == 1


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
    # Images, but none within tolerance of a grid time: empty, with UTC grid times.
    far = element_table([element("cam", "2026-09-25T15:15:00Z")], "cam")
    none = select_on_grid(far, every="30min", tolerance="5min")
    assert none.empty and str(none["grid_time"].dt.tz) == "UTC"


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


def test_download_command_skips_a_camera_without_stills(tmp_path: Path, monkeypatch, capsys):
    """An offline camera (no stills in the window) must not stop the others."""
    import wave_analysis.cli as cli
    import wave_analysis.ingest.downloader as downloader_module

    offline = {**ASSET, "data": {**ASSET["data"], "common": {"slug": "offline_cam"}}}
    offline["feeds"] = [
        {
            "products": [
                {
                    "data": {"common": {"slug": "one-minute-stills"}},
                    "services": [
                        {"data": {"common": {"slug": "offline_cam-one-minute-stills-s3"}}}
                    ],
                }
            ]
        }
    ]
    now = pd.Timestamp.now(tz="UTC").floor("min")

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def assets(self):
            return [offline, ASSET]

        def elements(self, service, start, end, **kwargs):
            if service.startswith("offline_cam"):
                return []
            return [element("oakisland_west", now.floor("30min"))]

    monkeypatch.setattr(cli, "_webcoos_client", lambda args: FakeClient())
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"jpeg"))
    real = downloader_module.Downloader
    monkeypatch.setattr(
        downloader_module,
        "Downloader",
        lambda **kw: real(**{**kw, "min_interval_s": 0, "transport": transport}),
    )
    rc = cli.main(
        [
            "webcoos",
            "download",
            "offline_cam",
            "oakisland_west",
            "--lookback",
            "2h",
            "--out",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "offline_cam: 0 stills listed" in out
    assert "oakisland_west: 1 indexed, 0 already archived, 1 downloaded" in out
    assert len(list((tmp_path / "oakisland_west").rglob("*.jpg"))) == 1


def test_camera_sets_expand_in_order_without_repeats():
    from wave_analysis.sources.webcoos import (
        CAMERA_SETS,
        NEAR_BUOY_CAMERAS,
        YIN2025_CAMERAS,
        expand_cameras,
    )

    assert expand_cameras(["@yin2025"]) == list(YIN2025_CAMERAS)
    both = expand_cameras(["@yin2025", "@near_buoy", "san_elijo"])
    assert both == [*YIN2025_CAMERAS, *NEAR_BUOY_CAMERAS]  # san_elijo already in the set
    assert not set(YIN2025_CAMERAS) & set(NEAR_BUOY_CAMERAS)
    assert expand_cameras(["oakisland_west", "@yin2025"])[0] == "oakisland_west"
    with pytest.raises(KeyError, match="unknown camera set"):
        expand_cameras(["@nope"])
    assert set(CAMERA_SETS) == {"yin2025", "near_buoy"}


def test_download_command_rejects_an_unknown_camera_set(capsys):
    from wave_analysis.cli import main

    assert main(["webcoos", "download", "@nope", "--list-only"]) == 2
    assert "unknown camera set" in capsys.readouterr().err


def test_collector_unit_uses_the_camera_sets():
    from wave_analysis.config import repo_root

    unit = (repo_root() / "deploy/systemd/wave-analysis-webcoos.service").read_text()
    assert "webcoos download @yin2025 @near_buoy --lookback 3D" in unit


# --------------------------------------------------------------------------- #
# On-premise archive
# --------------------------------------------------------------------------- #
COCOA = f"{ONPREM_BASE}/jpg/uncw/cocoabeach/"


def nginx_index(path: str, entries: dict[str, int | None]) -> str:
    """An nginx autoindex page; ``None`` marks a directory."""
    rows = ['<a href="../">../</a>']
    for name, size in entries.items():
        href = f"{name}/" if size is None else name
        rows.append(
            f'<a href="{href}">{href}</a>{" " * max(1, 51 - len(href))}'
            f"01-Dec-2023 11:52 {'-' if size is None else size:>19}"
        )
    body = "\n".join(rows)
    return f"<html><head><title>Index of {path}</title></head><body><h1>Index of {path}</h1><hr><pre>{body}\n</pre><hr></body></html>"


def still(t: str, cam: str = "cocoabeach") -> str:
    return f"{cam}-{pd.Timestamp(t):%Y-%m-%d-%H%M%S}Z.jpg"


#: A small archive: stills on 2023-01-31 and 2023-02-01, around midnight UTC.
TREE = {
    COCOA: {"2023": None, "cocoabeach.jpg": 9},
    f"{COCOA}2023/": {"01": None, "02": None},
    f"{COCOA}2023/01/": {"30": None, "31": None},
    f"{COCOA}2023/01/30/": {still("2023-01-30T15:00:10"): 5},
    f"{COCOA}2023/01/31/": {
        still("2023-01-31T23:29:50"): 5,
        still("2023-01-31T23:58:00"): 5,
        "cocoabeach.jpg_exiftool_tmp": 7,
    },
    f"{COCOA}2023/02/": {"01": None},
    f"{COCOA}2023/02/01/": {still("2023-02-01T00:01:00"): 5, still("2023-02-01T00:30:20"): 5},
}


def tree_handler(requests: list[httpx.Request]):
    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        url = str(req.url)
        if url in TREE:
            return httpx.Response(200, text=nginx_index(req.url.path, TREE[url]))
        if url.endswith(".jpg"):
            return httpx.Response(200, content=b"jpeg!")
        return httpx.Response(404)

    return handler


def test_parse_listing_reads_nginx_index():
    df = parse_listing(nginx_index("/x/", {"2023": None, "a.jpg": 12, "a b.jpg": 3}))
    assert list(df["name"]) == ["2023", "a.jpg", "a b.jpg"]
    assert list(df["is_dir"]) == [True, False, False]
    assert pd.isna(df["size_bytes"].iloc[0]) and list(df["size_bytes"].iloc[1:]) == [12, 3]
    assert parse_listing("").empty


def test_onprem_stills_take_the_time_from_the_name_and_skip_other_files():
    listing = parse_listing(
        nginx_index("/d/", TREE[f"{COCOA}2023/01/31/"] | {still("2023-01-31T12:00:00", "other"): 1})
    )
    df = onprem_stills(listing, "cocoabeach", f"{COCOA}2023/01/31/")
    assert list(df["time_utc"]) == [
        pd.Timestamp("2023-01-31T23:29:50Z"),
        pd.Timestamp("2023-01-31T23:58:00Z"),
    ]
    assert df["url"].iloc[0] == f"{COCOA}2023/01/31/cocoabeach-2023-01-31-232950Z.jpg"
    assert list(df["size_bytes"]) == [5, 5]


def test_onprem_archive_lists_only_the_days_in_range_and_sends_no_token():
    requests: list[httpx.Request] = []
    with Downloader(min_interval_s=0, transport=httpx.MockTransport(tree_handler(requests))) as dl:
        arc = OnPremArchive(dl)
        months = arc.months(
            "cocoabeach", pd.Timestamp("2023-01-31", tz="UTC"), pd.Timestamp("2023-02-02", tz="UTC")
        )
        assert [str(m) for m in months] == ["2023-01", "2023-02"]
        jan = arc.month_stills("cocoabeach", months[0], start=pd.Timestamp("2023-01-31", tz="UTC"))
        assert len(jan) == 2
        with pytest.raises(ValueError):
            arc.listing("https://elsewhere.example/archive/")
        with pytest.raises(ValueError):
            arc.camera_url("unknown_cam")
    listed = [str(r.url) for r in requests]
    assert f"{COCOA}2023/01/31/" in listed and f"{COCOA}2023/01/30/" not in listed
    assert all("authorization" not in r.headers for r in requests)


def test_select_between_gives_each_grid_time_one_still_across_months():
    t = [
        "2023-01-31T23:29:50Z",
        "2023-01-31T23:58:00Z",
        "2023-02-01T00:01:00Z",
        "2023-02-01T00:30:20Z",
    ]
    els = element_table([element("cam", x) for x in t], "cam")
    feb = pd.Timestamp("2023-02-01", tz="UTC")
    jan = select_between(els[els["time_utc"] < feb], pd.Timestamp("2023-01-01", tz="UTC"), feb)
    after = select_between(els[els["time_utc"] >= feb], feb, pd.Timestamp("2023-03-01", tz="UTC"))
    grid = list(jan["grid_time"]) + list(after["grid_time"])
    assert len(grid) == len(set(grid)) == 3  # 23:30, 00:00 (once), 00:30
    assert after["time_utc"].iloc[0] == pd.Timestamp("2023-02-01T00:01:00Z")


def test_ledger_path_per_host(tmp_path: Path):
    assert ledger_path(tmp_path, "cam").name == "cam.csv"
    assert ledger_path(tmp_path, "cam", "otacon").name == "cam.otacon.csv"
    with pytest.raises(ValueError):
        ledger_path(tmp_path, "cam", "../x")


def test_historical_command(tmp_path: Path, monkeypatch, capsys):
    """Needs the approval flag; skips stills already archived; indexes closed months once."""
    import wave_analysis.cli as cli
    import wave_analysis.ingest.downloader as downloader_module

    monkeypatch.setenv("WAVE_ANALYSIS_HOST", "Lab-PC")
    requests: list[httpx.Request] = []
    transport = httpx.MockTransport(tree_handler(requests))
    real = downloader_module.Downloader
    made: list[dict] = []

    def fake_downloader(**kw):
        made.append(kw)
        return real(**{**kw, "min_gap_s": 0, "max_bytes_per_s": None, "transport": transport})

    monkeypatch.setattr(downloader_module, "Downloader", fake_downloader)
    argv = ["webcoos", "historical", "cocoabeach", "--start", "2023-01-01", "--end", "2023-03-01"]
    argv += ["--out", str(tmp_path)]
    assert cli.main(argv) == 3 and not requests  # no approval, no request

    # One still was already fetched from S3 by the hourly collector.
    s3 = local_path(
        tmp_path, "cocoabeach", still("2023-01-31T23:29:50"), pd.Timestamp("2023-01-31T23:29:50Z")
    )
    s3.parent.mkdir(parents=True)
    s3.write_bytes(b"jpeg!")
    assert cli.main([*argv, "--historical-approved"]) == 0
    assert made[0]["min_gap_s"] >= 1.0 and made[0]["max_bytes_per_s"] <= 20 * 2**20
    out = capsys.readouterr().out
    assert "cocoabeach 2023-01: 3 images, 2 on the 30min grid, 1 already archived" in out
    files = sorted(p.name for p in (tmp_path / "cocoabeach").rglob("*.jpg"))
    assert files == [
        "cocoabeach-2023-01-30-150010Z.jpg",
        "cocoabeach-2023-01-31-232950Z.jpg",
        "cocoabeach-2023-02-01-000100Z.jpg",
        "cocoabeach-2023-02-01-003020Z.jpg",
    ]
    ledger = tmp_path / "_manifests" / "cocoabeach.lab-pc.csv"
    assert ledger.exists() and not (tmp_path / "_manifests" / "cocoabeach.csv").exists()
    assert len(list(read_manifest(ledger))) == 3
    assert month_index_path(tmp_path, "cocoabeach", pd.Period("2023-01", "M")).exists()

    # A second run reads the saved month listings and fetches nothing.
    requests.clear()
    assert cli.main([*argv, "--historical-approved"]) == 0
    assert [str(r.url) for r in requests] == [COCOA, f"{COCOA}2023/"]
    assert len(list(read_manifest(ledger))) == 3


TIMEX = f"{ONPREM_BASE}/timex/uncw/cocoabeach/"
TIMEX_TREE = {
    TIMEX: {"2024": None},
    f"{TIMEX}2024/": {"03": None},
    f"{TIMEX}2024/03/": {"05": None},
    f"{TIMEX}2024/03/05/": {
        "timex." + still("2024-03-05T15:00:09"): 6,
        "timex." + still("2024-03-05T15:10:09"): 6,
        "timex." + still("2024-03-05T15:20:09"): 6,
        "brt." + still("2024-03-05T15:00:09"): 9,  # another product: ignored
        still("2024-03-05T15:00:30"): 5,  # a still: ignored
    },
}


def test_timex_names_carry_the_product_prefix():
    assert still_time("timex.cam-2024-03-05-150009Z.jpg", "cam", "timex.") == pd.Timestamp(
        "2024-03-05T15:00:09Z"
    )
    assert still_time("cam-2024-03-05-150009Z.jpg", "cam", "timex.") is None
    assert still_time("brt.cam-2024-03-05-150009Z.jpg", "cam", "timex.") is None


def test_historical_timex_every_image(tmp_path: Path, monkeypatch, capsys):
    import wave_analysis.cli as cli
    import wave_analysis.ingest.downloader as downloader_module

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if url in TIMEX_TREE:
            return httpx.Response(200, text=nginx_index(req.url.path, TIMEX_TREE[url]))
        if url.endswith(".jpg"):
            return httpx.Response(200, content=b"timex!")
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real = downloader_module.Downloader
    monkeypatch.setattr(
        downloader_module,
        "Downloader",
        lambda **kw: real(
            **{**kw, "min_gap_s": 0, "max_bytes_per_s": None, "transport": transport}
        ),
    )
    out = tmp_path / "webcoos_timex"
    argv = ["webcoos", "historical", "cocoabeach", "--product", "timex", "--every", "all"]
    rc = cli.main([*argv, "--start", "2024-03-01", "--out", str(out), "--historical-approved"])
    assert rc == 0
    files = sorted(p.name for p in out.rglob("*.jpg"))
    assert files == [
        "timex.cocoabeach-2024-03-05-150009Z.jpg",
        "timex.cocoabeach-2024-03-05-151009Z.jpg",
        "timex.cocoabeach-2024-03-05-152009Z.jpg",
    ]
    assert "3 images, 3 kept (all), 0 already archived" in capsys.readouterr().out
    assert cli.main([*argv, "--product", "nope", "--out", str(out), "--historical-approved"]) == 2
