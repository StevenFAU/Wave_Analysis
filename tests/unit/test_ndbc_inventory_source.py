"""NDBC inventory parsing and adapter normalisation, fetch and validation (offline)."""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import EntryStatus
from wave_analysis.qc.flags import QCFlag
from wave_analysis.schemas.observation import validate_observation_frame
from wave_analysis.sources.ndbc import NDBCSource, assemble_spectrum
from wave_analysis.sources.ndbc import inventory as inv
from wave_analysis.sources.ndbc.products import PRODUCTS, historical_url, realtime_url

# ----------------------------------------------------------------------------- products / urls


def test_urls():
    assert realtime_url("41010", "swden").endswith("/realtime2/41010.data_spec")
    assert historical_url("41010", "swr1", 2023).endswith("/historical/swr1/41010j2023.txt.gz")
    assert historical_url("4cONF", "stdmet", 2020, filename="4cONFh2020.txt.gz").endswith(
        "4cONFh2020.txt.gz"
    )
    with pytest.raises(ValueError):
        realtime_url("41010", "adcp2")
    with pytest.raises(ValueError):
        historical_url("41010", "spec", 2020)


def test_every_historical_product_has_unique_code():
    codes = [p.historical_code for p in PRODUCTS.values() if p.historical_code]
    assert len(codes) == len(set(codes))


# ----------------------------------------------------------------------------- inventory


def test_historical_listing(ndbc_fixtures):
    page = (ndbc_fixtures / "index" / "historical_stdmet.html").read_text()
    listing = inv.parse_directory_listing(page)
    assert len(listing) > 50
    assert str(listing["last_modified"].dt.tz) == "UTC"
    df = inv.historical_listing_to_inventory(listing, "stdmet")
    assert df.attrs["unmatched"] == []
    assert {"41009", "41010", "4CONF", "EB01"} <= set(df["station_id"])
    assert df.loc[df["station_id"] == "41010", "year"].min() == 1988
    row = df[(df["station_id"] == "41010") & (df["year"] == 2023)].iloc[0]
    assert row["size_bytes"] == pytest.approx(219 * 1024)
    assert row["url"].endswith("/historical/stdmet/41010h2023.txt.gz")


def test_second_segment_files_are_kept():
    listing = pd.DataFrame(
        {
            "name": ["41013j2005.txt.gz", "41013jb2005.txt.gz", "41008b2010.txt.gz"],
            "last_modified": pd.Timestamp("2020-01-01", tz="UTC"),
            "size_bytes": 1.0,
        }
    )
    df = inv.historical_listing_to_inventory(listing, "swr1")
    assert df["period"].tolist() == ["2005", "2005b"]
    assert df["segment"].tolist() == ["a", "b"]
    assert (df["year"] == 2005).all() and (df["station_id"] == "41013").all()
    assert df.attrs["unmatched"] == ["41008b2010.txt.gz"]  # adcp2 code, wrong directory
    adcp2 = inv.historical_listing_to_inventory(listing.iloc[[2]], "adcp2")
    assert adcp2["station_id"].tolist() == ["41008"] and adcp2["period"].tolist() == ["2010"]


def test_realtime_listing(ndbc_fixtures):
    listing = inv.parse_directory_listing((ndbc_fixtures / "index" / "realtime2.html").read_text())
    df = inv.realtime_listing_to_inventory(listing)
    got = set(df.loc[df["station_id"] == "41010", "product"])
    assert {"stdmet", "spec", "swden", "swdir", "swdir2", "swr1", "swr2"} <= got
    assert (df["archive"] == "realtime").all()


def test_station_metadata_and_cameras(ndbc_fixtures):
    st = inv.parse_active_stations((ndbc_fixtures / "index" / "activestations.xml").read_text())
    row = st.set_index("station_id").loc["41010"]
    assert row["latitude"] == pytest.approx(28.86)
    assert row["has_met"]
    tab = inv.parse_station_table((ndbc_fixtures / "index" / "station_table.txt").read_text())
    assert "SCOOP" in tab.set_index("station_id").loc["41010", "payload"]
    cams = inv.parse_buoycams(
        (ndbc_fixtures / "index" / "buoycams.json").read_text(),
        pd.Timestamp("2026-09-26", tz="UTC"),
    )
    assert "41010" in set(cams["station_id"])
    # Real snapshot: some listed cameras have no current image (img is null) - kept, not dropped.
    urls = cams["latest_image_url"].dropna()
    assert len(urls) >= 0.8 * len(cams)
    assert urls.str.startswith("https://www.ndbc.noaa.gov/images/buoycam/").all()


def test_inventory_summary_logic():
    files = pd.DataFrame(
        [
            *[
                {"station_id": "A", "product": p, "archive": "historical", "year": 2020}
                for p in ("stdmet", "swden", "swdir", "swdir2", "swr1", "swr2")
            ],
            {"station_id": "A", "product": "swden", "archive": "historical", "year": 2021},
            {"station_id": "B", "product": "adcp", "archive": "historical", "year": 2019},
            {"station_id": "B", "product": "stdmet", "archive": "realtime", "year": pd.NA},
        ]
    )
    res = inv.InventoryResult(
        files=files,
        stations=pd.DataFrame({"station_id": ["A", "B"], "name": ["a", "b"]}),
        cameras=pd.DataFrame({"station_id": ["A"]}),
        manifest=[],
        snapshot_utc=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    s = res.station_summary()
    assert s.loc["A", "directional_spectrum_years"] == 1  # 2021 lacks directional files
    assert s.loc["A", "swden_years"] == 2
    assert s.loc["B", "adcp_years"] == 1
    assert bool(s.loc["A", "has_camera_now"]) and not bool(s.loc["B", "has_camera_now"])
    assert s.loc["B", "realtime_products"] == "stdmet"
    assert res.availability_matrix().loc["A", "swden"] == 2


# ----------------------------------------------------------------------------- normalise / validate


def test_normalize_stdmet_is_schema_valid(ndbc_fixtures):
    src = NDBCSource(Downloader(min_interval_s=0))
    f = ndbc_fixtures / "historical" / "41010h2023.txt.gz"
    obs = src.normalize(
        src.parse(f, "stdmet"), station_id="41010", source_file=f.name, source_checksum="0" * 64
    )
    validate_observation_frame(obs)
    assert set(obs["method"]) == {"provider_reported"}
    hs = obs[obs["variable"] == "wave_significant_height"]
    assert (hs["units"] == "m").all() and (hs["averaging_period_s"] == 1200).all()
    assert obs["observation_id"].is_unique


def test_unit_conversions_vis_and_tide():
    from wave_analysis.sources.ndbc.parsers import parse_stdmet

    text = (
        "#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE\n"
        "#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi    ft\n"
        "2026 01 01 00 00 360  5.0  6.0  1.00  8.00  6.00  90 1015.0  20.0  22.0  15.0  1.0  1.00\n"
    )
    src = NDBCSource(Downloader(min_interval_s=0))
    obs = src.normalize(parse_stdmet(text.encode()), station_id="x", source_file="t").set_index(
        "variable"
    )
    assert obs.loc["visibility", "value"] == pytest.approx(1852.0)
    assert obs.loc["water_level", "value"] == pytest.approx(0.3048)
    assert obs.loc["wind_from_direction", "value"] == 0.0  # 360 -> 0


def test_normalize_spec_and_spectrum(ndbc_fixtures):
    src = NDBCSource(Downloader(min_interval_s=0))
    spec_obs = src.normalize(
        src.parse(ndbc_fixtures / "realtime" / "41010.spec", "spec"),
        station_id="41010",
        source_file="41010.spec",
    )
    validate_observation_frame(spec_obs)
    assert "swell_significant_height" in set(spec_obs["variable"])

    blocks = {"swden": src.parse(ndbc_fixtures / "historical" / "41010w2023.txt.gz", "swden")[0]}
    long, derived = src.normalize_spectrum(
        assemble_spectrum(blocks), station_id="41010", source_file="w"
    )
    assert {"spectral_density_m2_hz", "bandwidth_hz"} <= set(long.columns)
    validate_observation_frame(derived)
    assert set(derived["processing_level"]) == {"derived"}
    assert "spectral_hm0" in set(derived["method"])


def test_validate_returns_qartod_results(ndbc_fixtures):
    src = NDBCSource(Downloader(min_interval_s=0))
    f = ndbc_fixtures / "historical" / "41010h2023.txt.gz"
    obs = src.normalize(src.parse(f, "stdmet"), station_id="41010", source_file=f.name)
    results = src.validate(obs, config={"expected_interval_s": 1800})
    names = {r.test for r in results}
    assert {"gap", "gross_range_hs", "flat_line", "rate_of_change", "steepness"} <= names
    n = (obs["variable"] == "wave_significant_height").sum()
    assert all(r.flags.shape == (n,) for r in results)
    gross = next(r for r in results if r.test == "gross_range_hs")
    assert (gross.flags == QCFlag.PASS).mean() > 0.9


# ----------------------------------------------------------------------------- fetch (mocked)


def test_fetch_records_every_attempt(tmp_path: Path):
    payload = b"\x1f\x8b fake gz bytes"

    def handler(req: httpx.Request) -> httpx.Response:
        if "41010h2023" in str(req.url):
            return httpx.Response(
                200, content=payload, headers={"last-modified": "Tue, 06 Feb 2024 16:35:00 GMT"}
            )
        return httpx.Response(404)

    dl = Downloader(min_interval_s=0, transport=httpx.MockTransport(handler))
    entries = NDBCSource(dl).fetch("41010", ["stdmet"], dest_root=tmp_path, years=[2023, 2024])
    statuses = [e.status for e in entries]
    assert statuses == [EntryStatus.VERIFIED, EntryStatus.NOT_FOUND]
    assert Path(entries[0].local_path).read_bytes() == payload
    assert entries[0].size_bytes == len(payload)
    assert entries[0].http_last_modified is not None
    assert np.all([e.retrieved_at.tzinfo is not None for e in entries])


# ----------------------------------------------------------------------------- buoycam archiver


def test_buoycam_filename_and_snapshot(tmp_path: Path, ndbc_fixtures):
    from wave_analysis.sources.ndbc.buoycam import (
        archive_index,
        parse_buoycam_filename,
        snapshot_cameras,
        summarize,
    )

    cam, ts = parse_buoycam_filename("W04A_2026_09_26_0010.jpg")
    assert cam == "W04A" and ts == pd.Timestamp("2026-09-26 00:10", tz="UTC")
    with pytest.raises(ValueError):
        parse_buoycam_filename("latest.jpg")

    listing = (ndbc_fixtures / "index" / "buoycams.json").read_bytes()

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("buoycams.php"):
            return httpx.Response(200, content=listing)
        return httpx.Response(200, content=b"\xff\xd8jpeg")

    dl = Downloader(min_interval_s=0, transport=httpx.MockTransport(handler))
    first_entries, cams = snapshot_cameras(dl, tmp_path)
    first = summarize(first_entries)
    second = summarize(snapshot_cameras(dl, tmp_path)[0])
    assert first["new_images"] >= 70 and len(cams) >= first["new_images"]
    assert second["new_images"] == 0  # de-duplicated by file name, no re-download
    idx = archive_index(tmp_path)
    assert len(idx) == first["new_images"] and "41010" in set(idx["station_id"])
    # dated layout: <root>/<station>/<YYYY>/<MM>/<file>
    p0 = Path(idx["path"].iloc[0]).relative_to(tmp_path)
    assert len(p0.parts) == 4 and p0.parts[1].isdigit() and len(p0.parts[2]) == 2


def _camera_table(*images: str, stations: tuple[str, ...] = ("41010", "46026")) -> pd.DataFrame:
    return pd.DataFrame({"station_id": list(stations[: len(images)]), "latest_image": list(images)})


def test_buoycam_backfill_candidates(tmp_path: Path):
    from wave_analysis.sources.ndbc.buoycam import backfill_candidates, image_path

    now = pd.Timestamp("2026-09-26 02:05", tz="UTC")
    cams = _camera_table("Z24A_2026_09_26_0110.jpg", "W04A_2026_09_26_0100.jpg")
    todo = backfill_candidates(cams, tmp_path, now=now, hours=5)
    # hours strictly before the latest image whose :10 stamp is newer than now - 5 h
    # (21:05); both minute stamps are candidates, the latest image's minute first
    z, w = "Z24A_2026_09_", "W04A_2026_09_"
    assert todo == [
        ("41010", (f"{z}26_0010.jpg", f"{z}26_0000.jpg")),
        ("41010", (f"{z}25_2310.jpg", f"{z}25_2300.jpg")),
        ("41010", (f"{z}25_2210.jpg", f"{z}25_2200.jpg")),
        ("41010", (f"{z}25_2110.jpg", f"{z}25_2100.jpg")),
        ("46026", (f"{w}26_0000.jpg", f"{w}26_0010.jpg")),
        ("46026", (f"{w}25_2300.jpg", f"{w}25_2310.jpg")),
        ("46026", (f"{w}25_2200.jpg", f"{w}25_2210.jpg")),
        ("46026", (f"{w}25_2100.jpg", f"{w}25_2110.jpg")),
    ]
    # an hour with any stamp on disk is skipped; known-missing names are dropped
    have = image_path(tmp_path, "41010", f"{z}26_0000.jpg")
    have.parent.mkdir(parents=True)
    have.write_bytes(b"x")
    missing = f"https://www.ndbc.noaa.gov/images/buoycam/{w}25_2300.jpg"
    todo2 = backfill_candidates(cams, tmp_path, now=now, hours=5, skip_urls=[missing])
    assert all(names[0] != f"{z}26_0010.jpg" for _, names in todo2)
    assert ("46026", (f"{w}25_2310.jpg",)) in todo2
    assert len(todo2) == 7
    with pytest.raises(ValueError):
        backfill_candidates(cams, tmp_path, now=now, hours=72)


def test_buoycam_archive_run_records_gaps(tmp_path: Path):
    """Latest + backfill; the other minute stamp is tried; 404s are recorded, not re-requested."""
    import json

    from wave_analysis.ingest.manifest import write_manifest
    from wave_analysis.sources.ndbc.buoycam import archive_cameras, manifest_path, summarize

    listing = json.dumps([{"id": "41010", "img": "Z24A_2026_09_26_0110.jpg"}]).encode()
    absent = ("2026_09_25_2310", "2026_09_25_2300", "2026_09_25_2210")
    requested: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requested.append(req.url.path.rsplit("/", 1)[-1])
        if req.url.path.endswith("buoycams.php"):
            return httpx.Response(200, content=listing)
        if any(a in req.url.path for a in absent):
            return httpx.Response(404)
        return httpx.Response(200, content=b"\xff\xd8jpeg")

    now = pd.Timestamp("2026-09-26 02:05", tz="UTC")
    dl = Downloader(min_interval_s=0, transport=httpx.MockTransport(handler))
    ledger = manifest_path(tmp_path, now)
    entries = archive_cameras(
        dl, tmp_path, backfill_hours=4, now=now, sink=lambda b: write_manifest(b, ledger)
    )
    assert sum(1 for _ in ledger.open()) == len(entries) + 1  # streamed to the ledger
    counts = summarize(entries)
    # latest 0110; hour 00 found at :10; hour 23 absent at both stamps; hour 22 found at :00
    assert counts["new_images"] == 3 and counts["not_found"] == 3
    assert "Z24A_2026_09_25_2200.jpg" in requested
    requested.clear()
    again = archive_cameras(dl, tmp_path, backfill_hours=4, now=now)
    assert summarize(again)["new_images"] == 0
    assert requested == ["buoycams.php"]  # nothing re-requested, including the 404 hour


def test_buoycam_ledger_is_named_per_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from wave_analysis.sources.ndbc.buoycam import collector_host, manifest_path

    now = pd.Timestamp("2026-10-08 02:00", tz="UTC")
    monkeypatch.setenv("WAVE_ANALYSIS_HOST", "Lab Desktop_2")
    assert collector_host() == "lab-desktop-2"
    assert manifest_path(tmp_path, now).name == "2026-10.lab-desktop-2.csv"
    assert manifest_path(tmp_path, now, host="laptop").name == "2026-10.laptop.csv"


def test_known_missing_urls_reads_every_host_for_three_months(tmp_path: Path):
    """Two hosts give several ledgers per month; the window is months, not files."""
    from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, write_manifest
    from wave_analysis.sources.ndbc.buoycam import known_missing_urls

    def missing(ledger: str) -> str:
        url = f"https://www.ndbc.noaa.gov/images/buoycam/Z24A_{ledger}.jpg"
        entry = ManifestEntry(
            source_id="ndbc",
            product="buoycam:image",
            station_id="41010",
            url=url,
            status=EntryStatus.NOT_FOUND,
            retrieved_at=pd.Timestamp("2026-10-08 02:00", tz="UTC"),
        )
        write_manifest([entry], tmp_path / "_manifests" / f"{ledger}.csv")
        return url

    missing("2026-07.a")
    kept = {missing(n) for n in ("2026-08", "2026-09.b", "2026-10.a", "2026-10.b")}
    assert known_missing_urls(tmp_path) == kept


def test_split_views_removes_caption():
    from wave_analysis.sources.ndbc.buoycam import caption_band_rows, split_views

    strip = np.zeros((300, 2880, 3), dtype=np.uint8)
    strip[-30:] = 255  # caption band
    views = split_views(strip)
    assert len(views) == 6
    assert all(v.shape == (270, 480, 3) for v in views)
    assert all(v.max() == 0 for v in views)
    assert caption_band_rows(246) == 25
    assert [v.shape[1] for v in split_views(np.zeros((246, 2304)))] == [384] * 6
    with pytest.raises(ValueError):
        split_views(np.zeros((300, 1000)))
