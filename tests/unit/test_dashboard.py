"""Dashboard exports: BibTeX reader, catalog, live status, sea state, site assembly (offline)."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import httpx
import pandas as pd
import pytest

from wave_analysis.cli import main
from wave_analysis.config import repo_root
from wave_analysis.dashboard import seastate as ss
from wave_analysis.dashboard.bibtex import ascii_fold, delatex, parse_bibtex
from wave_analysis.dashboard.catalog import (
    build_catalog,
    docs_url,
    find_mentions,
    load_camera_sites,
    site_records,
    station_records,
    to_json,
    verification_index,
)
from wave_analysis.dashboard.collections import CollectionInputs, build_collections
from wave_analysis.dashboard.site import build_live, build_site, check_site
from wave_analysis.dashboard.status import (
    StatusInputs,
    build_status,
    coverage_string,
    read_ledger,
)
from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, write_manifest

ROOT = repo_root(Path(__file__))
T0 = pd.Timestamp("2026-09-26T12:00:00Z")

# ----------------------------------------------------------------------------- BibTeX

BIB = r"""
% Header comment
% ------------------------------------------ core reviewed papers
@article{Smith2020,
  author  = {Smith, Jane and M{\"u}ller, J{\"o}rg and {\'E}lise Durand},
  title   = {Waves on {GPU}s: a {\em case} study},
  journal = {Ocean Eng.},
  pages   = {1--10},
  year    = {2020},
  doi     = {10.1/abc}
}
% -----------------------------------------------------------------
% Added by a later review (see docs/x.md),
% more notes.
% -----------------------------------------------------------------
@misc{Group2019,
  author = {{IAHR Working Group}},
  title  = "Quoted {title}",
  year   = 2019
}
"""


def test_bibtex_sections_fields_and_authors():
    entries = parse_bibtex(BIB)
    assert [e.key for e in entries] == ["Smith2020", "Group2019"]
    a, b = entries
    assert a.section == "Core reviewed papers"
    assert b.section == "Added by a later review"
    assert a.authors == ["Smith, Jane", "Müller, Jörg", "Élise Durand"]
    assert a.first_author_surname == "Smith"
    assert b.first_author_surname == "IAHR Working Group"  # corporate author kept whole
    assert delatex(a.fields["pages"]) == "1–10"
    assert a.year == 2020 and b.year == 2019
    assert b.fields["title"] == "Quoted {title}"
    assert a.raw.startswith("@article{Smith2020,") and a.raw.endswith("}")


def test_bibtex_rejects_what_it_cannot_read():
    with pytest.raises(ValueError, match="duplicate"):
        parse_bibtex("@misc{A1,\n title={x}}\n@misc{A1,\n title={y}}\n")
    with pytest.raises(ValueError, match="macros"):
        parse_bibtex("@article{A1,\n journal = jfm,\n}\n")
    with pytest.raises(ValueError, match="not supported"):
        parse_bibtex('@string{jfm = "J. Fluid Mech."}\n')
    with pytest.raises(ValueError, match="never closes"):
        parse_bibtex("@misc{A1,\n title={x}\n")


def test_ascii_fold():
    assert ascii_fold("Guimarães") == "Guimaraes"


# ----------------------------------------------------------------------------- repository sources


def test_project_bibliography_and_verification_index_agree():
    entries = parse_bibtex((ROOT / "docs/literature/bibliography.bib").read_text(encoding="utf-8"))
    keys = {e.key for e in entries}
    assert len(keys) == len(entries) >= 50
    levels = verification_index(
        (ROOT / "docs/literature/source_verification.md").read_text(encoding="utf-8")
    )
    assert levels["Kamagata2026"]["level"] == "V1"
    assert set(levels) <= keys, f"unknown keys in verification index: {set(levels) - keys}"
    # Keys cited in the landscape review's source list must exist (catches typos
    # such as a year that does not match the bibliography key).
    review = (ROOT / "docs/literature/landscape_review.md").read_text(encoding="utf-8")
    block = review[review.index("**V1 / V2 (read):**") : review.index("**Data services queried")]
    cited = set(re.findall(r"\b([A-Z][A-Za-z]+\d{4}[a-zA-Z]*)\b", block))
    cited = {c.rstrip("/") for c in cited} - {"SchwendemanThomson2015a"}  # "a/b" shorthand
    assert cited <= keys | {"SchwendemanThomson2015a"}, cited - keys


def test_verification_index_rejects_duplicates():
    md = "## Verification index\n\n| Key | Level | Basis |\n|---|---|---|\n| A1 | V1 | x |\n| A1 | V2 | y |\n"
    with pytest.raises(ValueError, match="twice"):
        verification_index(md)


def test_mkdocs_nav_keeps_external_links(tmp_path):
    from wave_analysis.dashboard.catalog import mkdocs_nav

    yml = tmp_path / "mkdocs.yml"
    yml.write_text(
        "nav:\n  - Home: index.md\n  - Site: https://example.org/\n  - G:\n      - A: a/b.md\n"
    )
    nav = mkdocs_nav(yml)
    assert nav[0]["url"] == "docs/"
    assert nav[1] == {"title": "Site", "url": "https://example.org/", "external": True}
    assert nav[2]["children"][0]["url"] == "docs/a/b/"


def test_docs_url_matches_mkdocs_directory_urls():
    assert docs_url("docs/methodology/models.md") == "docs/methodology/models/"
    assert docs_url("docs/index.md") == "docs/"
    assert docs_url("docs/decisions/README.md") == "docs/decisions/"
    assert docs_url("docs/ROADMAP.md#v02") == "docs/ROADMAP/#v02"


def test_mentions_by_key_and_author_year():
    entries = parse_bibtex(BIB)
    docs = {
        "docs/a.md": "As shown by Smith et al. (2020), and Mueller.",
        "docs/b.md": "See [Group2019].",
        "docs/c.md": "Smith 2019 is a different paper. Smith. 2020 spans a sentence.",
    }
    m = find_mentions(entries, docs)
    assert m["Smith2020"] == ["docs/a.md"]
    assert m["Group2019"] == ["docs/b.md"]


def test_camera_sites_resolve_and_distances_match_the_review():
    sites = load_camera_sites(ROOT / "data/registry/camera_sites.yaml")
    stations = station_records(
        pd.read_parquet(ROOT / "data/registry/stations.parquet"),
        pd.read_csv(ROOT / "data/registry/ndbc_station_summary.csv", dtype={"station_id": str}),
        [r.station_id for s in sites for r in s.references],
    )
    recs = {s["site_id"]: s for s in site_records(sites, stations)}
    km = {r["station_id"]: r["distance_km"] for r in recs["pacioos_waimea"]["references"]}
    assert km["51201"] == pytest.approx(6.3, abs=0.3)  # landscape review §5.3
    assert recs["fau_harbor_branch"]["references"][0]["distance_km"] == pytest.approx(14.6, abs=0.3)
    assert stations["51201"]["cdip_id"] == "106"
    for s in recs.values():
        for r in s["references"]:
            assert r["latitude"] is not None and r["distance_km"] >= 0


def test_site_reference_without_coordinates_is_an_error(tmp_path):
    reg = tmp_path / "sites.yaml"
    reg.write_text(
        "sites:\n  - site_id: x\n    name: X\n    provider: P\n    latitude: 1\n    longitude: 2\n"
        "    coordinate_source: s\n    period: p\n"
        "    references: [{station_id: NOPE, network: NDBC, role: primary}]\n"
    )
    with pytest.raises(ValueError, match="not in the station table"):
        site_records(load_camera_sites(reg), {})


def test_build_catalog_is_strict_json():
    cat = build_catalog(ROOT)
    text = to_json(cat)  # raises on NaN/inf
    assert cat["schema"].startswith("wave-analysis/dashboard-catalog@")
    assert len(cat["datasets"]) >= 25 and len(cat["references"]) >= 50
    assert cat["build"]["live_data_url"].startswith("https://raw.githubusercontent.com/")
    ref = next(r for r in cat["references"] if r["key"] == "Kamagata2026")
    assert ref["verification"]["level"] == "V1"
    assert ref["notes"] == "docs/literature/papers/kamagata2026.md"
    assert ref["notes"] not in ref["mentions"]
    assert "bibtex" in ref and ref["url"] == "https://arxiv.org/abs/2607.11998"
    assert any(d["sites"] for d in cat["datasets"])
    assert "/home/" not in text


# ----------------------------------------------------------------------------- live status


def _listing(root: Path, when: pd.Timestamp, items: list[dict[str, object]]) -> None:
    d = root / "_listings"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"buoycams_{when:%Y%m%dT%H%M%SZ}.json").write_text(json.dumps(items))


def _image(root: Path, sid: str, cam: str, t: pd.Timestamp, size: int = 1000) -> None:
    p = root / sid / f"{t:%Y}" / f"{t:%m}" / f"{cam}_{t:%Y_%m_%d_%H%M}.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\xff" * size)


def _entry(product: str, sid: str | None, period: str | None, status: str, at: pd.Timestamp):
    return ManifestEntry(
        source_id="ndbc",
        product=product,
        station_id=sid,
        period=period,
        url=f"https://example.invalid/{sid}/{period}",
        retrieved_at=at.to_pydatetime(),
        status=EntryStatus(status),
    )


@pytest.fixture
def archive(tmp_path: Path) -> StatusInputs:
    root = tmp_path / "raw" / "ndbc" / "buoycam"
    # Station 41010: images 08:10 .. 11:10 except 09 (404 at :10 and :00) and 10 (missed).
    for h in (8, 11):
        _image(root, "41010", "W04A", T0.replace(hour=h, minute=10))
    _image(root, "41010", "W04A", T0.replace(hour=7, minute=0))  # a :00 stamp
    listing = [
        {"id": "41010", "name": "CANAVERAL EAST", "lat": 28.86, "lng": -78.478,
         "img": "W04A_2026_09_26_1110.jpg", "width": 2880, "height": 300},
        {"id": "41099", "name": "NO IMAGE YET", "lat": 30.0, "lng": -70.0, "img": ""},
    ]  # fmt: skip
    _listing(root, T0.replace(minute=40), listing)
    run = T0.replace(minute=40)
    ledger = [
        _entry("buoycam:listing", None, "snapshot", "verified", run),
        _entry("buoycam:image", "41010", "2026-09-26T11:10Z", "verified", run + pd.Timedelta("1s")),
        _entry(
            "buoycam:image", "41010", "2026-09-26T09:10Z", "not_found", run + pd.Timedelta("2s")
        ),
        _entry(
            "buoycam:image", "41010", "2026-09-26T09:00Z", "not_found", run + pd.Timedelta("3s")
        ),
        # A 404 at :10 for an hour that has a :00 image is not "unpublished".
        _entry(
            "buoycam:image", "41010", "2026-09-26T07:10Z", "not_found", run + pd.Timedelta("4s")
        ),
    ]
    write_manifest(ledger, root / "_manifests" / "2026-09.csv")
    with (root / "_manifests" / "2026-09.csv").open("a") as fh:
        fh.write("ndbc,buoycam:image,41010,2026-09")  # truncated line being appended
    rt = tmp_path / "raw" / "ndbc" / "41010" / "stdmet"
    rt.mkdir(parents=True)
    (rt / "41010.txt.20260915T032000Z.gz").write_bytes(gzip.compress(b"x" * 10))
    return StatusInputs(
        buoycam_root=root, ndbc_root=tmp_path / "raw" / "ndbc", raw_root=tmp_path / "raw"
    )


def test_read_ledger_skips_truncated_rows(archive):
    led = read_ledger(archive.buoycam_root)
    assert len(led) == 5


def test_status_coverage_encoding(archive):
    st = build_status(archive, now=T0 + pd.Timedelta(minutes=45), window_hours=6)
    assert st["window"] == {
        "start": "2026-09-26T07:00:00Z",
        "hours": 6,
        "as_of": "2026-09-26T12:40:00Z",
    }
    s = {x["station_id"]: x for x in st["stations"]}
    # hours 07..12: 07 archived at :00, 08 at :10, 09 unpublished, 10 missed, 11 at :10, 12 pending
    assert s["41010"]["coverage"] == "01x?1."
    assert s["41010"]["camera"] == "W04A" and s["41010"]["images"] == 3
    assert len(s["41010"]["illumination"]) == 6 and set(s["41010"]["illumination"]) <= set("dltn")
    assert s["41099"]["listed"] and s["41099"]["images"] == 0 and s["41099"]["coverage"] == "......"
    assert st["archive"]["window_coverage"] == {"archived": 3, "not_published": 1, "gap": 1}
    assert st["archive"]["window_minute_stamps"] == {"0": 1, "10": 2}
    assert st["collector"]["runs"] == [
        {
            "at": "2026-09-26T12:40:00Z",
            "new_images": 1,
            "not_found": 3,
            "failed": 0,
            "duration_s": 4,
        }
    ]
    assert st["realtime"] == [
        {
            "date": "2026-09-15",
            "stations": 1,
            "products": ["stdmet"],
            "files": 1,
            "bytes": len(gzip.compress(b"x" * 10)),
        }
    ]
    assert st["offsite"] == {"last_sync": None}
    assert sum(st["daily"]["images"]) == 3
    assert str(archive.raw_root) not in to_json(st)


def test_status_lists_the_buoycam_collection(archive):
    st = build_status(archive, now=T0 + pd.Timedelta(minutes=45), window_hours=6)
    ids = [c["id"] for c in st["collections"]]
    assert ids == ["ndbc_buoycam", "ndbc_realtime"]  # no manifests_root: nothing else
    cam = st["collections"][0]
    assert cam["count"] == 3 and cam["not_found"] == 1 and cam["parts_count"] == 1
    # The fixture's ledger URLs are not image names, so every file is unledgered.
    assert cam["check"] == {"on_disk": 3, "ledger": 1, "missing_files": 1, "unledgered_files": 3}


def _ledger_rows(path: Path, rows: list[dict[str, object]]) -> None:
    at = T0.to_pydatetime()
    write_manifest(
        [ManifestEntry(retrieved_at=r.pop("at", at), **r) for r in rows],
        path,
    )


def test_collections_count_files_and_check_ledgers(tmp_path):
    raw, man = tmp_path / "raw", tmp_path / "manifests"
    pac = raw / "pacioos" / "beachcam"
    base = "https://pae-paha.pacioos.hawaii.edu/erddap/files/beachcam_003/2009/02/05"
    img = [
        ("beachcam_003.20090205T170000.jpg", "2009-02-06T03:00:00Z", "verified"),
        ("beachcam_003.20090205T180000.jpg", "2009-02-06T04:00:00Z", "verified"),  # not on disk
        ("beachcam_003.20090205T190000.jpg", "2009-02-06T05:00:00Z", "not_found"),
    ]
    _ledger_rows(
        pac / "_manifests" / "beachcam_003.csv",
        [
            {"source_id": "pacioos_beachcam", "product": "index", "station_id": "beachcam_003",
             "period": "all", "url": "https://pae-paha.pacioos.hawaii.edu/erddap/tabledap/x.csv",
             "status": EntryStatus.VERIFIED},
            *[
                {"source_id": "pacioos_beachcam", "product": "image", "station_id": "beachcam_003",
                 "period": period, "url": f"{base}/{name}", "size_bytes": 5, "status": EntryStatus(s)}
                for name, period, s in img
            ],
        ],
    )  # fmt: skip
    for name in ("beachcam_003.20090205T170000.jpg", "stray.jpg"):
        f = pac / "beachcam_003" / "2009" / "02" / "05" / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x" * 5)
    (pac / "_index").mkdir()
    (pac / "_index" / "ignored.jpg").write_bytes(b"x")  # under a "_" directory
    (man / "processed").mkdir(parents=True)
    (man / "processed" / "waimea_cdip106.summary.json").write_text(
        json.dumps(
            {
                "images": {
                    "exclusion": {"paired": 7, "no_buoy_within_30min": 2, "time_uncertain": 1},
                    "paired_days": 3,
                }
            }
        )
    )

    wc = raw / "webcoos"
    url = "https://s3.example.invalid/cam/2026/09/26/cam-2026-09-26-120000Z.jpg"
    _ledger_rows(
        wc / "_manifests" / "cam.csv",
        [
            {"source_id": "webcoos", "product": "image", "station_id": "cam",
             "period": "2026-09-26T12:00:00Z", "url": url, "status": EntryStatus.FAILED},
            {"source_id": "webcoos", "product": "image", "station_id": "cam",
             "period": "2026-09-26T12:00:00Z", "url": url, "size_bytes": 3,
             "status": EntryStatus.VERIFIED, "at": (T0 + pd.Timedelta("1h")).to_pydatetime()},
        ],
    )  # fmt: skip
    (wc / "cam" / "2026" / "09" / "26").mkdir(parents=True)
    (wc / "cam" / "2026" / "09" / "26" / "cam-2026-09-26-120000Z.jpg").write_bytes(b"x" * 3)

    era = "https://cds.climate.copernicus.eu/api/retrieve/v1/processes/x"
    _ledger_rows(
        man / "raw" / "era5.csv",
        [
            {"source_id": "era5_waves", "product": "wave_single_levels", "station_id": "site",
             "period": "2026-07", "url": era, "status": EntryStatus.VERIFIED,
             "note": "request ab; CDS request 1; preliminary (was missing)"},
            {"source_id": "era5_waves", "product": "wave_single_levels", "station_id": "site",
             "period": "2026-07", "url": era, "status": EntryStatus.CHANGED,
             "note": "request ab; CDS request 2; final (was preliminary)",
             "at": (T0 + pd.Timedelta("1h")).to_pydatetime()},
        ],
    )  # fmt: skip
    (raw / "era5" / "site").mkdir(parents=True)
    (raw / "era5" / "site" / "era5_waves_site_2026-07.nc").write_bytes(b"x" * 7)

    # A camera fetched only from the historical archive, by another host.
    old_url = "https://onprem.example.invalid/old/2024/01/02/old-2024-01-02-150000Z.jpg"
    _ledger_rows(
        wc / "_manifests" / "old.lab-pc.csv",
        [
            {"source_id": "webcoos", "product": "image", "station_id": "old",
             "period": "2024-01-02T15:00:00Z", "url": old_url, "size_bytes": 2,
             "status": EntryStatus.VERIFIED},
        ],
    )  # fmt: skip
    (wc / "old" / "2024" / "01" / "02").mkdir(parents=True)
    (wc / "old" / "2024" / "01" / "02" / "old-2024-01-02-150000Z.jpg").write_bytes(b"x" * 2)
    tx = raw / "webcoos_timex"
    tx_url = "https://onprem.example.invalid/timex/cam/2024/01/02/timex.cam-2024-01-02-150009Z.jpg"
    _ledger_rows(
        tx / "_manifests" / "cam.lab-pc.csv",
        [
            {"source_id": "webcoos", "product": "image", "station_id": "cam",
             "period": "2024-01-02T15:00:09Z", "url": tx_url, "size_bytes": 4,
             "status": EntryStatus.VERIFIED},
        ],
    )  # fmt: skip
    (tx / "cam" / "2024" / "01" / "02").mkdir(parents=True)
    (tx / "cam" / "2024" / "01" / "02" / "timex.cam-2024-01-02-150009Z.jpg").write_bytes(b"x" * 4)

    out = build_collections(CollectionInputs(raw, man))
    by = {c["id"]: c for c in out}
    # no CDIP or brt ledger: skipped
    assert list(by) == ["pacioos_waimea", "webcoos", "webcoos_timex", "era5_waves"]
    w = by["pacioos_waimea"]
    assert (w["count"], w["bytes"], w["not_found"]) == (2, 10, 1)
    assert w["check"] == {"on_disk": 2, "ledger": 2, "missing_files": 1, "unledgered_files": 1}
    assert (w["first"], w["last"]) == ("2009-02-06T03:00:00Z", "2009-02-06T04:00:00Z")
    assert w["pairing"]["trusted_time"] == 9 and w["pairing"]["paired"] == 7
    c = by["webcoos"]
    assert c["count"] == 2 and c["failed"] == 0  # the retry succeeded
    assert c["check"]["missing_files"] == c["check"]["unledgered_files"] == 0
    assert {p["id"]: p["hourly"] for p in c["parts"]} == {"cam": True, "old": False}
    t = by["webcoos_timex"]
    assert (t["mode"], t["count"], t["bytes"]) == ("on_request", 1, 4)
    assert t["check"]["missing_files"] == t["check"]["unledgered_files"] == 0
    e = by["era5_waves"]
    assert e["count"] == 1 and e["states"] == {"final": 1}
    assert e["check"] == {"on_disk": 1, "ledger": 1, "missing_files": 0, "unledgered_files": 0}
    assert str(tmp_path) not in to_json(out)


def test_collections_ndbc_history_and_external(tmp_path):
    raw, man, ext = tmp_path / "raw", tmp_path / "manifests", tmp_path / "external"
    hist = "https://www.ndbc.noaa.gov/data/historical/stdmet"
    _ledger_rows(
        man / "raw" / "ndbc.csv",
        [
            {"source_id": "ndbc", "product": "stdmet", "station_id": stn, "period": "2023",
             "url": f"{hist}/{stn}h2023.txt.gz", "size_bytes": 4, "status": EntryStatus.VERIFIED}
            for stn in ("41008", "41009")
        ],
    )  # fmt: skip
    (raw / "ndbc" / "41008" / "stdmet").mkdir(parents=True)
    (raw / "ndbc" / "41008" / "stdmet" / "41008h2023.txt.gz").write_bytes(b"x" * 4)
    # A realtime snapshot next to it and a buoy-camera image are not history files.
    (raw / "ndbc" / "41008" / "stdmet" / "41008.txt.20261001T122511Z.gz").write_bytes(b"x")
    (raw / "ndbc" / "buoycam" / "41008").mkdir(parents=True)
    (raw / "ndbc" / "buoycam" / "41008" / "41008h2023.txt.gz").write_bytes(b"x")

    repo = "https://digital.lib.washington.edu/server/api/core/bitstreams"
    rows = [
        {"source_id": "stereo_set", "product": name, "url": f"{repo}/{i}/content", "size_bytes": 6,
         "status": EntryStatus.VERIFIED}
        for i, name in enumerate(("readme.txt", "data.zip"))
    ]  # fmt: skip
    _ledger_rows(man / "raw" / "external.csv", rows)
    (ext / "stereo_set").mkdir(parents=True)
    for name in ("readme.txt", "data.zip"):
        (ext / "stereo_set" / name).write_bytes(b"x" * 6)
    (ext / "stereo_set" / "SOURCE.yaml").write_text("written by the downloader")

    out = build_collections(CollectionInputs(raw, man, ext))
    by = {c["id"]: c for c in out}
    assert list(by) == ["ndbc_history", "external"]
    h = by["ndbc_history"]
    assert (h["count"], h["bytes"], h["first"], h["last"], h["parts_count"]) == (
        2,
        4,
        "2023",
        "2023",
        2,
    )
    assert h["check"] == {"on_disk": 1, "ledger": 2, "missing_files": 1, "unledgered_files": 0}
    assert h["parts"] == [
        {"id": "stdmet", "count": 2, "bytes": 4, "first": "2023", "last": "2023", "stations": 2}
    ]
    e = by["external"]
    assert (e["count"], e["bytes"], e["dataset_id"]) == (2, 12, None)
    assert e["check"] == {"on_disk": 2, "ledger": 2, "missing_files": 0, "unledgered_files": 0}
    assert e["parts"][0]["id"] == "stereo_set"
    assert str(tmp_path) not in to_json(out)


def test_status_offsite_stamp(archive):
    (archive.raw_root / ".offsite_last_sync").write_text("2026-09-26T04:15:00Z\n")
    st = build_status(archive, now=T0, window_hours=2)
    assert st["offsite"] == {"last_sync": "2026-09-26T04:15:00Z"}


def test_coverage_string_respects_station_lifetime():
    hours = pd.date_range(T0, periods=4, freq="h")
    kw = {"as_of": T0 + pd.Timedelta(hours=5)}
    assert (
        coverage_string(
            hours, pd.Series([], dtype=object), set(), active_from=None, active_until=None, **kw
        )
        == "...."
    )
    # Removed from the listing after its last image at hour 1: later hours are not gaps.
    got = coverage_string(
        hours,
        pd.Series([T0 + pd.Timedelta(minutes=10)]),
        set(),
        active_from=T0,
        active_until=T0 + pd.Timedelta(hours=1),
        **kw,
    )
    assert got == "1?.."


# ----------------------------------------------------------------------------- sea state

TXT = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft
2026 09 26 11 50 350  6.0  8.0   2.8    15   8.9  11 1013.3  25.3  29.4  17.9   MM   MM    MM
2026 09 26 11 40 350  6.0  9.0    MM    MM    MM  MM 1013.3  25.3  29.4  18.0   MM   MM    MM
2026 09 26 11 20 360  6.0  8.0   3.0    13   9.2  32 1012.9  25.3  29.3  18.4   MM   MM    MM
2026 09 26 11 10 360  6.0  9.0   3.0    MM   9.2  32 1012.8  25.2  29"""  # cut mid-row
SPEC = """#YY  MM DD hh mm WVHT  SwH  SwP  WWH  WWP SwD WWD  STEEPNESS  APD MWD
#yr  mo dy hr mn    m    m  sec    m  sec  -  degT     -      sec degT
2026 09 26 11 40  2.9  2.8 12.9  0.7  4.3 NNE   N      SWELL  9.2  31
2026 09 26 11 10  3.0  2.9 12.9  0.8  5.0   N  NE    AVERAGE  9.2  32
"""


def test_parse_head_drops_the_partial_line():
    rows = ss.parse_head(ss.complete_lines(TXT.encode()), ss.PRODUCTS[0])
    assert len(rows) == 3  # the cut 11:10 row is dropped
    assert rows.index.is_monotonic_increasing
    assert rows["WVHT"].iloc[-1] == 2.8 and pd.isna(rows["DPD"].iloc[1])
    spec = ss.parse_head(ss.complete_lines(SPEC.encode()), ss.PRODUCTS[1])
    assert spec["SwD_deg"].tolist() == [0.0, 22.5] and spec["STEEPNESS"].iloc[-1] == "SWELL"
    assert ss.parse_head("#YY MM\n#yr mo\n", ss.PRODUCTS[0]).empty


def test_bytes_needed_bounds():
    p = ss.PRODUCTS[0]
    assert ss.bytes_needed(None, T0, p) == ss.MAX_BYTES
    assert ss.bytes_needed(T0 - pd.Timedelta(minutes=30), T0, p) == ss.MIN_BYTES
    assert ss.bytes_needed(T0 - pd.Timedelta(days=30), T0, p) == ss.MAX_BYTES
    mid = ss.bytes_needed(T0 - pd.Timedelta(hours=20), T0, p)
    assert ss.MIN_BYTES < mid < ss.MAX_BYTES


def _fetcher(calls: list[tuple[str, int]]):
    def fetch(url: str, n: int) -> bytes | None:
        calls.append((url, n))
        if "/41010." in url:
            return (TXT if url.endswith(".txt") else SPEC).encode()
        if "/41011." in url:
            raise httpx.ConnectError("boom")
        return None

    return fetch


def test_refresh_merges_and_isolates_failures(tmp_path):
    calls: list[tuple[str, int]] = []
    summary = ss.refresh(["41010", "41011", "41012"], tmp_path, _fetcher(calls), now=T0)
    assert summary.requests == 6 and summary.missing_files == 2 and len(summary.failures) == 2
    assert calls[0] == (f"{ss.REALTIME_BASE}/41010.txt", ss.MAX_BYTES)
    # Second run: smaller ranges, duplicates collapse.
    calls.clear()
    ss.refresh(["41010"], tmp_path, _fetcher(calls), now=T0)
    assert all(n == ss.MIN_BYTES for _, n in calls)
    caches = {p.name: ss.load_cache(tmp_path, p) for p in ss.PRODUCTS}
    assert len(caches["stdmet"]) == 3 and len(caches["spec"]) == 2
    # Rows older than KEEP_DAYS are trimmed.
    ss.refresh([], tmp_path, _fetcher([]), now=T0 + pd.Timedelta(days=ss.KEEP_DAYS + 1))
    assert ss.load_cache(tmp_path, ss.PRODUCTS[0]).empty


def test_payload_and_latest_values(tmp_path):
    ss.refresh(["41010"], tmp_path, _fetcher([]), now=T0)
    caches = {p.name: ss.load_cache(tmp_path, p) for p in ss.PRODUCTS}
    pl = ss.station_payload("41010", caches, now=T0)
    assert pl is not None
    std = pl["products"]["stdmet"]
    assert len(std["t"]) == 3 and std["WVHT"] == [3.0, None, 2.8] and std["MWD"][0] == 32
    assert pl["products"]["spec"]["STEEPNESS"] == ["AVERAGE", "SWELL"]
    assert ss.station_payload("99999", caches, now=T0) is None
    latest = ss.latest_values(caches, now=T0)["41010"]
    assert latest["WVHT"] == 2.8 and latest["wave_time"] == "2026-09-26T11:50:00Z"
    assert latest["WSPD"] == 6.0 and latest["wind_time"] == "2026-09-26T11:50:00Z"
    assert latest["SwH"] == 2.8 and latest["STEEPNESS"] == "SWELL"
    assert ss.latest_values(caches, now=T0 + pd.Timedelta(hours=5)) == {}
    json.dumps(pl, allow_nan=False)


def test_range_fetcher_asks_for_identity_encoding():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        if request.url.path.endswith("missing.txt"):
            return httpx.Response(404)
        return httpx.Response(206, content=b"abc")

    f = ss.RangeFetcher(min_interval_s=0)
    f._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert f("https://x.invalid/a.txt", 1024) == b"abc"
    assert seen["range"] == "bytes=0-1023" and seen["accept-encoding"] == "identity"
    assert f("https://x.invalid/missing.txt", 10) is None
    f.close()


# ----------------------------------------------------------------------------- site


def test_build_live_and_site_round_trip(archive, tmp_path):
    live = tmp_path / "live"
    res = build_live(
        ROOT,
        archive,
        live,
        cache_dir=tmp_path / "cache",
        fetch=_fetcher([]),
        now=T0 + pd.Timedelta(minutes=45),
    )
    assert res["seastate_files"] == 1
    status = json.loads((live / "status.json").read_text())
    assert status["seastate"]["stations"] == ["41010"]
    assert status["stations"][0]["latest_obs"]["WVHT"] == 2.8
    manifest = json.loads((live / "manifest.json").read_text())
    for f in manifest["files"]:
        assert hashlib.sha256((live / f["path"]).read_bytes()).hexdigest() == f["sha256"]

    site = tmp_path / "site"
    info = build_site(ROOT, site, live_dir=live)
    assert info["live_bundled"] and (site / ".nojekyll").exists()
    assert check_site(site) == []
    # Tampering with bundled data is detected.
    (site / "data/live/status.json").write_text(json.dumps({**status, "schema": "nope"}))
    problems = check_site(site)
    assert any("schema" in p for p in problems) and any("manifest" in p for p in problems)


def test_check_site_reports_missing_docs_and_assets(tmp_path):
    site = tmp_path / "site"
    build_site(ROOT, site)
    assert any("docs build did not produce" in p for p in check_site(site, require_docs=True))
    (site / "assets/js/lib/format.js").unlink()
    assert any("imports missing module" in p for p in check_site(site))


def test_cli_dashboard_catalog(tmp_path, capsys):
    assert main(["dashboard", "catalog", "--out", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "catalog.json").read_text())["schema"].startswith(
        "wave-analysis/"
    )
    assert "datasets" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Collector watchdog
# --------------------------------------------------------------------------- #

_WATCHED = {
    "generated_at": "2026-10-08T12:54:09Z",
    "collector": {"last_run": "2026-10-08T12:42:09Z", "runs": []},
    "archive": {"last_image": "2026-10-08T11:10:00Z"},
    "offsite": {"last_sync": "2026-10-08T08:18:56Z"},
}


def test_watchdog_flags_each_stale_timestamp():
    import datetime as dt

    from wave_analysis.dashboard.watchdog import alert_title, check_status, loss_deadline

    def at(s: str) -> dt.datetime:
        return dt.datetime.fromisoformat(s).replace(tzinfo=dt.UTC)

    assert all(c.ok for c in check_status(_WATCHED, at("2026-10-08 15:40")))
    late = check_status(_WATCHED, at("2026-10-08 15:50"))  # 3.1 h after the last run
    assert [c.name for c in late if not c.ok] == ["collector"]
    day = check_status(_WATCHED, at("2026-10-09 14:30"))
    assert [c.name for c in day if not c.ok] == ["collector", "publisher", "offsite"]
    assert alert_title(day) == (
        "Collector watchdog: collector, publisher, offsite silent since 2026-10-08 08:18 UTC"
    )
    # the 12:10 image is the first not archived; it leaves the 70 h window at 10:10 + 3 d
    assert loss_deadline(_WATCHED) == at("2026-10-11 10:10")
    # a missing or unreadable timestamp is a failure, not a pass
    broken = {**_WATCHED, "offsite": {"last_sync": None}, "generated_at": "garbage"}
    bad = check_status(broken, at("2026-10-08 13:00"))
    assert [c.name for c in bad if not c.ok] == ["publisher", "offsite"]
    assert alert_title(bad) == "Collector watchdog: publisher, offsite silent"


def test_watchdog_cli_exit_codes_and_report(tmp_path, capsys, monkeypatch):
    from wave_analysis.dashboard.watchdog import main as watchdog

    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    status = tmp_path / "status.json"
    status.write_text(json.dumps(_WATCHED))
    report = tmp_path / "report.md"
    assert watchdog([str(status), "--now", "2026-10-08T13:00Z", "--report", str(report)]) == 0
    assert capsys.readouterr().out.strip() == "healthy"
    assert "lost for good" not in report.read_text()
    assert watchdog([str(status), "--now", "2026-10-09T02:00Z", "--report", str(report)]) == 1
    assert capsys.readouterr().out.startswith("Collector watchdog: collector, publisher silent")
    text = report.read_text()
    assert "**2026-10-11 10:10 UTC** (56 h from now)" in text
    assert "github.com/StevenFAU/Wave_Analysis/blob/main/docs/operations" in text
    status.write_text("[]")
    assert watchdog([str(status)]) == 2
    assert watchdog([str(tmp_path / "missing.json")]) == 2
