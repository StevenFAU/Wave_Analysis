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
    yml.write_text("nav:\n  - Home: index.md\n  - Site: https://example.org/\n  - G:\n      - A: a/b.md\n")
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
