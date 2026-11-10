"""Integrity of the raw archive: verification against ledgers, durable writes, repairs."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
import pytest

from wave_analysis.cli import main
from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.durable import atomic_open
from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, write_manifest
from wave_analysis.ingest.offsite import compare_remote, decide, default_remote, repair
from wave_analysis.ingest.verify import (
    EMPTY_SHA256,
    hash_file,
    relative_to_raw,
    verify_archive,
    write_report,
)

JPEG = b"\xff\xd8\xff\xe0" + bytes(range(256)) * 8 + b"\xff\xd9"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(
    raw: Path, rel: str, data: bytes, *, note: str | None = None, when: str = ""
) -> ManifestEntry:
    return ManifestEntry(
        source_id="test",
        product="image",
        station_id="cam",
        url=f"https://example.org/{rel}",
        local_path=str(raw / rel),
        sha256=_sha(data),
        size_bytes=len(data),
        http_status=200,
        retrieved_at=datetime.fromisoformat(when) if when else datetime.now(UTC),
        status=EntryStatus.VERIFIED,
        note=note,
    )


def _put(raw: Path, rel: str, data: bytes) -> Path:
    p = raw / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """A raw root with one file of every status, and its ledger."""
    raw = tmp_path / "raw"
    ledger = raw / "cams" / "_manifests" / "cam.host.csv"
    entries = []
    for rel, data, stored in [
        ("cams/a/ok.jpg", JPEG, JPEG),
        ("cams/a/nul.jpg", JPEG, b"\0" * len(JPEG)),  # right size, zeros: invisible to size checks
        ("cams/a/empty.jpg", JPEG, b""),
        ("cams/a/short.jpg", JPEG, JPEG[:-10]),
        ("cams/a/upstream_empty.jpg", b"", b""),
    ]:
        _put(raw, rel, stored)
        entries.append(_entry(raw, rel, data))
    entries.append(_entry(raw, "cams/a/gone.jpg", JPEG))
    # stored gzip-compressed: the ledger describes the payload as served
    payload = b"#YY MM DD hh mm WVHT\n2026 11 10 12 00 1.2\n"
    _put(raw, "buoy/41002/41002.txt.20261110T120000Z.gz", gzip.compress(payload, mtime=0))
    entries.append(
        _entry(
            raw,
            "buoy/41002/41002.txt.20261110T120000Z.gz",
            payload,
            note="stored gzip-compressed; sha256/size_bytes are of the served payload",
        )
    )
    write_manifest(entries, ledger)
    _put(raw, "cams/a/no_row.jpg", JPEG)
    _put(raw, "cams/a/no_row_empty.jpg", b"")
    _put(raw, "cams/_listings/cam_20261110T120000Z.csv", b"name,url\n")
    _put(raw, "cams/a/.partial-abc123", JPEG[:5])
    _put(raw, ".offsite_last_sync", b"2026-11-04T10:05:00Z\n")
    return raw


def _statuses(raw: Path, **kw: object) -> dict[str, str]:
    report = verify_archive(raw, **kw)  # type: ignore[arg-type]
    return {r.path: r.status for r in report.results}


def test_verify_gives_every_file_one_status(archive: Path) -> None:
    got = _statuses(archive)
    assert got == {
        "cams/a/ok.jpg": "ok",
        "cams/a/nul.jpg": "sha256_mismatch",
        "cams/a/empty.jpg": "empty",
        "cams/a/short.jpg": "size_mismatch",
        "cams/a/upstream_empty.jpg": "empty_upstream",
        "cams/a/gone.jpg": "missing",
        "buoy/41002/41002.txt.20261110T120000Z.gz": "ok",
        "cams/a/no_row.jpg": "no_ledger",
        "cams/a/no_row_empty.jpg": "suspect",
        "cams/_listings/cam_20261110T120000Z.csv": "no_ledger",
        "cams/a/.partial-abc123": "partial",
    }


def test_verify_report_files(archive: Path, tmp_path: Path) -> None:
    report = verify_archive(archive)
    assert {r.path for r in report.damaged()} == {
        "cams/a/nul.jpg",
        "cams/a/empty.jpg",
        "cams/a/short.jpg",
        "cams/a/no_row_empty.jpg",
    }
    nul = next(r for r in report.results if r.path == "cams/a/nul.jpg")
    assert nul.note == "all NUL bytes" and nul.ledger_sha256 == _sha(JPEG)
    assert nul.md5 == hashlib.md5(b"\0" * len(JPEG)).hexdigest()
    out = write_report(report, tmp_path / "report")
    problems = pd.read_csv(out["problems"])
    assert "cams/a/ok.jpg" not in set(problems["path"])
    files = pd.read_csv(out["files"])
    assert len(files) == len(report.results)
    assert not list((tmp_path / "report").glob(".partial-*"))


def test_a_file_matching_any_ledger_version_passes(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    v1, v2 = JPEG, JPEG + b"v2"
    _put(raw, "cams/a/x.jpg", v1)
    # e.g. two hosts each fetched it; the newer row describes other bytes
    write_manifest(
        [
            _entry(raw, "cams/a/x.jpg", v1, when="2026-11-01T00:00:00+00:00"),
            _entry(raw, "cams/a/x.jpg", v2, when="2026-11-02T00:00:00+00:00"),
        ],
        raw / "cams" / "_manifests" / "cam.csv",
    )
    report = verify_archive(raw)
    (r,) = report.results
    assert r.status == "ok" and r.ledger_versions == 2


def test_damaged_ledger_is_reported_and_its_readable_rows_used(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _put(raw, "cams/a/x.jpg", JPEG)
    ledger = raw / "cams" / "_manifests" / "cam.csv"
    write_manifest([_entry(raw, "cams/a/x.jpg", JPEG)], ledger)
    with ledger.open("ab") as fh:
        fh.write(b"\0" * 2160)  # what the power cut of 2026-11-03 left
    report = verify_archive(raw)
    assert report.results[0].status == "ok"
    (bad,) = report.damaged_ledgers()
    assert bad["nul_bytes"] == 2160


def test_since_checks_recent_files_only(archive: Path) -> None:
    old = (datetime.now(UTC) - timedelta(days=10)).timestamp()
    for p in (archive / "cams" / "a").iterdir():
        if p.name != "nul.jpg":
            os.utime(p, (old, old))
    got = _statuses(archive, since=datetime.now(UTC) - timedelta(days=1))
    assert "cams/a/nul.jpg" in got and "cams/a/ok.jpg" not in got
    assert got["cams/a/gone.jpg"] == "missing"  # its row is recent


def test_paths_and_collections(archive: Path) -> None:
    got = _statuses(archive, paths=["cams/a/ok.jpg", "cams/a/nope.jpg"])
    assert got == {"cams/a/ok.jpg": "ok", "cams/a/nope.jpg": "missing"}
    assert set(_statuses(archive, collections=["buoy"])) == {
        "buoy/41002/41002.txt.20261110T120000Z.gz"
    }


def test_relative_to_raw(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    assert relative_to_raw("data/raw/webcoos/a.jpg", raw) == "webcoos/a.jpg"
    assert relative_to_raw(str(raw / "x" / "b.jpg"), raw) == "x/b.jpg"
    assert relative_to_raw("data/external/c.txt", raw) is None
    assert relative_to_raw("/elsewhere/d.jpg", raw) is None


def test_hash_file_flags_broken_gzip(tmp_path: Path) -> None:
    p = tmp_path / "x.gz"
    p.write_bytes(gzip.compress(b"payload" * 1000)[:-20])
    assert hash_file(p, gunzip=True).error
    q = tmp_path / "e"
    q.write_bytes(b"")
    assert hash_file(q).sha256 == EMPTY_SHA256


def test_cli_verify_exit_code_and_exclude_list(archive: Path, tmp_path: Path, capsys) -> None:
    excl = tmp_path / "exclude.txt"
    code = main(
        [
            "archive",
            "verify",
            "--root",
            str(archive),
            "--out",
            str(tmp_path / "rep"),
            "--exclude-list",
            str(excl),
            "--quiet",
        ]
    )
    assert code == 1
    assert "DAMAGED sha256_mismatch: cams/a/nul.jpg" in capsys.readouterr().out
    rules = excl.read_text().splitlines()
    assert "/cams/a/nul.jpg" in rules and "/cams/a/ok.jpg" not in rules


def test_cli_verify_since_last_exchange(archive: Path, tmp_path: Path, capsys) -> None:
    (archive / ".offsite_last_sync").write_text(
        (datetime.now(UTC) + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ") + "\n"
    )
    code = main(["archive", "verify", "--root", str(archive), "--out", str(tmp_path / "r"),
                 "--since", "last-exchange", "--quiet"])  # fmt: skip
    assert code == 0  # nothing modified within a day of that exchange
    assert "files modified since" in capsys.readouterr().out


# --------------------------------------------------------------------------- durable writes


def test_atomic_open_keeps_the_old_file_on_error(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    with atomic_open(p, "w") as fh:
        fh.write("one\n")

    def fail() -> None:
        with atomic_open(p, "w") as fh:
            fh.write("two\n")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        fail()
    assert p.read_text() == "one\n"
    assert [x.name for x in tmp_path.iterdir()] == ["a.csv"]


def test_downloader_syncs_the_file_before_the_rename(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd: int) -> None:
        events.append("fsync-dir" if os.path.isdir(f"/proc/self/fd/{fd}") else "fsync-file")
        real_fsync(fd)

    def replace(a: object, b: object) -> None:
        events.append("rename")
        real_replace(a, b)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    dl = Downloader(
        min_interval_s=0, transport=httpx.MockTransport(lambda r: httpx.Response(200, content=JPEG))
    )
    e = dl.fetch("https://example.org/x.jpg", tmp_path / "x.jpg", source_id="t", product="p")
    assert e.status == EntryStatus.VERIFIED
    assert events == ["fsync-file", "rename", "fsync-dir"]


def test_ledger_rows_are_synced(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real_fsync(fd)))
    raw = tmp_path / "raw"
    write_manifest([_entry(raw, "a.jpg", JPEG)], raw / "_manifests" / "l.csv")
    n_new = len(calls)
    write_manifest([_entry(raw, "b.jpg", JPEG)], raw / "_manifests" / "l.csv")
    assert n_new == 2  # the file, and its new directory entry
    assert len(calls) - n_new == 1  # an append syncs the file


# --------------------------------------------------------------------------- empty files are fetched again


def test_buoycam_empty_image_counts_as_missing(tmp_path: Path) -> None:
    from wave_analysis.sources.ndbc.buoycam import backfill_candidates, image_path

    now = pd.Timestamp("2026-09-26 02:05", tz="UTC")
    cams = pd.DataFrame({"station_id": ["41010"], "latest_image": ["Z24A_2026_09_26_0110.jpg"]})
    full = len(backfill_candidates(cams, tmp_path, now=now, hours=5))
    p = image_path(tmp_path, "41010", "Z24A_2026_09_26_0010.jpg")
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")  # left empty by a power cut
    assert len(backfill_candidates(cams, tmp_path, now=now, hours=5)) == full
    p.write_bytes(JPEG)
    assert len(backfill_candidates(cams, tmp_path, now=now, hours=5)) == full - 1


def test_webcoos_on_disk_ignores_empty_files_of_unknown_size(tmp_path: Path) -> None:
    from wave_analysis.sources.webcoos import local_path, on_disk

    t = pd.Timestamp("2026-08-08 18:28:36", tz="UTC")
    name = "oakisland_east-2026-08-08-182836Z.jpg"
    sel = pd.DataFrame(
        {"camera": ["oakisland_east"], "name": [name], "time_utc": [t], "size_bytes": [pd.NA]}
    )
    p = local_path(tmp_path, "oakisland_east", name, t)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    assert not on_disk(sel, tmp_path).iloc[0]
    p.write_bytes(JPEG)
    assert on_disk(sel, tmp_path).iloc[0]


# --------------------------------------------------------------------------- offsite copy


def test_decide() -> None:
    a, b = "a" * 64, "b" * 64
    assert decide("x", ("ok", a), ("ok", a))[0] == "none"
    assert decide("x", ("ok", a), ("sha256_mismatch", b))[0] == "push"
    assert decide("x", ("ok", a), ("missing", ""))[0] == "push"
    assert decide("x", ("empty", ""), ("ok", a))[0] == "pull"
    assert decide("x", ("empty", ""), ("empty", ""))[0] == "refetch"
    assert decide("x", ("ok", a), ("ok", b))[0] == "choose"
    assert decide("x", ("no_ledger", ""), ("ok", a))[0] == "unknown"


def test_compare_remote() -> None:
    files = [
        {"path": "a.jpg", "status": "ok", "size": "10", "md5": "m1"},
        {"path": "b.jpg", "status": "ok", "size": "10", "md5": "m2"},
        {"path": "c.jpg", "status": "ok", "size": "10", "md5": "m3"},
        {"path": "d.jpg", "status": "ok", "size": "10", "md5": "m4"},
        {"path": "e.jpg", "status": "partial", "size": "3", "md5": ""},
    ]
    listing = {
        "a.jpg": (10, "m1"),
        "b.jpg": (10, "zz"),  # same size, other bytes: what --size-only cannot see
        "c.jpg": (0, "d41d8cd98f00b204e9800998ecf8427e"),
        "x/_manifests/l.csv": (5, "m"),
        "era5/.era5.lock": (0, ""),
        "skip/y.jpg": (5, "m"),
        "z.jpg": (5, "m"),
        "d.jpg": (10, ""),
    }
    cmp = compare_remote(files, listing, skip_prefixes=["skip/"])
    got = {r["path"]: r["remote"] for r in cmp.rows}
    assert got == {
        "a.jpg": "same",
        "b.jpg": "differs",
        "c.jpg": "differs",
        "d.jpg": "same_size",
        "z.jpg": "remote_only",
    }


def test_default_remote(tmp_path: Path) -> None:
    env = tmp_path / "sync.env"
    env.write_text("WAVE_ANALYSIS_REMOTE=r2:bucket\n")
    assert default_remote({}, env) == "r2:bucket/raw"
    assert default_remote({"WAVE_ANALYSIS_REMOTE": "x:y/"}, env) == "x:y/raw"
    assert default_remote({}, tmp_path / "none") is None


@pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone is not installed")
def test_repair_uses_the_ledger_to_pick_the_right_copy(
    archive: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RCLONE_CONFIG", str(tmp_path / "rclone.conf"))
    remote = tmp_path / "remote"
    shutil.copytree(archive, remote, ignore=shutil.ignore_patterns("_manifests", ".*"))
    # offsite: nul.jpg is good there (pull), ok.jpg was damaged there (push)
    (remote / "cams" / "a" / "nul.jpg").write_bytes(JPEG)
    (remote / "cams" / "a" / "ok.jpg").write_bytes(b"\0" * len(JPEG))
    (remote / "cams" / "a" / "empty.jpg").write_bytes(b"")
    paths = ["cams/a/nul.jpg", "cams/a/ok.jpg", "cams/a/empty.jpg", "cams/a/upstream_empty.jpg"]

    dry = repair(paths, raw_root=archive, remote=str(remote), dry_run=True)
    assert {i.path: i.action for i in dry} == {
        "cams/a/nul.jpg": "pull",
        "cams/a/ok.jpg": "push",
        "cams/a/empty.jpg": "refetch",
        "cams/a/upstream_empty.jpg": "none",
    }
    assert (archive / "cams" / "a" / "nul.jpg").read_bytes() != JPEG  # nothing changed

    def fetch(url: str, dest: Path) -> Path | None:
        assert url == "https://example.org/cams/a/empty.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(JPEG)
        return dest

    log = tmp_path / "log.csv"
    items = repair(
        paths, raw_root=archive, remote=str(remote), refetch=True, fetch=fetch, log_path=log
    )
    outcome = {i.path: i.outcome for i in items}
    assert outcome["cams/a/nul.jpg"] == "pulled"
    assert outcome["cams/a/ok.jpg"] == "uploaded"
    assert outcome["cams/a/empty.jpg"] == "refetched; uploaded"
    for rel in ("nul.jpg", "ok.jpg", "empty.jpg"):
        assert (archive / "cams" / "a" / rel).read_bytes() == JPEG
        assert (remote / "cams" / "a" / rel).read_bytes() == JPEG
    assert len(pd.read_csv(log)) == 4
    after = _statuses(archive, paths=paths)
    assert set(after.values()) <= {"ok", "empty_upstream"}


@pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone is not installed")
def test_repair_reports_a_file_lost_everywhere(archive: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RCLONE_CONFIG", str(tmp_path / "rclone.conf"))
    remote = tmp_path / "remote"
    remote.mkdir()
    (item,) = repair(
        ["cams/a/empty.jpg"],
        raw_root=archive,
        remote=str(remote),
        refetch=True,
        fetch=lambda url, dest: None,
    )
    assert (item.action, item.outcome) == ("lost", "lost")
    assert (archive / "cams" / "a" / "empty.jpg").read_bytes() == b""  # annotate, don't delete


def test_rclone_is_called_with_raw_file_lists(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(cmd):  # type: ignore[no-untyped-def]
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    raw = tmp_path / "raw"
    _put(raw, "a/x.jpg", JPEG)
    write_manifest([_entry(raw, "a/x.jpg", JPEG)], raw / "a" / "_manifests" / "l.csv")
    (item,) = repair(["a/x.jpg"], raw_root=raw, remote="r2:b/raw", run=run, dry_run=True)
    assert item.action == "push"  # nothing came back from the fake remote
    assert calls[0][:4] == ["rclone", "copy", "r2:b/raw", calls[0][3]]
    assert "--files-from-raw" in calls[0]
