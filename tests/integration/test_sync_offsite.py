"""The offsite exchange script, run against a local folder instead of object storage."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "sync_offsite.sh"

pytestmark = pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone is not installed")


def _exchange(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    scratch = tmp_path / "tmp"
    scratch.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "WAVE_ANALYSIS_DATA": str(tmp_path / "data"),
        "WAVE_ANALYSIS_REMOTE": str(tmp_path / "remote"),
        # Not the user's rclone configuration: a local folder needs none.
        "RCLONE_CONFIG": str(tmp_path / "rclone.conf"),
        "TMPDIR": str(scratch),
        "WAVE_ANALYSIS_PYTHON": sys.executable,
    }
    env.pop("WAVE_ANALYSIS_SYNC_PULL_SKIP", None)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_exchange_succeeds_while_a_ledger_is_being_written(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    ledger = raw / "cams" / "_manifests" / "cam.host.csv"
    ledger.parent.mkdir(parents=True)
    row = "x" * 199 + "\n"
    start = row * 200_000  # 40 MB: long enough to copy that rows arrive meanwhile
    ledger.write_text(start)
    (raw / "cams" / "image.jpg").write_bytes(b"image")

    stop = threading.Event()

    def collector() -> None:
        while not stop.wait(0.001):
            with ledger.open("a") as fh:
                fh.write(row)

    writer = threading.Thread(target=collector)
    writer.start()
    try:
        run = _exchange(tmp_path)
    finally:
        stop.set()
        writer.join()

    assert run.returncode == 0, run.stderr
    remote = tmp_path / "remote" / "raw"
    assert (remote / "cams" / "image.jpg").read_bytes() == b"image"
    copy = (remote / "cams" / "_manifests" / "cam.host.csv").read_text()
    local = ledger.read_text()
    assert len(local) > len(start)  # rows did arrive during the exchange
    assert len(copy) >= len(start)
    assert local.startswith(copy)
    assert (raw / ".offsite_last_sync").exists()
    assert not list((tmp_path / "tmp").iterdir())  # the copy of the ledgers is removed

    # Once the collector has stopped, the next exchange uploads the rest.
    assert _exchange(tmp_path).returncode == 0
    assert (remote / "cams" / "_manifests" / "cam.host.csv").read_text() == local


JPEG = b"\xff\xd8\xff\xe0" + bytes(range(256)) * 4 + b"\xff\xd9"


def _ledger(raw: Path, rows: dict[str, bytes]) -> None:
    """A ledger naming each file with the SHA-256 of the bytes that were downloaded."""
    from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, write_manifest

    write_manifest(
        [
            ManifestEntry(
                source_id="t",
                product="image",
                url=f"https://example.org/{rel}",
                local_path=str(raw / rel),
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                retrieved_at=datetime.now(UTC),
                status=EntryStatus.VERIFIED,
            )
            for rel, data in rows.items()
        ],
        raw / "cams" / "_manifests" / "cam.host.csv",
    )


def test_a_damaged_file_is_held_back_and_the_rest_still_exchanged(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    good, damaged = "cams/a/good.jpg", "cams/a/nul.jpg"
    for rel in (good, damaged):
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
    (raw / good).write_bytes(JPEG)
    (raw / damaged).write_bytes(b"\0" * len(JPEG))  # the right size: invisible to --size-only
    _ledger(raw, {good: JPEG, damaged: JPEG})
    other = tmp_path / "remote" / "raw" / "cams" / "b" / "from_other_host.jpg"
    other.parent.mkdir(parents=True)
    other.write_bytes(JPEG)

    run = _exchange(tmp_path)

    remote = tmp_path / "remote" / "raw"
    assert run.returncode == 1
    assert "DAMAGED sha256_mismatch: cams/a/nul.jpg" in run.stdout
    assert "held back" in run.stderr
    assert (remote / good).exists() and not (remote / damaged).exists()
    assert (remote / "cams" / "_manifests" / "cam.host.csv").exists()
    assert (raw / "cams" / "b" / "from_other_host.jpg").read_bytes() == JPEG  # pulled
    assert not (raw / ".offsite_last_sync").exists()


def test_a_conflict_is_listed_with_the_command_that_settles_it(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    rel = "cams/a/x.jpg"
    (raw / rel).parent.mkdir(parents=True)
    (raw / rel).write_bytes(JPEG)
    _ledger(raw, {rel: JPEG})
    stale = tmp_path / "remote" / "raw" / rel
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"")  # an empty copy reached the remote after a power cut

    run = _exchange(tmp_path)

    conflicts = tmp_path / "data" / "interim" / "offsite" / "conflicts.txt"
    assert run.returncode == 1
    assert conflicts.read_text().splitlines() == [rel]
    assert "archive repair --from" in run.stderr
    assert (tmp_path / "remote" / "raw" / "cams" / "_manifests" / "cam.host.csv").exists()

    env = {**os.environ, "WAVE_ANALYSIS_DATA": str(tmp_path / "data")}
    env["RCLONE_CONFIG"] = str(tmp_path / "rclone.conf")
    fix = subprocess.run(
        [sys.executable, "-m", "wave_analysis.cli", "archive", "repair", "--from", str(conflicts),
         "--remote", str(tmp_path / "remote" / "raw")],
        env=env, capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    assert fix.returncode == 0, fix.stdout + fix.stderr
    assert "push" in fix.stdout and "uploaded" in fix.stdout
    assert stale.read_bytes() == JPEG
    assert _exchange(tmp_path).returncode == 0
    assert (raw / ".offsite_last_sync").exists()


def test_a_collection_whose_drive_is_not_mounted_stops_the_exchange(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    (raw / "cams").mkdir(parents=True)
    (raw / "cams" / "x.jpg").write_bytes(JPEG)
    (raw / "webcoos").symlink_to(tmp_path / "unmounted_drive" / "webcoos")
    remote = tmp_path / "remote" / "raw" / "webcoos" / "a.jpg"
    remote.parent.mkdir(parents=True)
    remote.write_bytes(JPEG)

    run = _exchange(tmp_path)

    assert run.returncode == 2
    assert "is its drive mounted" in run.stderr
    assert not (tmp_path / "remote" / "raw" / "cams").exists()  # nothing copied
    assert (raw / "webcoos").is_symlink() and not (tmp_path / "unmounted_drive").exists()
