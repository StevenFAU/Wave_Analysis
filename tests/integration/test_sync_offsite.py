"""The offsite exchange script, run against a local folder instead of object storage."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "sync_offsite.sh"

pytestmark = pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone is not installed")


def _exchange(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    scratch = tmp_path / "tmp"
    scratch.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "WAVE_ANALYSIS_DATA": str(tmp_path / "data"),
        "WAVE_ANALYSIS_REMOTE": str(tmp_path / "remote"),
        # Not the user's rclone configuration: a local folder needs none.
        "RCLONE_CONFIG": str(tmp_path / "rclone.conf"),
        "TMPDIR": str(scratch),
    }
    env.pop("WAVE_ANALYSIS_SYNC_PULL_SKIP", None)
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=300, check=False
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
