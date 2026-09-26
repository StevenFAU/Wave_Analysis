"""Downloader retries/atomicity, checksums and manifest round-trips."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from wave_analysis.ingest.checksum import sha256_bytes, sha256_file, verify_file
from wave_analysis.ingest.downloader import USER_AGENT, Downloader
from wave_analysis.ingest.manifest import (
    EntryStatus,
    ManifestEntry,
    latest_by_url,
    read_manifest,
    utcnow,
    write_manifest,
)


def test_sha256_known_vector(tmp_path: Path):
    assert (
        sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert sha256_file(p) == sha256_bytes(b"abc")
    assert verify_file(p, sha256_bytes(b"abc").upper())


def test_retry_then_success(tmp_path: Path):
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert req.headers["user-agent"] == USER_AGENT
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, content=b"data")

    dl = Downloader(min_interval_s=0, backoff_s=0, transport=httpx.MockTransport(handler))
    e = dl.fetch("https://x/y", tmp_path / "y", source_id="t", product="p")
    assert e.status == EntryStatus.VERIFIED and calls["n"] == 3
    assert not list(tmp_path.glob(".partial-*"))


def test_persistent_failure_is_recorded_not_raised(tmp_path: Path):
    dl = Downloader(
        min_interval_s=0,
        backoff_s=0,
        max_retries=1,
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
    )
    e = dl.fetch("https://x/y", tmp_path / "y", source_id="t", product="p")
    assert e.status == EntryStatus.FAILED
    assert not (tmp_path / "y").exists()


def test_change_detection(tmp_path: Path):
    dl = Downloader(
        min_interval_s=0,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"v2")),
    )
    same = dl.fetch(
        "https://x/y",
        tmp_path / "y",
        source_id="t",
        product="p",
        previous_sha256=sha256_bytes(b"v2"),
    )
    diff = dl.fetch(
        "https://x/y",
        tmp_path / "y",
        source_id="t",
        product="p",
        previous_sha256=sha256_bytes(b"v1"),
    )
    assert same.status == EntryStatus.UNCHANGED
    assert diff.status == EntryStatus.CHANGED


def test_compressed_storage_keeps_payload_checksum(tmp_path: Path):
    import gzip

    payload = b"#YY  MM DD hh mm\n" * 200
    dl = Downloader(
        min_interval_s=0,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=payload)),
    )
    e = dl.fetch(
        "https://x/41010.spec", tmp_path / "s.gz", source_id="t", product="p", compress=True
    )
    stored = (tmp_path / "s.gz").read_bytes()
    assert gzip.decompress(stored) == payload and len(stored) < len(payload)
    assert e.sha256 == sha256_bytes(payload) and e.size_bytes == len(payload)
    assert verify_file(tmp_path / "s.gz", e.sha256, gunzip=True)
    assert not verify_file(tmp_path / "s.gz", e.sha256)
    assert "gzip" in (e.note or "")


def test_no_overwrite(tmp_path: Path):
    (tmp_path / "y").write_bytes(b"old")
    dl = Downloader(
        min_interval_s=0,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"new")),
    )
    with pytest.raises(FileExistsError):
        dl.fetch("https://x/y", tmp_path / "y", source_id="t", product="p", overwrite=False)


def test_manifest_roundtrip_and_latest(tmp_path: Path):
    m = tmp_path / "m.csv"
    e1 = ManifestEntry(
        source_id="ndbc",
        product="stdmet",
        url="u",
        sha256="a" * 64,
        retrieved_at=utcnow(),
        status=EntryStatus.VERIFIED,
    )
    e2 = e1.model_copy(update={"sha256": "b" * 64, "retrieved_at": utcnow()})
    e3 = ManifestEntry(
        source_id="ndbc",
        product="stdmet",
        url="v",
        retrieved_at=utcnow(),
        status=EntryStatus.NOT_FOUND,
        http_status=404,
    )
    write_manifest([e1], m)
    write_manifest([e2, e3], m)  # appends without a second header
    back = list(read_manifest(m))
    assert len(back) == 3 and back[2].http_status == 404
    assert latest_by_url(m)["u"].sha256 == "b" * 64
    assert "v" not in latest_by_url(m)
    assert m.read_text().count("source_id,") == 1


class _DroppedStream(httpx.SyncByteStream):
    """A body that breaks off after the first chunk."""

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield b"partial"
        raise httpx.ReadError("connection dropped")


def test_body_larger_than_a_chunk_is_streamed_and_hashed(tmp_path: Path):
    payload = bytes(range(256)) * 20_000  # ~5 MB, several 1 MB chunks
    dl = Downloader(
        min_interval_s=0,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=payload)),
    )
    e = dl.fetch("https://x/big", tmp_path / "big", source_id="t", product="p")
    assert e.size_bytes == len(payload) and e.sha256 == sha256_bytes(payload)
    assert (tmp_path / "big").read_bytes() == payload


def test_connection_dropped_mid_body_is_retried_without_leftovers(tmp_path: Path):
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, stream=_DroppedStream())
        return httpx.Response(200, content=b"complete")

    dl = Downloader(min_interval_s=0, backoff_s=0, transport=httpx.MockTransport(handler))
    e = dl.fetch("https://x/y", tmp_path / "y", source_id="t", product="p")
    assert e.status == EntryStatus.VERIFIED and calls["n"] == 2
    assert (tmp_path / "y").read_bytes() == b"complete"
    assert not list(tmp_path.glob(".partial-*"))


def test_body_dropped_on_every_attempt_is_a_recorded_failure(tmp_path: Path):
    dl = Downloader(
        min_interval_s=0,
        backoff_s=0,
        max_retries=1,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=_DroppedStream())),
    )
    e = dl.fetch("https://x/y", tmp_path / "y", source_id="t", product="p")
    assert e.status == EntryStatus.FAILED and "connection dropped" in (e.note or "")
    assert not (tmp_path / "y").exists() and not list(tmp_path.glob(".partial-*"))
