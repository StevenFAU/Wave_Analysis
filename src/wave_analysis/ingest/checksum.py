"""Streaming SHA-256 checksums for raw and derived artefacts."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of an in-memory byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, gunzip: bool = False) -> str:
    """Hex SHA-256 of a file, streamed in 1 MiB chunks.

    With ``gunzip=True`` the hash is of the decompressed content, matching the
    manifest checksum of payloads stored with ``Downloader.fetch(compress=True)``.
    """
    h = hashlib.sha256()
    if gunzip:
        with gzip.open(path, "rb") as gz:
            while chunk := gz.read(_CHUNK):
                h.update(chunk)
    else:
        with Path(path).open("rb") as fh:
            while chunk := fh.read(_CHUNK):
                h.update(chunk)
    return h.hexdigest()


def verify_file(path: str | Path, expected_sha256: str, *, gunzip: bool = False) -> bool:
    """Return ``True`` if the file's SHA-256 matches ``expected_sha256`` (case-insensitive)."""
    return sha256_file(path, gunzip=gunzip) == expected_sha256.lower()
