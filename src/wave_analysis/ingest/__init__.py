"""Byte-preserving acquisition: downloads, SHA-256 checksums, manifests."""

from wave_analysis.ingest.checksum import sha256_bytes, sha256_file
from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, read_manifest, write_manifest

__all__ = [
    "Downloader",
    "EntryStatus",
    "ManifestEntry",
    "read_manifest",
    "sha256_bytes",
    "sha256_file",
    "write_manifest",
]
