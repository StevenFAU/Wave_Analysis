"""Resumable archiving of many files listed in an upstream index.

Used by collections that publish one file per record (PacIOOS and WebCOOS
camera images). Each archive keeps a per-request ledger (a manifest CSV) next
to its files. A later run skips any URL whose file exists with the size the
ledger recorded, and any URL the ledger records as 404, so an interrupted run
resumes where it stopped and a deleted or truncated file is fetched again.
"""

from __future__ import annotations

import os
import re
import socket
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, read_manifest

#: Receiver for manifest entries as they are produced.
Sink = Callable[[list[ManifestEntry]], object]


def collector_host() -> str:
    """Name of this collector host as used in ledger file names.

    ``WAVE_ANALYSIS_HOST`` overrides the system host name. The result is
    lower-cased and reduced to letters, digits and ``-``.
    """
    name = os.environ.get("WAVE_ANALYSIS_HOST") or socket.gethostname()
    return re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-") or "unknown"


@dataclass(frozen=True)
class ArchiveItem:
    """One file to archive."""

    url: str
    dest: Path
    period: str
    expected_size: int | None = None


def archived(
    ledger: Path, path_for: Callable[[str], Path], *, product: str = "image"
) -> tuple[set[str], set[str]]:
    """``(archived_urls, not_found_urls)`` from a ledger.

    ``path_for`` maps a URL to its local file. A URL counts as archived only if
    that file exists with the size the ledger recorded.
    """
    done: set[str] = set()
    missing: set[str] = set()
    if not ledger.exists():
        return done, missing
    for e in read_manifest(ledger):
        if e.product != product:
            continue
        if e.status == EntryStatus.NOT_FOUND:
            missing.add(e.url)
        elif e.sha256 is not None and e.size_bytes is not None:
            p = path_for(e.url)
            if p.exists() and p.stat().st_size == e.size_bytes:
                done.add(e.url)
    return done, missing - done


@dataclass
class ArchiveSummary:
    """Outcome of one :func:`archive_files` run."""

    dataset: str
    indexed: int = 0
    already_archived: int = 0
    known_missing: int = 0
    downloaded: int = 0
    not_found: int = 0
    failed: int = 0
    size_mismatch: int = 0
    bytes: int = 0
    seconds: float = 0.0
    failures: list[str] = field(default_factory=list)

    def line(self) -> str:
        """One-line report."""
        return (
            f"{self.dataset}: {self.indexed:,} indexed, {self.already_archived:,} already archived, "
            f"{self.downloaded:,} downloaded ({self.bytes / 1e6:,.1f} MB), "
            f"{self.not_found} not found, {self.failed} failed, "
            f"{self.size_mismatch} size mismatches, {self.seconds / 60:.1f} min"
        )


def archive_files(
    downloader: Downloader,
    items: Iterable[ArchiveItem],
    *,
    ledger: Path,
    path_for: Callable[[str], Path],
    source_id: str,
    collection: str,
    sink: Sink,
    product: str = "image",
    limit: int | None = None,
    progress: Callable[[ArchiveSummary], object] | None = None,
    progress_every: int = 500,
) -> ArchiveSummary:
    """Download the ``items`` that the ledger does not show as archived.

    Every request is passed to ``sink`` as soon as it completes, so an
    interrupted run keeps its ledger. A download whose size differs from
    ``expected_size`` is kept but noted in the ledger and counted in
    ``size_mismatch``. ``collection`` is recorded as the manifest's
    ``station_id`` (a camera or dataset name).
    """
    items = list(items)
    summary = ArchiveSummary(dataset=collection, indexed=len(items))
    done, missing = archived(ledger, path_for, product=product)
    summary.already_archived = sum(it.url in done for it in items)
    summary.known_missing = sum(it.url in missing for it in items)
    todo = [it for it in items if it.url not in done and it.url not in missing]
    if limit is not None:
        todo = todo[:limit]
    t0 = time.monotonic()
    for i, it in enumerate(todo, start=1):
        entry = downloader.fetch(
            it.url,
            it.dest,
            source_id=source_id,
            product=product,
            station_id=collection,
            period=it.period,
        )
        if entry.status == EntryStatus.VERIFIED:
            summary.downloaded += 1
            summary.bytes += entry.size_bytes or 0
            if it.expected_size is not None and entry.size_bytes != it.expected_size:
                summary.size_mismatch += 1
                entry = entry.model_copy(
                    update={"note": f"size differs from index ({it.expected_size} bytes)"}
                )
        elif entry.status == EntryStatus.NOT_FOUND:
            summary.not_found += 1
        else:
            summary.failed += 1
            summary.failures.append(f"{it.url}: {entry.http_status or entry.note}")
        sink([entry])
        summary.seconds = time.monotonic() - t0
        if progress is not None and i % progress_every == 0:
            progress(summary)
    summary.seconds = time.monotonic() - t0
    return summary
