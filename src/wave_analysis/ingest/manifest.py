"""Acquisition manifests: the ledger linking every raw file to its upstream source.

Raw data are not committed to Git (``data/raw/`` is ignored). Instead, every
file fetched from upstream is recorded here with its URL, SHA-256, size,
HTTP metadata, and retrieval time. A manifest is sufficient to (a) re-download
the same inputs, (b) detect silent upstream changes (checksum drift), and
(c) cite exactly which bytes an experiment used.

Manifests are stored as CSV (human-diffable, committed to Git under
``data/manifests/raw/``). Each row is a :class:`ManifestEntry`.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from wave_analysis.config import portable_path


class EntryStatus(StrEnum):
    """Lifecycle state of a manifest entry."""

    VERIFIED = "verified"  # downloaded and checksum recorded
    UNCHANGED = "unchanged"  # re-fetched; identical to previous checksum
    CHANGED = "changed"  # re-fetched; upstream bytes differ from previous checksum
    NOT_FOUND = "not_found"  # HTTP 404 - recorded so absence is itself documented
    FAILED = "failed"  # network or server error


class ManifestEntry(BaseModel):
    """One upstream file retrieval."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    product: str
    station_id: str | None = None
    period: str | None = Field(default=None, description="e.g. '2023', '2024-05', 'realtime'")
    url: str
    local_path: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = Field(default=None, ge=0)
    http_status: int | None = None
    http_last_modified: str | None = None
    http_etag: str | None = None
    retrieved_at: AwareDatetime
    status: EntryStatus
    software_version: str | None = None
    note: str | None = None


FIELDS: tuple[str, ...] = tuple(ManifestEntry.model_fields)


def write_manifest(
    entries: Iterable[ManifestEntry], path: str | Path, *, append: bool = True
) -> Path:
    """Write entries to a CSV manifest, creating it (with header) if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists() and p.stat().st_size > 0
    mode = "a" if append and exists else "w"
    with p.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        if mode == "w":
            writer.writeheader()
        for e in entries:
            row = e.model_dump(mode="json")
            if row.get("local_path"):
                row["local_path"] = portable_path(row["local_path"])
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return p


def read_manifest(path: str | Path) -> Iterator[ManifestEntry]:
    """Iterate over entries of a CSV manifest."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            yield ManifestEntry.model_validate(
                {k: (v if v != "" else None) for k, v in row.items()}
            )


def latest_by_url(path: str | Path) -> dict[str, ManifestEntry]:
    """Most recent successful entry per URL (for change detection on re-fetch)."""
    latest: dict[str, ManifestEntry] = {}
    p = Path(path)
    if not p.exists():
        return latest
    for e in read_manifest(p):
        if e.sha256 is None:
            continue
        prev = latest.get(e.url)
        if prev is None or e.retrieved_at >= prev.retrieved_at:
            latest[e.url] = e
    return latest


def utcnow() -> datetime:
    """Timezone-aware current UTC time (single point of truth for timestamps)."""
    return datetime.now(UTC)
