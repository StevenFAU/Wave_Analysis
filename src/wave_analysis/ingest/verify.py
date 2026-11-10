"""Check the raw archive against its ledgers.

Every file the pipeline downloads has a ledger row (a manifest CSV, see
:mod:`wave_analysis.ingest.manifest`) with the SHA-256 and size of the bytes
as served. A file that no longer matches its row was damaged after it was
written, most often by a power cut: ext4 can keep a rename whose data never
reached the disk, and the file then comes back empty or full of NUL bytes with
the right size (incidents of 2026-10-18 and 2026-11-03,
``docs/operations/data_collection.md``). Comparing sizes cannot see the
second kind; only the checksum can.

:func:`verify_archive` reads every ledger, walks ``data/raw`` (following the
links to collections on other drives), hashes each file once and gives it one
status:

``ok``
    Size and SHA-256 match a ledger row for this path.
``empty_upstream``
    0 bytes, and the ledger records that the source served 0 bytes (WebCOOS
    has a few such stills). Kept as evidence; not usable as an image.
``empty`` / ``size_mismatch`` / ``sha256_mismatch`` / ``unreadable``
    Damaged: 0 bytes where the ledger says otherwise, another size, the right
    size but other bytes, or a read error (I/O error, broken gzip stream).
``suspect``
    No ledger row, and the file is empty or ends in NUL bytes.
``no_ledger``
    No ledger row (e.g. the run lists that WebCOOS downloads write under
    ``_listings/``, which are records made here, not files fetched).
``missing``
    A ledger row names the file but it is not on disk.
``partial``
    A temporary file left by an interrupted download.

A file passes if it matches *any* row for its path: two hosts may each have
fetched the same image, and a re-fetch appends a row rather than replacing
one. Snapshots stored gzip-compressed (``Downloader.fetch(compress=True)``)
are compared after decompression, as their ledger rows describe the payload
as served. The MD5 of the stored bytes is recorded too: it is the ETag that
S3-compatible storage reports for single-part uploads, so the offsite copy can
be compared with these results without downloading it.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import os
import stat
import sys
import time
import zlib
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

#: SHA-256 of zero bytes.
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
#: Statuses that mean the file on disk is not the file that was downloaded.
DAMAGED = frozenset({"empty", "size_mismatch", "sha256_mismatch", "unreadable", "suspect"})
#: Statuses that need no action.
GOOD = frozenset({"ok", "empty_upstream"})
#: Directory holding a collection's ledgers.
LEDGER_DIR = "_manifests"
#: Every status, in report order.
STATUSES = (
    "ok",
    "empty_upstream",
    "empty",
    "size_mismatch",
    "sha256_mismatch",
    "unreadable",
    "suspect",
    "no_ledger",
    "missing",
    "partial",
)
#: Columns of the per-file report.
REPORT_FIELDS = (
    "path",
    "status",
    "size",
    "mtime",
    "sha256",
    "md5",
    "content_sha256",
    "ledger_size",
    "ledger_sha256",
    "ledger_versions",
    "ledger",
    "retrieved_at",
    "url",
    "note",
)

_CHUNK = 1 << 20
_NUL_TAIL = 64  # a file ending in this many NUL bytes is suspect
_COMPRESSED_NOTE = "stored gzip-compressed"


@dataclass(frozen=True)
class LedgerRow:
    """The part of a ledger row needed to check its file."""

    size: int
    sha256: str
    retrieved_at: str
    ledger: str
    url: str
    compressed: bool


@dataclass
class LedgerIndex:
    """Successful ledger rows by path relative to the raw root."""

    rows: dict[str, list[LedgerRow]] = field(default_factory=dict)
    ledgers: list[dict[str, object]] = field(default_factory=list)

    def add(self, rel: str, row: LedgerRow) -> None:
        """Record ``row`` for ``rel``: one entry per distinct version, the latest retrieval last."""
        have = self.rows.setdefault(rel, [])
        for i, r in enumerate(have):
            if (r.size, r.sha256, r.compressed) == (row.size, row.sha256, row.compressed):
                if row.retrieved_at > r.retrieved_at:
                    have[i] = row
                break
        else:
            have.append(row)
        if len(have) > 1:
            have.sort(key=lambda r: r.retrieved_at)


@dataclass
class FileResult:
    """One line of the report."""

    path: str
    status: str
    size: int | None = None
    mtime: str = ""
    sha256: str = ""
    md5: str = ""
    content_sha256: str = ""
    ledger_size: int | None = None
    ledger_sha256: str = ""
    ledger_versions: int = 0
    ledger: str = ""
    retrieved_at: str = ""
    url: str = ""
    note: str = ""

    def row(self) -> dict[str, object]:
        """As a CSV row."""
        return {k: ("" if (v := getattr(self, k)) is None else v) for k in REPORT_FIELDS}


@dataclass
class VerifyReport:
    """Outcome of :func:`verify_archive`."""

    root: Path
    results: list[FileResult]
    ledgers: list[dict[str, object]]
    started: str
    seconds: float
    bytes_read: int

    def counts(self) -> dict[str, int]:
        """Number of files per status (every status, zeros included)."""
        out = dict.fromkeys(STATUSES, 0)
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        return out

    def damaged(self) -> list[FileResult]:
        """Files whose bytes are not the bytes that were downloaded."""
        return [r for r in self.results if r.status in DAMAGED]

    def damaged_ledgers(self) -> list[dict[str, object]]:
        """Ledgers containing NUL bytes or rows that cannot be read."""
        return [lg for lg in self.ledgers if lg["nul_bytes"] or lg["bad_rows"]]

    def summary(self) -> str:
        """Multi-line text summary."""
        c = self.counts()
        lines = [
            f"verified {len(self.results):,} paths under {self.root} "
            f"({self.bytes_read / 1e9:,.1f} GB read, {self.seconds / 60:.1f} min)",
            "  " + ", ".join(f"{k} {v:,}" for k, v in c.items() if v),
        ]
        bad = self.damaged_ledgers()
        if bad:
            lines.append(
                f"  {len(bad)} damaged ledger(s): " + ", ".join(str(b["ledger"]) for b in bad)
            )
        by_top: dict[tuple[str, str], int] = {}
        for r in self.results:
            if r.status not in GOOD:
                top = r.path.split("/", 1)[0]
                by_top[(top, r.status)] = by_top.get((top, r.status), 0) + 1
        for (top, status), n in sorted(by_top.items()):
            lines.append(f"  {top}: {status} {n:,}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Ledgers
# --------------------------------------------------------------------------- #


def relative_to_raw(local_path: str, raw_root: Path, repo_root: Path | None = None) -> str | None:
    """Path of a ledger's ``local_path`` relative to the raw root, or ``None`` if outside it.

    Ledger paths are repository-relative (``data/raw/...``) when written inside
    the repository, absolute otherwise (e.g. with ``WAVE_ANALYSIS_DATA``).
    """
    p = Path(local_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        if repo_root is not None:
            candidates.append(repo_root / p)
        parts = PurePosixPath(local_path).parts
        for i in range(len(parts) - 1):
            if parts[i] == "data" and parts[i + 1] == "raw":
                return "/".join(parts[i + 2 :]) or None
    roots = (raw_root, raw_root.resolve())
    for cand in candidates:
        for root in roots:
            try:
                rel = cand.relative_to(root)
            except ValueError:
                continue
            return rel.as_posix()
    return None


def read_ledger(
    path: Path, index: LedgerIndex, raw_root: Path, *, name: str, repo_root: Path | None = None
) -> None:
    """Add the successful rows of one ledger CSV to ``index``.

    NUL bytes (left by a power cut while the ledger was being appended to)
    and rows that cannot be parsed are counted in ``index.ledgers``; the
    readable rows are still used. A last row without its line end is being
    written by a running collector and is skipped.
    """
    data = path.read_bytes()
    nul = data.count(b"\0")
    text = data.replace(b"\0", b"").decode("utf-8", errors="replace")
    info: dict[str, object] = {"ledger": name, "rows": 0, "bad_rows": 0, "nul_bytes": nul}
    lines = text.splitlines()
    if text and not text.endswith("\n") and len(lines) > 1:
        # A collector is appending to this ledger right now: its last row is
        # not finished. It is read at the next check, not counted as damage.
        lines.pop()
        info["note"] = "last row still being written"
    reader = csv.reader(lines)
    header = next(reader, None)
    if header is None:
        index.ledgers.append(info)
        return
    col = {k: i for i, k in enumerate(header)}
    absent = [k for k in ("local_path", "sha256", "size_bytes", "status") if k not in col]
    if absent:
        info["note"] = f"not a ledger (no {', '.join(absent)} column)"
        index.ledgers.append(info)
        return
    i_path, i_sha, i_size = col["local_path"], col["sha256"], col["size_bytes"]
    i_status, i_note = col["status"], col.get("note")
    i_time, i_url = col.get("retrieved_at"), col.get("url")
    rows = bad = 0
    for rec in reader:
        rows += 1
        if len(rec) != len(header):
            bad += 1
            continue
        sha = rec[i_sha]
        if not sha or not rec[i_path] or rec[i_status] in ("not_found", "failed"):
            continue
        try:
            size = int(rec[i_size])
        except ValueError:
            bad += 1
            continue
        rel = relative_to_raw(rec[i_path], raw_root, repo_root)
        if rel is None:
            continue
        note = rec[i_note] if i_note is not None else ""
        index.add(
            rel,
            LedgerRow(
                size=size,
                sha256=sha.lower(),
                retrieved_at=rec[i_time] if i_time is not None else "",
                ledger=name,
                url=rec[i_url] if i_url is not None else "",
                compressed=note.startswith(_COMPRESSED_NOTE),
            ),
        )
    info["rows"], info["bad_rows"] = rows, bad
    index.ledgers.append(info)


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def walk(
    raw_root: Path, ledger_dirs: set[str] | None = None
) -> Iterator[tuple[str, os.stat_result]]:
    """``(relative path, stat)`` of every file under ``raw_root``, following links.

    Hidden files (locks, ``.gitkeep``, the exchange's time stamp) are skipped,
    except the temporary files an interrupted download leaves (``.partial-*``,
    ``*.part``). Ledger folders are skipped (their paths are added to
    ``ledger_dirs``): ledgers are the reference, read separately.
    """
    seen: set[tuple[int, int]] = set()
    for dirpath, dirnames, filenames in os.walk(raw_root, followlinks=True):
        st = os.stat(dirpath)
        key = (st.st_dev, st.st_ino)
        if key in seen:  # a link loop
            dirnames[:] = []
            continue
        seen.add(key)
        rel_dir = os.path.relpath(dirpath, raw_root)
        if ledger_dirs is not None and LEDGER_DIR in dirnames:
            ledger_dirs.add(os.path.normpath(os.path.join(rel_dir, LEDGER_DIR)))
        dirnames[:] = sorted(d for d in dirnames if d != LEDGER_DIR and not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") and not (
                name.startswith(".partial-") or name.endswith(".part")
            ):
                continue
            full = os.path.join(dirpath, name)
            try:
                fst = os.stat(full)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(fst.st_mode):
                continue
            rel = name if rel_dir == "." else f"{rel_dir}/{name}".replace(os.sep, "/")
            yield rel, fst


def ledger_files(raw_root: Path, extra_dirs: Iterable[Path] = ()) -> list[tuple[str, Path]]:
    """``(name, path)`` of every ledger: ``_manifests/*.csv`` under the raw root, and ``extra_dirs``."""
    out: list[tuple[str, Path]] = []
    for dirpath, dirnames, _ in os.walk(raw_root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if os.path.basename(dirpath) == LEDGER_DIR:
            dirnames[:] = []
            for p in sorted(Path(dirpath).glob("*.csv")):
                out.append((p.relative_to(raw_root).as_posix(), p))
            continue
        # Only collection folders hold ledgers near their top; don't descend into image trees.
        depth = len(Path(dirpath).relative_to(raw_root).parts)
        if depth >= 2:
            dirnames[:] = [d for d in dirnames if d == LEDGER_DIR]
    for d in extra_dirs:
        for p in sorted(Path(d).glob("*.csv")):
            out.append((f"{d.name}/{p.name}" if d.name else p.name, p))
    return out


@dataclass
class _Digest:
    size: int
    sha256: str
    md5: str
    content_sha256: str = ""
    content_size: int | None = None
    all_nul: bool = False
    nul_tail: bool = False
    error: str = ""


def hash_file(path: str | Path, *, gunzip: bool = False) -> _Digest:
    """SHA-256 and MD5 of the stored bytes in one read; with ``gunzip`` also of the content."""
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    csha = hashlib.sha256() if gunzip else None
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS) if gunzip else None
    size = csize = 0
    all_nul = True
    tail = b""
    try:
        with open(path, "rb") as fh:
            while chunk := fh.read(_CHUNK):
                size += len(chunk)
                sha.update(chunk)
                md5.update(chunk)
                if all_nul and chunk.count(0) != len(chunk):
                    all_nul = False
                tail = (
                    chunk[-_NUL_TAIL:] if len(chunk) >= _NUL_TAIL else (tail + chunk)[-_NUL_TAIL:]
                )
                if dec is not None and csha is not None:
                    data = chunk
                    while data:
                        out = dec.decompress(data)
                        csha.update(out)
                        csize += len(out)
                        data = b""
                        if dec.eof and dec.unused_data:  # another gzip member follows
                            data = dec.unused_data
                            dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
        if dec is not None and size and not dec.eof:
            raise zlib.error("gzip stream ends early")
    except (OSError, zlib.error) as exc:
        return _Digest(size, "", "", error=f"{type(exc).__name__}: {exc}"[:300])
    return _Digest(
        size=size,
        sha256=sha.hexdigest(),
        md5=md5.hexdigest(),
        content_sha256=csha.hexdigest() if csha is not None else "",
        content_size=csize if gunzip else None,
        all_nul=all_nul and size > 0,
        nul_tail=len(tail) == _NUL_TAIL and tail.count(0) == _NUL_TAIL,
    )


def classify(rel: str, digest: _Digest, rows: list[LedgerRow]) -> tuple[str, str]:
    """``(status, note)`` of a hashed file against its ledger rows."""
    if digest.error:
        return "unreadable", digest.error
    if not rows:
        if digest.size == 0:
            return "suspect", "no ledger row; 0 bytes"
        if digest.all_nul:
            return "suspect", "no ledger row; all NUL bytes"
        if digest.nul_tail:
            return "suspect", "no ledger row; ends in NUL bytes"
        if "/_listings/" in f"/{rel}" or rel.endswith(".request.json"):
            return "no_ledger", "record written here, not downloaded"
        return "no_ledger", ""
    for r in rows:
        size, sha = (
            (digest.content_size, digest.content_sha256)
            if r.compressed
            else (digest.size, digest.sha256)
        )
        if size == r.size and sha == r.sha256:
            if r.size == 0:
                return "empty_upstream", "the source served 0 bytes"
            return "ok", ""
    latest = rows[-1]
    if digest.size == 0:
        return "empty", ""
    note = "all NUL bytes" if digest.all_nul else ("ends in NUL bytes" if digest.nul_tail else "")
    size = digest.content_size if latest.compressed else digest.size
    if all(size != r.size for r in rows):
        return "size_mismatch", note
    return "sha256_mismatch", note


def _check_one(raw_root: Path, rel: str, fst: os.stat_result, index: LedgerIndex) -> FileResult:
    mtime = datetime.fromtimestamp(fst.st_mtime, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    name = rel.rsplit("/", 1)[-1]
    if name.startswith(".partial-") or name.endswith(".part"):
        return FileResult(rel, "partial", size=fst.st_size, mtime=mtime)
    rows = index.rows.get(rel, [])
    digest = hash_file(raw_root / rel, gunzip=any(r.compressed for r in rows))
    status, note = classify(rel, digest, rows)
    latest = rows[-1] if rows else None
    return FileResult(
        rel,
        status,
        size=digest.size if not digest.error else fst.st_size,
        mtime=mtime,
        sha256=digest.sha256,
        md5=digest.md5,
        content_sha256=digest.content_sha256,
        ledger_size=latest.size if latest else None,
        ledger_sha256=latest.sha256 if latest else "",
        ledger_versions=len(rows),
        ledger=latest.ledger if latest else "",
        retrieved_at=latest.retrieved_at if latest else "",
        url=latest.url if latest else "",
        note=note,
    )


def build_index(
    raw_root: Path, *, manifests_dir: Path | None = None, repo_root: Path | None = None
) -> LedgerIndex:
    """Read every ledger that can name a file under ``raw_root``."""
    index = LedgerIndex()
    extra = [manifests_dir] if manifests_dir is not None and manifests_dir.is_dir() else []
    for name, path in ledger_files(raw_root, extra):
        read_ledger(path, index, raw_root, name=name, repo_root=repo_root)
    return index


def verify_archive(
    raw_root: Path,
    *,
    manifests_dir: Path | None = None,
    repo_root: Path | None = None,
    since: datetime | None = None,
    paths: Iterable[str] | None = None,
    collections: Iterable[str] | None = None,
    workers: int = 8,
    progress: Callable[[int, int, int], object] | None = None,
) -> VerifyReport:
    """Check files under ``raw_root`` against every ledger (see the module notes).

    Parameters
    ----------
    since
        Only files modified at or after this time, and ledger rows retrieved
        since then whose file is missing (the check before an exchange).
    paths
        Only these paths (relative to ``raw_root``).
    collections
        Only these top-level folders (e.g. ``webcoos_brt``).
    workers
        Files hashed in parallel (hashing releases the GIL).
    progress
        Called as ``progress(done, total, bytes_read)`` every few thousand files.
    """
    t0 = time.monotonic()
    started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    index = build_index(raw_root, manifests_dir=manifests_dir, repo_root=repo_root)
    tops = set(collections) if collections else None
    cutoff = since.timestamp() if since is not None else None

    def wanted(rel: str) -> bool:
        return tops is None or rel.split("/", 1)[0] in tops

    todo: list[tuple[str, os.stat_result]] = []
    results: list[FileResult] = []
    wanted_paths = (
        list(dict.fromkeys(p.strip().lstrip("/") for p in paths if p.strip()))
        if paths is not None
        else None
    )
    if wanted_paths is not None:
        for rel in wanted_paths:
            try:
                todo.append((rel, os.stat(raw_root / rel)))
            except FileNotFoundError:
                if rel not in index.rows:
                    results.append(FileResult(rel, "missing", note="no ledger row"))
    else:
        seen_ledgers: set[str] = set()
        for rel, fst in walk(raw_root, seen_ledgers):
            if wanted(rel) and (cutoff is None or fst.st_mtime >= cutoff):
                todo.append((rel, fst))
        read = {str(Path(str(lg["ledger"])).parent) for lg in index.ledgers}
        for d in sorted(seen_ledgers - read):
            if any(Path(raw_root, d).glob("*.csv")):
                raise RuntimeError(f"ledger folder {d} was not read; extend ledger_files()")

    bytes_read = 0
    batch = 2000
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for start in range(0, len(todo), batch):
            chunk = todo[start : start + batch]
            for res in pool.map(lambda item: _check_one(raw_root, item[0], item[1], index), chunk):
                results.append(res)
                bytes_read += res.size or 0
            if progress is not None:
                progress(min(start + batch, len(todo)), len(todo), bytes_read)

    on_disk = {r.path for r in results}
    for rel in wanted_paths if wanted_paths is not None else index.rows.keys():
        if rel in on_disk or not wanted(rel) or rel not in index.rows:
            continue
        latest = index.rows[rel][-1]
        if cutoff is not None and _iso_ts(latest.retrieved_at) < cutoff:
            continue
        if wanted_paths is None and (raw_root / rel).exists():
            continue  # outside the --since window, or skipped by the walk
        results.append(
            FileResult(
                rel,
                "missing",
                ledger_size=latest.size,
                ledger_sha256=latest.sha256,
                ledger_versions=len(index.rows[rel]),
                ledger=latest.ledger,
                retrieved_at=latest.retrieved_at,
                url=latest.url,
            )
        )
    results.sort(key=lambda r: r.path)
    return VerifyReport(
        root=raw_root,
        results=results,
        ledgers=index.ledgers,
        started=started,
        seconds=time.monotonic() - t0,
        bytes_read=bytes_read,
    )


def _iso_ts(text: str) -> float:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("inf")  # unknown time: never filtered out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def write_report(report: VerifyReport, out_dir: Path, *, all_files: bool = True) -> dict[str, Path]:
    """Write the report to ``out_dir``.

    * ``problems.csv`` - every path whose status is not ``ok``.
    * ``files.csv.gz`` - every path checked, with SHA-256 and MD5 (with ``all_files``).
    * ``ledgers.csv`` - rows read per ledger, NUL bytes and unreadable rows.
    * ``summary.txt`` - counts.

    Every file is written to a temporary name, synced and renamed.
    """
    from wave_analysis.ingest.durable import atomic_open

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    with atomic_open(out_dir / "problems.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REPORT_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in report.results:
            if r.status != "ok":
                w.writerow(r.row())
    written["problems"] = out_dir / "problems.csv"
    if all_files:
        with (
            atomic_open(out_dir / "files.csv.gz", "wb") as raw,
            gzip.open(raw, "wt", newline="", encoding="utf-8") as fh,
        ):
            w = csv.DictWriter(fh, fieldnames=REPORT_FIELDS, lineterminator="\n")
            w.writeheader()
            for r in report.results:
                w.writerow(r.row())
        written["files"] = out_dir / "files.csv.gz"
    with atomic_open(out_dir / "ledgers.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["ledger", "rows", "bad_rows", "nul_bytes", "note"],
            lineterminator="\n",
            extrasaction="ignore",
        )
        w.writeheader()
        for lg in report.ledgers:
            w.writerow({"note": "", **lg})
    written["ledgers"] = out_dir / "ledgers.csv"
    with atomic_open(out_dir / "summary.txt", "w") as fh:
        fh.write(f"started {report.started}\n{report.summary()}\n")
    written["summary"] = out_dir / "summary.txt"
    return written


def print_progress(done: int, total: int, nbytes: int) -> None:
    """Progress line for :func:`verify_archive` (to stderr)."""
    print(f"  {done:,}/{total:,} files, {nbytes / 1e9:,.1f} GB", file=sys.stderr, flush=True)


__all__ = [
    "DAMAGED",
    "EMPTY_SHA256",
    "GOOD",
    "REPORT_FIELDS",
    "STATUSES",
    "FileResult",
    "LedgerIndex",
    "LedgerRow",
    "VerifyReport",
    "build_index",
    "classify",
    "hash_file",
    "read_ledger",
    "relative_to_raw",
    "verify_archive",
    "walk",
    "write_report",
]
