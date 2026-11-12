"""The offsite copy of the raw archive: compare it with the local files, repair differences.

The daily exchange (``scripts/sync_offsite.sh``) copies raw files with
``rclone copy --immutable --size-only``: a file that exists on both sides is
never overwritten. That keeps a bad local file from replacing a good offsite
one, but it also means a damaged copy that reached the offsite store stays
there, and every later exchange stops on it ("immutable file modified").

The ledgers settle which copy is right. :func:`repair` checks the local file
and the offsite object against the ledger's SHA-256 and

* uploads the local file over the offsite object when only the local file is right,
* replaces the local file with the offsite object when only that one is right,
* with ``refetch``, downloads the file again from its source when neither is
  right and keeps it only if it matches the ledger,
* otherwise changes nothing and says why.

Every decision is appended to a repair log. :func:`compare_remote` lists the
offsite store (``rclone lsjson``; S3-compatible stores report the MD5 of a
single-part upload without a download) and compares it with the MD5 that
:mod:`wave_analysis.ingest.verify` recorded for each local file.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from wave_analysis.ingest.durable import fsync_dir, fsync_file
from wave_analysis.ingest.verify import (
    GOOD,
    LedgerIndex,
    build_index,
    classify,
    hash_file,
)

#: Where the exchange keeps its remote settings (``WAVE_ANALYSIS_REMOTE=...``).
SYNC_ENV = Path.home() / ".config" / "wave-analysis" / "sync.env"
#: Columns of the comparison report.
COMPARE_FIELDS = ("path", "local_status", "remote", "size", "remote_size", "md5", "remote_md5")
#: Columns of the repair log.
LOG_FIELDS = (
    "time",
    "host",
    "path",
    "local_status",
    "remote_status",
    "action",
    "sha256",
    "outcome",
    "note",
)

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), capture_output=True, text=True, check=False)


def default_remote(env: dict[str, str] | None = None, path: Path = SYNC_ENV) -> str | None:
    """``$WAVE_ANALYSIS_REMOTE/raw``, reading ``sync.env`` if the variable is not set."""
    env = dict(os.environ) if env is None else env
    remote = env.get("WAVE_ANALYSIS_REMOTE")
    if not remote and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == "WAVE_ANALYSIS_REMOTE":
                remote = value.strip().strip("\"'")
    return f"{remote.rstrip('/')}/raw" if remote else None


# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #


def list_remote(remote: str, *, run: Runner = _run) -> dict[str, tuple[int, str]]:
    """``{path: (size, md5)}`` of every object under ``remote`` (md5 ``""`` if unknown)."""
    cmd = [
        "rclone",
        "lsjson",
        "-R",
        "--files-only",
        "--fast-list",
        "--no-modtime",
        "--no-mimetype",
        "--hash-type",
        "MD5",
        remote,
    ]
    res = run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"rclone lsjson failed ({res.returncode}): {res.stderr.strip()[-500:]}")
    out: dict[str, tuple[int, str]] = {}
    for item in json.loads(res.stdout or "[]"):
        md5 = (item.get("Hashes") or {}).get("md5", "") or ""
        out[item["Path"]] = (int(item["Size"]), md5.lower())
    return out


@dataclass
class Comparison:
    """Local files (from a verify report) against the offsite listing."""

    rows: list[dict[str, object]] = field(default_factory=list)

    def by(self, remote: str) -> list[dict[str, object]]:
        """Rows whose ``remote`` field equals ``remote``."""
        return [r for r in self.rows if r["remote"] == remote]

    def counts(self) -> dict[str, int]:
        """Rows per ``remote`` value."""
        out: dict[str, int] = {}
        for r in self.rows:
            key = str(r["remote"])
            out[key] = out.get(key, 0) + 1
        return out


def compare_remote(
    files: Iterable[dict[str, str]],
    listing: dict[str, tuple[int, str]],
    *,
    skip_prefixes: Sequence[str] = (),
) -> Comparison:
    """Compare verify-report rows (``files.csv.gz``) with an offsite listing.

    ``remote`` is ``same`` (size and MD5 equal), ``same_size`` (size equal,
    no MD5 offsite), ``differs`` (size or MD5 differ), ``absent`` (not
    offsite) or ``remote_only`` (offsite, not here). Temporary files and the
    paths under ``skip_prefixes`` (collections this host does not pull) are
    left out.
    """
    cmp = Comparison()
    local: set[str] = set()
    for f in files:
        path = f["path"]
        if f["status"] in ("partial", "missing"):
            continue
        local.add(path)
        size = int(f["size"]) if f["size"] != "" else -1
        row: dict[str, object] = {
            "path": path,
            "local_status": f["status"],
            "size": size,
            "md5": f["md5"],
            "remote_size": "",
            "remote_md5": "",
        }
        if path not in listing:
            row["remote"] = "absent"
        else:
            rsize, rmd5 = listing[path]
            row["remote_size"], row["remote_md5"] = rsize, rmd5
            if rsize != size or (rmd5 and f["md5"] and rmd5 != f["md5"]):
                row["remote"] = "differs"
            elif not rmd5:
                row["remote"] = "same_size"
            else:
                row["remote"] = "same"
        cmp.rows.append(row)
    for path, (rsize, rmd5) in listing.items():
        if path in local or "/_manifests/" in f"/{path}" or path.rsplit("/", 1)[-1].startswith("."):
            continue  # ledgers are compared by the exchange; hidden files are locks and stamps
        if any(path.startswith(p) for p in skip_prefixes):
            continue
        cmp.rows.append(
            {
                "path": path,
                "local_status": "",
                "remote": "remote_only",
                "size": "",
                "md5": "",
                "remote_size": rsize,
                "remote_md5": rmd5,
            }
        )
    return cmp


def read_files_report(path: Path) -> Iterable[dict[str, str]]:
    """Rows of a verify report's ``files.csv.gz``."""
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def write_comparison(cmp: Comparison, out_dir: Path) -> dict[str, Path]:
    """Write ``remote_compare.csv`` (every row not ``same``) and ``repair_candidates.txt``.

    The candidates are the paths that differ offsite: :func:`repair` settles
    each of them with the ledger.
    """
    from wave_analysis.ingest.durable import atomic_open

    out_dir.mkdir(parents=True, exist_ok=True)
    with atomic_open(out_dir / "remote_compare.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COMPARE_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in cmp.rows:
            if r["remote"] != "same":
                w.writerow(r)
    with atomic_open(out_dir / "repair_candidates.txt", "w") as fh:
        for r in cmp.rows:
            if r["remote"] == "differs":
                fh.write(f"{r['path']}\n")
    return {
        "compare": out_dir / "remote_compare.csv",
        "candidates": out_dir / "repair_candidates.txt",
    }


# --------------------------------------------------------------------------- #
# Repair
# --------------------------------------------------------------------------- #


@dataclass
class RepairItem:
    """The decision for one path."""

    path: str
    local_status: str
    remote_status: str
    action: str  # none | push | pull | refetch | choose | unknown | lost
    sha256: str = ""
    outcome: str = ""
    note: str = ""


def _status(rel: str, file: Path, index: LedgerIndex) -> tuple[str, str, str]:
    """``(status, sha256 of the content, md5)`` of a file against the ledger rows for ``rel``."""
    if not file.exists():
        return "missing", "", ""
    rows = index.rows.get(rel, [])
    digest = hash_file(file, gunzip=any(r.compressed for r in rows))
    status, _ = classify(rel, digest, rows)
    content = digest.content_sha256 if any(r.compressed for r in rows) else digest.sha256
    return status, content, digest.md5


def decide(
    rel: str, local: tuple[str, str], remote: tuple[str, str], *, latest_sha256: str = ""
) -> tuple[str, str]:
    """``(action, note)`` from the local and offsite ``(status, sha256)``.

    When both copies match a ledger row but differ, the file was retrieved
    twice with different bytes: an upstream revision (an ERA5 month that was
    preliminary and is now final, a CDIP file reprocessed) or two hosts that
    fetched it at different times. The copy matching the newest retrieval
    (``latest_sha256``, the latest ledger row for the path) wins, as it does
    on the host that made it; the older version stays described by its ledger
    row.
    """
    lstat, lsha = local
    rstat, rsha = remote
    lgood, rgood = lstat in GOOD, rstat in GOOD
    if lstat == "no_ledger" or rstat == "no_ledger":
        return "unknown", "no ledger row names this path: compare the copies by hand"
    if lgood and rgood:
        if lsha == rsha:
            return "none", "both copies match the ledger"
        if latest_sha256 and lsha == latest_sha256:
            return "push", "both copies match a ledger row; the local one is the newest retrieval"
        if latest_sha256 and rsha == latest_sha256:
            return "pull", "both copies match a ledger row; the offsite one is the newest retrieval"
        return "choose", "both copies match a ledger row but differ (two retrievals)"
    if lgood:
        return "push", "the local copy matches the ledger"
    if rgood:
        return "pull", "the offsite copy matches the ledger"
    return "refetch", "neither copy matches the ledger"


def _install(src: Path, dest: Path) -> None:
    """Copy ``src`` over ``dest`` durably (temporary file in ``dest``'s folder, fsync, rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dest.parent, prefix=".partial-")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as out, src.open("rb") as inp:
            shutil.copyfileobj(inp, out, 1 << 20)
            fsync_file(out)
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_dir(dest.parent)


def repair(
    paths: Iterable[str],
    *,
    raw_root: Path,
    remote: str,
    manifests_dir: Path | None = None,
    repo_root: Path | None = None,
    dry_run: bool = False,
    refetch: bool = False,
    run: Runner = _run,
    fetch: Callable[[str, Path], Path | None] | None = None,
    log_path: Path | None = None,
    host: str = "",
    scratch: Path | None = None,
) -> list[RepairItem]:
    """Settle each path whose local and offsite copies may differ (see the module notes).

    ``fetch(url, dest)`` downloads a URL to ``dest`` (used with ``refetch``)
    and returns ``dest``, or ``None`` if the source no longer has it.
    """
    rels = list(dict.fromkeys(p.strip().lstrip("/") for p in paths if p.strip()))
    index = build_index(raw_root, manifests_dir=manifests_dir, repo_root=repo_root)
    items: list[RepairItem] = []
    with tempfile.TemporaryDirectory(prefix="wave-repair-", dir=scratch) as tmpdir:
        tmp = Path(tmpdir)
        listfile = tmp / "paths.txt"
        listfile.write_text("".join(f"{r}\n" for r in rels), encoding="utf-8")
        got = tmp / "remote"
        res = run(
            [
                "rclone",
                "copy",
                remote,
                str(got),
                "--files-from-raw",
                str(listfile),
                "--no-traverse",
                "--retries",
                "2",
            ]
        )
        if res.returncode not in (0, 3):  # 3: some listed files are not offsite
            missing_msgs = "not found" in res.stderr or "doesn't exist" in res.stderr
            if not missing_msgs:
                raise RuntimeError(f"rclone copy failed ({res.returncode}): {res.stderr[-500:]}")
        push: list[RepairItem] = []
        for rel in rels:
            lstat, lsha, _ = _status(rel, raw_root / rel, index)
            rstat, rsha, _ = _status(rel, got / rel, index)
            rows = index.rows.get(rel, [])
            latest = rows[-1].sha256 if rows else ""
            action, note = decide(rel, (lstat, lsha), (rstat, rsha), latest_sha256=latest)
            item = RepairItem(rel, lstat, rstat, action, note=note)
            item.sha256 = lsha if action == "push" else (rsha if action == "pull" else "")
            if action == "refetch":
                if not refetch or fetch is None or not rows:
                    item.outcome = "skipped" if rows else "lost"
                    if not rows:
                        item.action = "lost"
                elif not dry_run:
                    got_new = fetch(rows[-1].url, tmp / "refetch" / rel)
                    nstat, nsha, _ = (
                        _status(rel, got_new, index) if got_new else ("missing", "", "")
                    )
                    if nstat in GOOD and got_new is not None:
                        _install(got_new, raw_root / rel)
                        item.sha256 = nsha
                        item.outcome = "refetched"
                        item.action = "push"  # and upload it below
                    else:
                        item.action = "lost"
                        item.outcome = "lost"
                        item.note = f"the source no longer serves these bytes ({nstat})"
            if dry_run and item.action in ("push", "pull", "refetch"):
                item.outcome = "dry run"
            elif item.action == "pull":
                _install(got / rel, raw_root / rel)
                after, _, _ = _status(rel, raw_root / rel, index)
                item.outcome = "pulled" if after in GOOD else f"pull failed ({after})"
            elif item.action == "push":
                push.append(item)
            elif not item.outcome:
                item.outcome = "unchanged"
            items.append(item)
        if push and not dry_run:
            plist = tmp / "push.txt"
            plist.write_text("".join(f"{i.path}\n" for i in push), encoding="utf-8")
            res = run(
                [
                    "rclone",
                    "copy",
                    str(raw_root),
                    remote,
                    "--files-from-raw",
                    str(plist),
                    "--ignore-times",
                    "--copy-links",
                    "--no-traverse",
                ]
            )
            if res.returncode != 0:
                for i in push:
                    i.outcome = f"upload failed ({res.returncode})"
            else:
                check = tmp / "check"
                run(
                    [
                        "rclone",
                        "copy",
                        remote,
                        str(check),
                        "--files-from-raw",
                        str(plist),
                        "--no-traverse",
                    ]
                )
                for i in push:
                    after, sha, _ = _status(i.path, check / i.path, index)
                    ok = after in GOOD and sha == i.sha256
                    prefix = "refetched; " if i.outcome == "refetched" else ""
                    i.outcome = prefix + ("uploaded" if ok else f"upload not confirmed ({after})")
    if log_path is not None and not dry_run:
        write_log(items, log_path, host=host)
    return items


def write_log(items: Iterable[RepairItem], path: Path, *, host: str = "") -> None:
    """Append decisions to the repair log (a CSV, synced to disk)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        for i in items:
            w.writerow(
                {
                    "time": now,
                    "host": host,
                    "path": i.path,
                    "local_status": i.local_status,
                    "remote_status": i.remote_status,
                    "action": i.action,
                    "sha256": i.sha256,
                    "outcome": i.outcome,
                    "note": i.note,
                }
            )
        fsync_file(fh)


__all__ = [
    "Comparison",
    "RepairItem",
    "compare_remote",
    "decide",
    "default_remote",
    "list_remote",
    "read_files_report",
    "repair",
    "write_comparison",
    "write_log",
]
