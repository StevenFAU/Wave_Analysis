#!/usr/bin/env python
"""Download published research datasets listed in configs/sources/external_datasets.yaml.

    uv run python scripts/download_external.py                 # every dataset with URLs
    uv run python scripts/download_external.py carini_duck2016_ir_lidar
    uv run python scripts/download_external.py --verify        # check what is on disk
    uv run python scripts/download_external.py alberello_2017_agulhas_wass --from-zip ~/Downloads/x.zip

Files go to ``data/external/<id>/`` with a ``SOURCE.yaml`` copy of the
dataset's entry. Each file is checked against the size and, where the
repository publishes one, the MD5 in the config; a file that fails is deleted
and recorded as failed. Every download is a row in
``data/manifests/raw/external.csv`` (URL, SHA-256 of the bytes as served, size,
time). Files already present with the right size and MD5 are skipped, so an
interrupted run resumes.

Some repositories give files only after a form (AADC emails a one-time link).
For those, ``--from-zip`` takes the zip they provide and extracts the members
named in the config, by name only, into the dataset directory, with the same
checks and a ledger row per file.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml

from wave_analysis.config import data_dir, load_yaml, repo_root
from wave_analysis.ingest.checksum import sha256_file
from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, utcnow, write_manifest

CONFIG = repo_root() / "configs" / "sources" / "external_datasets.yaml"
LEDGER = data_dir("manifests") / "raw" / "external.csv"


def md5_file(path: Path) -> str:
    """MD5 of a file, read in 8 MB blocks (the checksum DSpace publishes)."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def check(path: Path, spec: dict[str, Any]) -> str | None:
    """``None`` if ``path`` matches the config's size and MD5, else the reason it does not."""
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size != spec["size_bytes"]:
        return f"size {size:,} != {spec['size_bytes']:,}"
    if spec.get("md5") and md5_file(path) != spec["md5"]:
        return "MD5 differs from the repository's"
    return None


def write_source(dest: Path, ds: dict[str, Any]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SOURCE.yaml").write_text(
        f"# Copied from {CONFIG.relative_to(repo_root())}; do not edit here.\n"
        + yaml.safe_dump(ds, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def download(ds: dict[str, Any], dl: Downloader) -> int:
    """Fetch every file of one dataset that has a URL; return the number of failures."""
    dest = data_dir("external") / ds["id"]
    write_source(dest, ds)
    failures = 0
    for spec in ds["files"]:
        path = dest / spec["name"]
        if not spec.get("url"):
            if check(path, spec):
                print(f"{ds['id']}/{spec['name']}: no URL ({ds.get('access_note', '').strip()})")
            continue
        if check(path, spec) is None:
            print(f"{ds['id']}/{spec['name']}: present and verified")
            continue
        print(
            f"{ds['id']}/{spec['name']}: downloading {spec['size_bytes'] / 1e6:,.1f} MB", flush=True
        )
        entry = dl.fetch(spec["url"], path, source_id=ds["id"], product=spec["name"])
        problem = None if entry.status != EntryStatus.VERIFIED else check(path, spec)
        if entry.status != EntryStatus.VERIFIED or problem:
            failures += 1
            path.unlink(missing_ok=True)
            entry = entry.model_copy(
                update={"status": EntryStatus.FAILED, "note": problem or entry.note}
            )
        elif spec.get("md5"):
            entry = entry.model_copy(update={"note": f"md5 {spec['md5']} matches the repository"})
        write_manifest([entry], LEDGER)
        print(f"  -> {entry.status.value}{': ' + entry.note if entry.note else ''}", flush=True)
    return failures


def from_zip(ds: dict[str, Any], archive: Path) -> int:
    """Extract the configured files of one dataset from a zip the repository provided."""
    dest = data_dir("external") / ds["id"]
    write_source(dest, ds)
    wanted = {spec["name"]: spec for spec in ds["files"]}
    sha = sha256_file(archive)
    failures = 0
    with zipfile.ZipFile(archive) as zf:
        members = {Path(i.filename).name: i for i in zf.infolist() if not i.is_dir()}
        for name, spec in wanted.items():
            info = members.get(name)
            if info is None:
                print(f"{ds['id']}/{name}: not in {archive.name}")
                failures += 1
                continue
            path = dest / name  # by name only: member paths are never used
            with zf.open(info) as src, path.open("wb") as out:
                shutil.copyfileobj(src, out, 8 << 20)
            problem = check(path, spec)
            if problem:
                failures += 1
                path.unlink(missing_ok=True)
            entry = ManifestEntry(
                source_id=ds["id"],
                product=name,
                url=ds["landing"],
                local_path=str(path) if not problem else None,
                sha256=None if problem else sha256_file(path),
                size_bytes=None if problem else spec["size_bytes"],
                retrieved_at=utcnow(),
                status=EntryStatus.FAILED if problem else EntryStatus.VERIFIED,
                note=f"extracted from {archive.name} (sha256 {sha})"
                + (f"; {problem}" if problem else ""),
            )
            write_manifest([entry], LEDGER)
            print(f"{ds['id']}/{name}: {entry.status.value}{'; ' + problem if problem else ''}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids", nargs="*", help="dataset ids (default: all)")
    ap.add_argument("--verify", action="store_true", help="only check the files on disk")
    ap.add_argument("--from-zip", type=Path, help="extract one dataset's files from this zip")
    ap.add_argument("--min-interval", type=float, default=1.0)
    args = ap.parse_args()
    datasets = load_yaml(CONFIG)["datasets"]
    known = {d["id"] for d in datasets}
    unknown = set(args.ids) - known
    if unknown:
        print(f"unknown dataset ids {sorted(unknown)}; known: {sorted(known)}", file=sys.stderr)
        return 2
    chosen = [d for d in datasets if not args.ids or d["id"] in args.ids]
    if args.from_zip:
        if len(chosen) != 1:
            print("--from-zip needs exactly one dataset id", file=sys.stderr)
            return 2
        return 1 if from_zip(chosen[0], args.from_zip) else 0
    if args.verify:
        bad = 0
        for ds in chosen:
            for spec in ds["files"]:
                problem = check(data_dir("external") / ds["id"] / spec["name"], spec)
                bad += problem is not None
                print(f"{ds['id']}/{spec['name']}: {problem or 'ok'}")
        return 1 if bad else 0
    failures = 0
    with Downloader(min_interval_s=args.min_interval, timeout_s=120.0) as dl:
        for ds in chosen:
            failures += download(ds, dl)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
