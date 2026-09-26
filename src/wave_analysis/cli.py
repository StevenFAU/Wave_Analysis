"""Command-line interface: ``wave-analysis <command> ...``.

Commands
--------
``ndbc inventory``
    Build the NDBC station/product/year availability inventory and write it to
    ``data/registry/`` (Parquet + CSV summaries) with raw index snapshots and a
    manifest.
``ndbc download``
    Download historical years and/or realtime files for stations/products,
    recording every attempt in ``data/manifests/raw/ndbc.csv``.
``ndbc standardize``
    Parse downloaded files and write standardized observation / spectral
    Parquet tables with provenance.
``ndbc cameras``
    Archive any new NDBC buoy-camera images and backfill the last ~70 h (run
    hourly to build an image archive that NDBC does not itself publish).
``registry validate``
    Validate ``data/registry/datasets.yaml`` against the registry schema.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from wave_analysis import __version__
from wave_analysis.config import data_dir, load_yaml, repo_root


def _cmd_ndbc_inventory(args: argparse.Namespace) -> int:
    from wave_analysis.datasets.provenance import build_provenance, write_provenance
    from wave_analysis.ingest.downloader import Downloader
    from wave_analysis.ingest.manifest import write_manifest
    from wave_analysis.sources.ndbc.inventory import run_inventory

    root = repo_root()
    snap = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    raw = data_dir("raw") / "ndbc" / "index" / snap
    out = Path(args.out) if args.out else root / "data" / "registry"
    out.mkdir(parents=True, exist_ok=True)
    with Downloader(min_interval_s=args.min_interval) as dl:
        res = run_inventory(dl, raw, historical_dirs=args.dirs or None)
    write_manifest(res.manifest, data_dir("manifests") / "raw" / "ndbc_index.csv")
    files_p = out / "ndbc_files.parquet"
    stations_p = out / "stations.parquet"
    summary_p = out / "ndbc_station_summary.csv"
    cams_p = out / f"ndbc_cameras_{snap}.csv"
    res.files.to_parquet(files_p, index=False)
    res.stations.to_parquet(stations_p, index=False)
    summary = res.station_summary()
    summary.to_csv(summary_p)
    res.cameras.to_csv(cams_p, index=False)
    prov = build_provenance(
        artifact_id=f"ndbc_inventory_{snap}",
        inputs=[e.local_path for e in res.manifest if e.local_path],
        outputs=[files_p, stations_p, summary_p, cams_p],
        config={"historical_dirs": args.dirs, "min_interval_s": args.min_interval},
    )
    write_provenance(prov, out / "ndbc_inventory.provenance.yaml")
    n_st = summary.shape[0]
    print(f"Inventory snapshot {res.snapshot_utc}: {len(res.files)} files, {n_st} stations")
    print(f"  stations with any swden year:            {(summary['swden_years'] > 0).sum()}")
    print(
        f"  stations with full directional spectra:  {(summary['directional_spectrum_years'] > 0).sum()}"
    )
    print(
        f"  stations with ADCP (adcp or adcp2):      {((summary['adcp_years'] + summary['adcp2_years']) > 0).sum()}"
    )
    print(f"  stations with a camera (snapshot):       {int(summary['has_camera_now'].sum())}")
    print(f"Wrote {files_p}, {stations_p}, {summary_p}, {cams_p}")
    return 0


def _parse_years(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    years: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return years


def _camera_station_ids(min_interval: float) -> list[str]:
    """Stations that currently list a buoy-camera image (fetches the listing)."""
    from wave_analysis.ingest.downloader import Downloader
    from wave_analysis.sources.ndbc.inventory import parse_buoycams
    from wave_analysis.sources.ndbc.products import BUOYCAM_URL

    with Downloader(min_interval_s=min_interval) as dl:
        resp = dl.get(BUOYCAM_URL)
        resp.raise_for_status()
    cams = parse_buoycams(resp.text, pd.Timestamp.now(tz="UTC"))
    return sorted(cams.loc[cams["latest_image"] != "", "station_id"].astype(str))


def _cmd_ndbc_download(args: argparse.Namespace) -> int:
    from wave_analysis.ingest.downloader import Downloader
    from wave_analysis.ingest.manifest import (
        EntryStatus,
        ManifestEntry,
        latest_by_url,
        write_manifest,
    )
    from wave_analysis.sources.ndbc.source import NDBCSource

    stations = list(args.stations)
    if args.camera_stations:
        stations += _camera_station_ids(args.min_interval)
    if not stations:
        print("no stations given (list them or pass --camera-stations)", file=sys.stderr)
        return 2
    manifest_path = (
        Path(args.manifest) if args.manifest else data_dir("manifests") / "raw" / "ndbc.csv"
    )
    previous = (
        {u: e.sha256 for u, e in latest_by_url(manifest_path).items() if e.sha256}
        if manifest_path.exists()
        else {}
    )
    dest = data_dir("raw") / "ndbc"
    with Downloader(min_interval_s=args.min_interval) as dl:
        src = NDBCSource(dl)
        entries: list[ManifestEntry] = []
        for st in dict.fromkeys(stations):
            entries += src.fetch(
                st,
                args.products,
                dest_root=dest,
                years=_parse_years(args.years),
                realtime=args.realtime,
                previous=previous,
                compress_realtime=args.gzip,
            )
    write_manifest(entries, manifest_path)
    counts = pd.Series([e.status.value for e in entries]).value_counts().to_dict()
    print(f"{len(entries)} requests: {counts}")
    failed = [e for e in entries if e.status == EntryStatus.FAILED]
    return 1 if failed else 0


def _cmd_ndbc_standardize(args: argparse.Namespace) -> int:
    from wave_analysis.datasets.provenance import build_provenance, write_provenance
    from wave_analysis.ingest.checksum import sha256_file
    from wave_analysis.sources.ndbc import SPECTRAL_SET, NDBCSource, assemble_spectrum
    from wave_analysis.sources.ndbc.inventory import HISTORICAL_FILENAME
    from wave_analysis.sources.ndbc.parsers import SpectralBlock

    src = NDBCSource()
    raw = data_dir("raw") / "ndbc"
    out = data_dir("processed") / "ndbc"
    for st in args.stations:
        sid = st.upper()
        std = raw / sid / "stdmet"
        written: list[Path] = []
        inputs: list[Path] = []
        for f in sorted(std.glob("*")) if std.exists() else []:
            table = src.parse(f, "stdmet")
            obs = src.normalize(
                table, station_id=sid, source_file=f.name, source_checksum=sha256_file(f)
            )
            p = out / sid / f"stdmet_{f.name.split('.')[0]}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            obs.to_parquet(p, index=False)
            written.append(p)
            inputs.append(f)
        # Spectra: group files by period across the five products.
        by_period: dict[str, dict[str, Path]] = {}
        for prod in SPECTRAL_SET:
            d = raw / sid / prod
            for f in sorted(d.glob("*.txt.gz")) if d.exists() else []:
                m = HISTORICAL_FILENAME.match(f.name)
                if m is None:
                    continue
                # "2005" and "2005b" are distinct segments (often different frequency grids).
                by_period.setdefault(m["year"] + m["variant"], {})[prod] = f
        for period, files in sorted(by_period.items()):
            if "swden" not in files:
                continue
            parsed: dict[str, list[SpectralBlock]] = {
                prod: src.parse(f, prod) for prod, f in files.items()
            }
            # A file can contain several frequency grids (payload change mid-period). Every
            # swden grid segment is written; directional blocks are matched by identical grid.
            for g, den in enumerate(parsed["swden"]):
                blocks: dict[str, SpectralBlock] = {"swden": den}
                for prod, bl in parsed.items():
                    match = [
                        b
                        for b in bl
                        if b.frequency.shape == den.frequency.shape
                        and np.allclose(b.frequency, den.frequency)
                    ]
                    if prod != "swden" and match:
                        blocks[prod] = match[0]
                spec = assemble_spectrum(
                    blocks, meta={"station_id": sid, "period": period, "grid_segment": g}
                )
                spec_long, derived = src.normalize_spectrum(
                    spec, station_id=sid, source_file=",".join(f.name for f in files.values())
                )
                tag = period if len(parsed["swden"]) == 1 else f"{period}_grid{g}"
                p1 = out / sid / f"spectrum_{tag}.parquet"
                p2 = out / sid / f"spectral_bulk_{tag}.parquet"
                p1.parent.mkdir(parents=True, exist_ok=True)
                spec_long.to_parquet(p1, index=False)
                derived.to_parquet(p2, index=False)
                written += [p1, p2]
            inputs += list(files.values())
        if written:
            write_provenance(
                build_provenance(
                    artifact_id=f"ndbc_standardized_{sid}", inputs=inputs, outputs=written
                ),
                out / sid / "provenance.yaml",
            )
        print(f"{sid}: wrote {len(written)} tables")
    return 0


def _cmd_ndbc_cameras(args: argparse.Namespace) -> int:
    from wave_analysis.ingest.downloader import Downloader
    from wave_analysis.ingest.manifest import write_manifest
    from wave_analysis.sources.ndbc.buoycam import archive_cameras, manifest_path, summarize

    dest = Path(args.out) if args.out else data_dir("raw") / "ndbc" / "buoycam"
    dest.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(dest / ".archiver.lock") as acquired:
        if not acquired:
            print(f"another archiver run holds {dest / '.archiver.lock'}; skipping")
            return 0
        ledger = manifest_path(dest)
        with Downloader(min_interval_s=args.min_interval) as dl:
            entries = archive_cameras(
                dl,
                dest,
                backfill_hours=args.backfill_hours or None,
                sink=lambda batch: write_manifest(batch, ledger),
            )
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ")
    print(f"{stamp} buoycam archive: {summarize(entries)} -> {dest}")
    return 0


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[bool]:
    """Non-blocking advisory lock; yields ``False`` if another process holds it."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX: no locking
        yield True
        return
    with path.open("w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _cmd_registry_validate(args: argparse.Namespace) -> int:
    from wave_analysis.registry import validate_registry

    path = Path(args.path) if args.path else repo_root() / "data" / "registry" / "datasets.yaml"
    problems = validate_registry(load_yaml(path))
    for p in problems:
        print(f"ERROR: {p}")
    print(f"{path}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(prog="wave-analysis", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="group", required=True)

    ndbc = sub.add_parser("ndbc", help="NOAA NDBC operations").add_subparsers(
        dest="cmd", required=True
    )
    inv = ndbc.add_parser("inventory", help="build the station/product/year availability inventory")
    inv.add_argument("--dirs", nargs="*", help="historical directories to scan (default: all)")
    inv.add_argument("--out", help="output directory (default: data/registry)")
    inv.add_argument("--min-interval", type=float, default=1.0, help="seconds between requests")
    inv.set_defaults(func=_cmd_ndbc_inventory)

    dl = ndbc.add_parser("download", help="download raw files for stations")
    dl.add_argument("stations", nargs="*")
    dl.add_argument(
        "--camera-stations",
        action="store_true",
        help="add every station currently listing a buoy-camera image",
    )
    dl.add_argument(
        "--products", nargs="+", default=["stdmet", "swden", "swdir", "swdir2", "swr1", "swr2"]
    )
    dl.add_argument("--years", help="e.g. 2020-2023,2025")
    dl.add_argument("--realtime", action="store_true", help="also fetch the 45-day realtime files")
    dl.add_argument("--gzip", action="store_true", help="store realtime snapshots gzip-compressed")
    dl.add_argument("--manifest", help="manifest CSV (default: data/manifests/raw/ndbc.csv)")
    dl.add_argument("--min-interval", type=float, default=1.0)
    dl.set_defaults(func=_cmd_ndbc_download)

    std = ndbc.add_parser("standardize", help="parse raw files into standardized tables")
    std.add_argument("stations", nargs="+")
    std.set_defaults(func=_cmd_ndbc_standardize)

    cams = ndbc.add_parser("cameras", help="archive new buoy-camera images (run hourly)")
    cams.add_argument("--out", help="archive root (default: data/raw/ndbc/buoycam)")
    cams.add_argument(
        "--backfill-hours",
        type=int,
        default=70,
        help="recover missed images from the last N hours (< 72 h retention; 0 disables)",
    )
    cams.add_argument("--min-interval", type=float, default=1.0)
    cams.set_defaults(func=_cmd_ndbc_cameras)

    reg = sub.add_parser("registry", help="dataset registry").add_subparsers(
        dest="cmd", required=True
    )
    val = reg.add_parser("validate", help="validate datasets.yaml")
    val.add_argument("--path")
    val.set_defaults(func=_cmd_registry_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
