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
``cdip discover | download | standardize``
    List a CDIP station's archive files; download historic, deployment or
    realtime netCDF files (``data/manifests/raw/cdip.csv``); write standardized
    bulk, spectral and derived tables for a time range.
``pacioos beachcam``
    Archive the PacIOOS beach-camera images (default: the Waimea Bay pair,
    2009-2013) from ERDDAP, with a per-request ledger; resumable.
``webcoos cameras | inventory | download | historical``
    List WebCOOS cameras; show a camera's still-image inventory; archive
    one-minute stills thinned to a time grid (token from
    ``$WEBCOOS_API_TOKEN`` or ``~/.config/wave-analysis/webcoos_token``);
    archive past years the same way from WebCOOS's on-premise archive
    (no token; only as agreed with WebCOOS).
``era5 download | standardize``
    Request ERA5 ocean-wave parameters, one month per request, for a box around
    camera sites (``data/registry/camera_sites.yaml``) from the Copernicus CDS
    (key in ``~/.cdsapirc``); write standardized tables at the nearest sea
    grid point.
``registry validate``
    Validate ``data/registry/datasets.yaml`` against the registry schema.
``dashboard catalog | live | build``
    Build the public dashboard: the catalog from repository files, the live
    status and sea-state data on the collector host, and the static site.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

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


def _cmd_cdip_discover(args: argparse.Namespace) -> int:
    from wave_analysis.ingest.downloader import Downloader
    from wave_analysis.sources.cdip import CDIPSource

    with Downloader(min_interval_s=args.min_interval) as dl:
        files = CDIPSource(dl).discover(args.stations)
    if args.out:
        files.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        cols = ["station_id", "product", "deployment", "file_name", "approx_size_bytes"]
        print(files[[*cols, "modified_utc"]].to_string(index=False))
    total = files["approx_size_bytes"].sum() / 1e9
    print(f"{len(files)} files, about {total:.1f} GB")
    return 0


def _cmd_cdip_download(args: argparse.Namespace) -> int:
    from wave_analysis.ingest.downloader import Downloader
    from wave_analysis.ingest.manifest import EntryStatus, latest_by_url, write_manifest
    from wave_analysis.sources.cdip import CDIPSource

    manifest_path = (
        Path(args.manifest) if args.manifest else data_dir("manifests") / "raw" / "cdip.csv"
    )
    previous = {u: e.sha256 for u, e in latest_by_url(manifest_path).items() if e.sha256}
    dest = data_dir("raw") / "cdip"
    deployments = _parse_years(args.deployments) or []
    failed = 0
    with Downloader(min_interval_s=args.min_interval, timeout_s=args.timeout) as dl:
        src = CDIPSource(dl)
        for st in dict.fromkeys(args.stations):
            # One station at a time, so a long run records what it has done so far.
            entries = src.fetch(
                st, args.products, dest_root=dest, deployments=deployments, previous=previous
            )
            write_manifest(entries, manifest_path)
            for e in entries:
                size = f"{e.size_bytes / 1e6:,.1f} MB" if e.size_bytes is not None else "-"
                print(f"{e.station_id} {e.period}: {e.status.value} ({size}) {e.url}")
            failed += sum(e.status == EntryStatus.FAILED for e in entries)
    return 1 if failed else 0


def _cmd_cdip_standardize(args: argparse.Namespace) -> int:
    from wave_analysis.datasets.provenance import build_provenance, write_provenance
    from wave_analysis.ingest.checksum import sha256_file
    from wave_analysis.schemas.observation import validate_observation_frame
    from wave_analysis.sources.cdip import CDIPSource, file_path, file_url, station_code

    src = CDIPSource()
    raw = data_dir("raw") / "cdip"
    out_root = data_dir("processed") / "cdip"
    tag = f"{args.start or 'start'}_{args.end or 'end'}".replace(":", "")
    for st in args.stations:
        stn = station_code(st)
        path = raw / stn / file_path(stn, "historic").rsplit("/", 1)[-1]
        if not path.exists():
            print(f"{stn}: {path} not found; run 'wave-analysis cdip download {stn}'")
            return 1
        checksum = sha256_file(path)
        ds = src.parse(path, start=args.start, end=args.end)
        url = file_url(file_path(stn, "historic"))
        obs = src.normalize(
            ds, station_id=stn, source_file=path.name, source_url=url, source_checksum=checksum
        )
        validate_observation_frame(obs)
        spec_long, derived = src.normalize_spectrum(
            src.spectrum(ds, station_id=stn),
            station_id=stn,
            source_file=path.name,
            source_url=url,
            source_checksum=checksum,
        )
        validate_observation_frame(derived)
        out = out_root / stn
        out.mkdir(parents=True, exist_ok=True)
        written = [
            out / f"bulk_{tag}.parquet",
            out / f"spectrum_{tag}.parquet",
            out / f"spectral_bulk_{tag}.parquet",
        ]
        obs.to_parquet(written[0], index=False)
        spec_long.to_parquet(written[1], index=False)
        derived.to_parquet(written[2], index=False)
        write_provenance(
            build_provenance(
                artifact_id=f"cdip_standardized_{stn}_{tag}",
                inputs=[path],
                outputs=written,
                config={"start": args.start, "end": args.end},
            ),
            out / f"provenance_{tag}.yaml",
        )
        n = int(ds.sizes["waveTime"])
        print(
            f"{stn}: {n:,} wave records -> {len(obs):,} bulk rows, {len(spec_long):,} spectral rows"
        )
    return 0


def _cmd_pacioos_beachcam(args: argparse.Namespace) -> int:
    from wave_analysis.ingest.downloader import Downloader
    from wave_analysis.ingest.manifest import write_manifest
    from wave_analysis.sources.pacioos import (
        BEACHCAMS,
        archive_images,
        fetch_index,
        index_summary,
        ledger_path,
    )

    unknown = sorted(set(args.datasets) - set(BEACHCAMS))
    if unknown:
        print(f"unknown datasets {unknown}; expected {sorted(BEACHCAMS)}", file=sys.stderr)
        return 2
    dest = Path(args.out) if args.out else data_dir("raw") / "pacioos" / "beachcam"
    dest.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M:%SZ} {msg}", flush=True)

    failed = 0
    with _exclusive_lock(dest / ".archiver.lock") as acquired:
        if not acquired:
            print(f"another run holds {dest / '.archiver.lock'}; skipping")
            return 0
        with Downloader(min_interval_s=args.min_interval) as dl:
            indexes = []
            for ds in args.datasets:
                ledger = ledger_path(dest, ds)
                entry, index = fetch_index(dl, ds, dest, start=args.start, end=args.end)
                write_manifest([entry], ledger)
                if index is None:
                    log(
                        f"{ds}: index request {entry.status.value} ({entry.http_status or entry.note})"
                    )
                    failed += 1
                    continue
                indexes.append(index)
            if indexes:
                with pd.option_context("display.width", 160):
                    log("index:\n" + index_summary(indexes).to_string(index=False))
            if args.index_only:
                return 1 if failed else 0
            for index in indexes:
                ds = str(index["dataset"].iloc[0])
                ledger = ledger_path(dest, ds)
                summary = archive_images(
                    dl,
                    index,
                    dest,
                    sink=functools.partial(write_manifest, path=ledger),
                    limit=args.limit,
                    progress=lambda s: log(s.line()),
                )
                log(summary.line())
                for f in summary.failures[:20]:
                    log(f"  failed: {f}")
                failed += summary.failed
    return 1 if failed else 0


def _webcoos_client(args: argparse.Namespace) -> Any:
    from wave_analysis.sources.webcoos import WebCOOSClient, load_token

    return WebCOOSClient(load_token(), min_interval_s=args.min_interval)


def _cmd_webcoos_cameras(args: argparse.Namespace) -> int:
    from wave_analysis.sources.webcoos import camera_table

    with _webcoos_client(args) as client:
        cams = camera_table(client.assets())
    if args.state:
        cams = cams[cams["state"].fillna("").str.contains(args.state, case=False)]
    if args.out:
        cams.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    cols = ["camera", "latitude", "longitude", "state", "status", "access_level", "label"]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(cams[cols].to_string(index=False))
    print(f"{len(cams)} cameras")
    return 0


def _cmd_webcoos_inventory(args: argparse.Namespace) -> int:
    from wave_analysis.sources.webcoos import STILLS_PRODUCT, camera_table

    with _webcoos_client(args) as client:
        cams = camera_table(client.assets()).set_index("camera")
        if args.camera not in cams.index:
            print(f"unknown camera {args.camera!r}", file=sys.stderr)
            return 2
        service = cams.loc[args.camera, "stills_service"]
        if not isinstance(service, str):
            print(f"{args.camera} has no {STILLS_PRODUCT} product", file=sys.stderr)
            return 2
        print(f"{service}: requesting inventory (WebCOOS can take several minutes)...", flush=True)
        inv = client.inventory(service)
    cols = ["bin_start", "has_data", "bin_end", "count", "bytes", "data_start", "data_end"]
    bins = pd.DataFrame(inv["results"][0]["values"], columns=cols)
    have = bins[bins["has_data"].astype(bool)]
    print(
        f"{len(bins)} bins, {len(have)} with data: {int(have['count'].sum()):,} images, "
        f"{have['bytes'].sum() / 1e9:,.1f} GB, {have['data_start'].min()} to {have['data_end'].max()}"
    )
    if args.out:
        bins.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


def _cmd_webcoos_download(args: argparse.Namespace) -> int:
    from wave_analysis.ingest.downloader import Downloader
    from wave_analysis.ingest.manifest import write_manifest
    from wave_analysis.sources.webcoos import (
        ACKNOWLEDGEMENT,
        HistoricalAccessError,
        archive_stills,
        camera_table,
        check_age,
        element_table,
        expand_cameras,
        ledger_path,
        select_on_grid,
        write_listing,
    )

    try:
        cameras = expand_cameras(args.cameras)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    dest = Path(args.out) if args.out else data_dir("raw") / "webcoos"
    dest.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(args.end or pd.Timestamp.now(tz="UTC"))
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(args.lookback)

    def log(msg: str) -> None:
        print(f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M:%SZ} {msg}", flush=True)

    failed = 0
    with _exclusive_lock(dest / ".archiver.lock") as acquired:
        if not acquired:
            print(f"another run holds {dest / '.archiver.lock'}; skipping")
            return 0
        with _webcoos_client(args) as client:
            cams = camera_table(client.assets()).set_index("camera")
            unknown = sorted(set(cameras) - set(cams.index))
            if unknown:
                print(f"unknown cameras {unknown}", file=sys.stderr)
                return 2
            selections = {}
            for cam in cameras:
                service = cams.loc[cam, "stills_service"]
                if not isinstance(service, str):
                    log(f"{cam}: no stills service; skipped")
                    continue
                days = max(1, int((end - start) / pd.Timedelta(days=1)))
                elements = element_table(
                    client.elements(
                        service,
                        start,
                        end,
                        progress=(
                            (
                                lambda t, n, cam=cam: log(
                                    f"{cam}: listed to {t:%Y-%m-%d}, {n:,} stills"
                                )
                            )
                            if days > 7
                            else None
                        ),
                    ),
                    cam,
                )
                sel = select_on_grid(
                    elements, every=args.every, tolerance=args.tolerance, offset=args.offset
                )
                log(
                    f"{cam}: {len(elements):,} stills listed, {len(sel):,} on the {args.every} grid"
                )
                if sel.empty:
                    # A camera can be offline for days (Jennette's Pier, 2026-09).
                    continue
                selections[cam] = sel
        everything = pd.concat(selections.values()) if selections else pd.DataFrame()
        try:
            old = check_age(
                everything.get("time_utc", pd.Series(dtype="datetime64[ns, UTC]")),
                historical_approved=args.historical_approved,
            )
        except HistoricalAccessError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 3
        if old:
            log(f"{old} selected images are older than 90 days")
        if args.list_only:
            return 0
        # Image files are public: this downloader carries no token.
        with Downloader(min_interval_s=args.min_interval) as dl:
            for cam, sel in selections.items():
                write_listing(sel, dest, cam)
                summary = archive_stills(
                    dl,
                    sel,
                    dest,
                    sink=functools.partial(write_manifest, path=ledger_path(dest, cam)),
                    limit=args.limit,
                    progress=lambda s: log(s.line()),
                )
                log(summary.line())
                for f in summary.failures[:20]:
                    log(f"  failed: {f}")
                failed += summary.failed
    log(f"acknowledge in publications: {ACKNOWLEDGEMENT}")
    return 1 if failed else 0


def _cmd_webcoos_historical(args: argparse.Namespace) -> int:
    import shutil

    import httpx

    from wave_analysis.ingest.archive import collector_host
    from wave_analysis.ingest.downloader import Downloader
    from wave_analysis.ingest.manifest import write_manifest
    from wave_analysis.sources.webcoos import (
        ACKNOWLEDGEMENT,
        CONTACT,
        ONPREM_GROUPS,
        ONPREM_MAX_BYTES_PER_S,
        ONPREM_PRODUCTS,
        ONPREM_WAIT_S,
        OnPremArchive,
        archive_stills,
        ledger_path,
        month_index_path,
        on_disk,
        read_month_index,
        select_between,
        write_month_index,
    )

    if not args.historical_approved:
        print(
            "refused: bulk downloads of past years need WebCOOS's agreement and a note to "
            f"them when the transfer starts ({CONTACT}). Then rerun with --historical-approved.",
            file=sys.stderr,
        )
        return 3
    unknown = sorted(set(args.cameras) - set(ONPREM_GROUPS))
    if unknown:
        print(f"unknown cameras {unknown}; known: {', '.join(ONPREM_GROUPS)}", file=sys.stderr)
        return 2
    if args.wait < ONPREM_WAIT_S:
        print(f"--wait must be at least {ONPREM_WAIT_S} s (WebCOOS's request)", file=sys.stderr)
        return 2
    if args.product not in ONPREM_PRODUCTS:
        print(
            f"unknown product {args.product!r}; known: {', '.join(ONPREM_PRODUCTS)}",
            file=sys.stderr,
        )
        return 2
    root = "webcoos" if args.product == "stills" else f"webcoos_{args.product}"
    dest = Path(args.out) if args.out else data_dir("raw") / root
    dest.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    start = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end = pd.Timestamp(args.end, tz="UTC") if args.end else now
    grid = {"every": args.every, "tolerance": args.tolerance, "offset": args.offset}
    if args.every != "all":
        pd.Timedelta(args.every)  # fail early on a bad step
    kept = "kept (all)" if args.every == "all" else f"on the {args.every} grid"
    host = collector_host()

    def log(msg: str) -> None:
        print(f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M:%SZ} {msg}", flush=True)

    failed = 0
    # Not the hourly collector's lock, so that it keeps running. One lock next
    # to the product roots, so that only one historical run (one connection to
    # WebCOOS) runs at a time, whatever the product.
    lock = dest.parent / ".webcoos-historical.lock"
    with _exclusive_lock(lock) as acquired:
        if not acquired:
            print(f"another historical run holds {lock}; skipping")
            return 0
        with Downloader(
            min_interval_s=0, min_gap_s=args.wait, max_bytes_per_s=ONPREM_MAX_BYTES_PER_S
        ) as dl:
            archive = OnPremArchive(dl, args.product)
            for cam in args.cameras:
                ledger = ledger_path(dest, cam, host)
                months = archive.months(cam, start, end)
                if not months:
                    log(f"{cam}: no months in range")
                    continue
                log(
                    f"{cam} {args.product}: {len(months)} months, {months[0]} to {months[-1]}; "
                    f"ledger {ledger.name}"
                )
                totals = {"listed": 0, "selected": 0, "on_disk": 0, "new": 0, "MB": 0.0}
                remaining = args.limit
                for month in months:
                    free_gb = shutil.disk_usage(dest).free / 1e9
                    if free_gb < args.min_free_gb and not args.list_only:
                        log(
                            f"stopping before {cam} {month}: {free_gb:,.0f} GB free on the "
                            f"drive of {dest}, less than --min-free-gb {args.min_free_gb:g}"
                        )
                        return 4
                    a = month.start_time.tz_localize("UTC")
                    b = (month + 1).start_time.tz_localize("UTC")
                    whole = (start is None or start <= a) and end >= b
                    a, b = (a if start is None else max(a, start)), min(b, end)
                    index = month_index_path(dest, cam, month)
                    try:
                        if whole and index.exists() and not args.relist:
                            stills = read_month_index(index)
                        else:
                            listed_at = pd.Timestamp.now(tz="UTC")
                            stills = archive.month_stills(cam, month, start=a, end=b)
                            # A whole month is indexed once it has ended (with a day's margin).
                            if whole and b < listed_at - pd.Timedelta("1D"):
                                write_month_index(stills, index, listed_at)
                    except (httpx.HTTPError, OSError) as exc:
                        log(f"{cam} {month}: listing failed ({exc}); rerun to retry")
                        failed += 1
                        continue
                    window = stills[(stills["time_utc"] >= a) & (stills["time_utc"] < b)]
                    if args.every == "all":
                        sel = window.assign(grid_time=window["time_utc"], offset_s=0.0)
                    else:
                        sel = select_between(window, a, b, **grid)
                    have = on_disk(sel, dest)
                    todo = sel[~have]
                    totals["listed"] += len(window)
                    totals["selected"] += len(sel)
                    totals["on_disk"] += int(have.sum())
                    msg = (
                        f"{cam} {month}: {len(window):,} images, {len(sel):,} {kept}, "
                        f"{int(have.sum()):,} already archived"
                    )
                    if args.list_only:
                        totals["MB"] += float(todo["size_bytes"].sum()) / 1e6
                        log(
                            f"{msg}, {len(todo):,} to fetch ({todo['size_bytes'].sum() / 1e6:,.0f} MB)"
                        )
                        continue
                    if todo.empty:
                        log(f"{msg}, none to fetch")
                        continue
                    summary = archive_stills(
                        dl,
                        todo,
                        dest,
                        ledger=ledger,
                        sink=functools.partial(write_manifest, path=ledger),
                        limit=remaining,
                        progress=lambda s: log(s.line()),
                    )
                    if remaining is not None:
                        remaining -= summary.downloaded + summary.not_found + summary.failed
                    totals["new"] += summary.downloaded
                    totals["MB"] += summary.bytes / 1e6
                    log(f"{msg}; {summary.line()}")
                    for f in summary.failures[:20]:
                        log(f"  failed: {f}")
                    failed += summary.failed
                    if remaining is not None and remaining <= 0:
                        log(f"{cam}: --limit reached")
                        break
                verb = "to fetch" if args.list_only else "downloaded"
                log(
                    f"{cam}: {totals['listed']:,} images, {totals['selected']:,} {kept}, "
                    f"{totals['on_disk']:,} already archived, {totals['MB']:,.0f} MB {verb}"
                    + ("" if args.list_only else f" ({totals['new']:,} files)")
                )
    log(f"acknowledge in publications: {ACKNOWLEDGEMENT}")
    return 1 if failed else 0


def _era5_sites(names: Sequence[str]) -> dict[str, tuple[float, float]]:
    from wave_analysis.dashboard.catalog import load_camera_sites

    sites = {
        s.site_id: (s.latitude, s.longitude)
        for s in load_camera_sites(repo_root() / "data" / "registry" / "camera_sites.yaml")
    }
    unknown = [n for n in names if n not in sites]
    if unknown:
        raise SystemExit(f"unknown sites {unknown}; known: {', '.join(sorted(sites))}")
    return {n: sites[n] for n in dict.fromkeys(names)}


def _cmd_era5_download(args: argparse.Namespace) -> int:
    from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, write_manifest
    from wave_analysis.sources.era5 import (
        CITATION,
        DATASET_PAGE,
        CDSConfigError,
        ERA5WaveSource,
        months,
    )

    sites = _era5_sites(args.sites)
    periods = months(args.start, args.end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m"))
    dest = Path(args.out) if args.out else data_dir("raw") / "era5"
    manifest = Path(args.manifest) if args.manifest else data_dir("manifests") / "raw" / "era5.csv"

    def log(msg: str) -> None:
        print(f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M:%SZ} {msg}", flush=True)

    src = ERA5WaveSource(log=log if args.verbose else None)
    plan = src.discover(sites, periods, dest_root=dest, box_deg=args.box, refresh=args.refresh)
    todo = plan[plan["request"]]
    log(
        f"{len(plan)} site-months, {len(todo)} to request "
        f"({todo['fields'].sum():,} fields; {plan['fields'].max():,} per request at most)"
    )
    if args.list_only:
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            print(plan.drop(columns="path").to_string(index=False))
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    failed = 0

    def record(batch: list[ManifestEntry]) -> None:
        # Called as each month completes, so a long run shows progress and keeps its ledger.
        write_manifest(batch, manifest)
        for e in batch:
            size = f"{e.size_bytes / 1e6:,.2f} MB" if e.size_bytes is not None else "-"
            log(f"{e.station_id} {e.period}: {e.status.value} ({size}) {e.note}")
            if e.status == EntryStatus.FAILED and "licen" in (e.note or "").lower():
                log(f"accept the dataset licence at {DATASET_PAGE} (Download tab)")

    with _exclusive_lock(dest / ".era5.lock") as acquired:
        if not acquired:
            print(f"another run holds {dest / '.era5.lock'}; skipping")
            return 0
        try:
            for site, (lat, lon) in sites.items():
                site_periods = [
                    pd.Period(p, freq="M") for p in todo.loc[todo["site_id"] == site, "period"]
                ]
                entries = src.fetch(
                    site,
                    lat,
                    lon,
                    site_periods,
                    dest_root=dest,
                    box_deg=args.box,
                    refresh=args.refresh,
                    sink=record,
                )
                failed += sum(e.status == EntryStatus.FAILED for e in entries)
        except CDSConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    log(f"cite: {CITATION}")
    return 1 if failed else 0


def _cmd_era5_standardize(args: argparse.Namespace) -> int:
    from wave_analysis.datasets.provenance import build_provenance, write_provenance
    from wave_analysis.ingest.checksum import sha256_file
    from wave_analysis.schemas.observation import validate_observation_frame
    from wave_analysis.sources.era5 import (
        ERA5WaveSource,
        month_status,
        months,
        nearest_sea_point,
        raw_path,
    )

    sites = _era5_sites(args.sites)
    periods = months(args.start, args.end)
    raw = data_dir("raw") / "era5"
    out_root = data_dir("processed") / "era5"
    tag = f"{periods[0]}_{periods[-1]}"
    src = ERA5WaveSource()
    for site, (lat, lon) in sites.items():
        frames, inputs, states = [], [], []
        point = None
        for p in periods:
            path = raw_path(raw, site, p)
            if not path.exists():
                continue
            ds = src.parse(path)
            point = nearest_sea_point(ds, lat, lon)
            frames.append(
                src.normalize(
                    ds,
                    site_id=site,
                    latitude=lat,
                    longitude=lon,
                    source_file=path.name,
                    source_checksum=sha256_file(path),
                )
            )
            inputs.append(path)
            states.append(month_status(path, p))
        if not frames or point is None:
            print(f"{site}: no raw files for {tag}; run 'wave-analysis era5 download {site}'")
            return 1
        obs = pd.concat(frames, ignore_index=True)
        validate_observation_frame(obs)
        out = out_root / site
        out.mkdir(parents=True, exist_ok=True)
        written = [out / f"bulk_{tag}.parquet"]
        obs.to_parquet(written[0], index=False)
        write_provenance(
            build_provenance(
                artifact_id=f"era5_standardized_{site}_{tag}",
                inputs=inputs,
                outputs=written,
                config={
                    "start": str(periods[0]),
                    "end": str(periods[-1]),
                    "site": {"latitude": lat, "longitude": lon},
                    "grid_point": {
                        "latitude": point.latitude,
                        "longitude": point.longitude,
                        "distance_km": round(point.distance_km, 2),
                    },
                },
            ),
            out / f"provenance_{tag}.yaml",
        )
        n_final = states.count("final")
        print(
            f"{site}: {len(inputs)} months ({n_final} final) -> {len(obs):,} rows at grid point "
            f"{point.latitude:.2f}, {point.longitude:.2f} ({point.distance_km:.1f} km from the site)"
        )
    return 0


def _cmd_registry_validate(args: argparse.Namespace) -> int:
    from wave_analysis.registry import validate_registry

    path = Path(args.path) if args.path else repo_root() / "data" / "registry" / "datasets.yaml"
    problems = validate_registry(load_yaml(path))
    for p in problems:
        print(f"ERROR: {p}")
    print(f"{path}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


def _cmd_dashboard_catalog(args: argparse.Namespace) -> int:
    from wave_analysis.dashboard.catalog import build_catalog, write_json

    catalog = build_catalog(repo_root())
    path = write_json(catalog, Path(args.out) / "catalog.json")
    print(
        f"{path}: {len(catalog['datasets'])} datasets, {len(catalog['sites'])} sites, "
        f"{len(catalog['references'])} references"
    )
    return 0


def _cmd_dashboard_live(args: argparse.Namespace) -> int:
    from wave_analysis.dashboard.seastate import RangeFetcher
    from wave_analysis.dashboard.site import build_live
    from wave_analysis.dashboard.status import StatusInputs

    raw = data_dir("raw")
    inputs = StatusInputs(
        buoycam_root=raw / "ndbc" / "buoycam",
        ndbc_root=raw / "ndbc",
        raw_root=raw,
        manifests_root=data_dir("manifests"),
        external_root=data_dir("external"),
    )
    cache = Path(args.cache) if args.cache else data_dir("interim") / "dashboard" / "seastate"
    fetcher = None if args.no_fetch else RangeFetcher(min_interval_s=args.min_interval)
    try:
        result = build_live(repo_root(), inputs, Path(args.out), cache_dir=cache, fetch=fetcher)
    finally:
        if fetcher is not None:
            fetcher.close()
    refresh = result["refresh"]
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ")
    print(
        f"{stamp} dashboard live data: {result['stations']} camera stations, "
        f"{result['seastate_files']} sea-state files"
        + (
            f"; realtime heads: {refresh['requests']} requests, "
            f"{refresh['bytes_received'] / 1e6:.1f} MB, {len(refresh['failures'])} failures"
            if refresh
            else ""
        )
    )
    for failure in (refresh or {}).get("failures", [])[:10]:
        print(f"  warning: {failure}")
    return 0


def _cmd_dashboard_check(args: argparse.Namespace) -> int:
    from wave_analysis.dashboard.site import check_site

    problems = check_site(Path(args.site), require_docs=args.require_docs)
    for p in problems:
        print(f"ERROR: {p}")
    print(f"{args.site}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


def _cmd_dashboard_build(args: argparse.Namespace) -> int:
    from wave_analysis.dashboard.site import build_site

    result = build_site(
        repo_root(), Path(args.out), live_dir=Path(args.live) if args.live else None
    )
    print(f"{args.out}: {result}")
    return 0


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

    cdip = sub.add_parser("cdip", help="CDIP Waverider buoys (Scripps)").add_subparsers(
        dest="cmd", required=True
    )
    cdisc = cdip.add_parser("discover", help="list a station's archive files (THREDDS catalog)")
    cdisc.add_argument("stations", nargs="+", help="CDIP station codes, e.g. 106")
    cdisc.add_argument("--out", help="also write the table to this CSV")
    cdisc.add_argument("--min-interval", type=float, default=1.0)
    cdisc.set_defaults(func=_cmd_cdip_discover)
    cdl = cdip.add_parser("download", help="download whole netCDF files for stations")
    cdl.add_argument("stations", nargs="+", help="CDIP station codes, e.g. 106")
    cdl.add_argument(
        "--products",
        nargs="+",
        default=["historic"],
        choices=["historic", "realtime", "deployment"],
        help="historic: all released records; deployment: per-deployment files with raw "
        "displacement (large); realtime: current deployment",
    )
    cdl.add_argument("--deployments", help="deployment numbers for 'deployment', e.g. 10-14")
    cdl.add_argument("--manifest", help="manifest CSV (default: data/manifests/raw/cdip.csv)")
    cdl.add_argument("--min-interval", type=float, default=1.0)
    cdl.add_argument("--timeout", type=float, default=120.0, help="per-request timeout [s]")
    cdl.set_defaults(func=_cmd_cdip_download)
    cstd = cdip.add_parser("standardize", help="write standardized tables from the historic file")
    cstd.add_argument("stations", nargs="+")
    cstd.add_argument("--start", help="first sample start (UTC), e.g. 2009-01-01")
    cstd.add_argument("--end", help="end of the range (UTC, exclusive), e.g. 2014-01-01")
    cstd.set_defaults(func=_cmd_cdip_standardize)

    pac = sub.add_parser("pacioos", help="PacIOOS beach cameras (Oahu)").add_subparsers(
        dest="cmd", required=True
    )
    pbc = pac.add_parser("beachcam", help="archive beach-camera images from ERDDAP (resumable)")
    pbc.add_argument(
        "datasets",
        nargs="*",
        default=["beachcam_003", "beachcam_004"],
        help="ERDDAP datasets beachcam_001..004 (default: the Waimea Bay pair, 003 and 004)",
    )
    pbc.add_argument("--start", help="first image time (UTC)")
    pbc.add_argument("--end", help="end of the range (UTC, exclusive)")
    pbc.add_argument("--limit", type=int, help="at most N new images per camera")
    pbc.add_argument("--index-only", action="store_true", help="fetch and check the index only")
    pbc.add_argument("--out", help="archive root (default: data/raw/pacioos/beachcam)")
    pbc.add_argument("--min-interval", type=float, default=1.0)
    pbc.set_defaults(func=_cmd_pacioos_beachcam)

    wc = sub.add_parser("webcoos", help="WebCOOS coastal webcams (API token)").add_subparsers(
        dest="cmd", required=True
    )
    wcam = wc.add_parser("cameras", help="list cameras with position, status and products")
    wcam.add_argument("--state", help="filter by state or territory, e.g. 'North Carolina'")
    wcam.add_argument("--out", help="also write the table to this CSV")
    wcam.add_argument("--min-interval", type=float, default=1.0)
    wcam.set_defaults(func=_cmd_webcoos_cameras)
    winv = wc.add_parser("inventory", help="a camera's one-minute-stills inventory (slow)")
    winv.add_argument("camera")
    winv.add_argument("--out", help="write the bins to this CSV")
    winv.add_argument("--min-interval", type=float, default=1.0)
    winv.set_defaults(func=_cmd_webcoos_inventory)
    wdl = wc.add_parser("download", help="archive one-minute stills thinned to a time grid")
    wdl.add_argument(
        "cameras",
        nargs="+",
        help="camera slugs (e.g. currituck_hampton_inn) or sets: @yin2025, @near_buoy",
    )
    wdl.add_argument("--start", help="UTC start (default: --end minus --lookback)")
    wdl.add_argument("--lookback", default="1D", help="window when --start is not given (1D)")
    wdl.add_argument("--end", help="UTC end (default: now)")
    wdl.add_argument("--every", default="30min", help="grid step (default 30min)")
    wdl.add_argument("--tolerance", default="5min", help="max distance from a grid time")
    wdl.add_argument("--offset", default="0min", help="grid offset from the hour")
    wdl.add_argument("--limit", type=int, help="at most N new images per camera")
    wdl.add_argument("--list-only", action="store_true", help="list and check; download nothing")
    wdl.add_argument(
        "--historical-approved",
        action="store_true",
        help="WebCOOS has agreed to a large download of data older than 90 days",
    )
    wdl.add_argument("--out", help="archive root (default: data/raw/webcoos)")
    wdl.add_argument("--min-interval", type=float, default=1.0)
    wdl.set_defaults(func=_cmd_webcoos_download)
    whi = wc.add_parser(
        "historical",
        help="archive past stills from WebCOOS's on-premise archive, thinned to a time grid",
    )
    whi.add_argument("cameras", nargs="+", help="camera slugs, e.g. currituck_hampton_inn")
    whi.add_argument(
        "--product",
        default="stills",
        help="stills (default), timex or brt (10-minute products, from 2024)",
    )
    whi.add_argument("--start", help="UTC start (default: the camera's first month)")
    whi.add_argument("--end", help="UTC end, exclusive (default: now)")
    whi.add_argument(
        "--every", default="30min", help="grid step (default 30min), or 'all' for every image"
    )
    whi.add_argument("--tolerance", default="5min", help="max distance from a grid time")
    whi.add_argument("--offset", default="0min", help="grid offset from the hour")
    whi.add_argument("--limit", type=int, help="at most N new images per camera")
    whi.add_argument("--list-only", action="store_true", help="list and count; download nothing")
    whi.add_argument(
        "--relist", action="store_true", help="list months again instead of using saved listings"
    )
    whi.add_argument(
        "--historical-approved",
        action="store_true",
        help="WebCOOS has agreed and has been told that the transfer starts",
    )
    whi.add_argument(
        "--wait", type=float, default=1.0, help="seconds between requests (at least 1)"
    )
    whi.add_argument(
        "--min-free-gb",
        type=float,
        default=50.0,
        help="stop when the archive's drive has less free space than this (default 50)",
    )
    whi.add_argument(
        "--out", help="archive root (default: data/raw/webcoos, webcoos_timex or webcoos_brt)"
    )
    whi.set_defaults(func=_cmd_webcoos_historical)

    era = sub.add_parser("era5", help="ERA5 reanalysis wave parameters (Copernicus CDS)")
    era_sub = era.add_subparsers(dest="cmd", required=True)
    edl = era_sub.add_parser("download", help="request monthly files for camera sites (resumable)")
    edl.add_argument("sites", nargs="+", help="site ids from data/registry/camera_sites.yaml")
    edl.add_argument("--start", required=True, help="first month, e.g. 2009-02")
    edl.add_argument("--end", help="last month, inclusive (default: this month)")
    edl.add_argument("--box", type=float, default=1.0, help="half-width of the box [deg]")
    edl.add_argument("--refresh", action="store_true", help="request months already final too")
    edl.add_argument("--list-only", action="store_true", help="print the plan; request nothing")
    edl.add_argument("--verbose", action="store_true", help="show the CDS client's messages")
    edl.add_argument("--out", help="raw directory (default: data/raw/era5)")
    edl.add_argument("--manifest", help="manifest CSV (default: data/manifests/raw/era5.csv)")
    edl.set_defaults(func=_cmd_era5_download)
    estd = era_sub.add_parser(
        "standardize", help="write standardized tables at the nearest sea point"
    )
    estd.add_argument("sites", nargs="+")
    estd.add_argument("--start", required=True, help="first month, e.g. 2009-02")
    estd.add_argument("--end", required=True, help="last month, inclusive")
    estd.set_defaults(func=_cmd_era5_standardize)

    dash = sub.add_parser("dashboard", help="public dashboard data and site").add_subparsers(
        dest="cmd", required=True
    )
    dcat = dash.add_parser("catalog", help="write catalog.json from the repository")
    dcat.add_argument("--out", required=True, help="output directory")
    dcat.set_defaults(func=_cmd_dashboard_catalog)
    dlive = dash.add_parser(
        "live", help="write live status and sea-state data (collector host, hourly)"
    )
    dlive.add_argument("--out", required=True, help="output directory (replaced)")
    dlive.add_argument("--cache", help="sea-state cache (default: data/interim/dashboard/seastate)")
    dlive.add_argument(
        "--no-fetch", action="store_true", help="use the cache only; no network requests"
    )
    dlive.add_argument("--min-interval", type=float, default=1.0)
    dlive.set_defaults(func=_cmd_dashboard_live)
    dbuild = dash.add_parser("build", help="assemble the static site (front end + catalog)")
    dbuild.add_argument("--out", required=True, help="site output directory")
    dbuild.add_argument("--live", help="live data directory to bundle as the offline fallback")
    dbuild.set_defaults(func=_cmd_dashboard_build)
    dcheck = dash.add_parser("check", help="validate a built site (JSON, assets, doc links)")
    dcheck.add_argument("--site", required=True, help="built site directory")
    dcheck.add_argument(
        "--require-docs", action="store_true", help="also require every linked docs page"
    )
    dcheck.set_defaults(func=_cmd_dashboard_check)

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
