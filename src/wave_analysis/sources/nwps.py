"""Archive published NWPS buoy spectra and system-tracking grids by forecast cycle."""

import argparse
import fcntl
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from wave_analysis.config import data_dir, load_yaml, repo_root
from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import read_manifest, write_manifest


def links(client, url, pattern):
    """Return safe matching basenames from an office or grid listing."""
    response = client.get(url)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    return sorted(
        {v for v in re.findall(r'href="([^"]+)"', response.text) if re.fullmatch(pattern, v)}
    )


def collect(config, end, days, grids=False):
    """Discover and archive the configured cycles, serialized with a filesystem lock."""
    root = data_dir("raw") / "nwps"
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "_manifests/nwps.csv"
    failures = []
    with (
        (root / ".collect.lock").open("a") as lock,
        Downloader(min_interval_s=1, min_gap_s=0.4) as client,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX)
        seen = (
            {r.local_path: r for r in read_manifest(ledger) if r.sha256} if ledger.exists() else {}
        )
        for age in reversed(range(days)):
            day = (end - timedelta(days=age)).strftime("%Y%m%d")
            n = size = 0
            for region, offices in config["offices"].items():
                for office in offices:
                    url = f"{config['base']}{region}.{day}/{office}/"
                    for cycle in links(client, url, r"[0-2][0-9]/"):
                        for grid, pattern in [
                            ("CG0", r"cg0_trkng"),
                            ("CG1", r"spc2d_[a-zA-Z0-9_-]+_cg1" + (r"|cg1" if grids else "")),
                        ]:
                            folder = f"{url}{cycle}{grid}/"
                            hour = cycle.rstrip("/")
                            names = links(
                                client,
                                folder,
                                rf"nwps\.t{hour}z\.(?:{pattern})\.{office}\.(?:grib2|txt)",
                            )
                            for name in names:
                                dest = root / day / office / hour / grid / name
                                relative = str(Path("data/raw/nwps") / dest.relative_to(root))
                                old = seen.get(relative)
                                if old and dest.is_file() and dest.stat().st_size == old.size_bytes:
                                    n += 1
                                    size += old.size_bytes
                                    continue
                                entry = client.fetch(
                                    folder + name,
                                    dest,
                                    source_id="noaa_nwps",
                                    product=grid,
                                    station_id=office,
                                    period=f"{day}T{hour}Z",
                                )
                                if entry.local_path:
                                    entry.local_path = relative
                                write_manifest([entry], ledger)
                                if entry.sha256:
                                    n += 1
                                    size += entry.size_bytes
                                else:
                                    failures.append((folder + name, entry.status))
            print(day, n, "files", size, "bytes on disk", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} unsuccessful downloads: {failures[:3]}")


def main():
    """Run the configured NWPS backfill."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--end", type=datetime.fromisoformat, default=datetime.now(UTC))
    p.add_argument("--days", type=int)
    p.add_argument("--include-grids", action="store_true", help="also collect CG1 GRIB2")
    args = p.parse_args()
    cfg = load_yaml(repo_root() / "configs/sources/nwps.yaml")
    days = cfg["retention_days"] if args.days is None else args.days
    if not 1 <= days <= cfg["retention_days"]:
        p.error("days must be within the configured retention window")
    collect(cfg, args.end.date(), days, args.include_grids or cfg.get("include_grids", False))


if __name__ == "__main__":
    main()
