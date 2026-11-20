"""Check or acquire the GFS-Wave station bulletin tar for each selected issue cycle."""

import argparse
from datetime import date, timedelta
from pathlib import Path
from time import sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from wave_analysis.config import data_dir
from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import read_manifest, write_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("first", type=date.fromisoformat)
    p.add_argument("last", type=date.fromisoformat)
    p.add_argument(
        "--hours", nargs="+", choices=["00", "06", "12", "18"], default=["00", "06", "12", "18"]
    )
    p.add_argument("--fetch", action="store_true", help="archive the full bulletin tars")
    a = p.parse_args()
    if a.last < a.first:
        p.error("last date precedes first")
    root = data_dir("raw") / "gfs_wave"
    ledger = root / "_manifests/bulletins.csv"
    seen = {r.url: r for r in read_manifest(ledger) if r.sha256} if ledger.exists() else {}
    print("date,cycle,http_status,bytes", flush=True)
    with Downloader(min_interval_s=1) as client:
        for offset in range((a.last - a.first).days + 1):
            day = (a.first + timedelta(days=offset)).strftime("%Y%m%d")
            for hh in a.hours:
                name = f"gfswave.t{hh}z.bull_tar"
                url = (
                    f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{day}/{hh}/wave/station/{name}"
                )
                dest = root / day / name
                if a.fetch:
                    old = seen.get(url)
                    if old and dest.exists() and dest.stat().st_size == old.size_bytes:
                        status, size = 200, old.size_bytes
                    else:
                        entry = client.fetch(
                            url,
                            dest,
                            source_id="noaa_gfs_wave",
                            product="bull_tar",
                            period=f"{day}T{hh}Z",
                        )
                        if entry.local_path:
                            entry.local_path = str(Path("data/raw/gfs_wave") / day / name)
                        write_manifest([entry], ledger)
                        status, size = entry.http_status, entry.size_bytes
                        if not entry.sha256 and status != 404:
                            raise RuntimeError(f"could not fetch {url}: {entry.note}")
                else:
                    try:
                        with urlopen(Request(url, method="HEAD"), timeout=30) as resp:
                            status, size = resp.status, resp.headers.get("Content-Length")
                    except HTTPError as exc:
                        if exc.code != 404:
                            raise
                        status, size = 404, 0
                    sleep(1)
                print(f"{day},{hh},{status},{size}", flush=True)


if __name__ == "__main__":
    main()
