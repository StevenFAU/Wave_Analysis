"""Acquire one day of a vertical waveform, saving the provider URL and byte checksum."""

import argparse
import fcntl
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

from wave_analysis.config import data_dir
from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import latest_by_url, write_manifest

CHANNELS = {"IU.KIP": "BHZ", "N4.Y60A": "HHZ"}


def fetch_day(client, station, day, root, ledger, seen):
    net, sta = station.split(".")
    chan = CHANNELS[station]
    query = urlencode(
        {
            "net": net,
            "sta": sta,
            "loc": "00",
            "cha": chan,
            "starttime": str(day),
            "endtime": str(day + timedelta(days=1)),
            "nodata": 404,
        }
    )
    file = Path(station) / f"{station}.00.{chan}.{day}.mseed"
    dest = root / file
    relative = str(Path("data/raw/earthscope") / file)
    url = "https://service.earthscope.org/fdsnws/dataselect/1/query?" + query
    old = seen.get(url)
    if old and dest.exists() and dest.stat().st_size == old.size_bytes:
        return
    entry = client.fetch(
        url,
        dest,
        source_id="earthscope_fdsn",
        product="miniseed",
        station_id=station,
        period=str(day),
    )
    if entry.local_path:
        entry.local_path = relative
    write_manifest([entry], ledger)
    if entry.http_status in (204, 404):
        print(relative, "no data", flush=True)
        return
    if not entry.sha256:
        raise SystemExit(f"{entry.status} ({entry.http_status})")
    print(relative, entry.size_bytes, entry.sha256, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("station", choices=CHANNELS)
    parser.add_argument("day", type=date.fromisoformat)
    parser.add_argument("--through", type=date.fromisoformat)
    args = parser.parse_args()
    root = data_dir("raw") / "earthscope"
    ledger = root / "_manifests" / f"{args.station}.csv"
    root.mkdir(parents=True, exist_ok=True)
    end = args.through or args.day
    if end < args.day:
        parser.error("end date precedes start")
    with (root / ".pull.lock").open("a") as lock, Downloader(min_gap_s=1, timeout_s=90) as client:
        fcntl.flock(lock, fcntl.LOCK_EX)
        seen = latest_by_url(ledger)
        for i in range((end - args.day).days + 1):
            fetch_day(client, args.station, args.day + timedelta(days=i), root, ledger, seen)


if __name__ == "__main__":
    main()
