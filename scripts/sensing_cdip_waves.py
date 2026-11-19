"""Build development-only CDIP individual-wave targets from small OPeNDAP slices."""

import argparse
import hashlib
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir, repo_root
from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import latest_by_url, write_manifest


def array(body, name):
    m = re.search(rf"^{name}\[(\d+)\](?:\[(\d+)\])?\s*\n(.+?)(?:\n\s*\n|\Z)", body, re.M | re.S)
    if not m:
        raise ValueError(f"missing {name}")
    n = int(m[1])
    if m[2]:
        v = [np.fromstring(line.split("],", 1)[1], sep=",") for line in m[3].strip().splitlines()]
        result = np.array(v)
        if result.shape != (n, int(m[2])):
            raise ValueError(f"truncated {name}")
        return result
    values = np.fromstring(m[3].strip(), sep=",")
    if len(values) != n:
        raise ValueError(f"truncated {name}")
    return values


def targets(body, start, end):
    bounds = array(body, "zcrossTimeBounds") / 1000
    peak, height, period, q = [
        array(body, n) for n in ("zcrossZPeak", "zcrossHeight", "zcrossPeriod", "zcrossFlagPrimary")
    ]
    # One full upcrossing wave extends through the following negative excursion.
    complete = (
        (peak[:-1] > 0)
        & (peak[1:] < 0)
        & (peak[1:] > -100)
        & (bounds[:-1, 0] >= start)
        & (bounds[1:, 1] <= end)
        & (np.abs(bounds[:-1, 1] - bounds[1:, 0]) < 0.002)
        & (height[:-1] > 0)
        & (height[:-1] <= 40)
        & (period[:-1] > 0)
    )
    result = {}
    for name, flags in [("good", [1]), ("incl_unevaluated", [1, 2])]:
        keep = complete & np.isin(q[:-1], flags) & np.isin(q[1:], flags)
        h = np.sort(height[:-1][keep])[::-1]
        res = {"wave_count": len(h)}
        if len(h) >= 20:
            res.update(
                mean_height_m=h.mean(),
                h_third_m=h[: (len(h) + 2) // 3].mean(),
                h_tenth_m=h[: (len(h) + 9) // 10].mean(),
                tz_s=period[:-1][keep].mean(),
            )
        result.update({name + "_" + k: v for k, v in res.items()})
    return result


class Subsets:
    def __init__(self, client):
        self.root = data_dir("raw") / "cdip/zcross_subsets"
        self.ledger = self.root / "_manifests/queries.csv"
        self.seen = latest_by_url(self.ledger)
        self.client = client

    def query(self, station, deploy, fields):
        base = (
            f"https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/zcross/archive/{station}p1/{deploy}"
        )
        url = base + ".ascii?" + quote(fields, safe=",:[]")
        name = hashlib.sha256(url.encode()).hexdigest() + ".ascii"
        dest = self.root / deploy / name
        old = self.seen.get(url)
        if not (old and dest.exists() and dest.stat().st_size == old.size_bytes):
            r = self.client.fetch(
                url,
                dest,
                source_id="cdip",
                product="zcross_subset",
                station_id=station,
                period=deploy,
            )
            if r.local_path:
                r.local_path = str(Path("data/raw/cdip/zcross_subsets") / deploy / name)
            write_manifest([r], self.ledger)
            if not r.sha256:
                raise RuntimeError(f"{deploy}: {r.status} HTTP {r.http_status}")
            self.seen[url] = r
        return dest.read_text().split("---------------------------------------------", 1)[1].strip()


def historic_windows(clips):
    """Match only development clip metadata to containing good archived records."""
    from netCDF4 import Dataset

    records = []
    for station, group in clips.groupby("cdip_station"):
        platform = str(station).zfill(3)
        path = data_dir("raw") / "cdip" / platform / f"{platform}p1_historic.nc"
        with Dataset(path) as ds:
            bounds = np.asarray(ds["waveTimeBounds"][:], float)
            if np.any(np.diff(bounds[:, 0]) < 0):
                raise ValueError("historic windows out of order")
            names = [v.tobytes().decode().strip("\x00 ") for v in ds["sourceFilename"][:]]
            source = ds["waveSourceIndex"]
            quality = ds["waveFlagPrimary"]
            for item in group.itertuples():
                mid = pd.Timestamp(item.mid_utc).timestamp()
                i = np.searchsorted(bounds[:, 0], mid, side="right") - 1
                r = {
                    "clip_id": item.clip_id,
                    "station": platform,
                    "status": "outside_historic_window",
                }
                if i >= 0 and bounds[i, 0] <= mid < bounds[i, 1]:
                    j = int(source[i]) - 1
                    if quality[i] == 1 and 0 <= j < len(names):
                        r.update(
                            status="archived",
                            deployment=names[j],
                            start_s=bounds[i, 0],
                            end_s=bounds[i, 1],
                        )
                    else:
                        r["status"] = "invalid_bulk_reference"
                records.append(r)
    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", type=Path, help="deployment and UTC bounds per clip")
    p.add_argument("--clips", type=Path, required=True)
    p.add_argument("--limit", type=int, help="number of distinct windows for acquisition QA")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--dates",
        type=Path,
        help="local dates to label (default: the development dates). Sealed dates (storm set, "
        "confirmation) only once their bars are committed: docs/experiments/sensing_per_wave_bars.md",
    )
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    days = set(
        (args.dates or repo_root() / "configs/experiments/webcoos_video_dev_dates.txt")
        .read_text()
        .split()
    )
    clips = pd.read_csv(
        args.clips,
        usecols=["clip_id", "local_day", "mid_utc", "cdip_station"],
        dtype={"cdip_station": str},
    )
    clips = clips[clips.local_day.isin(days)]
    windows = (
        pd.read_csv(args.windows, dtype={"station": str})
        if args.windows
        else historic_windows(clips)
    )
    frame = clips.merge(windows, on="clip_id", validate="1:1", how="left")
    print(
        "development clip coverage", frame.status.value_counts(dropna=False).to_dict(), flush=True
    )
    mid = pd.to_datetime(frame.mid_utc, utc=True, format="ISO8601").map(lambda t: t.timestamp())
    rows = frame[(frame.status == "archived") & mid.ge(frame.start_s) & mid.lt(frame.end_s)].copy()
    keys = ["station", "deployment", "start_s", "end_s"]
    unique = rows[keys].drop_duplicates()
    if args.limit:
        unique = unique.head(args.limit)
    labels = []
    with Downloader(min_interval_s=1, timeout_s=40) as dl:
        slices = Subsets(dl)
        for r in unique.itertuples():
            f = r.deployment.replace(".nc", "_zc.nc")
            origin = int(
                re.search(r"indexStartTime,\s*(\d+)", slices.query(r.station, f, "indexStartTime"))[
                    1
                ]
            )
            keys_ix = []
            for t in [r.start_s, r.end_s]:
                i = int(t) - origin
                if i < 0:
                    raise ValueError("window precedes deployment")
                ix = int(array(slices.query(r.station, f, f"indexKey[{i}]"), "indexKey")[0]) - 1
                keys_ix.append(ix)
            first, last = keys_ix
            if first < 0 or last <= first:
                raise ValueError("no valid crossing indexes")
            ss = f"[{first}:1:{last}]"
            fields = ["zcrossTimeBounds" + ss + "[0:1:1]"] + [
                n + ss for n in ["zcrossZPeak", "zcrossHeight", "zcrossPeriod", "zcrossFlagPrimary"]
            ]
            body = slices.query(r.station, f, ",".join(fields))
            labels.append({**{k: getattr(r, k) for k in keys}, **targets(body, r.start_s, r.end_s)})
    if not labels:
        raise ValueError("no eligible development windows")
    output = rows.merge(pd.DataFrame(labels), on=keys, validate="m:1")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False, mode="x")
    print(len(output), "clips;", len(labels), "independent reference windows")


if __name__ == "__main__":
    main()
