"""Index the local stereo bursts and identify CDIP displacement windows for filmed clips."""

import argparse
import bisect
import copy
import io
import re
import struct
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from wave_analysis.config import data_dir
from wave_analysis.physics.timeseries import zero_crossing_analysis


def stereo(out):
    rows = []
    base = data_dir("external") / "schwendeman_thomson_2017_ship_stereo"
    for archive in sorted(base.glob("StereoXYZData_*.zip")):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                parts = Path(member.filename).parts
                if (
                    len(parts) != 5
                    or parts[3] != "XYZResults"
                    or not re.fullmatch(r"ZGrid_\d+\.mat", parts[4])
                ):
                    continue
                group = "/".join(parts[1:3])
                rows.append(
                    {
                        "archive": archive.name,
                        "member": member.filename,
                        "burst": group,
                        "day": parts[1],
                        "minute": int(Path(parts[4]).stem.split("_")[-1]),
                        "unpacked_bytes": member.file_size,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no stereo surface fields found")
    df["overlapping_copy"] = df.duplicated(["burst", "minute"], keep=False)
    df.to_csv(out, index=False, mode="x")


@lru_cache(maxsize=2)
def local_offsets(path):
    """Recover physical local headers where ZIP32 offsets wrap inside these large archives."""
    offsets = {}
    with zipfile.ZipFile(path) as archive, path.open("rb") as raw:
        for entry in archive.infolist():
            guess = entry.header_offset % 2**32
            for offset in range(guess, archive.start_dir, 2**32):
                raw.seek(offset)
                header = raw.read(30)
                if len(header) != 30 or not header.startswith(b"PK\x03\x04"):
                    continue
                length = struct.unpack_from("<H", header, 26)[0]
                encoding = "utf-8" if entry.flag_bits & 0x800 else "cp437"
                if raw.read(length) == entry.orig_filename.encode(encoding):
                    offsets[entry.filename] = offset
                    break
            else:
                raise ValueError(f"no physical header for {entry.filename}")
        bounds = sorted([*offsets.values(), archive.start_dir])
    return {
        name: (start, bounds[bisect.bisect_right(bounds, start)]) for name, start in offsets.items()
    }


def read_stereo(row):
    path = data_dir("external") / "schwendeman_thomson_2017_ship_stereo" / row.archive
    with zipfile.ZipFile(path) as archive:
        entry = copy.copy(archive.getinfo(row.member))
        entry.header_offset, entry._end_offset = local_offsets(path)[entry.filename]
        payload = archive.read(entry)  # stdlib checks overlap, filename and CRC
        record = loadmat(io.BytesIO(payload), variable_names=["imgrid", "zgrid"])
    im = np.asarray(record["imgrid"])
    z = np.asarray(record["zgrid"])
    if im.shape != z.shape or im.shape[:2] != (161, 161):
        raise ValueError("unexpected stereo grid")
    return im, z


def stereo_sample(index, out):
    row = pd.read_csv(index).iloc[0]
    im, z = read_stereo(row)
    # Keep paired radiance and height at native cadence, plus a validity mask.
    mask = np.isfinite(im) & np.isfinite(z)
    np.savez_compressed(
        out,
        imgrid=im,
        zgrid=z,
        valid=mask,
        dx_m=0.25,
        fps=im.shape[-1] / 60,
        burst=row.burst,
        member=row.member,
    )


def windows(out):
    from netCDF4 import Dataset

    clips = pd.read_csv(
        data_dir("processed") / "packages/webcoos_video_v1/clips.csv", dtype={"cdip_station": str}
    )
    rows = []
    for sid, group in clips.groupby("cdip_station"):
        st = sid.zfill(3)
        name = st + "p1"
        with Dataset(data_dir("raw") / "cdip" / st / (name + "_historic.nc")) as ds:
            bounds = np.asarray(ds["waveTimeBounds"][:], float)
            middle = bounds.mean(axis=1)
            names = [r.tobytes().decode().strip("\x00 ") for r in ds["sourceFilename"][:]]
            indexes = np.asarray(ds["waveSourceIndex"][:], int) - 1
            good = np.asarray(ds["waveFlagPrimary"][:]) == 1
        for r in group.itertuples():
            mid = pd.Timestamp(r.mid_utc).timestamp()
            j = min(
                np.clip(
                    [middle.searchsorted(mid) - 1, middle.searchsorted(mid)], 0, len(middle) - 1
                ),
                key=lambda ix: abs(middle[ix] - mid),
            )
            row = {
                "clip_id": r.clip_id,
                "station": st,
                "historic_offset_s": middle[j] - mid,
                "status": "outside_historic_window",
            }
            if abs(middle[j] - mid) <= 1800 and good[j] and 0 <= indexes[j] < len(names):
                deployment = names[indexes[j]]
                row.update(
                    status="archived",
                    deployment=deployment,
                    start_s=bounds[j, 0],
                    end_s=bounds[j, 1],
                    opendap=f"https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/{name}/{deployment}",
                )
            rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False, mode="x")


def displacement(index, source, out):
    from netCDF4 import Dataset

    table = pd.read_csv(index)
    results = []
    for deployment, group in table[table.status.eq("archived")].groupby("deployment"):
        path = source / deployment
        if not path.exists():
            continue
        with Dataset(path) as ds:
            fs = float(ds["xyzSampleRate"][...])
            start = float(ds["xyzStartTime"][...]) - float(ds["xyzFilterDelay"][...])
            z = ds["xyzZDisplacement"]
            flags = ds["xyzFlagPrimary"]
            for r in group.itertuples():
                lo, hi = np.ceil((np.array([r.start_s, r.end_s]) - start) * fs - 1e-7).astype(int)
                row = {
                    "clip_id": r.clip_id,
                    "deployment": deployment,
                    "record_s": r.end_s - r.start_s,
                    "status": "bounds_or_qc",
                }
                if 0 <= lo < hi <= len(z):
                    eta = np.ma.asarray(z[lo:hi]).filled(np.nan)
                    q = np.ma.asarray(flags[lo:hi]).filled(9)
                    if np.isfinite(eta).all() and (q == 1).all():
                        try:
                            result = zero_crossing_analysis(eta, fs)
                            row.update(
                                status="ok",
                                mean_height_m=result.H_mean,
                                max_height_m=result.H_max,
                                tz_s=result.T_z,
                                n_waves=result.n_waves,
                            )
                        except ValueError:
                            row["status"] = "no_complete_waves"
                results.append(row)
    if not results:
        raise ValueError("no matched local deployment files")
    pd.DataFrame(results).to_csv(out, index=False, mode="x")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="task", required=True)
    for task in ["stereo", "stereo-sample", "cdip-windows", "cdip-displacement"]:
        p = sub.add_parser(task)
        p.add_argument("--out", type=Path, required=True)
        if task in ["stereo-sample", "cdip-displacement"]:
            p.add_argument("--index", type=Path, required=True)
        if task == "cdip-displacement":
            p.add_argument("--source", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.task == "stereo":
        stereo(a.out)
    elif a.task == "stereo-sample":
        stereo_sample(a.index, a.out)
    elif a.task == "cdip-windows":
        windows(a.out)
    else:
        displacement(a.index, a.source, a.out)


if __name__ == "__main__":
    main()
