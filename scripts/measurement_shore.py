"""Prepare transect annotation packets and extract 4 Hz time-distance images."""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
from scipy.signal import butter, find_peaks, sosfiltfilt

from wave_analysis.config import data_dir

FPS = 4


def archive(row):
    return data_dir("processed") / "video/windows_4fps" / row.camera / (row.clip_id + ".zip")


def sample_frames(row):
    with zipfile.ZipFile(archive(row)) as z:
        frames = sorted(n for n in z.namelist() if n.endswith(".jpg"))
        if frames != [f"{i:04d}.jpg" for i in range(len(frames))]:
            raise ValueError(f"{row.clip_id}: missing or unordered frames")
        for name in frames:
            with Image.open(io.BytesIO(z.read(name))) as im:
                yield im.convert("RGB")


def packet(out, per_camera):
    p = data_dir("processed") / "packages/webcoos_video_v1"
    clips = pd.read_csv(p / "clips.csv")
    # Use all *finished* local archives, including clips beyond the current packed whole-clip index.
    clips = clips[
        clips.apply(
            lambda r: archive(r).is_file() and archive(r).with_suffix(".json").is_file(), axis=1
        )
    ]
    clips = clips.sort_values(["camera", "sun_elevation_deg", "clip_id"]).drop_duplicates(
        ["camera", "local_day"], keep="last"
    )
    if clips.empty:
        raise ValueError("no packaged clips with completed 4 fps archives")
    out.mkdir(parents=True, exist_ok=False)
    tasks = []
    for camera, g in clips.groupby("camera"):
        for i in np.linspace(0, len(g) - 1, min(per_camera, len(g)), dtype=int):
            r = g.iloc[i]
            image = next(sample_frames(r))
            draw = ImageDraw.Draw(image)
            for frac in [0.2, 0.4, 0.6, 0.8]:
                x, y = int(frac * image.width), int(frac * image.height)
                draw.line([(x, 0), (x, image.height)], fill="yellow", width=1)
                draw.line([(0, y), (image.width, y)], fill="yellow", width=1)
            image.save(out / (r.clip_id + ".jpg"))
            tasks.append(
                {
                    "clip_id": r.clip_id,
                    "camera": camera,
                    "local_day": r.local_day,
                    "x0": np.nan,
                    "y0": np.nan,
                    "x1": np.nan,
                    "y1": np.nan,
                    "orientation": "",
                    "approved": False,
                }
            )
    pd.DataFrame(tasks).to_csv(out / "transects.csv", index=False)
    pd.DataFrame(columns=["clip_id", "reader", "start_s", "end_s", "crest_seconds"]).to_csv(
        out / "arrivals.csv", index=False
    )


def trace(row, max_s):
    coords = np.array([row.x0, row.y0, row.x1, row.y1], dtype=float)
    if (
        not np.isfinite(coords).all()
        or (coords < 0).any()
        or (coords > 1).any()
        or np.allclose(coords[:2], coords[2:])
    ):
        raise ValueError("transect requires distinct normalized endpoints inside image")
    signal = []
    for k, frame in enumerate(sample_frames(row)):
        if max_s and k >= FPS * max_s:
            break
        grey = np.asarray(frame.convert("L"), dtype=float)
        x = np.linspace(row.x0, row.x1, 64) * (frame.width - 1)
        y = np.linspace(row.y0, row.y1, 64) * (frame.height - 1)
        signal.append(map_coordinates(grey, [y, x], order=1))
    return np.stack(signal).astype(np.float32)


def arrivals(stack):
    # Arrival reference is the midpoint of the fixed transect; brighter crests give local maxima.
    s = stack[:, 30:34].mean(axis=1)
    if len(s) < 32:
        return np.array([])
    filt = sosfiltfilt(butter(3, [0.04, 0.4], fs=FPS, btype="bandpass", output="sos"), s)
    peaks, _ = find_peaks(filt, distance=2.5 * FPS, prominence=max(1e-6, 0.5 * filt.std()))
    return peaks / FPS


def stacks(table, out, seconds):
    rows = pd.read_csv(table)
    rows = rows[rows.approved.astype(str).str.lower().eq("true")]
    if rows.empty or rows.clip_id.duplicated().any():
        raise ValueError("one approved transect per clip is required for this first comparison")
    out.mkdir(parents=True, exist_ok=False)
    for row in rows.itertuples():
        stack = trace(row, seconds)
        np.save(out / (row.clip_id + ".npy"), stack)
        # Vertical coordinate is time down the page; arrow marks the arrival reference.
        lo, hi = np.percentile(stack, [2, 98])
        pic = Image.fromarray(np.uint8(255 * np.clip((stack - lo) / max(hi - lo, 1), 0, 1))).resize(
            (256, len(stack))
        )
        pic.save(out / (row.clip_id + ".png"))
    rows.to_csv(out / "index.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)
    p = sub.add_parser("packet")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-camera", type=int, default=3)
    p = sub.add_parser("stacks")
    p.add_argument("--transects", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=0, help="0 takes all complete windows")
    args = parser.parse_args()
    if args.task == "packet":
        packet(args.out, args.per_camera)
    else:
        stacks(args.transects, args.out, args.seconds)


if __name__ == "__main__":
    main()
