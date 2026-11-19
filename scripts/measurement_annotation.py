"""Export real observation packets for two independent crest readers and buoy foam masks."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from measurement_shore import FPS, sample_frames
from wave_analysis.config import data_dir


def crest(args):
    index = pd.read_csv(args.stacks / "index.csv")
    args.out.mkdir(parents=True, exist_ok=False)
    tasks = []
    for r in index.itertuples():
        n = len(np.load(args.stacks / (r.clip_id + ".npy"), mmap_mode="r"))
        video = args.out / (r.clip_id + ".mp4")
        stream = None
        try:
            for i, image in enumerate(sample_frames(r)):
                if i >= n:
                    break
                if stream is None:
                    stream = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-loglevel",
                            "error",
                            "-nostdin",
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            "rgb24",
                            "-s",
                            f"{image.width}x{image.height}",
                            "-r",
                            str(FPS),
                            "-i",
                            "pipe:0",
                            "-an",
                            "-vf",
                            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                            "-c:v",
                            "libx264",
                            "-threads",
                            "1",
                            "-pix_fmt",
                            "yuv420p",
                            str(video),
                        ],
                        stdin=subprocess.PIPE,
                    )
                x = (image.width - 1) * (r.x0 + r.x1) / 2
                y = (image.height - 1) * (r.y0 + r.y1) / 2
                draw = ImageDraw.Draw(image)
                draw.line([(x - 10, y), (x - 3, y)], fill="yellow", width=1)
                draw.line([(x + 3, y), (x + 10, y)], fill="yellow", width=1)
                draw.text(
                    (3, 2), f"{i / FPS:.2f} s", fill="yellow", stroke_width=1, stroke_fill="black"
                )
                stream.stdin.write(image.tobytes())
        finally:
            if stream:
                stream.stdin.close()
                if stream.wait():
                    raise RuntimeError(f"ffmpeg failed for {r.clip_id}")
        tasks.append({"clip_id": r.clip_id, "start_s": 0, "end_s": n / FPS, "crest_seconds": ""})
    for reader in ("A", "B"):
        pd.DataFrame(tasks).assign(reader=reader).to_csv(
            args.out / f"reader-{reader}.csv", index=False
        )


def foam(args):
    src = data_dir("processed") / "packages/ndbc_buoycam_spectral_v3cv"
    d = pd.read_csv(src / "labels.csv", low_memory=False).query("split_station == 'train'")
    horizons = pd.read_parquet(
        data_dir("processed") / "horizon/ndbc_buoycam_sea_state_v0/horizons.parquet"
    ).query("ok")
    d = d.merge(horizons[["sample_id", "view", "row_frac"]], on="sample_id")
    args.out.mkdir(parents=True, exist_ok=False)
    # Distinct station per lighting stratum until exhausted. Keep physical labels blind.
    chosen = []
    for _, g in d.groupby("illumination"):
        chosen.append(
            g.sample(frac=1, random_state=19).drop_duplicates("station_id").head(args.per_light)
        )
    packet = pd.concat(chosen)
    records = []
    for r in packet.itertuples():
        stem = f"{r.sample_id}_v{r.view}"
        with Image.open(src / r.image) as im:
            w = im.width // 6
            view = im.crop((w * r.view, 0, w * (r.view + 1), im.height))
            view.save(args.out / (stem + ".png"))
        records.append(
            {
                "sample_id": r.sample_id,
                "view": r.view,
                "illumination": r.illumination,
                "image": stem + ".png",
                "mask": stem + ".mask.png",
            }
        )
    pd.DataFrame(records).to_csv(args.out / "views.csv", index=False)


def reviewed_masks(packet):
    views = pd.read_csv(packet / "views.csv")
    for r in views.itertuples():
        path = packet / r.mask
        if not path.is_file():
            continue
        with Image.open(path) as im, Image.open(packet / r.image) as original:
            if im.size != original.size:
                raise ValueError(f"mask shape mismatch {path}")
            mask = np.asarray(im)
        if mask.ndim != 2 or not np.isin(mask, [0, 1, 2, 255]).all():
            raise ValueError(f"invalid mask codes {path}")
        yield r, mask, ((mask == 1) | (mask == 2))


def fractions(args):
    rows = []
    for r, mask, water in reviewed_masks(args.packet):
        if water.any():
            rows.append(
                {
                    "sample_id": r.sample_id,
                    "view": r.view,
                    "visible_water_pixels": int(water.sum()),
                    "foam_pixels": int((mask == 2).sum()),
                    "visible_foam_fraction": (mask == 2).sum() / water.sum(),
                }
            )
    if not rows:
        raise ValueError("no completed masks with labeled water")
    pd.DataFrame(rows).to_csv(args.out, index=False, mode="x")


def foam_probe(args):
    """Predict foam within reader-delineated visible water on held-out station groups."""
    from scipy.ndimage import uniform_filter

    from measurement_buoy_probe import ridge

    labels = pd.read_csv(args.labels, usecols=["sample_id", "station_group"]).set_index("sample_id")
    samples = []
    rng = np.random.default_rng(0)
    for r, mask, water in reviewed_masks(args.packet):
        if not water.any():
            continue
        with Image.open(args.packet / r.image) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
        light = rgb.mean(2)
        features = np.column_stack(
            [light[water], rgb[water], (light - uniform_filter(light, 9))[water]]
        )
        target = (mask[water] == 2).astype(float)
        subset = rng.choice(len(target), min(len(target), 2048), replace=False)
        samples.append((r, labels.loc[r.sample_id, "station_group"], features, target, subset))
    groups = {s[1] for s in samples}
    if len(groups) < 2:
        raise ValueError("need completed masks at two or more independent station groups")
    rows = []
    for group in sorted(groups):
        train = [s for s in samples if s[1] != group]
        x = np.concatenate([s[2][s[4]] for s in train])
        y = np.concatenate([s[3][s[4]] for s in train])
        for r, g, image, truth, _ in samples:
            if g != group:
                continue
            row = {
                "sample_id": r.sample_id,
                "view": r.view,
                "station_group": g,
                "illumination": r.illumination,
                "visible_water_pixels": len(truth),
                "visible_foam_fraction": truth.mean(),
                "training_mean": y.mean(),
            }
            for name, n in [("brightness", 1), ("color_texture", x.shape[1])]:
                row[name] = np.clip(ridge(x[:, :n], y, image[:, :n]), 0, 1).mean()
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.out, index=False, mode="x")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="task", required=True)
    c = sub.add_parser("crest")
    c.add_argument("--stacks", required=True, type=Path)
    f = sub.add_parser("foam")
    f.add_argument("--per-light", type=int, default=8)
    g = sub.add_parser("fractions")
    g.add_argument("--packet", required=True, type=Path)
    m = sub.add_parser("foam-probe")
    m.add_argument("--packet", required=True, type=Path)
    m.add_argument("--labels", required=True, type=Path)
    for a in (c, f, g, m):
        a.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    {"crest": crest, "foam": foam, "fractions": fractions, "foam-probe": foam_probe}[args.task](
        args
    )


if __name__ == "__main__":
    main()
