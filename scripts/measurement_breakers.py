"""Build wave-level Duck mask examples and the official site-held-out Yin clip index."""

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat

from wave_analysis.config import data_dir


def duck(out):
    source = data_dir("external") / "carini_duck2016_ir_lidar/lidarIR_sharedData.mat"
    keys = [
        "Hcat",
        "Tcat",
        "profilesMaskCat",
        "tcat",
        "tstartcat",
        "indNB",
        "indSS",
        "indSpill",
        "indPlunge",
    ]
    raw = loadmat(source, variable_names=keys, squeeze_me=True)
    starts = (np.asarray(raw["tstartcat"]) - 719529) * 86400
    type_of = {
        int(j) - 1: name
        for name, k in zip(
            ["nonbreaking", "steady", "spilling", "plunging"], keys[-4:], strict=True
        )
        for j in np.atleast_1d(raw[k])
    }
    masks = []
    rows = []
    for i, t in enumerate(raw["tcat"]):
        t = np.atleast_1d(t)
        mask = np.asarray(raw["profilesMaskCat"][i])
        if mask.ndim == 1:
            mask = mask[:, None]
        if mask.ndim != 2 or mask.shape[1] != len(t) or not len(t):
            continue
        run = int(np.searchsorted(starts, t.min() + 0.001, side="right") - 1)
        if run < 0 or t.max() - starts[run] > 310:
            continue
        h = np.asarray(raw["Hcat"][i], float)
        period = np.asarray(raw["Tcat"][i], float)
        if not np.isfinite(h).any() or not np.isfinite(period).any():
            continue
        # Area-resample the IR mask alone. Lidar profile geometry never enters the model.
        m = np.asarray(
            Image.fromarray((mask == 1).astype(np.float32)).resize((64, 64), Image.Resampling.BOX)
        )
        masks.append(m)
        rows.append(
            {
                "wave_id": i,
                "run_id": run,
                "utc_day": pd.Timestamp(t.min(), unit="s", tz="UTC").date(),
                "height_m": float(np.nanmedian(h)),
                "period_s": float(np.nanmedian(period)),
                "label": type_of.get(i, ""),
                "first_s": t.min(),
                "last_s": t.max(),
                "mask_fraction": m.mean(),
            }
        )
    if not rows:
        raise ValueError("no aligned tracks")
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "masks.npy", np.stack(masks))
    pd.DataFrame(rows).to_csv(out / "waves.csv", index=False)
    print(
        "Duck:",
        len(rows),
        "aligned tracks;",
        pd.Series([r["utc_day"] for r in rows]).value_counts().to_dict(),
    )


def yin(source, out):
    official = pd.read_csv(source / "loso_split_by_database_filename.csv")
    held = official[official.split.str.lower().eq("test")].copy()
    if held.empty or held.filename.duplicated().any() or not held.site.eq(held.held_out_site).all():
        raise ValueError("unexpected official held-site splits")
    with zipfile.ZipFile(source / "Set_A.zip") as z:
        members = [
            n
            for n in z.namelist()
            if n.lower().endswith(".mp4") and not Path(n).name.startswith("._")
        ]
    names = {Path(n).name: n for n in members}
    if len(names) != len(members):
        raise ValueError("duplicate clip basename in archive")
    held["zip_member"] = held.filename.map(names)
    if held.zip_member.isna().any():
        raise ValueError("official file missing in Set A")
    # Identify shared physical footage if using both datasets. A 30-second event can
    # overlap the end of the previous WebCOOS record or the start of the next one.
    shore = pd.read_csv(data_dir("processed") / "packages/webcoos_video_v1/clips.csv")
    shore["start"] = pd.to_datetime(shore.start_utc, utc=True)
    shore["end"] = shore.start + pd.to_timedelta(shore.duration_s, unit="s")
    overlaps = {}
    for cam, subset in held.groupby("database_camera"):
        records = shore[shore.camera == cam]
        for row in subset.itertuples():
            start = pd.Timestamp(row.timestamp_utc)
            match = records[
                (records.start < start + pd.Timedelta(seconds=30)) & (records.end > start)
            ]
            overlaps[row.filename] = "|".join(match.clip_id)
    held["webcoos_overlaps"] = held.filename.map(overlaps)
    out.mkdir(parents=True, exist_ok=False)
    held.to_csv(out / "clips.csv", index=False)
    print("Yin:", len(held), "clips;", held.groupby(["site", "true_label"]).size().to_dict())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="task", required=True)
    for name in ["duck", "yin"]:
        a = sub.add_parser(name)
        a.add_argument("--out", required=True, type=Path)
        if name == "yin":
            a.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    if args.task == "duck":
        duck(args.out)
    else:
        yin(args.source, args.out)


if __name__ == "__main__":
    main()
