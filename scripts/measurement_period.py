"""Train and compare crest-arrival detectors or frozen video period probes."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from torch import nn

from measurement_buoy_probe import ridge
from measurement_shore import FPS, arrivals
from train_video_probe import folds_by_date


def event_scores(ref, pred, tol):
    i = j = 0
    pairs = []
    while i < len(ref) and j < len(pred):
        if pred[j] < ref[i] - tol:
            j += 1
        elif pred[j] > ref[i] + tol:
            i += 1
        else:
            pairs.append((i, j))
            i += 1
            j += 1
    interval_errors = [
        abs((ref[b] - ref[a]) - (pred[d] - pred[c]))
        for (a, c), (b, d) in itertools.pairwise(pairs)
        if b == a + 1 and d == c + 1
    ]
    return {
        "hits": len(pairs),
        "observed": len(ref),
        "called": len(pred),
        "interval_abs_error_sum_s": sum(interval_errors),
        "matched_intervals": len(interval_errors),
    }


def blocks(frame, split):
    if split == "site":
        return frame.camera.replace(
            {
                "oakisland_east": "oakisland",
                "oakisland_west": "oakisland",
                "jennette_south": "jennette",
                "currituck_hampton_inn": "currituck",
            }
        )
    days = pd.to_datetime(frame.local_day)
    return days.dt.to_period("W").astype(str) if split == "week" else frame.local_day


def crest_net():
    return nn.Sequential(
        nn.Conv2d(1, 12, (9, 7), padding=(4, 3)),
        nn.ReLU(),
        nn.AvgPool2d((1, 4)),
        nn.Conv2d(12, 12, (9, 3), padding=(4, 1)),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((None, 1)),
        nn.Conv2d(12, 1, 1),
    )


def crest(args):
    index = pd.read_csv(args.stacks / "index.csv").set_index("clip_id", verify_integrity=True)
    annotations = pd.read_csv(args.annotations, keep_default_na=False)
    reviewed = annotations[annotations.reader.eq(args.reader) & annotations.crest_seconds.ne("")]
    if reviewed.clip_id.duplicated().any():
        raise ValueError("one reviewed interval per clip and reader required")
    grouping = blocks(index, args.split)
    records = []
    for row in reviewed.itertuples():
        stack = np.load(args.stacks / (row.clip_id + ".npy"))
        start, end = float(row.start_s), float(row.end_s)
        if not 0 <= start < end <= len(stack) / FPS:
            raise ValueError("invalid annotation window")
        ref = (
            np.array([])
            if row.crest_seconds == "NONE"
            else np.array([float(v) for v in row.crest_seconds.split()])
        )
        if np.any(np.diff(ref) <= 0) or ((ref < start) | (ref >= end)).any():
            raise ValueError("invalid crest times")
        i, j = round(start * FPS), round(end * FPS)
        if not np.allclose([i, j], [start * FPS, end * FPS]):
            raise ValueError("annotation limits must be frame aligned")
        s = stack[i:j]
        x = torch.from_numpy(((s - s.mean()) / max(s.std(), 1)).astype("float32"))[None, None]
        t = np.arange(len(s)) / FPS
        ref = ref - start
        # Half-second soft positives around reader timestamps.
        target = (
            np.exp(-((t[:, None] - ref) ** 2) / (2 * 0.5**2)).max(1)
            if len(ref)
            else np.zeros(len(s))
        )
        records.append(
            (
                row.clip_id,
                grouping.loc[row.clip_id],
                s,
                ref,
                x,
                torch.tensor(target, dtype=torch.float32),
            )
        )
    days = {r[1] for r in records}
    if len(days) < 2:
        raise ValueError("need reviewed intervals on at least two distinct local dates")
    scores = []
    for day in sorted(days):
        torch.manual_seed(0)
        model = crest_net().to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        train = [r for r in records if r[1] != day]
        if not any(len(r[3]) for r in train):
            raise ValueError("training dates contain no marked arrivals")
        for _ in range(args.epochs):
            for _, _, _, _, x, y in train:
                logits = model(x.to(args.device)).flatten()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, y.to(args.device), pos_weight=logits.new_tensor(3)
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        for clip, d, s, ref, x, _ in records:
            if day != d:
                continue
            with torch.no_grad():
                prob = model(x.to(args.device)).flatten().sigmoid().cpu().numpy()
            cnn = find_peaks(prob, height=0.5, distance=FPS * 2.5)[0] / FPS
            for arm, pred in [("detector", arrivals(s)), ("cnn", cnn)]:
                scores.append(
                    {
                        "clip_id": clip,
                        "held_block": day,
                        "arm": arm,
                        **event_scores(ref, pred, args.tolerance),
                    }
                )
    return pd.DataFrame(scores)


def tm02(args):
    clips = pd.read_csv(args.package / "clips.csv")
    pixel = pd.read_csv(args.package / "pixel_spectrum.csv")
    ids = pd.read_csv(args.features / "clips.csv")[["clip_id"]].assign(
        feature_row=lambda d: np.arange(len(d))
    )
    samples = clips.merge(pixel, on="clip_id", validate="1:1").merge(
        ids, on="clip_id", validate="1:1"
    )
    cols = ["log_T_pix", "log_T_m01", "log_T_m02", "rate_share", "log10_band_power"]
    target = "Tm02_label_s"
    extra = ["T_m02_s"]
    if getattr(args, "displacement", None):
        disp = pd.read_csv(args.displacement)
        target = args.target
        qc = ["good", "includes_unevaluated"] if args.allow_unevaluated else ["good"]
        disp = disp[disp.status.isin(qc) & np.isclose(disp.record_duration_s, args.duration)]
        extra += [
            "status",
            "record_duration_s",
            "qc_good_fraction",
            "deployment",
            "reference_start_s",
            "reference_end_s",
        ]
        samples = samples.merge(disp[["clip_id", target, *extra[1:]]], on="clip_id", validate="1:1")
    samples = samples.dropna(subset=[*cols, target]).reset_index(drop=True)
    if samples.local_day.nunique() < 2:
        raise ValueError("need two dates with matched inputs, target, duration and requested QC")
    y = samples[target].to_numpy()
    group = blocks(samples, args.split)
    if group.nunique() < 2:
        raise ValueError("need at least two independent blocks")
    folds = (
        pd.factorize(group, sort=True)[0]
        if args.split == "site"
        else folds_by_date(group.to_numpy(), 5)
    )
    data = {"pixel": samples[cols].to_numpy()}
    for arm in ("video", "still", "shuffled"):
        arr = np.load(args.features / (arm + ".npy"), mmap_mode="r")
        if len(arr) != len(ids):
            raise ValueError("feature count inconsistent with cache index")
        # Reduce each clip separately to keep host memory bounded.
        data[arm] = np.stack(
            [
                np.asarray(arr[i], dtype=np.float32).mean(axis=(0, 2)).ravel()
                for i in samples.feature_row
            ]
        )
    results = samples[["clip_id", "camera", "local_day", target, *extra]].copy()
    results["fold"] = folds
    for fold in np.unique(folds):
        tr = folds != fold
        te = ~tr
        if getattr(args, "displacement", None):
            refs = pd.MultiIndex.from_frame(samples[["deployment", "reference_start_s"]])
            tr &= ~refs.isin(refs[te])
            if tr.sum() < 2:
                raise ValueError("insufficient independent reference windows in a training fold")
        results.loc[te, "climatology"] = y[tr].mean()
        for arm, a in data.items():
            results.loc[te, arm] = np.maximum(0, ridge(a[tr], y[tr], a[te]))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="task", required=True)
    c = sub.add_parser("crest")
    c.add_argument("--stacks", type=Path, required=True)
    c.add_argument("--annotations", type=Path, required=True)
    c.add_argument("--reader", required=True)
    c.add_argument(
        "--tolerance",
        required=True,
        type=float,
        help="seconds, fix from independent reader agreement",
    )
    c.add_argument("--epochs", type=int, default=15)
    c.add_argument("--device", default="cpu")
    a = sub.add_parser("tm02")
    a.add_argument("--package", type=Path, required=True)
    a.add_argument(
        "--features",
        type=Path,
        required=True,
        help="registered midpoint feature cache with clips.csv",
    )
    a.add_argument("--displacement", type=Path, help="optional CDIP displacement.csv")
    a.add_argument(
        "--target", choices=["mean_height_m", "max_height_m", "tz_s"], default="mean_height_m"
    )
    a.add_argument(
        "--duration", type=float, default=1600, help="fixed displacement window in seconds"
    )
    a.add_argument(
        "--allow-unevaluated",
        action="store_true",
        help="include provider flag 2 in sensitivity study",
    )
    for sp in (a, c):
        sp.add_argument("--split", choices=["date", "week", "site"], default="date")
        sp.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if (
        args.task == "tm02"
        and not args.displacement
        and (args.allow_unevaluated or args.target != "mean_height_m")
    ):
        p.error("--displacement is required for displacement options")
    result = crest(args) if args.task == "crest" else tm02(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, mode="x")


if __name__ == "__main__":
    main()
