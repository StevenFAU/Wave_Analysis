"""Has the Currituck camera gone hazy, and does haze explain the calm-day misses?

Two checks on a finished run of ``scripts/train_sea_state.py`` (inference
only, no training):

1. **Sharpness of every still**: the variance of the Laplacian over the sea
   band (rows 27-60 % of the 900 x 509 package still, grey levels). Haze, a
   film on the housing or softer compression lower it. Reported by year as
   the share of stills below the 2022-2024 thirds.
2. **Fog test**: the run's val stills (weeks of 2022-2024) blurred (Gaussian,
   sigma in package pixels) and veiled (blended with a milky beige), then
   predicted again with the run's weights. If haze of the size seen in
   2025-2026 makes the model call calm seas "slight", its calm calls on
   buoy Hs 0.3-0.5 m drop here as they did on the test years.

Writes ``<run>/haze_check/sharpness.csv`` and ``report.md``.

    .venv-rocm/bin/python scripts/sea_state_haze_check.py --run runs/<year-split run>
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

from wave_analysis.config import load_yaml
from wave_analysis.evaluation.classification import classification_report
from wave_analysis.models.sea_state import TrainSettings, build_model, build_transforms, crop

BAND = (0.27, 0.60)
VEIL = np.array([205, 200, 190], dtype=np.float32)
#: (name, blur sigma in px, veil weight)
LEVELS = [
    ("none", 0.0, 0.0),
    ("blur 0.55", 0.55, 0.0),
    ("blur 0.7", 0.7, 0.0),
    ("blur 0.55 + veil 0.3", 0.55, 0.3),
    ("veil 0.3", 0.0, 0.3),
    ("blur 1.0", 1.0, 0.0),
    ("blur 1.5 + veil 0.4", 1.5, 0.4),
]


def sharpness(im: Image.Image) -> float:
    """Variance of the 4-neighbour Laplacian over the sea band, grey levels 0-255."""
    g = np.asarray(im.convert("L"), dtype=np.float32)
    h = g.shape[0]
    b = g[round(h * BAND[0]) : round(h * BAND[1])]
    lap = 4 * b[1:-1, 1:-1] - b[:-2, 1:-1] - b[2:, 1:-1] - b[1:-1, :-2] - b[1:-1, 2:]
    return float(lap.var())


def _sharpness_of(path: str) -> float:
    with Image.open(path) as im:
        return sharpness(im)


def haze(im: Image.Image, sigma: float, veil: float) -> Image.Image:
    """``im`` blurred by ``sigma`` px and blended with a milky veil of weight ``veil``."""
    if sigma > 0:
        im = im.filter(ImageFilter.GaussianBlur(sigma))
    if veil > 0:
        x = np.asarray(im, dtype=np.float32)
        im = Image.fromarray(((1 - veil) * x + veil * VEIL).clip(0, 255).astype(np.uint8))
    return im


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run directory (year split)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)
    run = Path(args.run)
    prov = load_yaml(run / "provenance.yaml")
    paths = [str(f["path"] if isinstance(f, dict) else f) for f in prov["input_files"]]
    package = Path(next(p for p in paths if p.endswith("labels.csv"))).parent
    out = run / "haze_check"
    out.mkdir(exist_ok=True)
    lines = [f"# Haze check, {run.name}\n"]

    lab = pd.read_csv(package / "labels.csv")
    with ProcessPoolExecutor(8) as ex:
        lab["sharpness"] = list(
            ex.map(_sharpness_of, [str(package / p) for p in lab["image"]], chunksize=256)
        )
    lab[["sample_id", "sharpness"]].to_csv(out / "sharpness.csv", index=False)
    early = lab[lab["local_year"] <= 2024]
    cuts = early["sharpness"].quantile([1 / 3, 2 / 3]).to_numpy()
    third = pd.cut(lab["sharpness"], [-np.inf, *cuts, np.inf], labels=["blurry", "mid", "sharp"])
    by_year = pd.crosstab(lab["local_year"], third, normalize="index").round(2)
    by_year.insert(0, "median", lab.groupby("local_year")["sharpness"].median().round(0))
    by_class = pd.crosstab(early["class_name"], third[early.index], normalize="index").round(2)
    lines += [
        f"Sea-band sharpness; thirds cut at {cuts[0]:.0f} and {cuts[1]:.0f} (2022-2024 stills).\n",
        "## By year (share of stills in each 2022-2024 third)\n",
        "```",
        by_year.to_string(),
        "```\n",
        "## 2022-2024, by class\n",
        "```",
        by_class.to_string(),
        "```\n",
    ]

    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    st = TrainSettings(**ck["settings"])
    model = build_model(st.backbone, len(ck["class_names"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    torch.backends.cudnn.benchmark = False  # MIOpen's run-time compile fails on ROCm
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).to(memory_format=torch.channels_last).eval()
    tf = build_transforms(st, tuple(ck["input_size_hw"]), train=False)
    pred = pd.read_csv(run / "predictions.csv")
    val = lab[lab["sample_id"].isin(pred.loc[pred["partition"] == "val", "sample_id"])]
    val = val.reset_index(drop=True)
    y = val["class"].to_numpy()
    calm = val["Hs_label_m"].between(0.3, 0.5).to_numpy()
    lines += [
        f"## Fog test on the val stills (n={len(val):,})\n",
        "```",
        "level                  median sharpness  calm (buoy 0.3-0.5 m) called smooth  "
        "recall smooth/slight/moderate/rough  bal. acc.",
    ]
    for name, sigma, veil in LEVELS:
        probs, batch, sharp = [], [], []
        for i, path in enumerate(val["image"]):
            with Image.open(package / path) as im:
                img = haze(im.convert("RGB"), sigma, veil)
            sharp.append(sharpness(img))
            batch.append(tf(crop(img, st)))
            if len(batch) == args.batch or i == len(val) - 1:
                x = torch.stack(batch).to(dev).to(memory_format=torch.channels_last)
                with torch.no_grad():
                    probs.append(torch.softmax(model(x).float(), dim=1).cpu().numpy())
                batch = []
        yhat = np.concatenate(probs).argmax(axis=1)
        rec = " / ".join(f"{np.mean(yhat[y == c] == c):.2f}" for c in range(4))
        ba = float(classification_report(y, yhat, n_classes=4)["balanced_accuracy"])  # type: ignore[arg-type]
        line = (
            f"{name:22s} {np.median(sharp):16.0f}  {np.mean(yhat[calm] == 0):35.2f}  "
            f"{rec:>35s}  {ba:9.3f}"
        )
        lines.append(line)
        print(line, flush=True)
    lines.append("```\n")
    text = "\n".join(lines)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
