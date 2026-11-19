"""Each view's test probabilities from a finished paper classification run.

For runs trained before the trainer saved per-view outputs (the three Choi VGG16
seeds): reloads the run's chosen epoch (``best.pt``) on the same test strips,
checks that the averaged probabilities reproduce its ``predictions.csv``, and
writes them again with ``view<k>_prob_<class>`` beside them. No training.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_sensing_papers import Strips, epoch, model_for
from wave_analysis.config import load_yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("package", "config", "out", "run"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--split", choices=["station"], required=True)
    p.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    p.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    p.add_argument("--num-workers", type=int, default=8)
    a = p.parse_args()
    if not torch.cuda.is_available():
        p.error("requires a GPU on the shared server")
    source = json.loads((a.run / "settings.json").read_text())
    sha = hashlib.sha256((a.package / "labels.csv").read_bytes()).hexdigest()
    if (source["split"], source["seed"], source["labels_sha256"]) != (a.split, a.seed, sha):
        raise ValueError("run, split, seed or package do not match")
    if source["arm"].startswith("yang"):
        raise ValueError("classification runs only")
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "predictions.csv").exists():
        raise FileExistsError(a.out)
    df = pd.read_csv(a.package / "labels.csv", low_memory=False, dtype={"station_id": str})
    test = Strips(a.package, df[df["split_" + a.split].eq("test")], source["arm"], None)
    loader = DataLoader(
        test, batch_size=load_yaml(a.config)["batch_size"], num_workers=a.num_workers
    )
    model = model_for(source["arm"])
    model.load_state_dict(torch.load(a.run / "best.pt", weights_only=True))
    pred, per_view = epoch(model.cuda(), loader, source["arm"], "cuda", a.amp)
    old = pd.read_csv(a.run / "predictions.csv", dtype={"station_id": str})
    out = test.frame[["sample_id", "station_group", "station_id"]].copy()
    out["target"] = test.target
    if not (out.sample_id.equals(old.sample_id) and np.allclose(out.target, old.target)):
        raise ValueError("test strips differ from the run's")
    for j, pr in enumerate(pred.T):
        out[f"prob_{j}"] = pr
    probs = sorted(old.filter(regex=r"^prob_\d+$").columns)
    gap = float(np.abs(out[probs].to_numpy() - old[probs].to_numpy()).max())
    agree = float((out[probs].to_numpy().argmax(1) == old[probs].to_numpy().argmax(1)).mean())
    for v in range(per_view.shape[1]):
        for j in range(per_view.shape[2]):
            out[f"view{v}_prob_{j}"] = per_view[:, v, j]
    settings = {
        **{k: source[k] for k in ("arm", "split", "seed", "labels_sha256", "recipe")},
        "source_run": a.run.name,
        "source_commit": source.get("commit"),
        "commit": os.environ.get("COMMIT"),
        "max_prob_difference": gap,
        "class_agreement": agree,
    }
    (a.out / "settings.json").write_text(json.dumps(settings, default=str, indent=2))
    out.to_csv(a.out / "predictions.csv", index=False)
    print(f"reproduced the run's test probabilities to {gap:.2e}; classes agree on {agree:.4f}")
    if agree < 0.999:
        raise ValueError("the reloaded model does not reproduce the run")


if __name__ == "__main__":
    main()
