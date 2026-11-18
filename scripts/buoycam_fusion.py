"""Test 2 of the buoy cameras: does the camera's wind-sea call add to ERA5 for the total sea state?

The camera sees the wind sea, not the swell (Q-M8); ERA5 carries the swell. Choices fixed
on 2026-11-18, before any v1 model was trained (``docs/experiments/ndbc_buoycam_sea_state.md``):

* **Camera:** a v1 run (wind-wave classes; normally the average of three seeds,
  ``scripts/ensemble_runs.py``) on the station split.
* **Target:** the total sea-state class of the same strip (the v0 package's label: WMO
  classes of the buoy's total significant wave height).
* **Combiner:** multinomial logistic regression (``wave_analysis.models.fusion``) on the
  camera's four class log-probabilities (clipped at 1e-6) and the logarithms of ERA5's
  total, wind-sea and swell heights at the buoy (interpolated to the strip, no value
  across a gap over 3 h); inverse class-frequency weights; L2 penalty 1e-3; no tuning.
* **Fitted on the val buoys** (strips where all three ERA5 heights exist), scored once on
  the test buoys.
* **The bar:** against the same regression on the three ERA5 heights alone, fitted on the
  same val strips, the camera + ERA5 combiner must gain in balanced accuracy *and*
  macro-F1, each with its 95 % interval above zero (paired bootstrap over whole test
  buoys). Reported beside it: the v0 package's ERA5 baseline (total height, line fitted
  on the training strips) and the camera alone is not comparable (other classes).

Known caveats: the val buoys carry less swell (mean share 0.40) than the test buoys
(0.67), so both combiners learn on calmer swell climates than they are scored on; and the
camera's epochs were chosen on the same val buoys, which makes the combiner trust the
camera slightly too much, against the fusion on the test buoys.

**Test 2b** (fixed 2026-11-18 after Test 2 was scored, before this variant was run):
everything as above except that *neither* combiner uses class weights
(``--class-weights none``). Test 2 failed on macro-F1 through smooth-sea false alarms,
which the weights encourage; 2b asks whether the camera's gains survive without them.
Because Test 2's result suggested it, 2b is a follow-up, not a confirmation.

    python scripts/buoycam_fusion.py --run runs/<v1 ensemble> [--class-weights none]

Writes ``<run>/fusion/`` (Test 2) or ``<run>/fusion_unweighted/`` (Test 2b):
``fusion.json`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir, load_yaml, repo_root
from wave_analysis.datasets.sea_state import nowcast_at
from wave_analysis.evaluation.classification import classification_report
from wave_analysis.evaluation.sea_state import format_line, report
from wave_analysis.evaluation.uncertainty import paired_block_bootstrap_ci
from wave_analysis.models.fusion import fit_logistic, inverse_frequency_weights

L2 = 1e-3
P_FLOOR = 1e-6
MAX_GAP = "3h"
GAINS = ("balanced_accuracy", "f1_macro")
ERA5 = ("era5_hs_m", "era5_windsea_m", "era5_swell_m")


def _metric(name: str, k: int) -> Any:
    def f(a: np.ndarray, b: np.ndarray) -> float:
        rep = classification_report(a.astype(np.int64), b.astype(np.int64), n_classes=k)
        return float(rep[name])  # type: ignore[arg-type]

    return f


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path, help="a v1 station-split run or ensemble")
    ap.add_argument(
        "--package-v0",
        type=Path,
        default=repo_root() / "data/processed/packages/ndbc_buoycam_sea_state_v0",
    )
    ap.add_argument(
        "--era5", type=Path, default=data_dir("processed") / "era5/buoycam/era5_at_buoys.parquet"
    )
    ap.add_argument(
        "--class-weights", choices=("inverse_frequency", "none"), default="inverse_frequency",
        help="weights of both combiners: Test 2 (inverse_frequency) or Test 2b (none)",
    )  # fmt: skip
    args = ap.parse_args(argv)
    run = args.run
    info = json.loads((run / "metrics.json").read_text())["run"]
    if info["split"] != "station":
        raise SystemExit("Test 2 is fixed on the station split")
    cfg0 = load_yaml(args.package_v0 / "config.yaml")
    names = list(cfg0["task"]["classes"]["names"])
    k = len(names)
    cfg1 = load_yaml(repo_root() / "configs/experiments" / f"{info['config']}.yaml")
    boot = cfg1["evaluation"]["bootstrap"]
    kw = {"n_boot": int(boot["n_boot"]), "alpha": float(boot["alpha"]), "seed": int(boot["seed"])}

    cam = pd.read_csv(run / "predictions.csv")
    cam = cam[cam["partition"].isin(["val", "test"])]
    pcols = [c for c in cam.columns if c.startswith("p_")]
    lab0 = pd.read_csv(
        args.package_v0 / "labels.csv",
        usecols=["sample_id", "station_id", "time_utc", "class", "near_boundary", "split_station"],
        dtype={"station_id": str},
    )
    base0 = pd.read_csv(args.package_v0 / "baselines.csv", usecols=["sample_id", "era5_station"])
    d = (
        cam[["sample_id", "partition", *pcols]]
        .merge(lab0, on="sample_id")
        .merge(base0, on="sample_id")
    )
    if not (d["partition"] == d["split_station"]).all():
        raise SystemExit("the run's partitions differ from the v0 package's")
    d["time_utc"] = pd.to_datetime(d["time_utc"], utc=True)
    era = pd.read_parquet(args.era5)
    missing = [c for c in ERA5 if c not in era.columns]
    if missing:
        raise SystemExit(
            f"{args.era5} lacks {missing}: download those ERA5 parameters, then extract"
        )
    for c in ERA5:
        x = np.full(len(d), np.nan)
        for sid, g in era.groupby("station_id"):
            rows = (d["station_id"] == str(sid)).to_numpy()
            if rows.any():
                x[rows] = nowcast_at(
                    d.loc[rows, "time_utc"], g.set_index("time_utc")[c].dropna(), max_gap=MAX_GAP
                )
        d[c] = x
    ok = np.all([(d[c] > 0).to_numpy() & np.isfinite(d[c].to_numpy()) for c in ERA5], axis=0)
    coverage = {p: float(ok[(d["partition"] == p).to_numpy()].mean()) for p in ("val", "test")}
    d = d[ok].reset_index(drop=True)

    x_era = np.log(d[list(ERA5)].to_numpy())
    x_cam = np.log(np.clip(d[pcols].to_numpy(), P_FLOOR, None))
    x_both = np.hstack([x_cam, x_era])
    y = d["class"].to_numpy(dtype=np.int64)
    fit = (d["partition"] == "val").to_numpy()
    w = inverse_frequency_weights(y[fit], k) if args.class_weights == "inverse_frequency" else None
    both = fit_logistic(x_both[fit], y[fit], k, sample_weight=w, l2=L2)
    alone = fit_logistic(x_era[fit], y[fit], k, sample_weight=w, l2=L2)

    te = (d["partition"] == "test").to_numpy()
    t = d[te]
    preds = {
        "camera + ERA5": both.predict(x_both[te]),
        "ERA5 alone, refitted on val (the bar)": alone.predict(x_era[te]),
        "ERA5 baseline of v0 (total height, line on training strips)": t["era5_station"].to_numpy(
            dtype=np.int64
        ),
    }
    g = t["station_id"].to_numpy()
    # A baseline is scored only on the strips it covers (-1: none, e.g. the v0
    # package's ERA5 ended earlier than the ERA5 table used here); the camera +
    # ERA5 combiner is scored beside it on the same strips.
    covered = {n: p >= 0 for n, p in preds.items()}
    reps = {
        n: report(t["class"][c], p[c], groups=g[c], names=names,
                  near_boundary=t["near_boundary"][c], **kw)
        for (n, p), c in zip(preds.items(), covered.values(), strict=True)
    }  # fmt: skip
    gains: dict[str, Any] = {}
    fused = preds["camera + ERA5"]
    for other in list(preds)[1:]:
        c = covered[other]
        for m in GAINS:
            v, lo, hi = paired_block_bootstrap_ci(
                t["class"].to_numpy()[c],
                fused[c],
                preds[other][c],
                _metric(m, k),
                groups=g[c],
                **kw,
            )
            gains[f"{m} vs {other}"] = {"value": v, "lo": lo, "hi": hi, "n": int(c.sum())}
    bar_keys = [f"{m} vs ERA5 alone, refitted on val (the bar)" for m in GAINS]
    met = all(gains[key]["lo"] > 0 for key in bar_keys)
    results = {
        "run": str(run),
        "choices": {
            "l2": L2,
            "p_floor": P_FLOOR,
            "fit_on": "val buoys",
            "class_weights": args.class_weights,
            "features": "log p_camera, log ERA5 total, wind sea, swell",
            "max_gap": MAX_GAP,
        },
        "coverage_all_three_era5": coverage,
        "fit_strips": int(fit.sum()),
        "fit_buoys": int(d.loc[fit, "station_id"].nunique()),
        "test_strips": int(te.sum()),
        "test_buoys": int(t["station_id"].nunique()),
        "metrics": reps,
        "gains_of_camera_plus_era5": gains,
        "bar_met": met,
    }
    test = "Test 2" if args.class_weights == "inverse_frequency" else "Test 2b"
    out = run / ("fusion" if args.class_weights == "inverse_frequency" else "fusion_unweighted")
    out.mkdir(exist_ok=True)
    (out / "fusion.json").write_text(json.dumps(results, indent=1, default=str))
    lines = [
        f"# {test}, camera + ERA5 (class weights: {args.class_weights}), {run.name}\n",
        f"Combiners fitted on {results['fit_strips']:,} val strips ({results['fit_buoys']} buoys); "
        f"scored on {results['test_strips']:,} test strips ({results['test_buoys']} buoys). "
        f"ERA5 coverage: {coverage}.\n",
        "```",
    ]
    lines += [format_line(n, r, groups="buoys") for n, r in reps.items()]
    lines += [""] + [
        f"gain in {key} (n={v['n']:,}): {v['value']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
        for key, v in gains.items()
    ]
    lines += ["```", f"\n**Bar {'met' if met else 'not met'}.**\n"]
    text = "\n".join(lines)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
