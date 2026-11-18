"""Test 3 of the buoy cameras: the best estimate from everything at the buoy - camera, wind and ERA5.

Choices fixed on 2026-11-18, before any v2 run was scored (``docs/experiments/
ndbc_buoycam_sea_state.md``, "Test 3"):

* **Inputs per strip:** the camera's three heights (the v2 arm with the lower
  *validation* RMSE of H_s, its seeds' geometric mean), the logs of ERA5's total,
  wind-sea and swell heights at the buoy (interpolated, no value across 3 h), and the
  buoy's own wind (log of max(speed, 0.5 m/s), log of max(gust, 0.5 m/s)).
* **Combiner:** ridge regression on standardised features, alpha 1.0, one per target
  (log H_s, log wind-wave height); fitted on the **val buoys**, scored once on the test
  buoys, on strips where every input exists.
* **The bar:** camera + ERA5 + wind must beat **ERA5 + wind** (the same ridge without
  the camera, fitted on the same strips) in RMSE of H_s **and** of the wind-wave height,
  each difference's 95 % interval below zero (paired bootstrap over whole test buoys).
  This asks whether the camera adds anything a buoy's anemometer and a wave model
  do not already give.
* **Reported beside it:** ERA5 alone (ridge), the camera alone, and ERA5 with a line on
  the training strips (the v2 V2-2 comparison).

    python scripts/buoycam_combined.py --arm images runs/<A seeds> --arm horizon runs/<B seeds> --out runs/<name>

Writes ``<out>/combined.json`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir, load_yaml, repo_root
from wave_analysis.datasets.sea_state import nowcast_at
from wave_analysis.evaluation.regression import rmse
from wave_analysis.evaluation.uncertainty import paired_block_bootstrap_ci

TARGETS = {"hs": "Hs_label_m", "wind_wave": "wind_wave_label_m"}
ERA5 = ("era5_hs_m", "era5_windsea_m", "era5_swell_m")
ALPHA = 1.0
U_MIN = 0.5


def geometric_mean(runs: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(r / "predictions.csv", dtype={"station_id": str}).sort_values("sample_id").reset_index(drop=True)
              for r in runs]  # fmt: skip
    out = frames[0][["sample_id", "partition", "station_id"]].copy()
    for h in ("hs", "wind_wave", "swell"):
        out[f"cam_{h}"] = np.exp(
            np.mean([np.log(f[f"pred_{h}_m"].to_numpy()) for f in frames], axis=0)
        )
    return out


def val_rmse(d: pd.DataFrame, labels: pd.DataFrame) -> float:
    v = d[d["partition"] == "val"].merge(labels[["sample_id", "Hs_label_m"]], on="sample_id")
    return rmse(v["Hs_label_m"].to_numpy(), v["cam_hs"].to_numpy())


def ridge_fit(
    x: np.ndarray, y: np.ndarray, alpha: float = ALPHA
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    ym = y.mean()
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - ym))
    return mu, sd, coef, ym


def ridge_predict(model: tuple, x: np.ndarray) -> np.ndarray:
    mu, sd, coef, ym = model
    return ((x - mu) / sd) @ coef + ym


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", nargs="+", action="append", required=True, metavar=("NAME", "RUN"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs/experiments/ndbc_buoycam_heights_v2.yaml",
    )
    ap.add_argument(
        "--era5", type=Path, default=data_dir("processed") / "era5/buoycam/era5_at_buoys.parquet"
    )
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    b = cfg["evaluation"]["bootstrap"]
    kw = {"n_boot": int(b["n_boot"]), "alpha": float(b["alpha"]), "seed": int(b["seed"])}
    labels = pd.read_csv(
        repo_root() / cfg["package"]["out"] / "labels.csv", dtype={"station_id": str}
    )

    arms = {a[0]: geometric_mean([Path(r) for r in a[1:]]) for a in args.arm}
    vr = {n: val_rmse(d, labels) for n, d in arms.items()}
    chosen = min(vr, key=vr.get)
    d = arms[chosen].merge(
        labels[["sample_id", "time_utc", *TARGETS.values(), "wind_speed_mps", "wind_gust_mps"]],
        on="sample_id",
    )
    d["time_utc"] = pd.to_datetime(d["time_utc"], utc=True)
    era = pd.read_parquet(args.era5)
    for c in ERA5:
        x = np.full(len(d), np.nan)
        for sid, g in era.groupby("station_id"):
            rows = (d["station_id"] == str(sid)).to_numpy()
            if rows.any():
                x[rows] = nowcast_at(
                    d.loc[rows, "time_utc"], g.set_index("time_utc")[c].dropna(), max_gap="3h"
                )
        d[c] = x
    f_era = [np.log(np.maximum(d[c].to_numpy(), 0.05)) for c in ERA5]
    f_wind = [
        np.log(np.maximum(d[c].to_numpy(), U_MIN)) for c in ("wind_speed_mps", "wind_gust_mps")
    ]
    f_cam = [np.log(d[f"cam_{h}"].to_numpy()) for h in ("hs", "wind_wave", "swell")]
    sets = {
        "camera + ERA5 + wind": np.column_stack(f_cam + f_era + f_wind),
        "ERA5 + wind (the bar)": np.column_stack(f_era + f_wind),
        "ERA5 alone": np.column_stack(f_era),
        "camera + wind": np.column_stack(f_cam + f_wind),
    }
    res: dict[str, Any] = {"arm": chosen, "val_rmse_hs_by_arm": vr, "alpha": ALPHA, "targets": {}}
    for tname, tcol in TARGETS.items():
        y = np.log(np.maximum(d[tcol].to_numpy(dtype=float), 0.05))
        ok = np.isfinite(y) & np.all([np.isfinite(x).all(axis=1) for x in sets.values()], axis=0)
        fit = ok & (d["partition"] == "val").to_numpy()
        te = ok & (d["partition"] == "test").to_numpy()
        truth = np.exp(y[te])
        preds = {
            n: np.exp(ridge_predict(ridge_fit(x[fit], y[fit]), x[te])) for n, x in sets.items()
        }
        preds["camera alone"] = d.loc[te, f"cam_{tname}"].to_numpy()
        g = d.loc[te, "station_id"].to_numpy()
        r: dict[str, Any] = {"n_fit": int(fit.sum()), "n_test": int(te.sum()), "test_buoys": len(set(g)),
                             "rmse_m": {n: rmse(truth, p) for n, p in preds.items()}, "gain_vs": {}}  # fmt: skip
        best = preds["camera + ERA5 + wind"]
        for other in [n for n in preds if n != "camera + ERA5 + wind"]:
            v, lo, hi = paired_block_bootstrap_ci(truth, best, preds[other], rmse, groups=g, **kw)
            r["gain_vs"][other] = [float(v), float(lo), float(hi)]
        r["beats_bar"] = bool(r["gain_vs"]["ERA5 + wind (the bar)"][2] < 0)
        res["targets"][tname] = r
    res["bar_met"] = all(res["targets"][t]["beats_bar"] for t in TARGETS)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "combined.json").write_text(json.dumps(res, indent=1, default=str))
    lines = [f"# Test 3: camera + ERA5 + wind (camera arm: {chosen}, chosen on val RMSE {vr})\n"]
    for tname, r in res["targets"].items():
        lines.append(
            f"## {tname}: fitted on {r['n_fit']:,} val strips, scored on {r['n_test']:,} test strips ({r['test_buoys']} buoys)\n"
        )
        lines += [f"- {n}: RMSE {v:.3f} m" for n, v in r["rmse_m"].items()]
        lines += [
            f"- camera + ERA5 + wind minus {n}: {v[0]:+.3f} m [{v[1]:+.3f}, {v[2]:+.3f}]"
            for n, v in r["gain_vs"].items()
        ]
        lines.append("")
    lines.append(f"**Test 3 {'met' if res['bar_met'] else 'not met'}.**")
    text = "\n".join(lines) + "\n"
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
