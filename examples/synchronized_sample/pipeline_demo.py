"""Offline end-to-end demo of the v0.1 pipeline on real NDBC fixture spectra (mock images)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wave_analysis.datasets import build_manifest, check_leakage, time_block_split
from wave_analysis.evaluation.regression import mae, regression_report
from wave_analysis.evaluation.uncertainty import block_bootstrap_ci
from wave_analysis.models import ClimatologyBaseline, RidgeBaseline
from wave_analysis.processing.solar import illumination_category, solar_position
from wave_analysis.processing.spectra import bulk_parameters
from wave_analysis.processing.synchronize import pair_nearest
from wave_analysis.sources.ndbc import assemble_spectrum, parse_spectral

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "ndbc" / "historical"
LAT, LON = 28.86, -78.478  # NDBC 41010


def run(verbose: bool = True) -> dict[str, float]:
    codes = {"swden": "w", "swdir": "d", "swdir2": "i", "swr1": "j", "swr2": "k"}
    blocks = {p: parse_spectral(FIX / f"41010{c}2023.txt.gz", p)[0] for p, c in codes.items()}
    waves = bulk_parameters(assemble_spectrum(blocks))[["time_utc", "Hm0", "Tp_smooth", "Dm"]]

    # Mock image table: minute 10 of every hour, like NDBC buoycam file names.
    t = pd.date_range("2023-03-01 00:10", "2023-03-05 23:10", freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    elev, _ = solar_position(t, LAT, LON)
    images = pd.DataFrame(
        {
            "sample_id": [f"41010_{x:%Y%m%dT%H%M}" for x in t],
            "station_id": "41010",
            "time_utc": t,
            "illumination": illumination_category(elev),
            "mean_luma": np.clip(
                0.05 + 0.5 * np.clip(elev, 0, None) / 60 + rng.normal(0, 0.02, len(t)), 0, 1
            ),
        }
    )

    paired = pair_nearest(images, waves, tolerance="45min", suspect_offset="25min")
    paired["day"] = paired["time_utc"].dt.strftime("%Y-%m-%d")
    paired["month"] = paired["time_utc"].dt.month
    split = time_block_split(
        paired, "time_utc", val_start="2023-03-04", test_start="2023-03-05", embargo="6h"
    )
    built = build_manifest(
        paired,
        id_col="sample_id",
        target_cols=["Hm0"],
        filters=[
            {"column": "sync_qc_flag", "op": "in", "value": [1, 3]},
            {"column": "time_offset_s", "op": "abs_le", "value": 1800},
        ],
        split=split,
    )
    audit = check_leakage(
        paired, split, time_col="time_utc", time_group_col="station_id", min_gap="3h"
    )

    m = built.manifest
    train, test = m[m["split"] == "train"], m[m["split"] == "test"]
    clim = ClimatologyBaseline(["station_id", "month"]).fit(train, "Hm0").predict(test)
    ridge = (
        RidgeBaseline(alpha=1.0)
        .fit(train[["mean_luma"]], train["Hm0"])
        .predict(test[["mean_luma"]])
    )
    rep_clim = regression_report(test["Hm0"], clim)
    point, lo, hi = block_bootstrap_ci(
        test["Hm0"].to_numpy(), ridge, mae, block_length=6, n_boot=500
    )

    if verbose:
        print(f"spectra: {len(waves)}  images: {len(images)}  paired rows: {len(paired)}")
        print(
            f"|Δt| median: {paired['time_offset_s'].abs().median():.0f} s; "
            f"sync flags: {paired['sync_qc_flag'].value_counts().to_dict()}"
        )
        print(f"illumination: {images['illumination'].value_counts().to_dict()}")
        print(
            f"manifest: {len(m)} samples, {len(built.exclusions)} excluded "
            f"({built.exclusions['excluded_by'].value_counts().to_dict()})"
        )
        print(f"split: {split.summary()}  leakage ok: {audit.ok}")
        print(
            f"climatology baseline on test: MAE {rep_clim['mae']:.3f} m, bias {rep_clim['bias']:+.3f} m"
        )
        print(
            f"ridge(mean_luma) baseline MAE {point:.3f} m [95% block-bootstrap CI {lo:.3f}, {hi:.3f}]"
        )
        print(
            "(Mock brightness carries no wave information, so neither baseline should show skill.)"
        )
    return {"n_manifest": float(len(m)), "leakage_ok": float(audit.ok), "clim_mae": rep_clim["mae"]}


if __name__ == "__main__":
    run()
