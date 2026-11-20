#!/usr/bin/env python
"""Do the wave partitions add as energies? NDBC: WVHT² = SwH² + WWH²; ERA5: swh² = shww² + shts².

Stage 2 of the architecture review (partition-aware fusion) builds the total height
from ERA5's swell and the camera's wind sea, H_s² = swell² + wind-wave², which holds
only if the two parts split one spectrum. NDBC computes WVHT = 4 sqrt(m0) on board and
splits the same spectrum at a separation frequency into swell (SwH) and wind waves
(WWH) (NDBC Technical Document 96-01; Gilhousen & Hervey 2001), so the identity should
hold up to the 0.1 m rounding of the three reported values. This checks it on every
realtime ``.spec`` record of the camera buoys (latest snapshot wins), and the same for
ERA5's total, wind-sea and swell heights at the buoys.

* **Rounding bound** of a record: |WVHT² - SwH² - WWH²| <= 0.1 (WVHT + SwH + WWH) +
  0.0075 m², the most three values each off by at most 0.05 m can move it.
* Reported: the share of records inside that bound, the relative error of
  sqrt(SwH² + WWH²) against WVHT (median, 95th and 99th percentiles), and the same for
  ERA5 (no rounding in its archived values).

    python scripts/buoycam_partition_check.py [--out runs/partition-check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir
from wave_analysis.datasets.buoycam import spec_records


def summary(total: np.ndarray, a: np.ndarray, b: np.ndarray, rounding: float | None) -> dict:
    ok = np.isfinite(total) & np.isfinite(a) & np.isfinite(b) & (total > 0)
    t, a, b = total[ok], a[ok], b[ok]
    rel = np.sqrt(a**2 + b**2) / t - 1
    out = {
        "n": int(ok.sum()),
        "rel_error_median": float(np.median(rel)),
        "rel_error_abs_p95": float(np.percentile(np.abs(rel), 95)),
        "rel_error_abs_p99": float(np.percentile(np.abs(rel), 99)),
        "share_within_1pct": float(np.mean(np.abs(rel) <= 0.01)),
        "rel_error_abs_p99_hs_ge_2m": float(np.percentile(np.abs(rel[t >= 2.0]), 99)),
        "rel_error_abs_max_hs_ge_2m": float(np.max(np.abs(rel[t >= 2.0]))),
    }
    if rounding is not None:
        d = t**2 - a**2 - b**2
        bound = 2 * rounding * (t + a + b) + 3 * rounding**2
        out["share_within_rounding"] = float(np.mean(np.abs(d) <= bound + 1e-9))
        big = t >= 1.0
        out["share_within_rounding_hs_ge_1m"] = float(np.mean(np.abs(d[big]) <= bound[big] + 1e-9))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path("runs/partition-check"))
    ap.add_argument(
        "--labels",
        type=Path,
        default=data_dir("processed") / "packages/ndbc_buoycam_heights_v2cv/labels.csv",
    )
    ap.add_argument(
        "--era5", type=Path, default=data_dir("processed") / "era5/buoycam/era5_at_buoys.parquet"
    )
    args = ap.parse_args(argv)
    stations = sorted(
        pd.read_csv(args.labels, usecols=["station_id"], dtype=str)["station_id"].unique()
    )
    ndbc = data_dir("raw") / "ndbc"
    rec = pd.concat(
        [spec_records(sorted((ndbc / s / "spec").glob(f"{s}.spec.*Z.gz")), s) for s in stations],
        ignore_index=True,
    )
    res = {
        "ndbc": summary(rec["Hs_m"].to_numpy(float), rec["swell_height_m"].to_numpy(float),
                        rec["wind_wave_height_m"].to_numpy(float), rounding=0.05),
        "ndbc_stations": int(rec["station_id"].nunique()),
        "ndbc_period": [str(rec["time_utc"].min()), str(rec["time_utc"].max())],
    }  # fmt: skip
    era = pd.read_parquet(args.era5)
    res["era5"] = summary(era["era5_hs_m"].to_numpy(float), era["era5_windsea_m"].to_numpy(float),
                          era["era5_swell_m"].to_numpy(float), rounding=None)  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "partition_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
