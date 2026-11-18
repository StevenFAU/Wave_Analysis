"""The buoy camera as a motion sensor: does the horizon's tilt follow the buoy's own wave slope? (Q-M5)

Exploratory; nothing is trained. Questions fixed on 2026-11-18, before the answers were
looked at:

A. **Motion against slope.** At one buoy, do strips whose horizons tilt more (roll) or
   sit further from their usual height (pitch) come from steeper seas? The slope is the
   buoy's own: the mean-square slope of its measured spectrum, mss = sum S(f) k(f)^2 df,
   k = (2 pi f)^2 / g (deep water), averaged over the records within 30 min of the strip
   (ADR 0010). Measured as the Spearman correlation within each buoy, pooled by ranking
   within buoys, with a 95 % interval from a bootstrap over whole buoys.
B. **Which waves move the hull.** The same correlation of the motion with the wind-wave
   height and the swell height (NDBC's split). The image texture follows the wind sea
   (Q-M8); if the tilt also follows the swell, it carries what the texture misses.
C. **One instant or six.** Do adjacent views of one strip agree on the buoy's motion
   (correlation of their roll), as one rigid ring of cameras photographed at one instant
   would? If not, a strip holds up to six separate samples of the motion.

Per view (``wave_analysis.processing.horizon.detect_horizon``): the horizon's angle and
height; views the config's light rule rejects, or whose fit is unclear, are left out.
Each camera's mounting is removed by subtracting its median angle and height over all
its strips (per buoy, camera and view). A strip's motion: the root mean square of its
remaining roll (degrees) and pitch (share of the view's height) over at least three
clear views.

    python scripts/buoycam_horizon.py [--package <dir>] [--workers N]

Writes ``data/processed/horizon/<package>/{horizons.parquet,strips.parquet,summary.json}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

from wave_analysis.config import data_dir, load_yaml, repo_root
from wave_analysis.datasets.buoycam import snapshot_stamp, usable_view_mask
from wave_analysis.processing.horizon import detect_horizon
from wave_analysis.processing.spectra import frequency_bandwidths
from wave_analysis.processing.synchronize import label_from_candidates, pair_window
from wave_analysis.sources.ndbc.buoycam import N_VIEWS
from wave_analysis.sources.ndbc.parsers import parse_spectral

G = 9.80665
MIN_VIEWS = 3
MIN_STRIPS_PER_BUOY = 100


def _detect(job: tuple[str, str, list[bool]]) -> list[dict]:
    sample_id, path, usable = job
    gray = np.asarray(Image.open(path).convert("L"))
    w = gray.shape[1] // N_VIEWS
    rows = []
    for v in range(N_VIEWS):
        if not usable[v]:
            continue
        f = detect_horizon(gray[:, v * w : (v + 1) * w])
        rows.append({"sample_id": sample_id, "view": v, "angle_deg": f.angle_deg,
                     "row_frac": f.row_mid / gray.shape[0], "inlier_share": f.inlier_share,
                     "step": f.step, "ok": f.ok()})  # fmt: skip
    return rows


def spectral_slope(ndbc: Path, station: str) -> pd.DataFrame:
    """Mean-square slope and m0 of every realtime ``swden`` record of a station (latest snapshot wins)."""
    frames = []
    for p in sorted(
        (ndbc / station / "swden").glob(f"{station}.data_spec.*Z.gz"), key=snapshot_stamp
    ):
        for b in parse_spectral(p, "swden", realtime=True):
            s = np.nan_to_num(np.asarray(b.values, dtype=float), nan=0.0)
            f = np.asarray(b.frequency, dtype=float)
            df = frequency_bandwidths(f)
            k2 = ((2 * np.pi * f) ** 2 / G) ** 2
            frames.append(pd.DataFrame({
                "time_utc": pd.DatetimeIndex(b.time).tz_localize("UTC") if pd.DatetimeIndex(b.time).tz is None else pd.DatetimeIndex(b.time),
                "mss": s @ (k2 * df), "m0": s @ df, "snapshot": snapshot_stamp(p),
            }))  # fmt: skip
    if not frames:
        return pd.DataFrame(columns=["station_id", "time_utc", "mss", "m0", "start_utc", "end_utc"])
    rec = (
        pd.concat(frames)
        .sort_values(["time_utc", "snapshot"])
        .groupby("time_utc")
        .last()
        .reset_index()
    )
    rec.insert(0, "station_id", station)
    rec["start_utc"] = rec["time_utc"] - pd.Timedelta(minutes=10)
    rec["end_utc"] = rec["time_utc"] + pd.Timedelta(minutes=10)
    return rec


def within_buoy_spearman(
    d: pd.DataFrame, x: str, y: str, n_boot: int = 1000, seed: int = 0
) -> dict:
    """Spearman correlation of x and y after ranking within each buoy, with a bootstrap over buoys."""
    d = d[np.isfinite(d[x]) & np.isfinite(d[y])]
    counts = d["station_id"].value_counts()
    d = d[d["station_id"].isin(counts[counts >= MIN_STRIPS_PER_BUOY].index)].copy()
    for c in (x, y):
        d[f"r_{c}"] = d.groupby("station_id")[c].rank(pct=True)
    rho = float(np.corrcoef(d[f"r_{x}"], d[f"r_{y}"])[0, 1])
    per = d.groupby("station_id").apply(
        lambda g: stats.spearmanr(g[x], g[y])[0], include_groups=False
    )
    keys = d["station_id"].unique()
    parts = {k: d[d["station_id"] == k] for k in keys}
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        b = pd.concat([parts[k] for k in rng.choice(keys, keys.size)])
        reps.append(np.corrcoef(b[f"r_{x}"], b[f"r_{y}"])[0, 1])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"rho": rho, "ci": [float(lo), float(hi)], "buoys": int(keys.size), "strips": len(d),
            "per_buoy_median": float(per.median()), "per_buoy_positive": float((per > 0).mean())}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--package",
        type=Path,
        default=repo_root() / "data/processed/packages/ndbc_buoycam_sea_state_v0",
    )
    ap.add_argument("--ndbc", type=Path, default=data_dir("raw") / "ndbc")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    pkg = args.package
    out = data_dir("processed") / "horizon" / pkg.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(pkg / "config.yaml")
    lab = pd.read_csv(pkg / "labels.csv", dtype={"station_id": str})
    lab["time_utc"] = pd.to_datetime(lab["time_utc"], utc=True)
    usable = usable_view_mask(
        lab,
        min_mean=float(cfg["images"]["min_view_mean"]),
        min_std=float(cfg["images"]["min_view_std"]),
    )

    hz_path = out / "horizons.parquet"
    if hz_path.exists():
        hz = pd.read_parquet(hz_path)
    else:
        jobs = [(s, str(pkg / im), list(map(bool, u)))
                for s, im, u in zip(lab["sample_id"], lab["image"], usable, strict=True)]  # fmt: skip
        with Pool(args.workers) as pool:
            hz = pd.DataFrame([r for rows in pool.imap(_detect, jobs, chunksize=64) for r in rows])
        hz.to_parquet(hz_path, index=False)
    hz = hz.merge(lab[["sample_id", "station_id", "camera_code"]], on="sample_id")
    ok = hz[hz["ok"]].copy()
    key = ["station_id", "camera_code", "view"]
    ok["roll"] = ok["angle_deg"] - ok.groupby(key)["angle_deg"].transform("median")
    ok["pitch"] = ok["row_frac"] - ok.groupby(key)["row_frac"].transform("median")
    strip = ok.groupby("sample_id").agg(
        n_views=("roll", "size"), roll_rms_deg=("roll", lambda v: float(np.sqrt(np.mean(v**2)))),
        pitch_rms=("pitch", lambda v: float(np.sqrt(np.mean(v**2)))),
    ).reset_index()  # fmt: skip
    strip = strip[strip["n_views"] >= MIN_VIEWS]

    # the buoy's own slope, by the label rule (mean of the records within 30 min)
    slopes = []
    for st, g in lab.groupby("station_id"):
        rec = spectral_slope(args.ndbc, st)
        if rec.empty:
            continue
        pw = pair_window(g, rec, window="60min")
        cand = pd.concat([g.loc[pw["target_index"], ["sample_id"]].reset_index(drop=True),
                          pw.drop(columns=["target_index", "reference_index"]),
                          rec.loc[pw["reference_index"], ["mss"]].reset_index(drop=True)], axis=1)  # fmt: skip
        lab_mss = label_from_candidates(
            cand, "mss", rule="window_mean", target="sample_id", max_offset="30min"
        )
        slopes.append(lab_mss["label"].rename("mss"))
    mss = pd.concat(slopes)
    s = strip.merge(lab[["sample_id", "station_id", "Hs_label_m", "swell_height_m", "wind_wave_height_m"]],
                    on="sample_id").merge(mss, left_on="sample_id", right_index=True, how="left")  # fmt: skip
    s["slope_rms"] = np.sqrt(s["mss"])
    s.to_parquet(out / "strips.parquet", index=False)

    summary: dict = {
        "views_tried": len(hz), "views_clear": int(hz["ok"].sum()),
        "strips_with_motion": len(s), "strips_with_slope": int(s["slope_rms"].notna().sum()),
        "roll_rms_deg_median": float(s["roll_rms_deg"].median()),
        "A_roll_vs_slope": within_buoy_spearman(s, "roll_rms_deg", "slope_rms"),
        "A_pitch_vs_slope": within_buoy_spearman(s, "pitch_rms", "slope_rms"),
        "B_roll_vs_wind_wave": within_buoy_spearman(s, "roll_rms_deg", "wind_wave_height_m"),
        "B_roll_vs_swell": within_buoy_spearman(s, "roll_rms_deg", "swell_height_m"),
        "B_roll_vs_total": within_buoy_spearman(s, "roll_rms_deg", "Hs_label_m"),
        "B_pitch_vs_wind_wave": within_buoy_spearman(s, "pitch_rms", "wind_wave_height_m"),
        "B_pitch_vs_swell": within_buoy_spearman(s, "pitch_rms", "swell_height_m"),
    }  # fmt: skip
    # C: adjacent views of one strip, de-mounted roll
    wide = ok.pivot_table(index="sample_id", columns="view", values="roll")
    adj = [wide[[v, (v + 1) % N_VIEWS]].dropna() for v in range(N_VIEWS)]
    summary["C_adjacent_view_roll_corr"] = [
        round(float(np.corrcoef(a.iloc[:, 0], a.iloc[:, 1])[0, 1]), 3) for a in adj
    ]
    summary["C_opposite_view_roll_corr"] = [
        round(float(np.corrcoef(*wide[[v, v + 3]].dropna().to_numpy().T)[0, 1]), 3)
        for v in range(3)
    ]
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
