"""Recreate the historical wind-wave figure and station 41040 image comparison.

From the repo root, run:
    uv run --locked --with matplotlib==3.11.2 python scripts/buoycam_fusion_review.py
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from wave_analysis.config import data_dir, repo_root
from wave_analysis.ingest.checksum import sha256_file

WIND = "true_wind_wave_m"
TOTAL = "true_hs_m"
STATION = "41040"
TEAL, RUST, INK, GRAY = "#0d706c", "#ae4d2c", "#18303c", "#58676f"


def group_scores(rows):
    rows = rows.copy()
    arms = ["forecast", "camera", "fusion"]
    squared = rows[arms].sub(rows["observed"], axis=0).pow(2)
    groups = [rows.target, rows.station_group]
    rmses = squared.groupby(groups).mean().pow(0.5).add_suffix("_rmse_m")
    counts = rows.groupby(["target", "station_group"]).agg(
        strips=("sample_id", "size"), buoys=("station_id", "nunique")
    )
    scores = counts.join(rmses)
    scores["reduction_cm"] = 100 * (scores.forecast_rmse_m - scores.fusion_rmse_m)
    return scores.reset_index()


def main_figure(scores, registered, output):
    fig = plt.figure(figsize=(10.5, 5.1), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.97,
        bottom=0.20,
        top=0.78,
        width_ratios=[0.92, 1.65],
        height_ratios=[1.8, 1],
        hspace=0.68,
        wspace=0.35,
    )
    fig.text(0.055, 0.94, "What buoy cameras add to the forecast", fontsize=18, weight="bold")
    fig.text(
        0.055,
        0.879,
        "ERA5 + GFS-Wave + measured wind    /    34,723 historical strips at 71 held-out buoys",
        color=GRAY,
    )
    ax = fig.add_subplot(grid[0, 0])
    ax.set_title("A   Pooled RMSE reduction", loc="left", pad=12, weight="bold")
    for y, target in [(1, WIND), (0, TOTAL)]:
        diff, lo, hi = registered[target]["by_group"]["fusion - forecast"]
        delta, lower, upper = -100 * diff, -100 * hi, -100 * lo
        percent = delta / registered[target]["rmse"]["forecast"]
        ax.hlines(y, lower, upper, color=TEAL, lw=3)
        ax.plot(delta, y, "o", color=TEAL, ms=7)
        ax.text(
            max(delta, 0.73),
            y + 0.24,
            f"{delta:.2f} cm ({percent:.1f}%)",
            ha="center",
            fontsize=10,
            weight="bold",
        )
    ax.set_yticks([0, 1], ["Total", "Wind-wave"])
    ax.set(
        ylim=(-0.4, 1.62),
        xlim=(-0.12, 2.87),
        xticks=[0, 1, 2],
        xlabel="Forecast RMSE − fusion RMSE (cm)",
    )
    ax.axvline(0, color=GRAY, lw=0.9)
    ax.grid(axis="x", color="#e0e6e8")
    ax.set_axisbelow(True)

    right = fig.add_subplot(grid[:, 1])
    s = scores[scores.target == WIND].sort_values("reduction_cm").reset_index(drop=True)
    rank = np.arange(1, len(s) + 1)
    gain = s.reduction_cm.to_numpy()
    right.set_title("B   Wind-wave reduction by station group", loc="left", pad=12, weight="bold")
    right.bar(rank, gain, width=0.78, color=np.where(gain < 0, RUST, TEAL))
    right.axhline(0, color=INK, lw=0.9)
    right.set(
        ylabel="RMSE reduction (cm)",
        xlabel="Station groups ranked by RMSE reduction",
        xlim=(0, len(s) + 1),
        xticks=[1, 10, 20, 30, len(s)],
    )
    right.grid(axis="y", color="#e0e6e8")
    right.set_axisbelow(True)
    i = s.index[s.station_group == "g" + STATION].item()
    right.annotate(
        f"Buoy {STATION}\n{gain[i]:+.2f} cm",
        (rank[i], gain[i]),
        xytext=(rank[i] + 5, gain[i] + 0.45),
        ha="left",
        va="center",
        fontsize=9,
        color=RUST,
        arrowprops={"arrowstyle": "-", "color": RUST},
    )
    right.text(
        0.99,
        0.045,
        f"{sum(gain > 0)} of {len(s)} groups improved",
        transform=right.transAxes,
        ha="right",
        weight="bold",
        fontsize=10,
    )

    fresh = fig.add_subplot(grid[1, 0])
    fresh.axis("off")
    fresh.text(0, 0.70, "C   Fresh-strip confirmation", weight="bold", fontsize=11)
    fresh.text(0, 0.23, "Pending the registered ERA5 cutoff.", color=GRAY, fontsize=10)
    fig.text(
        0.055,
        0.061,
        "Positive values indicate lower error with the camera. Panel A intervals are paired 95% station-group bootstrap intervals.",
        fontsize=9,
        color=GRAY,
    )
    fig.savefig(output / "forecast-fusion.png", dpi=260)
    fig.savefig(output / "forecast-fusion.svg", metadata={"Date": None})
    plt.close(fig)


def station_data(rows, labels, package):
    selected = rows[(rows.station_id == STATION) & (rows.target == WIND)]
    labels = labels[labels.station_id == STATION]
    site = selected.merge(labels, on=["sample_id", "station_id"], validate="1:1")
    if len(site) != len(selected) or not np.allclose(
        site.observed, site.wind_wave_label_m, atol=1e-5
    ):
        raise ValueError("station labels differ from the registered scoring rows")
    site["error_m"] = site.fusion - site.observed
    views = pd.read_csv(package / "views.csv")
    views = views[views.sample_id.isin(site.sample_id) & views.ok].copy()
    views["abs_roll_deg"] = views.roll_deg.abs()
    views["abs_pitch_frac"] = views.pitch_frac.abs()
    posture = views.groupby("sample_id")[["abs_roll_deg", "abs_pitch_frac"]].median()
    site = site.join(posture, on="sample_id")
    means = site[[f"view_mean_{k}" for k in range(6)]].to_numpy()
    stds = site[[f"view_std_{k}" for k in range(6)]].to_numpy()
    usable = (means >= 20) & (stds >= 4)
    site["brightness"] = np.nanmean(np.where(usable, means, np.nan), axis=1)
    site["contrast"] = np.nanmean(np.where(usable, stds, np.nan), axis=1)
    return site


def examples(site):
    """Three worst dates, each paired to a different accurate strip at comparable sea height."""
    cases = site.sort_values(["error_m", "sample_id"]).drop_duplicates("station_day").head(3)
    accurate = site[site.error_m.abs() <= 0.10].copy()
    selected = []
    for number, (_, case) in enumerate(cases.iterrows(), start=1):
        pool = accurate[
            (accurate.station_day != case.station_day)
            & (accurate.illumination == case.illumination)
        ].copy()
        pool["height_gap"] = (pool.observed - case.observed).abs()
        pool["sun_gap"] = (pool.sun_elevation_deg - case.sun_elevation_deg).abs()
        control = pool.sort_values(["height_gap", "sun_gap", "sample_id"]).iloc[0]
        accurate = accurate[accurate.sample_id != control.sample_id]
        for row, kind in [(case, "Underestimate"), (control, "Accurate")]:
            selected.append({**row.to_dict(), "pair": number, "selection": kind})
    return pd.DataFrame(selected)


def image_sheet(pairs, package, output):
    fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 10.2))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.035, top=0.91, hspace=0.5)
    fig.suptitle("Buoy 41040  /  wind-wave height", x=0.025, ha="left", fontsize=18, weight="bold")
    fig.text(
        0.025,
        0.944,
        "Three large underestimates and three same-buoy comparisons. Heights in metres.",
        color=GRAY,
    )
    for ax, row in zip(axes, pairs.itertuples(), strict=True):
        path = package / row.image
        if sha256_file(path) != row.sha256_package:
            raise ValueError(f"strip checksum mismatch for {row.sample_id}")
        with Image.open(path) as strip:
            ax.imshow(np.asarray(strip))
        ax.set_anchor("W")
        ax.set_title(
            f"{row.pair}   {row.selection}    ·    {row.time_utc[:16].replace('T', ' ')} UTC"
            f"      |      buoy {row.observed:.2f}     forecast {row.forecast:.2f}     camera {row.camera:.2f}     fusion {row.fusion:.2f}",
            loc="left",
            fontsize=10,
            color=RUST if row.selection == "Underestimate" else INK,
        )
        ax.axis("off")
    fig.savefig(output / "41040-strips.png", dpi=175, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=repo_root() / "reports/figures/buoycam-fusion")
    args = parser.parse_args()
    run = repo_root() / "runs/forecast-fusion-nested-v1"
    package = data_dir("processed") / "packages/ndbc_buoycam_heights_v2cv"
    rows = pd.read_parquet(run / "rows.parquet").query(
        "arrangement == 'nested' and baseline == 'model_wind'"
    )
    ids = rows.groupby("target").sample_id.agg(set)
    if rows.duplicated(["sample_id", "target"]).any() or ids[WIND] != ids[TOTAL]:
        raise ValueError("expected paired and unique scoring strips")
    registered = json.loads((run / "summary.json").read_text())["test1"]["nested"]["model_wind"]
    scores = group_scores(rows)
    for target in (WIND, TOTAL):
        block = scores[scores.target == target]
        for arm in ("forecast", "fusion"):
            pooled = np.sqrt(np.average(block[f"{arm}_rmse_m"] ** 2, weights=block.strips))
            if round(pooled, 4) != registered[target]["rmse"][arm]:
                raise ValueError(f"{target} {arm} disagrees with registered RMSE")

    args.out.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "buoycam-fusion-review",
            "font.family": "DejaVu Sans",
        }
    )
    main_figure(scores, registered, args.out)
    scores.to_csv(args.out / "station-groups.csv", index=False, float_format="%.6f")
    labels = pd.read_csv(package / "labels.csv", dtype={"station_id": str}, low_memory=False)
    wind_rows = rows[rows.target == WIND].merge(
        labels[["sample_id", "wind_speed_mps"]], on="sample_id", how="left", validate="1:1"
    )
    if wind_rows.wind_speed_mps.isna().any():
        raise ValueError("missing measured wind on registered strips")
    windy_others = wind_rows[(wind_rows.station_id != STATION) & (wind_rows.wind_speed_mps > 7)]
    group_scores(windy_others).to_csv(
        args.out / "other-groups-above-7mps.csv", index=False, float_format="%.6f"
    )
    site = station_data(rows, labels, package)
    pairs = examples(site)
    image_sheet(pairs, package, args.out)
    columns = [
        "sample_id",
        "time_utc",
        "image",
        "observed",
        "forecast",
        "camera",
        "fusion",
        "error_m",
        "wind_speed_mps",
        "swell_label_m",
        "sun_elevation_deg",
        "illumination",
        "usable_views",
        "brightness",
        "contrast",
        "abs_roll_deg",
        "abs_pitch_frac",
    ]
    site[columns].merge(
        pairs[["sample_id", "pair", "selection"]], how="left", on="sample_id", validate="1:1"
    ).to_csv(args.out / "41040-strips.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
