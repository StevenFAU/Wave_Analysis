"""Count the development shore labels built by ``sensing_cdip_waves.py`` (test 2 inventory).

Reads the per-clip label table and, optionally, the interim video run's per-clip
predictions (only their clip list, CDIP H_s, T_p and Tm02 labels, local dates and folds)
and prints the counts that ``docs/experiments/sensing_per_wave_bars.md`` reports:
independent station/deployment/windows, clips, dates and clips per Kim class of measured
mean wave height in 0.75-3.25 m, and the good-flag tier beside the tier that also admits
waves the provider left unevaluated (flag 2). Labels only: no prediction is read.

    python scripts/sensing_cdip_inventory.py runs/cdip-development-v1.csv \\
        [--interim <run>/predictions_whole_clip.csv] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KIM_EDGES = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25]  # mean individual wave height (m), Kim et al. 2023
TIERS = ("good", "incl_unevaluated")
MIN_WAVES = 20
WINDOW = ["station", "deployment", "start_s", "end_s"]


def kim_class(h: pd.Series) -> pd.Series:
    """Kim's class 0-4 of measured mean height; NaN outside 0.75-3.25 m or unlabelled."""
    h = h.astype(float)
    inside = (h >= KIM_EDGES[0]) & (h < KIM_EDGES[-1])
    return pd.Series(np.where(inside, np.digitize(h, KIM_EDGES[1:-1]), np.nan), index=h.index)


def camera(clip_id: pd.Series) -> pd.Series:
    return clip_id.str.replace(r"-\d{4}-\d{2}-\d{2}-\d{6}Z$", "", regex=True)


def tier_counts(d: pd.DataFrame, tier: str) -> dict:
    ok = d[f"{tier}_mean_height_m"].notna()
    win = d[ok].drop_duplicates(WINDOW)
    k = kim_class(d.loc[ok, f"{tier}_mean_height_m"])
    dk = d.loc[ok].assign(kim=k)
    by_class = {
        f"{KIM_EDGES[c]:.2f}-{KIM_EDGES[c + 1]:.2f}": {
            "dates": int(dk.loc[dk.kim == c, "local_day"].nunique()),
            "clips": int((dk.kim == c).sum()),
            "windows": int(dk[dk.kim == c].drop_duplicates(WINDOW).shape[0]),
        }
        for c in range(5)
    }
    inside = dk.kim.notna()
    return {
        "windows": len(win),
        "windows_by_station": win.station.astype(str).value_counts().sort_index().to_dict(),
        "clips": int(ok.sum()),
        "clips_by_camera": camera(d.loc[ok, "clip_id"]).value_counts().sort_index().to_dict(),
        "dates": int(d.loc[ok, "local_day"].nunique()),
        "kim_range": {
            "clips": int(inside.sum()),
            "dates": int(dk.loc[inside, "local_day"].nunique()),
            "windows": int(dk[inside].drop_duplicates(WINDOW).shape[0]),
            "by_class": by_class,
            "below_0.75_clips": int((dk[f"{tier}_mean_height_m"] < KIM_EDGES[0]).sum()),
            "above_3.25_clips": int((dk[f"{tier}_mean_height_m"] >= KIM_EDGES[-1]).sum()),
        },
        "wave_count_median": float(win[f"{tier}_wave_count"].median()),
        "wave_count_min": int(win[f"{tier}_wave_count"].min()),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("labels", type=Path)
    p.add_argument("--interim", type=Path, help="the interim run's predictions_whole_clip.csv")
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    d = pd.read_csv(a.labels, dtype={"station": str, "cdip_station": str})
    windows = d.drop_duplicates(WINDOW)
    res: dict = {
        "rows": len(d),
        "clip_ids_unique": int(d.clip_id.nunique()),
        "dates": int(d.local_day.nunique()),
        "windows": len(windows),
        "windows_by_station": windows.station.astype(str).value_counts().sort_index().to_dict(),
        "clips_per_window_max": int(d.groupby(WINDOW).size().max()),
        "status": d.status.value_counts().to_dict(),
    }
    for t in TIERS:
        res[t] = tier_counts(d, t)
    # flag sensitivity: how many windows and waves the unevaluated flag adds
    g, u = windows["good_wave_count"], windows["incl_unevaluated_wave_count"]
    res["flag"] = {
        "windows_where_tiers_differ": int((g != u).sum()),
        "waves_good": int(g.sum()),
        "waves_incl_unevaluated": int(u.sum()),
        "windows_good_lt_min_but_incl_ge_min": int(((g < MIN_WAVES) & (u >= MIN_WAVES)).sum()),
        "windows_below_min_both": int((u < MIN_WAVES).sum()),
    }
    if a.interim:
        it = pd.read_csv(
            a.interim,
            usecols=["clip_id", "local_day", "fold", "Hs_label_m", "Tp_label_s", "Tm02_label_s"],
        )
        m = it.merge(d, on=["clip_id", "local_day"], how="left", validate="1:1")
        ok = m["good_mean_height_m"].notna()
        mm = m[ok]
        res["interim"] = {
            "clips": len(it),
            "dates": int(it.local_day.nunique()),
            "labelled_good": int(ok.sum()),
            "labelled_incl_unevaluated": int(m["incl_unevaluated_mean_height_m"].notna().sum()),
            "unlabelled_status": m.loc[~ok, "status"]
            .fillna("not in label table")
            .value_counts()
            .to_dict(),
            "good": tier_counts(mm, "good"),
            # label sanity on common clips: time-domain against the bulk spectral labels
            "ratio_median": {
                "mean_height_over_Hs": float((mm.good_mean_height_m / mm.Hs_label_m).median()),
                "h_third_over_Hs": float((mm.good_h_third_m / mm.Hs_label_m).median()),
                "h_tenth_over_h_third": float((mm.good_h_tenth_m / mm.good_h_third_m).median()),
                "tz_over_Tm02": float((mm.good_tz_s / mm.Tm02_label_s).median()),
                "tz_over_Tp": float((mm.good_tz_s / mm.Tp_label_s).median()),
            },
            "ratio_iqr_mean_height_over_Hs": [
                float(q) for q in (mm.good_mean_height_m / mm.Hs_label_m).quantile([0.25, 0.75])
            ],
            # the interim run's proxy (0.631 H_s) against the measured class, on common clips
            "proxy_class_agreement": float(
                (
                    kim_class(0.631 * mm.Hs_label_m).fillna(-1)
                    == kim_class(mm.good_mean_height_m).fillna(-1)
                ).mean()
            ),
            "proxy_kim_range_clips": int(kim_class(0.631 * mm.Hs_label_m).notna().sum()),
        }
    text = json.dumps(res, indent=2, default=str)
    print(text)
    if a.json:
        a.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
