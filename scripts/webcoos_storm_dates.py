#!/usr/bin/env python
"""The storm set of the shore-video work: storm dates chosen from CDIP records alone, then sealed.

    python scripts/webcoos_storm_dates.py   # -> configs/experiments/webcoos_video_storm_dates.csv

**Why.** The video sample is too calm for Kim et al.'s five classes: of 1,444 labelled
clips, 1,133 have a mean wave height below 0.75 m and only 5 reach 1.75-2.25 m
(``docs/experiments/sensing_per_wave_bars.md`` on the shore-labels branch). Kim's mean
height is about 0.631 H_s, so the upper classes start at H_s of about 2.0 m (mean 1.25
m), 2.8 m (1.75 m) and 3.6 m (2.25 m). The storm set adds days with high seas, chosen by
the buoys' records only: stratified sampling, which leaves every clip's label unseen.

**Rule** (fixed 2026-11-19, before any storm clip was downloaded):

1. Camera-days: each of the four cameras of the video sample, on local days
   (America/New_York) from 2024-03-01 (the video archive's start) to 2026-11-18,
   within its buoy's records (``webcoos_pairs.PAIRINGS``; CDIP 200 ends 2026-07-20).
2. Daylight H_s of a camera-day: the mean H_s of its CDIP records whose centre has the
   sun at or above 0 deg at the camera (the run's own daylight rule), if there are at
   least 10 such records.
3. A storm camera-day has a daylight H_s of 2.0 m or more.
4. Excluded: the development dates (``webcoos_video_dev_dates.txt``) and the sample's
   dates, the test-set days and six weeks of ``scripts/webcoos_video_sample.sh``, each
   widened by a day on both sides (clips near midnight UTC fall on the next local day).
5. A storm date is a site and local day with at least one storm camera-day. North:
   Currituck (CDIP 433) and Jennette (CDIP 243); Oak Island: its two views (CDIP 200).
   Its cameras are its storm camera-days; its H_s is the largest of their daylight H_s.
6. Order: four strata, north with H_s >= 2.8 m, Oak Island >= 2.8 m, north 2.0-2.8 m and
   Oak Island 2.0-2.8 m. Within a stratum the seasons take turns (DJF, MAM, JJA, SON),
   each season's dates by H_s, largest first. The strata then take turns, one date each,
   skipping a stratum once it is empty. A date within three days of an earlier date of the
   same site in this order is dropped: one date per storm and site, so the dates are
   closer to independent.
7. Download (``scripts/webcoos_storm_sample.sh``) in this order, hourly clips of the
   storm date's cameras, until 24 dates have video (at least 6 clips from one of their
   cameras), 120 GB is reached, or the list ends. A date without video is skipped.

**Sealed.** The storm clips go to their own archive root (``data/raw/webcoos_video_storm``),
which the video sample's pairing never reads, so they cannot join the final run's
development or confirmation dates. No per-clip label, feature or score of a storm date is
opened until the bars that use them are committed (the shore-labels session writes them).
The list below records each date's daylight H_s at the buoy: the stratification variable.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd

from wave_analysis.config import repo_root
from wave_analysis.processing.solar import solar_position
from wave_analysis.sources.webcoos import ONPREM_POSITIONS

FIRST, LAST = "2024-03-01", "2026-11-18"
HS_STORM, HS_HIGH, MIN_RECORDS, SEPARATION_DAYS = 2.0, 2.8, 10, 3
SITES = {
    "north": ("currituck_hampton_inn", "jennette_south"),
    "oak": ("oakisland_east", "oakisland_west"),
}
# The video sample (scripts/webcoos_video_sample.sh): UTC ranges [start, end).
SAMPLE = [("2026-09-24", "2026-09-25"), ("2024-08-04", "2024-08-05"), ("2024-04-12", "2024-04-13"),
          ("2024-08-14", "2024-08-15"), ("2024-04-08", "2024-04-15"), ("2024-07-08", "2024-07-15"),
          ("2024-10-07", "2024-10-14"), ("2025-01-13", "2025-01-20"), ("2025-04-14", "2025-04-21"),
          ("2026-01-12", "2026-01-19")]  # fmt: skip
SEASONS = ("DJF", "MAM", "JJA", "SON")
STRATA = (("north", "high"), ("oak", "high"), ("north", "mid"), ("oak", "mid"))
# Mean GB of a whole camera-day of the sample at 30-min clips (video ledgers, 85 camera-days,
# 2026-11-19); hourly clips take half.
GB_PER_DAY_30MIN = {"currituck_hampton_inn": 1.9, "jennette_south": 6.8,
                    "oakisland_east": 2.2, "oakisland_west": 1.8}  # fmt: skip
OUT = repo_root() / "configs" / "experiments" / "webcoos_video_storm_dates.csv"


def _pairs_module():
    spec = importlib.util.spec_from_file_location(
        "webcoos_pairs", repo_root() / "scripts" / "webcoos_pairs.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["webcoos_pairs"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def daylight_hs(camera: str) -> pd.DataFrame:
    """Daylight mean H_s and record count of the camera's buoy, per local day (rules 1-2)."""
    wp = _pairs_module()
    station = str(wp.PAIRINGS[camera]["cdip"])
    lat, lon = ONPREM_POSITIONS[camera]
    rec, _ = wp.cdip_records(station, lat, lon)
    rec = rec[rec["Hs_m"].notna()]
    t = rec["start_utc"] + pd.to_timedelta(rec["averaging_period_s"] / 2, unit="s")
    el, _ = solar_position(t, np.full(len(t), lat), np.full(len(t), lon))
    day = t.dt.tz_convert(str(wp.PAIRINGS[camera]["timezone"])).dt.strftime("%Y-%m-%d")
    d = pd.DataFrame({"local_day": day, "Hs_m": rec["Hs_m"].to_numpy()})[el >= 0]
    g = d.groupby("local_day")["Hs_m"].agg(hs="mean", n_records="size").reset_index()
    return g.assign(camera=camera, cdip=station)


def excluded_days() -> set[str]:
    """Development dates and the sample's dates, each sample range widened by a day (rule 4)."""
    days = set(
        (repo_root() / "configs/experiments/webcoos_video_dev_dates.txt").read_text().split()
    )
    for a, b in SAMPLE:
        span = pd.date_range(pd.Timestamp(a) - pd.Timedelta("1D"), pd.Timestamp(b), freq="D")
        days |= {x.strftime("%Y-%m-%d") for x in span}
    return days


def season(day: str) -> str:
    return SEASONS[(int(day[5:7]) % 12) // 3]


def storm_dates() -> pd.DataFrame:
    """Every storm date in download order (rules 3-6), one per storm and site."""
    cams = pd.concat([daylight_hs(c) for cs in SITES.values() for c in cs], ignore_index=True)
    cams = cams[cams["local_day"].between(FIRST, LAST) & (cams["n_records"] >= MIN_RECORDS)
                & (cams["hs"] >= HS_STORM) & ~cams["local_day"].isin(excluded_days())]  # fmt: skip
    cams["site"] = cams["camera"].map({c: s for s, cs in SITES.items() for c in cs})
    d = (cams.sort_values("camera").groupby(["site", "local_day"])
         .agg(hs_daylight_m=("hs", "max"), cameras=("camera", ";".join)).reset_index())  # fmt: skip
    d["season"] = d["local_day"].map(season)
    d["band"] = np.where(d["hs_daylight_m"] >= HS_HIGH, "high", "mid")
    d["est_gb_hourly"] = [
        round(sum(GB_PER_DAY_30MIN[c] / 2 for c in cs.split(";")), 2) for cs in d["cameras"]
    ]
    queues = {}
    for site, band in STRATA:
        s = d[(d["site"] == site) & (d["band"] == band)].copy()
        s["k"] = s.groupby("season")["hs_daylight_m"].rank(ascending=False, method="first")
        s["s"] = s["season"].map(SEASONS.index)
        queues[(site, band)] = list(s.sort_values(["k", "s"]).index)
    order = []
    while any(queues.values()):
        for key in STRATA:
            if queues[key]:
                order.append(queues[key].pop(0))
    kept: dict[str, list[pd.Timestamp]] = {site: [] for site in SITES}
    keep = []
    for i in order:  # one date per storm and site
        day, site = pd.Timestamp(d.loc[i, "local_day"]), d.loc[i, "site"]
        if all(abs((day - k).days) > SEPARATION_DAYS for k in kept[site]):
            kept[site].append(day)
            keep.append(i)
    out = d.loc[keep].reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["hs_daylight_m"] = out["hs_daylight_m"].round(2)
    return out[
        ["rank", "site", "local_day", "season", "band", "hs_daylight_m", "cameras", "est_gb_hourly"]
    ]


def main() -> int:
    d = storm_dates()
    d.to_csv(OUT, index=False)
    first = d.head(24)
    print(f"{len(d)} storm dates ({(d['band'] == 'high').sum()} with H_s >= {HS_HIGH} m) -> {OUT}")
    print(f"first 24: {first['site'].value_counts().to_dict()}, {first['band'].value_counts().to_dict()}, "
          f"{first['season'].value_counts().to_dict()}, {first['est_gb_hourly'].sum():.0f} GB hourly")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
