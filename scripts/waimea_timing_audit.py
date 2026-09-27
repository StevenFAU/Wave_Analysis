#!/usr/bin/env python
"""Capture times of the Waimea Bay images, and checks that they are right.

Run after ``scripts/waimea_extract.py``. Reads the burned-in caption of every
image, decides which time to trust, and tests that time against the sun and
against CDIP 106. Writes ``data/interim/waimea_audit/image_times.parquet``
(one row per archived image) and ``summary.json``; prints the report.

Caption
    Each image carries ``YYYY.MM.DD HH:MM:SS   F: YYYYMMDDTHHMMSS`` in a
    monospaced 7-px font at a fixed position (41 cells from x = 7, rows
    1238-1252). Digits are read by template matching. The templates are
    learnt only from the year digits and from the month digits of images in
    the middle of a month, which do not depend on the time zone or clock
    being tested. The two copies of the time are read independently; they
    must agree to within a few seconds.

Time source
    ``caption`` when the caption falls within the hour named in the file name
    (the normal case: photos at hh:55, or hh:23-24 in 2010-2011). Otherwise
    ``uncertain``: every image of 23 May - 27 July 2013, when captions were
    often 1-10 h after the named hour and some daylight images are stamped
    19:30-23:40 HST (the camera clock was wrong), and the two images of
    27 Feb 2010 08:00 stamped about 18:00.

Checks (on ``caption`` images)
    * time zone: sun elevation at each capture if the caption is HST vs UTC;
    * clock offset: dark/bright images at dawn and dusk against sun elevation
      for clock shifts of -60..+60 min;
    * date: Spearman correlation between image white-water fraction (water
      blocks chosen from the view, not from the data) and CDIP 106 Hs for
      offsets of -10..+10 days.

Usage::

    uv run python scripts/waimea_timing_audit.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from wave_analysis.config import data_dir
from wave_analysis.processing.solar import solar_position
from wave_analysis.sources.pacioos import HST_TO_UTC_HOURS

LAT, LON = 21.643578, -158.06339  # beachcam_003/004 (ERDDAP)
CELL0, PITCH = 7, 7
DATE_CELLS = {"year": [0, 1, 2, 3], "month": [5, 6], "day": [8, 9]}
TIME_CELLS = {"hour": [12, 13], "minute": [15, 16], "second": [18, 19]}
F_CELLS = {
    "year": [26, 27, 28, 29],
    "month": [30, 31],
    "day": [32, 33],
    "hour": [35, 36],
    "minute": [37, 38],
    "second": [39, 40],
}
#: Water blocks on the 16 x 16 grid, chosen from the camera views.
ROI = {
    "beachcam_003": [(slice(8, 12), slice(6, 16)), (slice(12, 13), slice(9, 13))],
    "beachcam_004": [(slice(8, 11), slice(0, 12)), (slice(11, 13), slice(5, 11))],
}
DARK_MAX_BLOCK = 0.6  # no 1/16 block brighter than this: twilight or night
#: File-name span in which captions were often outside the named hour (first and
#: last such image; the archive has no images between 2013-07-27 and 2013-07-31).
#: The camera clock is not trusted for any image in it.
CLOCK_ANOMALY = ("2013-05-23", "2013-07-28")


def _unit(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)


def read_captions(caption: np.ndarray, names: pd.Series, ok: np.ndarray) -> pd.DataFrame:
    """Caption and F: times (naive HST) and the smallest template margin per image."""
    cap = caption.astype(np.float32)

    def cells(k: int) -> np.ndarray:
        x = CELL0 + PITCH * k
        return cap[:, :, x : x + PITCH].reshape(len(cap), -1)

    fdate = names.str.extract(r"\.(\d{8})T")[0]
    day = pd.to_numeric(fdate.str[6:8], errors="coerce").to_numpy()
    train = ok & (day >= 3) & (day <= 27)
    xs, ys = [], []
    for k, pos in zip([0, 1, 2, 3, 5, 6], [0, 1, 2, 3, 4, 5], strict=True):
        xs.append(_unit(cells(k)[train]))
        ys.append(fdate[train].str[pos].astype(int).to_numpy())
    x, y = np.concatenate(xs), np.concatenate(ys)
    if set(np.unique(y)) != set(range(10)):
        raise RuntimeError("training cells do not cover all ten digits")
    templates = _unit(np.stack([x[y == d].mean(axis=0) for d in range(10)]))
    if (np.argmax(x @ templates.T, axis=1) != y).any():
        raise RuntimeError("templates misread their own training cells")

    digits: dict[int, np.ndarray] = {}
    margins = []
    used = {k for f in (DATE_CELLS, TIME_CELLS, F_CELLS) for ks in f.values() for k in ks}
    for k in sorted(used):
        s = _unit(cells(k)) @ templates.T
        srt = np.sort(s, axis=1)
        digits[k] = np.argmax(s, axis=1)
        margins.append(srt[:, -1] - srt[:, -2])

    def stamp(fields: dict[str, list[int]]) -> pd.Series:
        parts = {}
        for name, ks in fields.items():
            v = np.zeros(len(cap), int)
            for k in ks:
                v = v * 10 + digits[k]
            parts[name] = v
        return pd.to_datetime(pd.DataFrame(parts), errors="coerce")

    out = pd.DataFrame(
        {
            "caption_hst": stamp({**DATE_CELLS, **TIME_CELLS}),
            "f_hst": stamp(F_CELLS),
            "ocr_margin": np.min(np.stack(margins, axis=1), axis=1),
        }
    )
    out.loc[~ok, ["caption_hst", "f_hst", "ocr_margin"]] = np.nan
    return out


def main() -> int:
    d = data_dir("interim") / "waimea_audit"
    t = pd.read_parquet(d / "images.parquet")
    feats = np.load(d / "features.npz")
    ok = t["features_ok"].fillna(False).to_numpy(bool)
    t = pd.concat([t, read_captions(feats["caption"], t["name"], ok)], axis=1)
    t["file_hst"] = pd.to_datetime(
        t["name"].str.extract(r"\.(\d{8}T\d{6})")[0], format="%Y%m%dT%H%M%S"
    )
    off_min = (t["caption_hst"] - t["file_hst"]).dt.total_seconds() / 60
    t["caption_minus_file_min"] = off_min
    in_hour = (off_min >= 0) & (off_min < 60)
    in_anomaly = (t["file_hst"] >= CLOCK_ANOMALY[0]) & (t["file_hst"] < CLOCK_ANOMALY[1])
    trusted = in_hour & ~in_anomaly
    t["time_source"] = np.where(~ok, "none", np.where(trusted, "caption", "uncertain"))
    t["time_utc"] = (t["caption_hst"] + pd.Timedelta(hours=HST_TO_UTC_HOURS)).dt.tz_localize("UTC")
    t.loc[t["time_source"] != "caption", "time_utc"] = pd.NaT
    bl = feats["block_luma"].reshape(len(t), -1)
    t["dark"] = bl.max(axis=1) < DARK_MAX_BLOCK
    bw = feats["block_white"]
    t["white"] = [
        np.concatenate([bw[i][r].ravel() for r in ROI[c]]).mean() if ok[i] else np.nan
        for i, c in enumerate(t["camera"])
    ]

    s: dict[str, object] = {}
    good = t[ok]
    s["images"] = int(ok.sum())
    s["caption_read"] = int(good["caption_hst"].notna().sum())
    agree = (good["caption_hst"] - good["f_hst"]).dt.total_seconds()
    s["caption_minus_F_seconds"] = {str(int(k)): int(v) for k, v in agree.value_counts().items()}
    s["ocr_min_margin"] = round(float(good["ocr_margin"].min()), 3)
    s["time_source"] = t["time_source"].value_counts().to_dict()
    s["outside_named_hour"] = int((ok & ~in_hour).sum())
    s["outside_named_hour_outside_anomaly"] = t.loc[ok & ~in_hour & ~in_anomaly, "name"].tolist()
    s["in_clock_anomaly"] = int((ok & in_anomaly).sum())
    cap = t[t["time_source"] == "caption"].copy()
    s["caption_minute_of_hour_top"] = {
        str(k): int(v) for k, v in cap["caption_hst"].dt.minute.value_counts().head(3).items()
    }

    el_hst = solar_position(cap["time_utc"], LAT, LON)[0]
    el_utc = solar_position(cap["caption_hst"].dt.tz_localize("UTC"), LAT, LON)[0]
    s["sun_below_-6_if_HST"] = int((el_hst < -6).sum())
    s["sun_below_-6_if_UTC"] = int((el_utc < -6).sum())
    cap["el"] = el_hst
    tw = cap[cap["el"] < 8]
    scan = []
    for shift in range(-60, 61, 2):
        el = solar_position(tw["time_utc"] + pd.Timedelta(minutes=shift), LAT, LON)[0]
        dk = tw["dark"].to_numpy()
        errs = min(
            int(((el > th) & dk).sum() + ((el <= th) & ~dk).sum()) for th in np.arange(-10, 2, 0.25)
        )
        scan.append((shift, errs))
    scan_df = pd.DataFrame(scan, columns=["clock_shift_min", "misclassified"])
    best = scan_df[scan_df["misclassified"] == scan_df["misclassified"].min()]
    s["twilight_images"] = len(tw)
    s["twilight_dark"] = int(tw["dark"].sum())
    s["clock_shift_consistent_min"] = [
        int(best["clock_shift_min"].min()),
        int(best["clock_shift_min"].max()),
    ]

    c = pd.read_parquet(
        data_dir("processed") / "cdip" / "106" / "bulk_2009-01-01_2014-01-01.parquet"
    )
    c = c[(c["variable"] == "wave_significant_height") & (c["method"] == "provider_reported")]
    centre = c["time_utc"] + pd.to_timedelta(c["averaging_period_s"] / 2, unit="s")
    hs = pd.Series(c["value"].to_numpy(), index=centre).sort_index()
    hs = hs[~hs.index.duplicated()]
    hs_ns, hs_v = hs.index.as_unit("ns").asi8, hs.to_numpy()

    def hs_at(x: np.ndarray) -> np.ndarray:
        v = np.interp(x, hs_ns, hs_v, left=np.nan, right=np.nan)
        j = np.clip(np.searchsorted(hs_ns, x), 1, len(hs_ns) - 1)
        v[(hs_ns[j] - hs_ns[j - 1]) > 3600e9] = np.nan
        return v

    lags = np.arange(-10, 10.25, 0.25)
    i0 = int(np.flatnonzero(lags == 0)[0])

    def lag_curve(g: pd.DataFrame) -> list[float]:
        tn = g["time_utc"].dt.as_unit("ns").astype("int64").to_numpy()
        w = g["white"].to_numpy()
        r = []
        for lag in lags:
            v = hs_at(tn + int(lag * 86400e9))
            m = np.isfinite(v)
            r.append(float(spearmanr(w[m], v[m])[0]) if m.sum() > 50 else np.nan)
        return r

    lag_rows = {}
    s["day_lag"] = {}
    for cam, g in cap.groupby("camera"):
        r = lag_curve(g)
        lag_rows[cam] = r
        k = int(np.nanargmax(r))
        per_year = {}
        for yr, gy in g.groupby(g["time_utc"].dt.year):
            ry = lag_curve(gy)
            ky = int(np.nanargmax(ry))
            per_year[str(yr)] = {
                "n": len(gy),
                "peak_rho": round(ry[ky], 2),
                "peak_lag_days": float(lags[ky]),
            }
        # Hour scale: changes between consecutive hourly images of a day, with the
        # median change for that hour of day and month removed (illumination).
        g = g.sort_values("time_utc")
        tn = g["time_utc"].dt.as_unit("ns").astype("int64").to_numpy()
        dt_h = np.diff(tn) / 3.6e12
        pair = (dt_h > 0.75) & (dt_h < 1.25)
        dw = np.diff(g["white"].to_numpy())[pair]
        hod = g["caption_hst"].dt.hour.to_numpy()[1:][pair]
        mon = g["caption_hst"].dt.month.to_numpy()[1:][pair]
        dw = dw - pd.Series(dw).groupby([hod, mon]).transform("median").to_numpy()
        dh = hs_at(tn[1:][pair]) - hs_at(tn[:-1][pair])
        m = np.isfinite(dh)
        s["day_lag"][cam] = {
            "peak_rho": round(r[k], 3),
            "peak_lag_days": float(lags[k]),
            "rho_at_0": round(r[i0], 3),
            "rho_at_pm1d": [round(r[i0 - 4], 3), round(r[i0 + 4], 3)],
            "rho_at_pm7d": [round(r[i0 - 28], 3), round(r[i0 + 28], 3)],
            "per_year": per_year,
            "hourly_change_rho_at_0": round(float(spearmanr(dw[m], dh[m])[0]), 3),
            "hourly_pairs": int(m.sum()),
        }
    pd.DataFrame(lag_rows, index=lags).rename_axis("lag_days").to_csv(d / "lag_days.csv")
    scan_df.to_csv(d / "clock_scan.csv", index=False)

    cols = [
        "camera",
        "name",
        "url",
        "file_hst",
        "caption_hst",
        "f_hst",
        "caption_minus_file_min",
        "time_source",
        "time_utc",
        "ocr_margin",
        "dark",
        "white",
        "sha256",
        "size_bytes",
    ]
    t.loc[ok, cols].to_parquet(d / "image_times.parquet", index=False)
    (d / "summary.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    print(json.dumps(s, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
