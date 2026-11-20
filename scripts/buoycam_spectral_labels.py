#!/usr/bin/env python
"""Stage 3 labels: periods, mean direction and the wind-sea spectrum of every strip, from its own buoy.

Rank 5 of ``reports/Wave camera architectures beyond backbones.md``. Fixed 2026-11-18,
before any stage-3 run (``docs/experiments/ndbc_buoycam_sea_state.md``, "Stage 3").
Every value comes from the buoy that carries the camera: its realtime ``.spec``
summaries and its spectral records (``swden`` density with NDBC's separation
frequency; ``swdir`` alpha1 and ``swr1`` r1 for direction), latest snapshot first.

Per record (one 20-min wave record; the spectral files stamp it 10 min after its
``.spec`` stamp, so each spectral record joins the ``.spec`` record nearest in time,
at most 15 min apart):

* **Periods**: the peak period T_p = 1 / (frequency of the highest density) (ERA5
  ``pp1d``); the energy period Tm-1,0 = m-1 / m0 (ERA5 ``mwp``); NDBC's APD, a
  zero-crossing period from m0 and m2 (ERA5 ``mp2``); NDBC's wind-wave and swell
  periods WWP and SwP (peak periods of each part); and the wind sea's own Tm-1,0 over
  the frequencies at or above NDBC's separation frequency (ERA5 ``mpww``).
* **Mean direction** the waves come from, weighted by energy over the whole spectrum
  (Kuik et al. 1988): atan2(sum S r1 sin a1 df, sum S r1 cos a1 df), ERA5 ``mwd``'s
  definition; NDBC's ``.spec`` MWD (the direction at the dominant period) beside it.
* **Wind-sea spectrum shape**: each of the 46 frequency bins' share of the wind-sea
  energy (S df over the bins at or above the separation frequency, summing to 1); the
  energy itself is (WWH / 4)².

Per strip, as the heights (ADR 0010): the mean over the records stamped within 30 min
of the strip (directions as the mean of unit vectors; shapes renormalised). The
direction is also given relative to the sun's azimuth at the strip (cos and sin of
direction minus sun azimuth): the caption carries no compass heading (a fixed "90°"
at every buoy and hour, "290°" at 44008), so the sun, which the views show, is the only
compass in the image.

Writes ``data/processed/packages/ndbc_buoycam_spectral_v3cv``: every file of
``ndbc_buoycam_heights_v2cv`` (hard links) except ``labels.csv``, which gains the
columns below, plus ``frequencies.json`` (the bins and their widths); ``SHA256SUMS``
rewritten.

    python scripts/buoycam_spectral_labels.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir
from wave_analysis.datasets.buoycam import snapshot_stamp, spec_records
from wave_analysis.processing.spectra import frequency_bandwidths
from wave_analysis.sources.ndbc.parsers import parse_spectral

SRC = "ndbc_buoycam_heights_v2cv"
OUT = "ndbc_buoycam_spectral_v3cv"
WINDOW = pd.Timedelta(minutes=30)
JOIN = pd.Timedelta(minutes=15)
N_BINS = 46
SCALARS = ["tp_s", "tm10_s", "apd_s", "wwp_s", "swp_s", "tm10_ww_s", "sep_freq_hz", "hs_spectrum_m"]


def spectral_table(ndbc: Path, station: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Per spectral record: time, density (46 bins), separation frequency, alpha1, r1."""
    out: dict[str, pd.DataFrame] = {}
    freq = None
    for prod in ("swden", "swdir", "swr1"):
        pattern = f"{station}.data_spec.*Z.gz" if prod == "swden" else f"{station}.{prod}.*Z.gz"
        frames = []
        for p in sorted((ndbc / station / prod).glob(pattern), key=snapshot_stamp):
            for b in parse_spectral(p, prod, realtime=True):
                f = np.asarray(b.frequency, dtype=float)
                if f.size != N_BINS:
                    continue
                freq = f if freq is None else freq
                if not np.allclose(f, freq):
                    raise SystemExit(f"{p.name}: another frequency grid")
                t = pd.DatetimeIndex(b.time)
                t = t.tz_localize("UTC") if t.tz is None else t
                d = pd.DataFrame(np.asarray(b.values, dtype=float), index=t)
                d["snapshot"] = snapshot_stamp(p)
                if prod == "swden":
                    d["sep"] = np.asarray(b.separation_frequency, dtype=float)
                frames.append(d)
        if frames:
            a = pd.concat(frames).sort_values("snapshot")
            out[prod] = a[~a.index.duplicated(keep="last")].drop(columns="snapshot").sort_index()
    if "swden" not in out:
        return pd.DataFrame(), None
    return pd.concat(out, axis=1, join="outer").sort_index(), freq


def per_record(spec: pd.DataFrame, sp: pd.DataFrame, freq: np.ndarray) -> pd.DataFrame:
    """Every ``.spec`` record with its spectral quantities (NaN where no spectral record joins)."""
    df = frequency_bandwidths(freq)
    rec = spec[["time_utc", "Hs_m", "wind_wave_height_m", "swell_period_s", "wind_wave_period_s",
                "mean_period_s", "mean_direction_deg"]].copy().sort_values("time_utc")  # fmt: skip
    rec = rec.rename(columns={"swell_period_s": "swp_s", "wind_wave_period_s": "wwp_s",
                              "mean_period_s": "apd_s", "mean_direction_deg": "mwd_ndbc_deg"})  # fmt: skip
    for c in ["tp_s", "tm10_s", "tm10_ww_s", "sep_freq_hz", "hs_spectrum_m", "mwd_deg"]:
        rec[c] = np.nan
    shape = np.full((len(rec), N_BINS), np.nan)
    if sp.empty:
        return rec.assign(**{f"wsf_{j:02d}": shape[:, j] for j in range(N_BINS)})
    times = pd.DataFrame({"sp_time": sp.index}).sort_values("sp_time")
    j = pd.merge_asof(rec[["time_utc"]].reset_index(), times, left_on="time_utc", right_on="sp_time",
                      direction="nearest", tolerance=JOIN)  # fmt: skip
    have = j["sp_time"].notna().to_numpy()
    rows = sp.loc[j.loc[have, "sp_time"]]
    s = np.nan_to_num(rows["swden"][list(range(N_BINS))].to_numpy(float), nan=0.0)
    s = np.maximum(s, 0.0)
    e = s * df
    m0 = e.sum(1)
    ok = m0 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        tm10 = np.where(ok, (e / freq).sum(1) / m0, np.nan)
        tp = np.where(ok, 1.0 / freq[np.argmax(s, axis=1)], np.nan)
        sep = rows["swden"]["sep"].to_numpy(float)
        sep = np.where((sep > 0) & (sep < 1.0), sep, np.nan)  # NDBC writes 9.999 for none
        ww = freq[None, :] >= sep[:, None]
        eww = e * ww
        m0w = eww.sum(1)
        okw = m0w > 0
        tm10w = np.where(okw, (eww / freq).sum(1) / m0w, np.nan)
        shp = np.where(okw[:, None], eww / m0w[:, None], np.nan)
    mwd = np.full(len(rows), np.nan)
    if "swdir" in rows and "swr1" in rows:
        a1 = np.deg2rad(rows["swdir"][list(range(N_BINS))].to_numpy(float))
        r1 = rows["swr1"][list(range(N_BINS))].to_numpy(float)
        w = np.where(np.isfinite(a1) & np.isfinite(r1), e * r1, 0.0)
        a, b = (w * np.nan_to_num(np.cos(a1))).sum(1), (w * np.nan_to_num(np.sin(a1))).sum(1)
        good = (np.abs(a) + np.abs(b)) > 0
        mwd[good] = np.rad2deg(np.arctan2(b[good], a[good])) % 360
    idx = np.flatnonzero(have)
    rec.iloc[idx, rec.columns.get_loc("tp_s")] = tp
    rec.iloc[idx, rec.columns.get_loc("tm10_s")] = tm10
    rec.iloc[idx, rec.columns.get_loc("tm10_ww_s")] = tm10w
    rec.iloc[idx, rec.columns.get_loc("sep_freq_hz")] = sep
    rec.iloc[idx, rec.columns.get_loc("hs_spectrum_m")] = 4 * np.sqrt(m0)
    rec.iloc[idx, rec.columns.get_loc("mwd_deg")] = mwd
    shape[idx] = shp
    return rec.assign(**{f"wsf_{k:02d}": shape[:, k] for k in range(N_BINS)})


def circular_mean(deg: np.ndarray) -> float:
    d = deg[np.isfinite(deg)]
    if d.size == 0:
        return float("nan")
    r = np.deg2rad(d)
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def window_labels(strips: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    """Per strip: means over the records stamped within 30 min (directions: circular)."""
    rt = rec["time_utc"].dt.tz_convert(None).to_numpy()
    st = strips["time_utc"].dt.tz_convert(None).to_numpy()
    lo = np.searchsorted(rt, st - WINDOW.to_timedelta64(), side="left")
    hi = np.searchsorted(rt, st + WINDOW.to_timedelta64(), side="right")
    cols = [
        "tp_s",
        "tm10_s",
        "apd_s",
        "wwp_s",
        "swp_s",
        "tm10_ww_s",
        "sep_freq_hz",
        "hs_spectrum_m",
    ]
    vals = rec[cols].to_numpy(float)
    shp = rec[[f"wsf_{k:02d}" for k in range(N_BINS)]].to_numpy(float)
    out = np.full((len(st), len(cols)), np.nan)
    shape = np.full((len(st), N_BINS), np.nan)
    mwd, mwd_n, n = np.full(len(st), np.nan), np.full(len(st), np.nan), np.zeros(len(st), int)
    for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
        if b <= a:
            continue
        n[i] = b - a
        v = vals[a:b]
        cnt = np.isfinite(v).sum(axis=0)
        out[i] = np.where(cnt > 0, np.nansum(v, axis=0) / np.maximum(cnt, 1), np.nan)
        good = np.isfinite(shp[a:b]).all(axis=1)
        if good.any():
            m = shp[a:b][good].mean(axis=0)
            shape[i] = m / m.sum()
        mwd[i] = circular_mean(rec["mwd_deg"].to_numpy(float)[a:b])
        mwd_n[i] = circular_mean(rec["mwd_ndbc_deg"].to_numpy(float)[a:b])
    lab = pd.DataFrame(
        out, columns=[f"{c.rsplit('_', 1)[0]}_label_{c.rsplit('_', 1)[1]}" for c in cols]
    )
    lab["mwd_label_deg"], lab["mwd_ndbc_label_deg"] = mwd, mwd_n
    lab["n_spectral_records"] = n
    rel = np.deg2rad(mwd - strips["sun_azimuth_deg"].to_numpy(float))
    lab["dir_sun_rel_cos"], lab["dir_sun_rel_sin"] = np.cos(rel), np.sin(rel)
    for k in range(N_BINS):
        lab[f"wsf_{k:02d}"] = shape[:, k]
    return lab


def main() -> int:
    src, out = data_dir("processed") / "packages" / SRC, data_dir("processed") / "packages" / OUT
    labels = pd.read_csv(src / "labels.csv", dtype={"station_id": str}, low_memory=False)
    labels["time_utc"] = pd.to_datetime(labels["time_utc"], utc=True)
    ndbc = data_dir("raw") / "ndbc"
    parts, checks, freq_all = [], [], None
    for sid, strips in labels.groupby("station_id"):
        spec = spec_records(sorted((ndbc / sid / "spec").glob(f"{sid}.spec.*Z.gz")), sid)
        sp, freq = spectral_table(ndbc, sid)
        freq_all = freq if freq is not None else freq_all
        rec = per_record(spec, sp, freq if freq is not None else np.linspace(0.033, 0.485, N_BINS))
        rec = rec.sort_values("time_utc").reset_index(drop=True)
        strips = strips.sort_values("time_utc")
        lab = window_labels(strips, rec)
        lab.insert(0, "sample_id", strips["sample_id"].to_numpy())
        parts.append(lab)
        both = rec["hs_spectrum_m"].notna() & rec["Hs_m"].notna() & (rec["Hs_m"] >= 0.5)
        checks.append({"station_id": sid, "records": len(rec), "with_spectrum": int(rec["tp_s"].notna().sum()),
                       "with_direction": int(rec["mwd_deg"].notna().sum()),
                       "hs_spectrum_over_wvht_median": float((rec.loc[both, "hs_spectrum_m"] / rec.loc[both, "Hs_m"]).median())
                       if both.any() else float("nan")})  # fmt: skip
    new = pd.concat(parts, ignore_index=True).set_index("sample_id").loc[labels["sample_id"]]
    if out.exists():
        shutil.rmtree(out)
    for f in src.rglob("*"):
        if f.is_file() and f.name not in ("labels.csv", "SHA256SUMS"):
            dest = out / f.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.hardlink_to(f)
    rows = (src / "labels.csv").read_text().splitlines()
    if len(rows) != len(labels) + 1:
        raise SystemExit("labels.csv has quoted line breaks: append the columns another way")
    txt = new.to_csv(index=False, header=True, float_format="%.6g", na_rep="").splitlines()
    (out / "labels.csv").write_text(
        "\n".join(r + "," + t for r, t in zip(rows, txt, strict=True)) + "\n"
    )
    assert freq_all is not None
    (out / "frequencies.json").write_text(json.dumps(
        {"frequency_hz": freq_all.round(4).tolist(), "bandwidth_hz": frequency_bandwidths(freq_all).round(5).tolist(),
         "source": "NDBC realtime swden grid, the same at all 79 camera buoys"}, indent=1))  # fmt: skip
    sums = {
        n: hashlib.sha256((out / n).read_bytes()).hexdigest()
        for n in ("labels.csv", "frequencies.json")
    }
    lines = []
    for line in (src / "SHA256SUMS").read_text().splitlines():
        _sha, name = line.split(None, 1)
        lines.append(
            f"{sums['labels.csv']}  {name.strip()}" if name.strip() == "labels.csv" else line
        )
    lines.append(f"{sums['frequencies.json']}  frequencies.json")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    ck = pd.DataFrame(checks)
    ck.to_csv(out / "spectral_checks.csv", index=False)
    cov = new.notna().mean()
    print(f"{out}: {len(new):,} strips")
    print("coverage:", {c: round(float(cov[c]), 3) for c in ["tp_label_s", "tm10_label_s", "apd_label_s",
          "wwp_label_s", "swp_label_s", "tm10_ww_label_s", "mwd_label_deg", "mwd_ndbc_label_deg", "wsf_20"]})  # fmt: skip
    print("buoys without spectra:", list(ck.loc[ck["with_spectrum"] == 0, "station_id"]))
    print("buoys without direction:", list(ck.loc[ck["with_direction"] == 0, "station_id"]))
    print("4 sqrt(m0) / WVHT, median over buoys:", round(float(ck["hs_spectrum_over_wvht_median"].median()), 3),
          "range", round(float(ck["hs_spectrum_over_wvht_median"].min()), 3), round(float(ck["hs_spectrum_over_wvht_median"].max()), 3))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
