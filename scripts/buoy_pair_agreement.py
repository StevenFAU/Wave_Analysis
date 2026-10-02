#!/usr/bin/env python
"""Measure how wave-height agreement between two buoys falls off with distance.

Shore cameras are paired with wave buoys 2-40 km away
(``docs/literature/landscape_review.md`` section 5.3,
``data/registry/camera_sites.yaml``). How much does a label degrade with that
separation? Two buoys a known distance apart bound the answer: their
disagreement is the spatial variability of the sea state alone, before any
shoaling, refraction or breaking between a buoy and a surf zone, so a camera
label taken from a buoy at that distance can only be worse.

For one year (default 2023) the script takes every station in the committed
NDBC inventory with both a historical ``stdmet`` and a spectral-density
(``swden``) file that year (NDBC, CDIP and partner wave buoys), forms every
pair between ``MIN_KM`` and ``--max-km`` apart, downloads the stations'
``stdmet`` files (raw bytes under ``data/raw/ndbc/``, one manifest row each in
``data/manifests/raw/ndbc.csv``), averages WVHT (the spectral H_m0, see
``docs/datasets/ndbc.md``) into hourly bins, and compares the hours both report.

Per pair (``data/manifests/processed/buoy_pair_agreement_<year>.csv``):

``n_hours``, ``pearson_r``
    Overlap and linear correlation of hourly H_s.
``median_abs_rel_diff``, ``p90_abs_rel_diff``
    |a - b| / mean(a, b): the typical and the 90th-percentile mismatch.
``systematic_pct``
    100 * |exp(median log(b/a)) - 1|: a constant ratio between the two sites.
    A site-specific model could learn it; it does not transfer to other sites.
``random_pct``
    100 * (exp(1.4826 * MAD(log(b/a))) - 1): the robust scatter left after
    removing that ratio. No model can remove it.
``best_lag_h``, ``r_best_lag``
    The lag in [-3, 3] h that maximises the correlation; positive when
    station a's record follows station b's.

Per distance bin (``.summary.json``): pair counts and the median of each metric.
Positions are the stations' current positions from the inventory, so a station
moved since ``year`` has a slightly wrong distance.

    uv run python scripts/buoy_pair_agreement.py [--year 2023] [--max-km 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir, repo_root
from wave_analysis.datasets.provenance import build_provenance, write_provenance
from wave_analysis.ingest.downloader import Downloader
from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, latest_by_url, write_manifest
from wave_analysis.processing.geospatial import haversine_m
from wave_analysis.sources.ndbc.source import NDBCSource

#: Distance bins [km] for the summary.
BINS_KM = (0, 10, 20, 30, 50, 100, 200)
#: Pairs closer than this are treated as one site (renamed or relocated station).
MIN_KM = 0.5
#: Minimum overlap for a pair to be reported (about a month of hourly values).
MIN_HOURS = 720
LAGS_H = range(-3, 4)
METRICS = [
    "n_hours",
    "pearson_r",
    "median_abs_rel_diff",
    "p90_abs_rel_diff",
    "systematic_pct",
    "random_pct",
    "best_lag_h",
    "r_best_lag",
]


def wave_stations(year: int) -> pd.DataFrame:
    """Wave stations with a ``stdmet`` file in ``year``, with position and file name.

    A wave station has an NDBC ``swden`` file that year or is a CDIP Waverider.
    NDBC distributes CDIP buoys' bulk parameters (WVHT in ``stdmet``) but not
    their spectra, so CDIP buoys are recognised by owner (``SCRIPPS``) or by
    the CDIP number that ends their name, e.g. ``Fort Pierce, FL (134)``.
    Stations whose file holds no WVHT are dropped after reading.
    """
    reg = repo_root() / "data" / "registry"
    files = pd.read_parquet(reg / "ndbc_files.parquet")
    hist = files[(files["archive"] == "historical") & (files["year"] == year)]
    st = pd.read_parquet(reg / "stations.parquet")
    st["label"] = st["name"].fillna(st["table_name"])
    cdip = st["owner"].eq("SCRIPPS") | st["label"].fillna("").str.contains(r"\(\d{3}\)\s*$")
    st["network"] = np.where(cdip, "CDIP", st["owner"].fillna(""))
    swden = set(hist.loc[hist["product"] == "swden", "station_id"])
    wave = swden | set(st.loc[cdip, "station_id"])
    std = hist[(hist["product"] == "stdmet") & (hist["segment"] == "a")]
    std = std[std["station_id"].isin(wave)].drop_duplicates("station_id")
    out = std[["station_id", "filename"]].merge(
        st[["station_id", "label", "network", "latitude", "longitude"]], on="station_id"
    )
    return out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def station_pairs(st: pd.DataFrame, max_km: float) -> pd.DataFrame:
    """Every pair of stations between ``MIN_KM`` and ``max_km`` apart."""
    lat, lon = st["latitude"].to_numpy(), st["longitude"].to_numpy()
    d = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) / 1000
    i, j = np.nonzero(np.triu((d >= MIN_KM) & (d <= max_km), k=1))
    a = st.iloc[i].reset_index(drop=True).add_suffix("_a")
    b = st.iloc[j].reset_index(drop=True).add_suffix("_b")
    out = pd.concat([a, b], axis=1)
    out["distance_km"] = d[i, j].round(2)
    return out.sort_values("distance_km").reset_index(drop=True)


def hourly_hs(path: Path, src: NDBCSource) -> pd.Series:
    """Hourly mean WVHT [m] from one historical stdmet file."""
    wvht = src.parse(path, "stdmet").data["WVHT"].dropna()
    wvht = wvht[wvht > 0]
    return wvht.resample("1h").mean().dropna()


def compare(a: pd.Series, b: pd.Series) -> dict[str, float] | None:
    """Agreement metrics for two hourly H_s series (``None`` if the overlap is short)."""
    df = pd.concat({"a": a, "b": b}, axis=1, sort=True).asfreq("h")
    both = df.dropna()
    if len(both) < MIN_HOURS:
        return None
    x, y = both["a"].to_numpy(), both["b"].to_numpy()
    rel = np.abs(x - y) / ((x + y) / 2)
    lr = np.log(y / x)
    med = float(np.median(lr))
    mad = float(np.median(np.abs(lr - med)))
    lags = {lag: df["a"].corr(df["b"].shift(lag)) for lag in LAGS_H}
    best = max(lags, key=lambda k: -np.inf if np.isnan(lags[k]) else lags[k])
    return {
        "n_hours": len(both),
        "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 3),
        "median_abs_rel_diff": round(float(np.median(rel)), 3),
        "p90_abs_rel_diff": round(float(np.quantile(rel, 0.9)), 3),
        "systematic_pct": round(100 * abs(np.exp(med) - 1), 1),
        "random_pct": round(100 * (np.exp(1.4826 * mad) - 1), 1),
        "best_lag_h": int(best),
        "r_best_lag": round(float(lags[best]), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--max-km", type=float, default=100.0)
    ap.add_argument("--min-interval", type=float, default=1.0)
    args = ap.parse_args()

    st = wave_stations(args.year)
    pairs = station_pairs(st, args.max_km)
    needed = st[st["station_id"].isin(set(pairs["station_id_a"]) | set(pairs["station_id_b"]))]
    print(f"{len(st)} wave stations in {args.year}; {len(pairs)} pairs within {args.max_km} km; "
          f"{len(needed)} stations to read")  # fmt: skip

    raw = data_dir("raw") / "ndbc"
    manifest_path = data_dir("manifests") / "raw" / "ndbc.csv"
    previous = {u: e.sha256 for u, e in latest_by_url(manifest_path).items() if e.sha256}
    src = NDBCSource()
    entries: list[ManifestEntry] = []
    series: dict[str, pd.Series] = {}
    inputs: list[Path] = []
    with Downloader(min_interval_s=args.min_interval) as dl:
        src_dl = NDBCSource(dl)
        for row in needed.itertuples(index=False):
            sid = str(row.station_id).upper()
            path = raw / sid / "stdmet" / str(row.filename)
            if not path.exists():
                got = src_dl.fetch(
                    sid,
                    ["stdmet"],
                    dest_root=raw,
                    years=[args.year],
                    filenames={("stdmet", args.year): str(row.filename)},
                    previous=previous,
                )
                entries += got
                if not any(e.local_path for e in got):
                    continue
            hs = hourly_hs(path, src)
            if len(hs):
                series[str(row.station_id)] = hs
                inputs.append(path)
    if entries:
        write_manifest(entries, manifest_path)
    failed = [e for e in entries if e.status == EntryStatus.FAILED]

    rows = []
    for p in pairs.itertuples(index=False):
        a, b = series.get(str(p.station_id_a)), series.get(str(p.station_id_b))
        m = None if a is None or b is None else compare(a, b)
        if m is not None:
            rows.append({**p._asdict(), **m})
    res = pd.DataFrame(rows)
    res["distance_bin_km"] = pd.cut(res["distance_km"], BINS_KM, right=False).astype(str)
    keep = [
        "station_id_a", "label_a", "network_a", "station_id_b", "label_b", "network_b",
        "distance_km", "distance_bin_km", *METRICS,
    ]  # fmt: skip
    res = res[keep]

    summary: dict[str, object] = {
        "year": args.year,
        "max_km": args.max_km,
        "min_km": MIN_KM,
        "min_hours": MIN_HOURS,
        "wave_stations": len(st),
        "stations_read": len(series),
        "pairs_within_max_km": len(pairs),
        "pairs_reported": len(res),
        "by_distance_bin": {
            str(k): {"pairs": len(g), **{m: round(float(g[m].median()), 3) for m in METRICS}}
            for k, g in res.groupby("distance_bin_km", sort=False)
        },
    }
    out = data_dir("manifests") / "processed"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"buoy_pair_agreement_{args.year}"
    paths = {"pairs": out / f"{stem}.csv", "summary": out / f"{stem}.summary.json"}
    res.to_csv(paths["pairs"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=1) + "\n")
    write_provenance(
        build_provenance(
            artifact_id=stem,
            inputs=inputs,
            outputs=list(paths.values()),
            command="python scripts/buoy_pair_agreement.py " + " ".join(sys.argv[1:]),
            config={
                "year": args.year,
                "max_km": args.max_km,
                "min_km": MIN_KM,
                "min_hours": MIN_HOURS,
                "bins_km": list(BINS_KM),
            },
            notes="WVHT hourly means; positions from data/registry/stations.parquet.",
        ),
        out / f"{stem}.provenance.yaml",
    )
    print(json.dumps(summary, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
