#!/usr/bin/env python
"""Summarise NDBC ADCP availability from the committed inventory.

Reads ``data/registry/ndbc_files.parquet`` (produced by ``inventory_ndbc.py``)
and writes ``data/registry/ndbc_adcp_availability.csv``: one row per station
with ADCP (``adcp`` and/or ``adcp2``) years, co-located directional-spectrum
years, and whether a camera is currently present. Answers the technical
summary's §47 questions about which stations can pair currents with waves.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "data" / "registry"


def main() -> None:
    files = pd.read_parquet(REG / "ndbc_files.parquet")
    summary = pd.read_csv(REG / "ndbc_station_summary.csv", dtype={"station_id": str}).set_index(
        "station_id"
    )
    hist = files[files["archive"] == "historical"]
    adcp = hist[hist["product"].isin(["adcp", "adcp2"])]
    years = adcp.groupby("station_id")["year"].agg(
        adcp_first="min", adcp_last="max", adcp_years="nunique"
    )
    spec_years = (
        hist[hist["product"] == "swden"].groupby("station_id")["year"].apply(set).rename("spec_set")
    )
    adcp_sets = adcp.groupby("station_id")["year"].apply(set).rename("adcp_set")
    j = years.join(adcp_sets).join(spec_years)
    j["adcp_and_spectrum_years"] = [
        len(a & b) if isinstance(b, set) else 0
        for a, b in zip(j["adcp_set"], j["spec_set"], strict=True)
    ]
    j = j.drop(columns=["adcp_set", "spec_set"]).join(
        summary[["name", "latitude", "longitude", "has_camera_now", "realtime_products"]],
        how="left",
    )
    out = REG / "ndbc_adcp_availability.csv"
    j.sort_values("adcp_years", ascending=False).to_csv(out)
    print(
        f"{len(j)} stations with ADCP records; "
        f"{(j['adcp_and_spectrum_years'] > 0).sum()} with same-year spectra; "
        f"{int(j['has_camera_now'].fillna(False).sum())} with a camera now -> {out}"
    )


if __name__ == "__main__":
    main()
