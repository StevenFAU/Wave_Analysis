"""Check missing totals, overflow flags and short-lead timestamps in station bulletins."""

import importlib
from pathlib import Path

import pandas as pd

from wave_analysis.sources.gfs_wave import short_leads


def test_first_seven_hours():
    text = "Location : 41025\nCycle : 20260930 18 UTC\n"
    for lead in range(7):
        day, hour = (30, 18 + lead) if lead < 6 else (1, 0)
        cells = [
            f"{day} {hour}",
            "0" if lead == 0 else "1.30 7 1",
            "* 1.20 7.4 250" if lead else "",
            "",
            "",
            "",
            "",
            "",
        ]
        text += " | " + " | ".join(cells) + " |\n"
    rows = short_leads(text)
    assert rows[0]["hs_m"] is None and rows[0]["systems"] == 0
    assert rows[1]["system1_wind"] and rows[1]["extra_systems"] == 1
    assert rows[-1]["valid_utc"].month == 10 and rows[-1]["lead_h"] == 6


def test_strip_match_uses_only_issued_cycle_and_keeps_absent_stations(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    match = importlib.import_module("gfs_wave_match").match_strips
    strips = pd.DataFrame(
        {
            "sample_id": list("abcd"),
            "station_id": ["1", "1", "1", "2"],
            "time_utc": [f"2026-11-01T{hh}Z" for hh in ("05:30", "05:50", "06:00", "06:00")],
        }
    )
    points = pd.DataFrame(
        {
            "station_id": ["1"] * 3,
            "issue_utc": pd.to_datetime(["2026-11-01T00Z"] * 2 + ["2026-11-01T06Z"], utc=True),
            "valid_utc": pd.to_datetime(["2026-11-01T05Z"] + ["2026-11-01T06Z"] * 2, utc=True),
            "hs_m": [0.5, 0.6, 0.7],
        }
    )
    result = match(strips, points)
    assert result.hs_m.iloc[:3].tolist() == [0.5, 0.6, 0.7]
    assert result.offset_s.tolist() == [-1800, 600, 0, 0]
    assert pd.isna(result.hs_m.iloc[3]) and not result.bulletin_present.iloc[3]
