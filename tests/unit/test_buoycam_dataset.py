"""Building blocks of the NDBC buoy-camera sea-state dataset (wave_analysis.datasets.buoycam)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from wave_analysis.datasets.buoycam import (
    crop_strip,
    snapshot_stamp,
    spatial_groups,
    spec_records,
    station_positions,
    strip_stats,
    usable_views,
)

HEADER = (
    "#YY  MM DD hh mm WVHT  SwH  SwP  WWH  WWP SwD WWD  STEEPNESS  APD MWD\n"
    "#yr  mo dy hr mn    m    m  sec    m  sec  -  degT     -      sec degT\n"
)


def _spec(path: Path, rows: list[str]) -> Path:
    with gzip.open(path, "wt") as fh:
        fh.write(HEADER + "\n".join(rows) + "\n")
    return path


def test_spec_records_merge_snapshots_latest_wins(tmp_path: Path):
    old = _spec(
        tmp_path / "41004.spec.20261101T082107Z.gz",
        [
            "2026 11 01 07 40  2.1  0.8  8.3  2.0  7.1   E ENE      STEEP  5.3  66",
            "2026 11 01 07 10  2.0  0.8  8.3  1.9  6.7   E ENE      STEEP  5.1  69",
        ],
    )
    new = _spec(
        tmp_path / "41004.spec.20261114T153312Z.gz",
        [
            "2026 11 01 08 10   MM   MM   MM   MM   MM  MM  MM         MM   MM  MM",
            "2026 11 01 07 40  2.3  0.8  8.3  2.0  7.1   E ENE      STEEP  5.3  66",
        ],
    )
    rec = spec_records([new, old], "41004")  # order of the paths does not matter
    assert list(rec["time_utc"].dt.strftime("%H:%M")) == ["07:10", "07:40", "08:10"]
    r = rec.set_index(rec["time_utc"].dt.strftime("%H:%M"))
    assert r.loc["07:40", "Hs_m"] == 2.3  # the later snapshot's value
    assert r.loc["07:40", "n_snapshots"] == 2
    assert bool(r.loc["07:40", "snapshots_disagree"]) is True
    assert bool(r.loc["07:10", "snapshots_disagree"]) is False
    assert np.isnan(r.loc["08:10", "Hs_m"])
    assert (rec["end_utc"] - rec["start_utc"]).eq(pd.Timedelta(minutes=20)).all()
    assert (rec["start_utc"] + pd.Timedelta(minutes=10)).eq(rec["time_utc"]).all()
    assert r.loc["07:40", "steepness"] == "STEEP"
    assert snapshot_stamp(old) == "20261101T082107Z"
    with pytest.raises(ValueError):
        snapshot_stamp(tmp_path / "41004h2023.txt.gz")
    assert spec_records([], "41004").empty


def test_station_positions_median_and_movement(tmp_path: Path):
    for i, (lat, lon) in enumerate([(32.50, -79.10), (32.51, -79.10), (32.50, -79.09)]):
        items = [{"id": "41004", "lat": lat, "lng": lon}, {"id": "x1", "lat": None, "lng": 1.0}]
        (tmp_path / f"buoycams_{i}.json").write_text(json.dumps(items))
    (tmp_path / "broken.json").write_text("")  # an interrupted listing download
    pos = station_positions(sorted(tmp_path.glob("*.json"))).set_index("station_id")
    assert list(pos.index) == ["41004"]
    assert pos.loc["41004", "latitude"] == pytest.approx(32.50)
    assert pos.loc["41004", "n_listings"] == 3
    assert 0.9 < pos.loc["41004", "moved_km"] < 1.2  # 0.01 deg of latitude


def _strip(path: Path, h: int = 300, w: int = 2880) -> Path:
    a = np.zeros((h, w, 3), dtype=np.uint8)
    view = w // 6
    for i in range(6):
        a[: h - 30, i * view : (i + 1) * view] = 40 * i  # view i has grey 40*i
    a[: h - 30 : 2, : view * 3] += 10  # stripes give views 0-2 some contrast
    a[h - 30 :] = 255  # the caption band
    Image.fromarray(a).save(path, quality=100)
    return path


def test_strip_stats_and_usable_views(tmp_path: Path):
    s = strip_stats(_strip(tmp_path / "s.jpg"))
    assert s["readable"] and (s["width"], s["height"]) == (2880, 300)
    assert [round(s[f"view_mean_{i}"]) for i in range(3, 6)] == [120, 160, 200]
    assert s["view_std_0"] > 4 and s["view_std_5"] < 1  # the caption is not counted
    assert len(s["sha256"]) == 64
    table = pd.DataFrame([s])
    # views 1 and 2 are bright enough and striped; 0 is too dark, 3-5 flat
    assert usable_views(table, min_mean=20, min_std=4).tolist() == [2]
    (tmp_path / "bad.jpg").write_bytes(b"not a jpeg")
    assert strip_stats(tmp_path / "bad.jpg")["readable"] is False
    Image.new("RGB", (1000, 300)).save(tmp_path / "odd.jpg")
    assert strip_stats(tmp_path / "odd.jpg")["readable"] is False  # not six views


def test_crop_strip_cuts_the_caption(tmp_path: Path):
    for h, w in ((300, 2880), (246, 2304)):
        src = _strip(tmp_path / f"s{h}.jpg", h, w)
        sha, n, cw, ch = crop_strip(src, tmp_path / "out" / f"c{h}.jpg", quality=95)
        assert (cw, ch) == (w, h - 30)
        with Image.open(tmp_path / "out" / f"c{h}.jpg") as im:
            bottom = np.asarray(im.convert("L"))[-1, : w // 6]
        assert bottom.max() < 60  # view 0's dark rows, not the white caption
        assert n == (tmp_path / "out" / f"c{h}.jpg").stat().st_size and len(sha) == 64


def test_spatial_groups_chain_and_stable_labels():
    ids = ["B", "A", "C", "D"]
    # A-B 111 km apart, B-C 111 km, C-D 1,000 km away
    lat = [1.0, 0.0, 2.0, 11.0]
    lon = [0.0, 0.0, 0.0, 0.0]
    g = spatial_groups(ids, lat, lon, km=150)
    assert g == {"A": "gA", "B": "gA", "C": "gA", "D": "gD"}  # A-C chained through B
    assert spatial_groups(ids, lat, lon, km=100) == {"A": "gA", "B": "gB", "C": "gC", "D": "gD"}
    assert spatial_groups(["X"], [0.0], [0.0], km=300) == {"X": "gX"}
    # across the date line: 179.5 E and 179.5 W are 111 km apart at the equator
    assert len(set(spatial_groups(["E", "W"], [0, 0], [179.5, -179.5], km=150).values())) == 1
