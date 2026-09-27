"""Synchronisation, splits, leakage audit, manifest builder, provenance."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from wave_analysis.datasets.builder import FilterRule, build_manifest
from wave_analysis.datasets.provenance import (
    build_provenance,
    config_hash,
    git_state,
    write_provenance,
    write_ro_crate,
)
from wave_analysis.datasets.splits import (
    EXCLUDED,
    TEST,
    TRAIN,
    VAL,
    check_leakage,
    extreme_condition_split,
    grouped_split,
    holdout_groups_split,
    leave_one_group_out,
    time_block_split,
)
from wave_analysis.processing.geospatial import haversine_m, initial_bearing_deg
from wave_analysis.processing.synchronize import (
    pair_interval_overlap,
    pair_nearest,
    pair_window,
    propagation_lag,
)
from wave_analysis.qc.flags import QCFlag


def _samples(n_days=20, per_day=24, stations=("A", "B", "C", "D")):
    rows = []
    for s in stations:
        for d in range(n_days):
            for h in range(per_day):
                rows.append(
                    {
                        "station_id": s,
                        "time_utc": pd.Timestamp("2024-01-01", tz="UTC")
                        + pd.Timedelta(days=d, hours=h),
                    }
                )
    df = pd.DataFrame(rows)
    df["day"] = df["time_utc"].dt.strftime("%Y-%m-%d")
    df["Hs"] = 1 + np.sin(np.arange(len(df)) / 50.0) ** 2 * 3
    df["sample_id"] = [f"s{i}" for i in range(len(df))]
    return df


# ----------------------------------------------------------------------------- synchronisation


def test_pair_nearest_keeps_offsets_and_unmatched():
    imgs = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(
                ["2018-04-02 11:10", "2018-04-02 12:10", "2018-04-02 20:00"], utc=True
            ),
            "img": ["a", "b", "c"],
        }
    )
    waves = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(["2018-04-02 10:40", "2018-04-02 11:40"], utc=True),
            "Hs": [1.0, 1.2],
        }
    )
    out = pair_nearest(imgs, waves, tolerance="45min", suspect_offset="20min")
    assert out["img"].tolist() == ["a", "b", "c"]  # input order preserved, nothing dropped
    assert out["time_offset_s"].iloc[0] == 1800  # the Choi et al. (2020) 30-minute offset
    assert out["sync_qc_flag"].tolist() == [
        int(QCFlag.SUSPECT),
        int(QCFlag.SUSPECT),
        int(QCFlag.MISSING),
    ]
    back = pair_nearest(imgs, waves, tolerance="45min", direction="backward")
    assert (back["time_offset_s"].dropna() >= 0).all()


def test_pair_interval_overlap():
    clips = pd.DataFrame(
        {
            "s": pd.to_datetime(["2024-01-01 00:05"], utc=True),
            "e": pd.to_datetime(["2024-01-01 00:08"], utc=True),
        }
    )
    recs = pd.DataFrame(
        {
            "s": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:07"], utc=True),
            "e": pd.to_datetime(["2024-01-01 00:20", "2024-01-01 00:27"], utc=True),
        }
    )
    out = pair_interval_overlap(
        clips, recs, target_start="s", target_end="e", reference_start="s", reference_end="e"
    )
    assert len(out) == 1 and out["overlap_fraction"].iloc[0] == pytest.approx(1.0)
    assert out["center_offset_s"].iloc[0] == pytest.approx((6.5 - 10) * 60)


def _cdip_like_records(starts, index=None):
    s = pd.to_datetime(starts, utc=True)
    return pd.DataFrame({"start_utc": s, "end_utc": s + pd.Timedelta(seconds=1600)}, index=index)


def test_pair_window_keeps_every_record_with_offsets():
    # CDIP-like 1600-s samples every 30 min, one missing (01:30).
    recs = _cdip_like_records(
        ["2024-01-01 00:00", "2024-01-01 00:30", "2024-01-01 01:00", "2024-01-01 02:00"],
        index=[10, 11, 12, 13],
    )
    imgs = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(
                ["2024-01-01 00:55:00", None, "2024-01-01 06:00:00", "2024-01-01 00:28:20"],
                utc=True,
            )
        },
        index=["a", "no_time", "far", "tie"],
    )
    out = pair_window(imgs, recs, window="1h")
    assert set(out["target_index"]) == {"a", "tie"}  # no rows without a time or a record
    a = out[out["target_index"] == "a"].set_index("reference_index")
    # centres 00:13:20, 00:43:20, 01:13:20; 02:13:20 is 78 min away
    assert a["offset_s"].to_dict() == {10: 2500.0, 11: 700.0, 12: -1100.0}
    assert a["in_record"].to_dict() == {10: False, 11: True, 12: False}
    assert a["rank"].to_dict() == {11: 0, 12: 1, 10: 2}
    tie = out[out["target_index"] == "tie"].set_index("reference_index")
    # 00:28:20 is 15 min from two centres and inside neither sample (the 200-s gap)
    assert tie.loc[10, "offset_s"] == 900.0 and tie.loc[11, "offset_s"] == -900.0
    assert tie.loc[10, "rank"] == 0 and tie.loc[11, "rank"] == 1  # earlier record wins
    assert not tie["in_record"].any()


def test_pair_window_validates_and_handles_empty():
    recs = _cdip_like_records(["2024-01-01 00:00"])
    empty = pair_window(pd.DataFrame({"time_utc": pd.to_datetime([], utc=True)}), recs, window="1h")
    assert empty.empty and list(empty.columns) == [
        "target_index",
        "reference_index",
        "offset_s",
        "in_record",
        "rank",
    ]
    bad = recs.assign(end_utc=recs["start_utc"] - pd.Timedelta(seconds=1))
    with pytest.raises(ValueError, match="ends before"):
        pair_window(pd.DataFrame({"time_utc": recs["start_utc"]}), bad, window="1h")


def test_propagation_lag_deep_water():
    # 10 km at Tp = 10 s in deep water: cg = gT/(4 pi) ~ 7.8 m/s -> ~21 min
    assert propagation_lag(10_000, 10.0) / 60 == pytest.approx(21.4, abs=0.2)


def test_geodesy():
    assert haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)
    assert initial_bearing_deg(0, 0, 1, 0) == pytest.approx(0.0, abs=1e-9)
    assert initial_bearing_deg(0, 0, 0, 1) == pytest.approx(90.0, abs=1e-9)


# ----------------------------------------------------------------------------- splits


def test_grouped_split_no_group_overlap_and_reproducible():
    df = _samples()
    a = grouped_split(df, ["station_id", "day"], seed=3)
    b = grouped_split(df, ["station_id", "day"], seed=3)
    pd.testing.assert_series_equal(a.labels, b.labels)
    keys = df["station_id"] + df["day"]
    assert (keys.groupby(a.labels.values).apply(set).apply(len).sum()) == keys.nunique()
    frac = a.labels.value_counts(normalize=True)
    assert frac[TRAIN] == pytest.approx(0.7, abs=0.05)
    assert check_leakage(df, a, group_cols=["day"]).shared_groups[
        "day"
    ]  # days shared across stations...
    assert check_leakage(df.assign(k=keys), a, group_cols=["k"]).ok  # ...but not station-days


def test_time_block_split_embargo():
    df = _samples()
    r = time_block_split(
        df, "time_utc", val_start="2024-01-13", test_start="2024-01-17", embargo="12h"
    )
    t = df["time_utc"]
    assert (r.labels[t < "2024-01-13"] == TRAIN).all()
    assert (r.labels[(t >= "2024-01-13") & (t < "2024-01-13 12:00")] == EXCLUDED).all()
    assert (r.labels[t >= "2024-01-17 12:00"] == TEST).all()
    rep = check_leakage(df, r, time_col="time_utc", time_group_col="station_id", min_gap="6h")
    assert rep.temporal_neighbours == 0


def test_random_frame_split_is_detected_as_leaky():
    df = _samples()
    rng = np.random.default_rng(0)
    from wave_analysis.datasets.splits import SplitResult

    labels = pd.Series(np.where(rng.random(len(df)) < 0.8, TRAIN, TEST), index=df.index)
    rep = check_leakage(
        df,
        SplitResult(labels, "random"),
        time_col="time_utc",
        time_group_col="station_id",
        min_gap="2h",
    )
    assert not rep.ok and rep.temporal_neighbours > 100


def test_holdout_and_logo():
    df = _samples()
    r = holdout_groups_split(df, "station_id", test_groups=["D"], val_groups=["C"])
    assert set(df.loc[r.labels == TEST, "station_id"]) == {"D"}
    assert set(df.loc[r.labels == VAL, "station_id"]) == {"C"}
    with pytest.raises(ValueError):
        holdout_groups_split(df, "station_id", test_groups=["D"], val_groups=["D"])
    folds = list(leave_one_group_out(df, "station_id"))
    assert [g for g, _ in folds] == ["A", "B", "C", "D"]


def test_extreme_split_holds_out_tail_by_group():
    df = _samples()
    df["event"] = df["station_id"] + df["day"]
    r = extreme_condition_split(df, "Hs", quantile=0.9, group_col="event")
    thr = r.params["threshold"]
    assert df.loc[r.labels != TEST, "Hs"].max() <= thr
    ev_test = set(df.loc[r.labels == TEST, "event"])
    assert not ev_test & set(df.loc[r.labels != TEST, "event"])


def test_near_duplicate_detection_in_audit():
    df = pd.DataFrame({"h": [0b1111, 0b1110, 0xFF00FF]})
    from wave_analysis.datasets.splits import SplitResult

    r = SplitResult(pd.Series([TRAIN, TEST, TEST]), "x")
    assert check_leakage(df, r, hash_col="h").near_duplicate_pairs == 1


# ----------------------------------------------------------------------------- builder / provenance


def test_manifest_builder_records_exclusions(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d", "e"],
            "Hs": [1.0, 2.0, np.nan, 1.5, 3.0],
            "sync_qc_flag": [1, 1, 1, 3, 1],
            "time_offset_s": [0, 2400, 0, 0, 60],
        }
    )
    built = build_manifest(
        df,
        id_col="sample_id",
        target_cols=["Hs"],
        filters=[
            {"column": "sync_qc_flag", "op": "in", "value": [1]},
            {"column": "time_offset_s", "op": "abs_le", "value": 1800},
        ],
    )
    assert built.manifest["sample_id"].tolist() == ["a", "e"]
    reasons = dict(zip(built.exclusions["sample_id"], built.exclusions["excluded_by"], strict=True))
    assert reasons == {
        "b": "time_offset_s abs_le 1800",
        "c": "Hs notna None",
        "d": "sync_qc_flag in [1]",
    }
    paths = built.write(tmp_path, "m")
    assert all(p.exists() for p in paths.values())
    with pytest.raises(ValueError):
        FilterRule.from_mapping({"column": "x", "op": "bogus"})


def test_provenance_and_ro_crate(tmp_path: Path):
    inp = tmp_path / "in.txt"
    out = tmp_path / "out.txt"
    inp.write_text("a")
    out.write_text("b")
    rec = build_provenance(artifact_id="x", inputs=[inp], outputs=[out], config={"b": 1, "a": 2})
    assert rec["pipeline"]["config_sha256"] == config_hash({"a": 2, "b": 1})
    assert rec["input_files"][0]["sha256"].startswith("ca978112")
    p = write_provenance(rec, tmp_path / "p.yaml")
    assert yaml.safe_load(p.read_text())["artifact_id"] == "x"
    crate = write_ro_crate(
        tmp_path,
        name="t",
        description="d",
        license_id="https://creativecommons.org/licenses/by/4.0/",
        data_files=[out],
        author_name="Steven Cohen",
    )
    graph = json.loads(crate.read_text())["@graph"]
    assert any(e.get("@id") == "out.txt" and e.get("sha256") for e in graph)


def test_git_state_ignores_untracked_files(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.org",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "x",
    )
    (tmp_path / "untracked.txt").write_text("local")
    assert git_state(tmp_path)["dirty"] is False
    (tmp_path / "tracked.txt").write_text("a")
    git("add", "tracked.txt")
    assert git_state(tmp_path)["dirty"] is True
