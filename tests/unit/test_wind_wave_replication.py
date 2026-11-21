"""Disjoint confirmation windows and replay of frozen, fold-specific weights."""

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import wind_wave_replication as replication


def cut(stamp):
    return {"until_utc": stamp, "rule_met": True}


def test_cutoffs_require_two_complete_disjoint_weeks():
    start, end = replication.closed_window(cut("2026-11-21T16:00Z"), cut("2026-12-12T00:00Z"))
    assert start == pd.Timestamp("2026-11-28T00:00Z") and end == pd.Timestamp("2026-12-12T00:00Z")
    late = cut("2026-11-30T00:00Z")
    with pytest.raises(ValueError, match="follow-up coverage"):
        replication.closed_window(late, cut("2026-12-14T23:59Z"))
    start, _ = replication.closed_window(late, cut("2026-12-15T00:00Z"))
    assert start == pd.Timestamp("2026-12-01T00:00Z")
    with pytest.raises(ValueError, match="completed"):
        replication.closed_window({"until_utc": "2026-11-25T10:00Z", "rule_met": False}, late)


def test_frozen_followup_uses_the_same_strips_for_every_arm(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    stamps = ["2026-11-27T23:59Z", "2026-11-28T00:00Z", "2026-12-11T23:59Z", "2026-12-12T00:00Z"]
    labels = pd.DataFrame(
        {
            "sample_id": list("abcd"),
            "station_id": ["40111", "40111", "40122", "40122"],
            "cv_fold": ["0", "0", "1", "1"],
            "station_group": ["g1", "g1", "g2", "g2"],
            "time_utc": stamps,
            "wind_speed_mps": [1.0, 2.0, 3.0, 4.0],
            "wind_wave_label_m": [1.0, 2.0, 3.0, 4.0],
        }
    )
    rows = labels[replication.KEYS].assign(
        arrangement="fresh",
        baseline="model_wind",
        target=replication.wr.WIND,
        forecast=1.0,
        camera=2.0,
        fusion=1.2,
        varweight=1.4,
        observed=labels.wind_wave_label_m,
    )
    rows.to_parquet(base / "c4b_rows.parquet", index=False)
    metadata = {"cut": cut(stamps[-1]), "C4b": {"model_wind": {replication.wr.WIND: {"strips": 4}}}}
    (base / "claims.json").write_text(json.dumps(metadata))
    args = Namespace(
        november_cut=tmp_path / "november.json",
        c4b_run=base,
        labels=tmp_path / "labels.csv",
        out=tmp_path / "result",
    )
    args.november_cut.write_text(json.dumps(cut("2026-11-22T05:00Z")))
    labels = pd.concat([labels, labels.iloc[[1]].assign(sample_id="e")], ignore_index=True)
    labels.to_csv(args.labels, index=False)
    frozen = json.loads(replication.WEIGHTS.read_text())
    replication.score_followup(args, frozen)
    meta = json.loads((args.out / "summary.json").read_text())
    assert (meta["available_package_strips"], meta["matched_strips"], meta["matched_groups"]) == (
        3,
        2,
        2,
    )
    scored = pd.read_parquet(args.out / "rows.parquet").reset_index(drop=True)
    assert list(scored.sample_id) == ["b", "c"]
    assert set(pd.read_csv(args.out / "scores.csv").query("scope == 'overall'").strips) == {2}
    # The predictions can be reproduced from the three inputs and fold without a reference label.
    blind = scored[["cv_fold", "camera", "forecast", "wind"]]
    replay = replication.apply_frozen(blind, frozen["folds"])
    for arm in ["constant", "two_inputs", "reliability"]:
        np.testing.assert_array_equal(scored[arm], replay[arm])
