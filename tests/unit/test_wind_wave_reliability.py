"""Reliability blending, nested folds, and availability of local labels."""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import wind_wave_reliability as wr


def test_weight_bounds_and_empty_cells():
    train = pd.DataFrame(
        {"forecast": [1.0, 2.0, 3.0], "disagreement": [0.2, 0.4, 1.0], "wind": [2.0, 6.0, 8.0]}
    )
    train["observed"] = train.forecast + 0.25 * train.disagreement
    fit = wr.fit_weights(train, wr.INPUTS)
    assert fit["constant"] == pytest.approx(0.25)
    assert 0 in fit["counts"]
    np.testing.assert_allclose(fit["weights"], 0.25)
    fit = wr.fit_weights(train.assign(disagreement=0), wr.INPUTS)
    np.testing.assert_array_equal(fit["weights"], 0.0)
    fit = wr.fit_weights(train.assign(observed=-1), wr.INPUTS)
    np.testing.assert_array_equal(fit["weights"], 0.0)


def test_outer_labels_do_not_set_transfer_weights():
    outer = pd.DataFrame(
        {
            "cv_fold": list("012"),
            "sample_id": list("abc"),
            "station_group": list("ABC"),
            "forecast": 1.0,
            "disagreement": 0.4,
            "wind": 5.0,
            "observed": 1.2,
        }
    )
    inner = pd.concat(
        [
            outer[outer.cv_fold.isin([a, b])].assign(pair=f"pair{a}{b}")
            for a, b in combinations("012", 2)
        ]
    )
    result, _ = wr.transfer(inner, outer.assign(observed=20.0))
    np.testing.assert_allclose(result.reliability, 1.2)
    # Changing labels in the pair's rows for fold 0 cannot affect its own fit.
    inner.loc[inner.cv_fold == "0", "observed"] = 50.0
    changed, _ = wr.transfer(inner, outer)
    np.testing.assert_allclose(changed.loc[changed.cv_fold == "0", "reliability"], 1.2)
    with pytest.raises(ValueError, match="calibration population"):
        wr.transfer(pd.concat([inner, inner.iloc[[0]]]), outer)


def test_local_offset_uses_only_the_first_week():
    stamps = list(pd.date_range("2026-10-01", periods=24, freq="h", tz="UTC"))
    stamps.extend(pd.to_datetime(["2026-10-08", "2026-10-10", "2026-10-11"], utc=True))
    site = pd.DataFrame(
        {
            "station_id": "A",
            "cv_fold": "2",
            "time_utc": stamps,
            "observed": 1.5,
            "forecast": 1.0,
            "disagreement": 1.0,
            "constant": 1.2,
            "reliability": 1.25,
            "weight_constant": 0.2,
            "weight_reliability": 0.25,
            "inner_mean_d2": 0.1,
        }
    )
    later, counts = wr.local_calibration(site)
    assert counts.loc[0, ["early", "embargo", "later"]].tolist() == [24, 1, 2]
    assert later.local_reliability.iloc[0] == pytest.approx(1.25 + 24 * 0.25 / 29)
    site.loc[24:, "observed"] = -100
    short = site.drop(index=1).assign(station_id="B")
    later2, counts2 = wr.local_calibration(pd.concat([site, short], ignore_index=True))
    # Keep the same start when removing one calibration sample.
    assert set(later2.station_id) == {"A"}
    assert list(counts2.eligible) == [True, False]
    np.testing.assert_array_equal(later.local_reliability, later2.local_reliability)


def test_forward_fits_exclude_cutoff_gap_and_scoring_labels():
    times = [wr.EARLY_END - pd.Timedelta(seconds=1), wr.EARLY_END, wr.LATER_START]
    outer = pd.DataFrame(
        [
            {
                "cv_fold": fold,
                "station_group": fold,
                "sample_id": f"{fold}-{i}",
                "time_utc": stamp,
                "cmp_hs": 2.0,
                "forecast": 1.0,
                "camera": 1.4,
                "disagreement": 0.4,
                "wind": 5.0,
                "observed": 1.1,
            }
            for fold in "012"
            for i, stamp in enumerate(times)
        ]
    )
    inner = pd.concat(
        [
            outer[outer.cv_fold.isin([a, b])].assign(pair=f"pair{a}{b}")
            for a, b in combinations("012", 2)
        ],
        ignore_index=True,
    )
    rows, fits = wr.transfer(inner, outer, forward=True)
    assert set(rows.sample_id) == {"0-2", "1-2", "2-2"}
    assert all(f["reliability"]["rows"] == 2 for f in fits.values())
    inner.loc[inner.time_utc >= wr.EARLY_END, "observed"] = 50.0
    changed, new_fits = wr.transfer(inner, outer.assign(observed=20.0), forward=True)
    assert fits == new_fits
    pd.testing.assert_frame_equal(rows[wr.FORWARD_ARMS], changed[wr.FORWARD_ARMS])
