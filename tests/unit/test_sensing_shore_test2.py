"""Test 2's scorer on synthetic clips: ratios from training labels only, classes, bars."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd


def _clips(n_days: int = 15, per_day: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for d in range(n_days):
        hs_day = [1.4, 2.2, 3.0][
            d % 3
        ] + 0.05 * d  # each date its own sea; 3 classes on 5 dates each
        for i in range(per_day):
            hs = hs_day * (1 + 0.05 * rng.standard_normal())
            rows.append({
                "clip_id": f"cam-{d:02d}-{i:02d}", "camera": "cam", "local_day": f"2024-01-{d + 1:02d}",
                "fold": d % 5, "Hs_label_m": hs, "Tp_label_s": 8.0 + 0.3 * d,
                "pred_hs_clip": hs * 1.02, "pred_hs_window": hs * 1.1, "pred_hs_window_image": hs_day * 1.3,
                "pred_tp_clip": 9.0,
                **{c: rng.standard_normal() for c in ["log_T_pix", "log_T_m01", "log_T_m02", "frac_25_16s",
                   "frac_16_12s", "frac_12_9s", "frac_9_6s", "frac_6_3s", "log10_band_power", "rate_share"]},
                "good_mean_height_m": 0.63 * hs, "good_h_third_m": 0.98 * hs, "good_h_tenth_m": 1.25 * hs,
                "good_tz_s": 6.0 + 0.2 * d, "good_wave_count": 200,
            })  # fmt: skip
    d = pd.DataFrame(rows)
    for c in ["mean_height_m", "h_third_m", "h_tenth_m", "tz_s", "wave_count"]:
        d[f"incl_unevaluated_{c}"] = d[f"good_{c}"]
    return d


def test_test2_arms_and_bars(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    m = importlib.import_module("sensing_shore_test2")
    d = _clips()
    o = m.arms(d, "good")
    # the ratio comes from training labels: 0.63 exactly, whatever the predictions are
    assert np.allclose(o["ratio_mean_height"], 0.63)
    assert np.allclose(o["pred_mean_height_clip"], 0.63 * d["pred_hs_clip"])
    # the majority class is the training fold's in-range mode, never the test fold's
    assert set(o["kim_majority"]) <= set(range(5))
    r = m.score(o, "good")
    assert r["kim"]["assessable"]
    # too few dates: unassessable, recorded as such rather than as a miss
    few = m.score(m.arms(d[d["local_day"] <= "2024-01-06"], "good"), "good")
    assert not few["kim"]["assessable"]
    assert few["bars"]["T2-A (class accuracy, clip - majority)"] == "unassessable (coverage)"
    # near-perfect clip estimates beat the majority class and the day-level snapshot
    assert r["kim"]["accuracy"]["clip"] > r["kim"]["accuracy"]["majority"]
    assert r["bars"]["T2-A (class accuracy, clip - majority)"] == "met"
    assert r["regression"]["mean_height"]["clip"] < r["regression"]["mean_height"]["snapshot"]
