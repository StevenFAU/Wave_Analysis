"""QARTOD tests, physical checks, imagery metrics, solar geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wave_analysis.physics.spectra_models import cos2s_moments
from wave_analysis.processing.solar import glint_risk, illumination_category, solar_position
from wave_analysis.qc import qartod
from wave_analysis.qc.flags import QCFlag, aggregate
from wave_analysis.qc.imagery import difference_hash, hamming_distance, image_quality
from wave_analysis.qc.waves import (
    directional_moment_test,
    spectral_consistency_test,
    steepness_test,
)

P, NE, S, F, M = (
    int(x) for x in (QCFlag.PASS, QCFlag.NOT_EVALUATED, QCFlag.SUSPECT, QCFlag.FAIL, QCFlag.MISSING)
)


def test_aggregate_precedence():
    out = aggregate([P, P, NE, S, P, NE], [NE, S, NE, F, M, P])
    assert out.tolist() == [P, S, NE, F, M, P]


def test_gross_range():
    r = qartod.gross_range_test([0.5, 30.0, 0.01, np.nan, 12.0], (0.0, 25.0), (0.05, 10.0))
    assert r.flags.tolist() == [P, F, S, M, S]
    assert r.reference.endswith("Test 19")


def test_height_failure_propagates():
    assert qartod.propagate_height_failure([F, P, F], [P, P, M]).tolist() == [F, P, M]


def test_flat_line_follows_manual_counts():
    v = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    r = qartod.flat_line_test(v, suspect_count=3, fail_count=5)
    assert r.flags.tolist() == [P, P, S, S, F, P, P, P, S, S, S]  # zeros never FAIL


def test_rate_of_change_and_scaling():
    t = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    assert qartod.rate_of_change_test([1.0, 1.5, 4.0, 4.1], t, 2.0).flags.tolist() == [P, P, F, P]
    t2 = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 06:00"], tz="UTC")
    # 2.5 m change over 6 h is allowed at 2 m per 3 h
    assert qartod.rate_of_change_test([1.0, 3.5], t2, 2.0, per_seconds=3 * 3600).flags.tolist() == [
        P,
        P,
    ]


def test_gap_test():
    t = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 04:00", "2024-01-01 04:00"], tz="UTC"
    )
    assert qartod.gap_test(t, 3600).flags.tolist() == [P, P, S, F]


def test_mean_std_flags_outlier():
    t = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
    v = 1.0 + 0.05 * np.sin(np.arange(30))
    v[25] = 5.0
    r = qartod.mean_std_test(v, t, window="12h", n_std=3)
    assert r.flags[25] == S and (r.flags[:25] != S).all()


def test_spectral_tests():
    f = np.array([0.01, 0.03, 0.1, 0.45, 0.7])
    assert qartod.operational_frequency_range_test(f, (0.02, 0.6), (0.03, 0.4)).flags.tolist() == [
        F,
        P,
        P,
        S,
        F,
    ]
    S2 = np.array([[1.0, 0.0, 0.0], [1.0, 5.0, 0.0]])
    r = qartod.low_frequency_energy_test(
        [0.02, 0.04, 0.2], S2, [0.01, 0.01, 0.01], f_cutoff=0.05, energy_span=(0, 0.02)
    )
    assert r.flags.tolist() == [P, S]


def test_neighbor():
    r = qartod.neighbor_test([1.0, 2.0, 1.0], [1.1, 3.0, np.nan], 0.5)
    assert r.flags.tolist() == [P, S, NE]


def test_steepness():
    r = steepness_test([2.0, 2.0, 4.0, np.nan], [10.0, 5.0, 3.0, 8.0])
    # s_p = 0.0128 (pass), 0.0513 (pass), 0.285 (fail), missing
    assert r.flags.tolist() == [P, P, F, M]
    assert steepness_test([2.5], [4.5]).flags[0] == S  # s_p ~ 0.079


def test_spectral_consistency():
    r = spectral_consistency_test([1.0, 1.0, 3.0, np.nan], [1.05, 1.3, 3.2, 1.0])
    assert r.flags.tolist() == [P, S, P, NE]


def test_directional_moment_realisability():
    r1, r2 = cos2s_moments(5.0)
    ok = directional_moment_test([10.0], [10.0], [r1], [r2])
    assert ok.flags.tolist() == [P]
    # r2 = 1 with r1 = 0 is realisable (bimodal); r1 = 0.9 with r2 = 0 is not.
    bad = directional_moment_test(
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.9, 1.2], [1.0, 0.0, 0.5]
    )
    assert bad.flags.tolist() == [P, S, F]


def test_image_quality_metrics():
    dark = np.zeros((64, 64, 3), dtype=np.uint8)
    q = image_quality(dark)
    assert q.dark_fraction == 1.0 and q.saturation_fraction == 0.0 and q.sharpness == 0.0
    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    qw = image_quality(white)
    assert qw.saturation_fraction == 1.0 and qw.glare_fraction == 1.0
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert image_quality(noise).sharpness > 0.01
    assert image_quality(noise).entropy_bits > 7


def test_dhash_near_duplicates():
    rng = np.random.default_rng(1)
    img = rng.random((120, 160))
    near = np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1)
    other = rng.random((120, 160))
    assert hamming_distance(difference_hash(img), difference_hash(near)) <= 4
    assert hamming_distance(difference_hash(img), difference_hash(other)) > 10


def test_solar_position_reference_values():
    # Greenwich near the June solstice at 12:00 UTC: elevation ~ 90 - 51.4769 + 23.44.
    el, az = solar_position(pd.Timestamp("2024-06-20 12:00", tz="UTC"), 51.4769, 0.0)
    assert el[0] == pytest.approx(61.96, abs=0.15)
    assert az[0] == pytest.approx(180.0, abs=2.0)
    # Midnight at the same site is night.
    el2, _ = solar_position(pd.Timestamp("2024-06-20 00:00", tz="UTC"), 51.4769, 0.0)
    assert illumination_category(el2)[0] == "night" or el2[0] < -6


def test_illumination_and_glint():
    cats = illumination_category([30, 3, -3, -9, -20])
    assert cats.tolist() == ["day", "low_sun", "civil_twilight", "nautical_twilight", "night"]
    assert glint_risk(
        [10, 10, 60, -5], [90, 200, 90, 90], camera_azimuth_deg=90, camera_hfov_deg=60
    ).tolist() == [
        True,
        False,
        False,
        False,
    ]
