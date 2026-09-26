"""Evaluation metrics against hand-computed and reference values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from wave_analysis.evaluation import circular as C
from wave_analysis.evaluation import regression as R
from wave_analysis.evaluation.classification import (
    bin_values,
    classification_report,
    quadratic_weighted_kappa,
)
from wave_analysis.evaluation.stratified import metrics_by_stratum, sea_state_bins
from wave_analysis.evaluation.uncertainty import (
    block_bootstrap_ci,
    calibration_curve,
    gaussian_crps,
    interval_coverage,
)

Y = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
P = np.array([1.5, 1.5, 3.5, 3.5, 2.0])


def test_basic_regression_metrics_ignore_nan():
    assert R.bias(Y, P) == pytest.approx(0.0)
    assert R.mae(Y, P) == pytest.approx(0.5)
    assert R.rmse(Y, P) == pytest.approx(0.5)
    assert R.r2(Y, P) == pytest.approx(1 - 1.0 / 5.0)
    assert R.pearson(Y, P) == pytest.approx(stats.pearsonr(Y[:4], P[:4]).statistic)
    assert R.regression_report(Y, P)["n"] == 4


def test_nrmse_normalisers_and_kamagata_convention():
    y = np.array([0.85, 1.5, 2.80])
    p = y + np.array([0.5618, -0.5618, 0.5618])
    assert R.nrmse(y, p, "range") == pytest.approx(0.5618 / 1.95)  # = 28.81 %
    assert R.nrmse(y, p, "mean") == pytest.approx(0.5618 / y.mean())


def test_bland_altman():
    rng = np.random.default_rng(0)
    y = rng.uniform(0.5, 4, 5000)
    p = y + 0.1 + rng.normal(0, 0.2, y.size)
    ba = R.bland_altman(y, p)
    assert ba.bias == pytest.approx(0.1, abs=0.01)
    assert ba.loa_upper - ba.loa_lower == pytest.approx(2 * 1.96 * 0.2, rel=0.05)
    assert ba.proportional_p > 0.001  # no proportional bias injected
    # Regressing on the pair mean manufactures a slope when only the prediction is noisy.
    assert R.bland_altman(y, p, against="mean").proportional_p < 1e-4
    # A genuine proportional error is detected.
    assert R.bland_altman(y, y * 1.2).proportional_slope == pytest.approx(0.2)


def test_constant_prediction_gives_nan_correlation_without_warning(recwarn):
    y = np.array([1.0, 2.0, 3.0, 4.0])
    rep = R.regression_report(y, np.full(4, 2.5))
    assert np.isnan(rep["pearson"]) and np.isnan(rep["spearman"])
    assert rep["slope"] == pytest.approx(0.0)
    assert not recwarn.list


def test_regression_dilution_slope():
    y = np.linspace(0.85, 2.8, 100)
    p = y.mean() + 0.12 * (y - y.mean())
    assert R.regression_slope(y, p)[0] == pytest.approx(0.12)


def test_circular_metrics_wrap():
    assert C.angular_difference(359, 1) == pytest.approx(2.0)
    assert C.angular_difference(1, 359) == pytest.approx(-2.0)
    assert C.circular_mae([359, 1], [1, 359]) == pytest.approx(2.0)
    assert C.circular_mean([350, 10]) % 360 == pytest.approx(0.0, abs=1e-9)
    assert C.circular_bias([0, 90, 180], [10, 100, 190]) == pytest.approx(10.0)
    np.testing.assert_allclose(
        C.decode_angles(C.encode_angles([0, 90, 359.5])), [0, 90, 359.5], atol=1e-9
    )
    # Linear RMSE on raw degrees is badly wrong across north; circular RMSE is not.
    y, p = np.array([355.0, 5.0]), np.array([5.0, 355.0])
    assert R.rmse(y, p) == pytest.approx(350.0)
    assert C.circular_rmse(y, p) == pytest.approx(10.0)


def test_circular_correlation():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 360, 500)
    assert C.circular_correlation(a, a + rng.normal(0, 5, a.size)) > 0.95
    assert abs(C.circular_correlation(a, rng.uniform(0, 360, a.size))) < 0.15


def test_classification_report_ordinal():
    y = [0, 1, 1, 2, 3, 4, 4]
    p = [0, 1, 2, 2, 2, 4, 3]
    rep = classification_report(y, p, n_classes=5)
    assert rep["accuracy"] == pytest.approx(4 / 7)
    assert rep["within_one_accuracy"] == pytest.approx(1.0)
    assert rep["ordinal_mae"] == pytest.approx(3 / 7)
    cm = np.array(rep["confusion_matrix"])
    assert cm.sum() == 7 and cm[3, 2] == 1
    assert quadratic_weighted_kappa(np.eye(3, dtype=np.int64) * 5) == pytest.approx(1.0)


def test_bin_values_excludes_out_of_range():
    # Choi et al. (2020) binary task: <1.52 m low, >3.05 m high, middle excluded.
    idx = bin_values([1.0, 2.0, 3.5, np.nan], [0, 1.52, 3.05, 99])
    assert idx.tolist() == [0, 1, 2, -1]


def test_block_bootstrap_wider_than_iid_for_autocorrelated_errors():
    rng = np.random.default_rng(0)
    n = 2000
    e = np.zeros(n)
    for i in range(1, n):  # AR(1) errors, phi = 0.95
        e[i] = 0.95 * e[i - 1] + rng.normal(0, 0.1)
    y = np.ones(n)
    p = y + e
    _, lo_b, hi_b = block_bootstrap_ci(y, p, R.bias, block_length=100, n_boot=400)
    _, lo_i, hi_i = block_bootstrap_ci(y, p, R.bias, block_length=1, n_boot=400)
    assert (hi_b - lo_b) > 2 * (hi_i - lo_i)


def test_probabilistic_scores():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 20000)
    cal = calibration_curve(y, np.zeros_like(y), np.ones_like(y))
    for nominal, emp in cal.items():
        assert emp == pytest.approx(nominal, abs=0.01)
    assert gaussian_crps([0.0], [0.0], [1.0]) == pytest.approx(
        (np.sqrt(2) - 1) / np.sqrt(np.pi), rel=1e-6
    )
    assert interval_coverage([1, 2, 3], [0, 2.5, 2], [2, 3, 4]) == pytest.approx(2 / 3)


def test_metrics_by_stratum():
    df = pd.DataFrame(
        {"y": [1.0, 2, 3, 4], "p": [1.0, 2, 3.5, 4.5], "illum": ["day", "day", "night", "night"]}
    )
    out = metrics_by_stratum(
        df, true_col="y", pred_col="p", strata="illum", metrics={"mae": R.mae}
    ).set_index("illum")
    assert out.loc["day", "mae"] == 0 and out.loc["night", "mae"] == pytest.approx(0.5)
    bins = sea_state_bins(pd.Series([0.2, 1.0, 7.0]))
    assert bins.tolist()[0] == "[0, 0.5)" and bins.tolist()[-1] == "[6, inf)"
