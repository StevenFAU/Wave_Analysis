"""The camera + hindcast combiner and the paired day bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from wave_analysis.evaluation.uncertainty import paired_block_bootstrap_ci
from wave_analysis.models.fusion import fit_logistic, inverse_frequency_weights


def test_logistic_recovers_ordered_classes_from_one_feature() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 4, 2000)
    y = np.digitize(x, [1.0, 2.0, 3.0])  # four classes, thresholds 1, 2, 3
    m = fit_logistic(x, y, 4, l2=1e-6)
    assert (m.predict(x[:, None]) == y).mean() > 0.97
    lp = m.log_proba(np.array([[0.5], [3.5]]))
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0)
    assert lp[0].argmax() == 0 and lp[1].argmax() == 3


def test_class_weights_move_the_boundary_towards_the_common_class() -> None:
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, 900), rng.normal(2, 1, 100)])
    y = np.r_[np.zeros(900, int), np.ones(100, int)]
    w = inverse_frequency_weights(y, 2)
    assert w.mean() == pytest.approx(1.0) and w[-1] / w[0] == pytest.approx(9.0)
    plain = fit_logistic(x, y, 2).predict(x[:, None]).mean()
    weighted = fit_logistic(x, y, 2, sample_weight=w).predict(x[:, None]).mean()
    assert weighted > plain  # the rare class is predicted more often


def test_paired_bootstrap_of_a_difference() -> None:
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    groups = np.repeat(np.arange(4), 2)
    acc = lambda a, b: float(np.mean(a == b))  # noqa: E731
    same = paired_block_bootstrap_ci(y, y, y, acc, groups=groups, n_boot=200)
    assert same == (0.0, 0.0, 0.0)
    worse = np.where(np.arange(8) == 0, 1 - y, y)
    v, lo, hi = paired_block_bootstrap_ci(y, y, worse, acc, groups=groups, n_boot=500)
    assert v == pytest.approx(0.125) and 0.0 <= lo <= v <= hi
