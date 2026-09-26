"""Statistical baselines (no optional dependencies)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wave_analysis.models import (
    ClimatologyBaseline,
    MeanBaseline,
    PersistenceBaseline,
    RidgeBaseline,
)


def test_mean_and_climatology():
    df = pd.DataFrame({"st": ["a", "a", "b", "b"], "m": [1, 1, 1, 2], "y": [1.0, 3.0, 5.0, 7.0]})
    assert MeanBaseline().fit(df["y"]).predict(2).tolist() == [4.0, 4.0]
    clim = ClimatologyBaseline(["st", "m"]).fit(df, "y")
    pred = clim.predict(pd.DataFrame({"st": ["a", "b", "c"], "m": [1, 2, 1]}))
    assert pred.tolist() == [2.0, 7.0, 4.0]  # unseen group -> global mean


def test_persistence_is_causal():
    ref_t = pd.Series(pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"], utc=True))
    tgt_t = pd.Series(
        pd.to_datetime(["2024-01-01 00:30", "2024-01-01 00:59", "2024-01-01 06:00"], utc=True)
    )
    out = PersistenceBaseline().predict(tgt_t, ref_t, [1.0, 2.0], max_age="3h")
    assert out[:2].tolist() == [1.0, 1.0] and np.isnan(out[2])


def test_ridge_recovers_linear_relation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = 2 * X[:, 0] - X[:, 2] + 0.5
    m = RidgeBaseline(alpha=1e-6).fit(X, y)
    np.testing.assert_allclose(m.predict(X), y, atol=1e-4)
