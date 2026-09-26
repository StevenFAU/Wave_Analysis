"""Non-learned and linear baselines - the floor every vision model must beat.

A vision model's skill is only meaningful relative to what is achievable
*without* looking at the image. These baselines answer that question:

* :class:`MeanBaseline` - predict the training mean (R^2 = 0 by construction on
  the training distribution). Kamagata et al. (2026) report R^2 = 0.107-0.134
  for Hs, i.e. only modestly above this floor.
* :class:`ClimatologyBaseline` - per-group (station x month) mean; captures
  seasonal and site climatology, the natural competitor for cross-time tests.
* :class:`PersistenceBaseline` - last available reference value (for
  forecasting-style or gap-filling evaluations with a causal lag).
* :class:`RidgeBaseline` - closed-form ridge regression on tabular features
  (e.g. image-quality statistics, wind speed); tests whether simple covariates
  already explain the target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass
class MeanBaseline:
    """Predict the mean of the training targets."""

    mean_: float = float("nan")

    def fit(self, y: ArrayLike) -> MeanBaseline:
        """Store the finite-sample mean of ``y``."""
        self.mean_ = float(np.nanmean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, n: int) -> FloatArray:
        """Return ``n`` copies of the training mean."""
        return np.full(n, self.mean_)


@dataclass
class ClimatologyBaseline:
    """Mean target per group (e.g. ``["station_id", "month"]``), falling back to the global mean."""

    keys: Sequence[str]
    table_: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    global_: float = float("nan")

    def fit(self, df: pd.DataFrame, target: str) -> ClimatologyBaseline:
        """Compute group means on the training frame."""
        self.table_ = df.groupby(list(self.keys), observed=True)[target].mean()
        self.global_ = float(df[target].mean())
        return self

    def predict(self, df: pd.DataFrame) -> FloatArray:
        """Look up each row's group mean."""
        idx = (
            pd.MultiIndex.from_frame(df[list(self.keys)])
            if len(self.keys) > 1
            else df[self.keys[0]]
        )
        vals = self.table_.reindex(idx).to_numpy(dtype=np.float64)
        return np.where(np.isfinite(vals), vals, self.global_)


@dataclass
class PersistenceBaseline:
    """Most recent reference value at least ``lag`` before each target time (causal)."""

    lag: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(0))

    def predict(
        self,
        target_times: pd.Series,
        ref_times: pd.Series,
        ref_values: ArrayLike,
        *,
        max_age: str = "3h",
    ) -> FloatArray:
        """Backward as-of lookup with a maximum age; NaN when no recent reference exists."""
        tt = (pd.to_datetime(target_times, utc=True) - self.lag).dt.as_unit("ns")
        t = pd.DataFrame({"t": tt, "_i": range(len(target_times))})
        r = pd.DataFrame(
            {
                "t": pd.to_datetime(ref_times, utc=True).dt.as_unit("ns"),
                "v": np.asarray(ref_values, dtype=float),
            }
        )
        m = pd.merge_asof(
            t.sort_values("t"),
            r.sort_values("t"),
            on="t",
            direction="backward",
            tolerance=pd.Timedelta(max_age),
        )
        return m.sort_values("_i")["v"].to_numpy(dtype=np.float64)


@dataclass
class RidgeBaseline:
    """Ridge regression with feature standardisation, solved in closed form."""

    alpha: float = 1.0
    coef_: FloatArray = field(default_factory=lambda: np.empty(0))
    intercept_: float = 0.0
    mu_: FloatArray = field(default_factory=lambda: np.empty(0))
    sd_: FloatArray = field(default_factory=lambda: np.empty(0))

    def fit(self, X: ArrayLike, y: ArrayLike) -> RidgeBaseline:
        """Fit on complete rows only."""
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        ok = np.all(np.isfinite(Xa), axis=1) & np.isfinite(ya)
        Xa, ya = Xa[ok], ya[ok]
        self.mu_ = Xa.mean(axis=0)
        self.sd_ = np.where(Xa.std(axis=0) > 0, Xa.std(axis=0), 1.0)
        Z = (Xa - self.mu_) / self.sd_
        self.intercept_ = float(ya.mean())
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(A, Z.T @ (ya - self.intercept_))
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict (NaN for rows with missing features)."""
        Z = (np.asarray(X, dtype=np.float64) - self.mu_) / self.sd_
        return np.asarray(Z @ self.coef_ + self.intercept_, dtype=np.float64)
