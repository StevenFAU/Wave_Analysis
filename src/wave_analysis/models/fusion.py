"""Combining a camera classifier with a wave hindcast: weighted multinomial logistic regression.

The question it answers (``docs/experiments/currituck_sea_state.md``): does
the camera add information to the hindcast, or only repeat it? A combiner of
the camera's class log-probabilities and the hindcast's log Hs is compared
with the same combiner fitted on the hindcast alone, on the same stills, so
that the gain from refitting is not counted as the camera's.

NumPy and SciPy only (no scikit-learn): standardized features, inverse
class-frequency weights, an L2 penalty on the coefficients (not the
intercepts), fitted with L-BFGS on the exact gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import log_softmax

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LogisticModel:
    """A fitted combiner: ``softmax(((x - mean) / scale) @ coef + intercept)``."""

    mean: FloatArray
    scale: FloatArray
    coef: FloatArray  # (n_features, n_classes)
    intercept: FloatArray  # (n_classes,)

    def log_proba(self, x: ArrayLike) -> FloatArray:
        """Class log-probabilities, one row per sample."""
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return np.asarray(log_softmax(z @ self.coef + self.intercept, axis=1), dtype=np.float64)

    def predict(self, x: ArrayLike) -> NDArray[np.int64]:
        """The most probable class of each sample."""
        return np.asarray(self.log_proba(x).argmax(axis=1), dtype=np.int64)


def inverse_frequency_weights(y: ArrayLike, n_classes: int) -> FloatArray:
    """Weight of each sample: the inverse frequency of its class, scaled to mean 1 over samples."""
    yy = np.asarray(y, dtype=np.int64)
    counts = np.bincount(yy, minlength=n_classes).astype(np.float64)
    per_class = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
    w = per_class[yy]
    return np.asarray(w / w.mean(), dtype=np.float64)


def fit_logistic(
    x: ArrayLike,
    y: ArrayLike,
    n_classes: int,
    *,
    sample_weight: ArrayLike | None = None,
    l2: float = 1e-3,
) -> LogisticModel:
    """Minimize the weighted mean cross-entropy plus ``l2 * ||coef||^2``."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx[:, None]
    yy = np.asarray(y, dtype=np.int64)
    n, d = xx.shape
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    w = w / w.sum()
    mean = xx.mean(axis=0)
    scale = xx.std(axis=0)
    scale[scale == 0] = 1.0
    z = (xx - mean) / scale
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), yy] = 1.0

    def loss_grad(theta: FloatArray) -> tuple[float, FloatArray]:
        coef = theta[: d * n_classes].reshape(d, n_classes)
        b = theta[d * n_classes :]
        lp = log_softmax(z @ coef + b, axis=1)
        loss = -float(np.sum(w * lp[np.arange(n), yy])) + l2 * float(np.sum(coef**2))
        r = (np.exp(lp) - onehot) * w[:, None]
        g_coef = z.T @ r + 2 * l2 * coef
        return loss, np.concatenate([g_coef.ravel(), r.sum(axis=0)])

    theta0 = np.zeros(d * n_classes + n_classes)
    res = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    if not res.success:
        raise RuntimeError(f"logistic fit did not converge: {res.message}")
    coef = res.x[: d * n_classes].reshape(d, n_classes)
    return LogisticModel(mean, scale, coef, res.x[d * n_classes :])


__all__ = ["LogisticModel", "fit_logistic", "inverse_frequency_weights"]
