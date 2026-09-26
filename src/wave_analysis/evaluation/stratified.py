"""Stratified evaluation: metrics by station, sea state, illumination, weather, geometry.

"One global metric is not enough" (technical summary §43). A model whose
global RMSE looks acceptable may fail entirely at night or above 3 m. This
module computes any metric per stratum with sample counts and (optionally)
dependence-aware bootstrap intervals, producing the tables required by
``docs/methodology/evaluation.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise

import numpy as np
import pandas as pd

from wave_analysis.evaluation.uncertainty import block_bootstrap_ci

Metric = Callable[[np.ndarray, np.ndarray], float]


def metrics_by_stratum(
    df: pd.DataFrame,
    *,
    true_col: str,
    pred_col: str,
    strata: str | Sequence[str],
    metrics: Mapping[str, Metric],
    bootstrap_group_col: str | None = None,
    n_boot: int = 0,
    min_n: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute ``metrics`` per stratum.

    If ``n_boot > 0``, adds ``<metric>_lo``/``<metric>_hi`` columns from a
    cluster bootstrap over ``bootstrap_group_col`` (or a moving-block bootstrap
    if it is ``None``). Strata with fewer than ``min_n`` rows are reported with
    NaN metrics rather than dropped, so thin coverage stays visible.
    """
    keys = [strata] if isinstance(strata, str) else list(strata)
    rows = []
    for key, g in df.groupby(keys, dropna=False, observed=True):
        rec: dict[str, object] = dict(
            zip(keys, key if isinstance(key, tuple) else (key,), strict=True)
        )
        rec["n"] = len(g)
        y, p = g[true_col].to_numpy(dtype=float), g[pred_col].to_numpy(dtype=float)
        for name, fn in metrics.items():
            if len(g) < min_n:
                rec[name] = np.nan
                continue
            if n_boot > 0:
                groups = g[bootstrap_group_col].to_numpy() if bootstrap_group_col else None
                pt, lo, hi = block_bootstrap_ci(y, p, fn, groups=groups, n_boot=n_boot, seed=seed)
                rec[name], rec[f"{name}_lo"], rec[f"{name}_hi"] = pt, lo, hi
            else:
                rec[name] = fn(y, p)
        rows.append(rec)
    return pd.DataFrame(rows)


def sea_state_bins(
    hs: pd.Series, edges: Sequence[float] = (0, 0.5, 1.25, 2.5, 4.0, 6.0, np.inf)
) -> pd.Series:
    """Bin Hs into sea-state classes (defaults follow WMO sea-state code boundaries)."""
    labels = [f"[{a:g}, {b:g})" for a, b in pairwise(edges)]
    return pd.cut(hs, bins=list(edges), labels=labels, right=False)
