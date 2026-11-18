"""Grouped folds of the buoy-camera heights (scripts/buoycam_cv_folds.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _folds():
    spec = importlib.util.spec_from_file_location(
        "buoycam_cv_folds", ROOT / "scripts/buoycam_cv_folds.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_assign_folds_balances_strips_and_keeps_groups_whole():
    strips = pd.Series(
        {"west": 9440, "alaska": 3852, "ne": 3055, **{f"g{i}": 600 for i in range(40)}}
    )
    folds = _folds().assign_folds(strips, 5)
    assert set(folds) == set(strips.index)  # each group in exactly one fold
    load = strips.groupby(pd.Series(folds)).sum()
    assert folds["west"] == 0 and load[0] == 9440  # the largest group stands alone
    others = load.drop(0)
    assert others.max() - others.min() <= 600
