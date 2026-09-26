"""Leakage-aware dataset splitting.

Adjacent video frames, overlapping clips, and consecutive buoy records are
strongly autocorrelated. A random frame-level split places frame *n* in
training and frame *n+1* in test and reports an optimistic, meaningless score.
All splitters here therefore assign **whole independent units** (station,
deployment, day, event, sequence, region) to a partition, and time-based
splits insert an **embargo gap** so that autocorrelation cannot bridge the
boundary.

Benchmark protocols (``docs/methodology/dataset_splitting.md``):

=====================  ==========================================================
Protocol               Question answered
=====================  ==========================================================
``iid_grouped``        Known stations, independent temporal blocks (e.g. days).
``cross_time``         Does performance hold on *later* data (drift, seasons)?
``cross_station``      Does the model transfer to *unseen cameras/buoys*?
``cross_region``       Does it transfer to unseen coastlines / wave climates?
``extreme``            Can it extrapolate to the highest sea states?
=====================  ==========================================================

Every splitter returns a :class:`SplitResult` recording its parameters, so the
exact split can be regenerated and cited.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TRAIN, VAL, TEST, EXCLUDED = "train", "val", "test", "excluded"


@dataclass
class SplitResult:
    """Partition labels for each row plus the recipe that produced them."""

    labels: pd.Series  # index aligned with the input frame; values in {train, val, test, excluded}
    protocol: str
    params: dict[str, Any] = field(default_factory=dict)

    def indices(self, part: str) -> pd.Index:
        """Row index of one partition."""
        return self.labels.index[self.labels == part]

    def summary(self) -> dict[str, int]:
        """Row counts per partition."""
        return {str(k): int(v) for k, v in self.labels.value_counts().items()}

    def to_frame(self) -> pd.DataFrame:
        """Two-column frame for manifest storage."""
        return self.labels.rename("split").to_frame()


def _check_fractions(fractions: Sequence[float]) -> tuple[float, float, float]:
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0) or min(fractions) < 0:
        raise ValueError(
            "fractions must be three non-negative numbers summing to 1 (train, val, test)"
        )
    return float(fractions[0]), float(fractions[1]), float(fractions[2])


def grouped_split(
    df: pd.DataFrame,
    group_col: str | Sequence[str],
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    *,
    seed: int = 0,
    protocol: str = "iid_grouped",
) -> SplitResult:
    """Assign whole groups to train/val/test, matching target fractions by sample count.

    Groups are shuffled with ``seed`` and filled greedily into the partition
    furthest below its target share; this handles very unequal group sizes
    better than splitting the group list by count.
    """
    ftr, fva, fte = _check_fractions(fractions)
    cols = [group_col] if isinstance(group_col, str) else list(group_col)
    keys = df[cols].astype(str).agg("|".join, axis=1)
    sizes = keys.value_counts()
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.index.to_numpy())
    targets = np.array([ftr, fva, fte]) * len(df)
    filled = np.zeros(3)
    assign: dict[str, str] = {}
    names = (TRAIN, VAL, TEST)
    for g in order:
        deficit = np.where(targets > 0, (targets - filled) / np.maximum(targets, 1), -np.inf)
        k = int(np.argmax(deficit))
        assign[g] = names[k]
        filled[k] += sizes[g]
    labels = keys.map(assign).rename("split")
    return SplitResult(
        labels, protocol, {"group_col": cols, "fractions": [ftr, fva, fte], "seed": seed}
    )


def time_block_split(
    df: pd.DataFrame,
    time_col: str,
    *,
    val_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    embargo: str | pd.Timedelta = "0h",
    group_col: str | None = None,
) -> SplitResult:
    """Cross-time split: train < ``val_start`` <= val < ``test_start`` <= test.

    Samples within ``embargo`` *after* each boundary are labelled
    ``excluded`` (kept in the table, not used), preventing autocorrelated
    neighbours from straddling partitions. If ``group_col`` is given (e.g. a
    storm-event id), a group that straddles a boundary is excluded entirely.
    """
    t = pd.to_datetime(df[time_col], utc=True)
    vs, ts = pd.Timestamp(val_start), pd.Timestamp(test_start)
    vs = vs.tz_localize("UTC") if vs.tzinfo is None else vs
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    if ts <= vs:
        raise ValueError("test_start must be after val_start")
    emb = pd.Timedelta(embargo)
    labels = pd.Series(TRAIN, index=df.index, name="split", dtype=object)
    labels[t >= vs] = VAL
    labels[t >= ts] = TEST
    labels[((t >= vs) & (t < vs + emb)) | ((t >= ts) & (t < ts + emb))] = EXCLUDED
    if group_col is not None:
        n_parts = labels[labels != EXCLUDED].groupby(df[group_col]).nunique()
        straddling = n_parts[n_parts > 1].index
        labels[df[group_col].isin(straddling)] = EXCLUDED
    return SplitResult(
        labels,
        "cross_time",
        {"val_start": str(vs), "test_start": str(ts), "embargo": str(emb), "group_col": group_col},
    )


def holdout_groups_split(
    df: pd.DataFrame,
    group_col: str,
    *,
    test_groups: Sequence[str],
    val_groups: Sequence[str] = (),
    protocol: str = "cross_station",
) -> SplitResult:
    """Explicit hold-out of named groups (stations, regions, cameras).

    Used for ``cross_station`` (``group_col='station_id'``) and
    ``cross_region`` (``group_col='region'``). Explicit lists (rather than a
    random draw) make the benchmark stable and citable.
    """
    overlap = set(test_groups) & set(val_groups)
    if overlap:
        raise ValueError(f"groups in both val and test: {sorted(overlap)}")
    g = df[group_col].astype(str)
    labels = pd.Series(TRAIN, index=df.index, name="split", dtype=object)
    labels[g.isin([str(x) for x in val_groups])] = VAL
    labels[g.isin([str(x) for x in test_groups])] = TEST
    missing = set(map(str, test_groups)) - set(g)
    return SplitResult(
        labels,
        protocol,
        {
            "group_col": group_col,
            "test_groups": list(map(str, test_groups)),
            "val_groups": list(map(str, val_groups)),
            "test_groups_absent_from_data": sorted(missing),
        },
    )


def extreme_condition_split(
    df: pd.DataFrame,
    value_col: str,
    *,
    quantile: float = 0.9,
    group_col: str | None = None,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> SplitResult:
    """Hold out the upper tail of ``value_col`` (e.g. Hs) as the test set.

    Tests extrapolation: the model never sees sea states above the
    ``quantile`` threshold during training. When ``group_col`` is given, any
    group containing an extreme sample goes entirely to test (so a storm's
    ramp-up cannot leak its peak), and the remaining groups are split
    train/val with :func:`grouped_split`.
    """
    thresh = float(df[value_col].quantile(quantile))
    extreme = df[value_col] > thresh
    labels = pd.Series(TRAIN, index=df.index, name="split", dtype=object)
    if group_col is not None:
        ext_groups = set(df.loc[extreme, group_col])
        is_test = df[group_col].isin(ext_groups)
    else:
        is_test = extreme
    labels[is_test] = TEST
    rest = df[~is_test]
    if len(rest) and val_fraction > 0:
        sub = (
            grouped_split(rest, group_col, (1 - val_fraction, val_fraction, 0.0), seed=seed)
            if group_col
            else SplitResult(
                pd.Series(
                    np.where(
                        np.random.default_rng(seed).random(len(rest)) < val_fraction, VAL, TRAIN
                    ),
                    index=rest.index,
                ),
                "tmp",
            )
        )
        labels[sub.labels.index] = sub.labels
    return SplitResult(
        labels,
        "extreme",
        {
            "value_col": value_col,
            "quantile": quantile,
            "threshold": thresh,
            "group_col": group_col,
            "seed": seed,
        },
    )


def leave_one_group_out(df: pd.DataFrame, group_col: str) -> Iterator[tuple[str, SplitResult]]:
    """Yield one cross-station/region fold per group (for small numbers of sites)."""
    for g in sorted(df[group_col].astype(str).unique()):
        yield g, holdout_groups_split(df, group_col, test_groups=[g], protocol=f"logo:{group_col}")


# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #


@dataclass
class LeakageReport:
    """Result of :func:`check_leakage`. ``ok`` is False if any check found a violation."""

    shared_groups: dict[str, list[str]] = field(default_factory=dict)
    temporal_neighbours: int = 0
    near_duplicate_pairs: int = 0

    @property
    def ok(self) -> bool:
        """True if no leakage was detected."""
        return (
            not any(self.shared_groups.values())
            and not self.temporal_neighbours
            and not self.near_duplicate_pairs
        )


def check_leakage(
    df: pd.DataFrame,
    split: SplitResult,
    *,
    group_cols: Sequence[str] = (),
    time_col: str | None = None,
    time_group_col: str | None = None,
    min_gap: str | pd.Timedelta = "0h",
    hash_col: str | None = None,
    max_hamming: int = 4,
) -> LeakageReport:
    """Audit a split for the three common leakage modes in wave/vision datasets.

    1. **Shared groups** - any value of ``group_cols`` present in both train and
       test (or val and test).
    2. **Temporal neighbours** - train and test samples from the same
       ``time_group_col`` (e.g. station) closer in time than ``min_gap``.
    3. **Near-duplicate frames** - image hashes (``hash_col``, e.g. from
       :func:`wave_analysis.qc.imagery.difference_hash`) within
       ``max_hamming`` bits across train and test.
    """
    from wave_analysis.qc.imagery import hamming_distance

    lab = split.labels
    report = LeakageReport()
    for col in group_cols:
        tr = set(df.loc[lab == TRAIN, col]) | set(df.loc[lab == VAL, col])
        te = set(df.loc[lab == TEST, col])
        report.shared_groups[col] = sorted(map(str, tr & te))

    gap = pd.Timedelta(min_gap)
    if time_col is not None and gap > pd.Timedelta(0):
        t = pd.to_datetime(df[time_col], utc=True)
        groups = df[time_group_col] if time_group_col else pd.Series("all", index=df.index)
        n = 0
        for _, idx in groups.groupby(groups).groups.items():
            tt = t.loc[idx]
            ll = lab.loc[idx]
            train_t = np.sort(tt[ll == TRAIN].to_numpy())
            test_t = tt[ll == TEST].to_numpy()
            if train_t.size == 0 or test_t.size == 0:
                continue
            pos = np.searchsorted(train_t, test_t)
            left = np.abs(test_t - train_t[np.clip(pos - 1, 0, train_t.size - 1)])
            right = np.abs(train_t[np.clip(pos, 0, train_t.size - 1)] - test_t)
            n += int((np.minimum(left, right) < gap.to_timedelta64()).sum())
        report.temporal_neighbours = n

    if hash_col is not None:
        tr_h = df.loc[lab == TRAIN, hash_col].astype("int64").tolist()
        te_h = df.loc[lab == TEST, hash_col].astype("int64").tolist()
        report.near_duplicate_pairs = sum(
            1 for a in te_h for b in tr_h if hamming_distance(int(a), int(b)) <= max_hamming
        )
    return report
