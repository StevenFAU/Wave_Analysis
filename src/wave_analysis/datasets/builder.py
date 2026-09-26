"""ML manifest construction: task-specific filtering applied *transparently*.

The builder is where the preservation principle pays off. Upstream layers keep
every observation and annotate it; here a **declarative task specification**
decides what enters a given experiment. The builder writes:

* the manifest (one row per sample, with its split label);
* an **exclusion ledger** (one row per excluded sample and the rule that
  excluded it), so the population a result applies to is always explicit;
* a provenance record.

Example task specification (see ``configs/experiments/baseline_hs.yaml``)::

    task:
      target: Hs_m
      filters:
        - {column: sync_qc_flag, op: in, value: [1]}
        - {column: time_offset_s, op: abs_le, value: 1800}
        - {column: illumination, op: in, value: [day, low_sun]}
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wave_analysis.datasets.splits import SplitResult

_OPS = {
    "eq": lambda s, v: s == v,
    "ne": lambda s, v: s != v,
    "lt": lambda s, v: s < v,
    "le": lambda s, v: s <= v,
    "gt": lambda s, v: s > v,
    "ge": lambda s, v: s >= v,
    "in": lambda s, v: s.isin(v),
    "not_in": lambda s, v: ~s.isin(v),
    "abs_le": lambda s, v: s.abs() <= v,
    "notna": lambda s, v: s.notna(),
    "between": lambda s, v: s.between(v[0], v[1]),
}


@dataclass(frozen=True)
class FilterRule:
    """One declarative inclusion rule: keep rows where ``op(column, value)`` is true."""

    column: str
    op: str
    value: Any = None

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> FilterRule:
        """Build from a config mapping ``{column, op, value}``."""
        if m["op"] not in _OPS:
            raise ValueError(f"unknown filter op {m['op']!r}; choose from {sorted(_OPS)}")
        return cls(m["column"], m["op"], m.get("value"))

    def __str__(self) -> str:
        return f"{self.column} {self.op} {self.value!r}"

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Boolean mask of rows that satisfy the rule (NaN never satisfies)."""
        if self.column not in df:
            raise KeyError(f"filter column {self.column!r} not in dataset")
        mask = _OPS[self.op](df[self.column], self.value)
        return pd.Series(np.asarray(mask.fillna(False), dtype=bool), index=df.index)


@dataclass
class BuiltManifest:
    """Output of :func:`build_manifest`."""

    manifest: pd.DataFrame
    exclusions: pd.DataFrame
    rules: list[FilterRule]

    def write(self, directory: str | Path, name: str) -> dict[str, Path]:
        """Write manifest and exclusion ledger as Parquet + CSV (CSV for review/diffs)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        paths = {
            "manifest": d / f"{name}.parquet",
            "manifest_csv": d / f"{name}.csv",
            "exclusions": d / f"{name}.exclusions.csv",
        }
        self.manifest.to_parquet(paths["manifest"], index=False)
        self.manifest.to_csv(paths["manifest_csv"], index=False)
        self.exclusions.to_csv(paths["exclusions"], index=False)
        return paths


def build_manifest(
    samples: pd.DataFrame,
    *,
    id_col: str,
    target_cols: Sequence[str],
    filters: Sequence[Mapping[str, Any] | FilterRule] = (),
    split: SplitResult | None = None,
) -> BuiltManifest:
    """Apply filters in order, record the first rule excluding each sample, attach splits.

    Rows with a missing target are excluded by an implicit final rule
    ``target notna`` so that the ledger also accounts for label gaps.
    """
    if samples[id_col].duplicated().any():
        raise ValueError(f"{id_col} is not unique")
    rules = [r if isinstance(r, FilterRule) else FilterRule.from_mapping(r) for r in filters]
    rules += [FilterRule(t, "notna") for t in target_cols]
    keep = pd.Series(True, index=samples.index)
    reason = pd.Series(pd.NA, index=samples.index, dtype="string")
    for rule in rules:
        ok = rule.apply(samples)
        newly = keep & ~ok
        reason[newly] = str(rule)
        keep &= ok
    manifest = samples.loc[keep].copy()
    if split is not None:
        manifest["split"] = split.labels.reindex(manifest.index)
        manifest["split_protocol"] = split.protocol
    exclusions = pd.DataFrame(
        {id_col: samples.loc[~keep, id_col], "excluded_by": reason[~keep]}
    ).reset_index(drop=True)
    return BuiltManifest(manifest.reset_index(drop=True), exclusions, rules)
