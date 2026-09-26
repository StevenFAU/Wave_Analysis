"""Dataset construction: leakage-aware splits, ML manifests, provenance."""

from wave_analysis.datasets.builder import BuiltManifest, FilterRule, build_manifest
from wave_analysis.datasets.provenance import build_provenance, write_provenance, write_ro_crate
from wave_analysis.datasets.splits import (
    SplitResult,
    check_leakage,
    extreme_condition_split,
    grouped_split,
    holdout_groups_split,
    leave_one_group_out,
    time_block_split,
)

__all__ = [
    "BuiltManifest",
    "FilterRule",
    "SplitResult",
    "build_manifest",
    "build_provenance",
    "check_leakage",
    "extreme_condition_split",
    "grouped_split",
    "holdout_groups_split",
    "leave_one_group_out",
    "time_block_split",
    "write_provenance",
    "write_ro_crate",
]
