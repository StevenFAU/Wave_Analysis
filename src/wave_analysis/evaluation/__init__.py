"""Evaluation: regression, circular, classification, agreement, uncertainty, stratified.

See ``docs/methodology/evaluation.md`` for the required reporting set.
"""

from wave_analysis.evaluation.circular import circular_report
from wave_analysis.evaluation.classification import classification_report
from wave_analysis.evaluation.regression import bland_altman, regression_report
from wave_analysis.evaluation.stratified import metrics_by_stratum
from wave_analysis.evaluation.uncertainty import block_bootstrap_ci

__all__ = [
    "bland_altman",
    "block_bootstrap_ci",
    "circular_report",
    "classification_report",
    "metrics_by_stratum",
    "regression_report",
]
