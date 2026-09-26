"""Layered quality control: provider -> QARTOD/physical -> imagery -> synchronisation -> ML.

See ``docs/QUALITY_CONTROL.md``. No function in this package removes data.
"""

from wave_analysis.qc.flags import QCFlag, QCTestResult, aggregate

__all__ = ["QCFlag", "QCTestResult", "aggregate"]
