"""Quality-control flag vocabulary and aggregation.

The project uses the QARTOD / UNESCO (2013) primary-level flag scheme
throughout, for *all* layers (provider, physical, temporal, sensor, visual,
synchronisation, ML). Using one vocabulary everywhere means flags can be
aggregated and filtered uniformly at task-construction time.

========  =====  ==========================================================
Flag      Value  Meaning (IOOS QARTOD Waves Manual v2.1, Table 3-1)
========  =====  ==========================================================
PASS      1      Passed critical real-time QC tests.
NOT_EVAL  2      Not QC-tested, or quality information not available.
SUSPECT   3      Suspect *or of high interest* - kept, never deleted.
FAIL      4      Failed one or more critical QC checks.
MISSING   9      Data are missing; placeholder.
========  =====  ==========================================================

Note "suspect **or of high interest**": the QARTOD definition itself supports
this project's principle that hard or unusual observations are annotated, not
discarded (``docs/QUALITY_CONTROL.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class QCFlag(IntEnum):
    """QARTOD primary flag values."""

    PASS = 1
    NOT_EVALUATED = 2
    SUSPECT = 3
    FAIL = 4
    MISSING = 9


# Aggregation precedence: higher rank wins.
_RANK = {
    QCFlag.NOT_EVALUATED: 0,
    QCFlag.PASS: 1,
    QCFlag.SUSPECT: 2,
    QCFlag.FAIL: 3,
    QCFlag.MISSING: 4,
}
_RANK_LUT = np.zeros(10, dtype=np.int8)
_INV_LUT = np.zeros(5, dtype=np.int8)
for _flag, _rank in _RANK.items():
    _RANK_LUT[int(_flag)] = _rank
    _INV_LUT[_rank] = int(_flag)


def aggregate(*flags: ArrayLike) -> NDArray[np.int8]:
    """Combine flag arrays element-wise into one aggregate flag.

    Precedence: MISSING > FAIL > SUSPECT > PASS > NOT_EVALUATED. A value is
    PASS only if at least one test was evaluated and none flagged it; this is
    the same logic as ``ioos_qc`` "qartod_aggregate".
    """
    if not flags:
        raise ValueError("at least one flag array is required")
    arrs = np.broadcast_arrays(*(np.asarray(f, dtype=np.int8) for f in flags))
    ranks = np.stack([_RANK_LUT[a] for a in arrs])
    return np.asarray(_INV_LUT[ranks.max(axis=0)], dtype=np.int8)


@dataclass(frozen=True)
class QCTestResult:
    """Outcome of one QC test applied to one variable.

    Keeping per-test results (not only the aggregate) makes every exclusion
    explainable: a training manifest can state *which* test removed a sample
    and with which parameters.
    """

    test: str
    layer: str
    flags: NDArray[np.int8]
    params: dict[str, Any] = field(default_factory=dict)
    reference: str = ""

    def counts(self) -> dict[str, int]:
        """Number of values in each flag category."""
        vals, n = np.unique(self.flags, return_counts=True)
        return {QCFlag(int(v)).name: int(c) for v, c in zip(vals, n, strict=True)}
