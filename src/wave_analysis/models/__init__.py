"""Models: statistical baselines (always available) and reference deep architectures (``ml`` extra).

Deep-learning modules import ``torch`` and are therefore **not** imported here;
import them explicitly, e.g. ``from wave_analysis.models.baselines.choi2020 import Choi2020Regressor``.
See ``docs/methodology/models.md`` for what each reference implementation
reproduces and which details remain unverified.
"""

from wave_analysis.models.baselines.statistical import (
    ClimatologyBaseline,
    MeanBaseline,
    PersistenceBaseline,
    RidgeBaseline,
)

__all__ = ["ClimatologyBaseline", "MeanBaseline", "PersistenceBaseline", "RidgeBaseline"]
