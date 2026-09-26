"""Ocean-wave physics: linear dispersion, wave statistics, parametric spectra.

These routines are deliberately free of provider-specific assumptions so they
can be validated against analytic results and textbook values (see
``tests/unit/test_dispersion.py`` and ``docs/methodology/wave_parameters.md``).
"""

from wave_analysis.physics.constants import G
from wave_analysis.physics.dispersion import (
    group_velocity,
    phase_velocity,
    wave_steepness,
    wavelength,
    wavenumber,
)
from wave_analysis.physics.timeseries import ZeroCrossingResult, zero_crossing_analysis

__all__ = [
    "G",
    "ZeroCrossingResult",
    "group_velocity",
    "phase_velocity",
    "wave_steepness",
    "wavelength",
    "wavenumber",
    "zero_crossing_analysis",
]
