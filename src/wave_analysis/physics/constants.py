"""Physical constants used throughout the package.

Values are SI. Where a quantity varies with location (e.g. gravitational
acceleration with latitude, seawater density with temperature and salinity),
the constant here is a documented nominal value; functions accept overrides.
"""

from __future__ import annotations

#: Standard gravitational acceleration [m s^-2] (CGPM 1901 conventional value).
G: float = 9.80665

#: Nominal seawater density [kg m^-3].
RHO_SEAWATER: float = 1025.0

#: Kinematic surface tension of seawater, sigma/rho [m^3 s^-2]; only relevant
#: for capillary waves (f >~ 5 Hz), which are outside the scope of this package.
SURFACE_TENSION_KINEMATIC: float = 7.4e-5

#: Mean Earth radius [m] (IUGG), used for great-circle distances.
EARTH_RADIUS_M: float = 6_371_008.8
