"""Canonical variable registry: project keys, CF standard names, canonical units.

Rule (``docs/DATA_MODEL.md``): *whenever a CF standard name exists, use it*.
Every variable that enters the standardized layer is described here once, so
units, names, and provenance notes are never re-invented per adapter.

CF standard names were checked against the CF Standard Name Table
(https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html).
Where a provider's definition does not map cleanly onto a CF name, the
``note`` field records the discrepancy rather than hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariableSpec:
    """Definition of one canonical variable."""

    key: str
    standard_name: str | None
    units: str
    long_name: str
    note: str = ""
    is_angle: bool = False


_VARS: tuple[VariableSpec, ...] = (
    # ---- waves -------------------------------------------------------------
    VariableSpec(
        "wave_significant_height",
        "sea_surface_wave_significant_height",
        "m",
        "Significant wave height",
        "Estimator (spectral Hm0 vs time-domain H1/3) recorded in the observation 'method' field. "
        "NDBC WVHT is documented as H1/3 but is reproduced to rounding by 4*sqrt(m0) of the "
        "published spectrum (scripts/validate_ndbc_hm0.py).",
    ),
    VariableSpec(
        "wave_peak_period",
        "sea_surface_wave_period_at_variance_spectral_density_maximum",
        "s",
        "Peak (dominant) wave period",
        "NDBC DPD. Quantised by the frequency grid.",
    ),
    VariableSpec(
        "wave_mean_period",
        "sea_surface_wave_mean_period",
        "s",
        "Average wave period",
        "NDBC APD, described by NDBC as the average period of all waves in the 20-min record. "
        "Exact estimator (Tm02 vs zero-crossing) should be confirmed per payload.",
    ),
    VariableSpec(
        "wave_mean_period_tm01",
        "sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment",
        "s",
        "Mean wave period m0/m1",
    ),
    VariableSpec(
        "wave_mean_period_tm02",
        "sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment",
        "s",
        "Mean wave period sqrt(m0/m2)",
    ),
    VariableSpec(
        "wave_zero_upcrossing_period",
        "sea_surface_wave_zero_upcrossing_period",
        "s",
        "Mean zero-upcrossing period (time domain)",
    ),
    VariableSpec(
        "wave_maximum_height",
        "sea_surface_wave_maximum_height",
        "m",
        "Maximum individual wave height",
    ),
    VariableSpec(
        "wave_from_direction_at_peak",
        "sea_surface_wave_from_direction_at_variance_spectral_density_maximum",
        "degree",
        "Wave direction (from) at the spectral peak",
        "NDBC MWD: direction from which waves at DPD are coming, degrees true.",
        is_angle=True,
    ),
    VariableSpec(
        "wave_mean_from_direction",
        "sea_surface_wave_from_direction",
        "degree",
        "Energy-weighted mean wave direction (from)",
        is_angle=True,
    ),
    VariableSpec(
        "swell_significant_height",
        "sea_surface_swell_wave_significant_height",
        "m",
        "Swell significant height",
        "NDBC SwH; partition by NDBC separation frequency.",
    ),
    VariableSpec(
        "swell_period",
        "sea_surface_swell_wave_period",
        "s",
        "Swell period",
        "NDBC SwP.",
    ),
    VariableSpec(
        "swell_from_direction",
        "sea_surface_swell_wave_from_direction",
        "degree",
        "Swell direction (from)",
        "NDBC SwD, reported as a 16-point compass string; converted to degrees (22.5 deg resolution).",
        is_angle=True,
    ),
    VariableSpec(
        "wind_wave_significant_height",
        "sea_surface_wind_wave_significant_height",
        "m",
        "Wind-wave significant height",
        "NDBC WWH.",
    ),
    VariableSpec(
        "wind_wave_period", "sea_surface_wind_wave_period", "s", "Wind-wave period", "NDBC WWP."
    ),
    VariableSpec(
        "wind_wave_from_direction",
        "sea_surface_wind_wave_from_direction",
        "degree",
        "Wind-wave direction (from)",
        "NDBC WWD, 16-point compass string converted to degrees.",
        is_angle=True,
    ),
    VariableSpec(
        "wave_spectral_density",
        "sea_surface_wave_variance_spectral_density",
        "m2 s",
        "Wave variance spectral density S(f)",
        "m2 s == m2/Hz.",
    ),
    # ---- meteorology -------------------------------------------------------
    VariableSpec(
        "wind_speed",
        "wind_speed",
        "m s-1",
        "Wind speed",
        "NDBC WSPD; 8-min (buoy) or 2-min (C-MAN) average.",
    ),
    VariableSpec(
        "wind_from_direction",
        "wind_from_direction",
        "degree",
        "Wind direction (from)",
        is_angle=True,
    ),
    VariableSpec("wind_speed_of_gust", "wind_speed_of_gust", "m s-1", "Peak 5- or 8-s gust speed"),
    VariableSpec(
        "air_pressure_at_sea_level",
        "air_pressure_at_mean_sea_level",
        "hPa",
        "Sea-level pressure",
        "NDBC PRES (historical header BAR before 2007).",
    ),
    VariableSpec(
        "air_pressure_tendency",
        "tendency_of_air_pressure",
        "hPa",
        "3-hour pressure tendency",
        "NDBC PTDY; units are hPa per 3 h.",
    ),
    VariableSpec("air_temperature", "air_temperature", "degree_Celsius", "Air temperature"),
    VariableSpec(
        "sea_surface_temperature",
        "sea_surface_temperature",
        "degree_Celsius",
        "Sea surface temperature",
        "NDBC WTMP; sensor depth varies by hull.",
    ),
    VariableSpec(
        "dew_point_temperature", "dew_point_temperature", "degree_Celsius", "Dew-point temperature"
    ),
    VariableSpec(
        "visibility",
        "visibility_in_air",
        "m",
        "Station visibility",
        "NDBC reports nmi (realtime) or mi (historical); normalised to m.",
    ),
    VariableSpec(
        "water_level",
        "water_surface_height_above_reference_datum",
        "m",
        "Water level relative to MLLW",
        "NDBC TIDE, feet relative to MLLW; normalised to m.",
    ),
)

VARIABLES: dict[str, VariableSpec] = {v.key: v for v in _VARS}


def get(key: str) -> VariableSpec:
    """Look up a canonical variable, raising ``KeyError`` with a helpful message."""
    try:
        return VARIABLES[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown canonical variable {key!r}; add it to schemas/variables.py"
        ) from exc
