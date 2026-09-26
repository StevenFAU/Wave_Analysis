"""Common observation schema (standardized layer).

All in-situ scalar sources are normalised to a **long ("tidy") table** with one
row per (time, platform, sensor, variable). Long format keeps heterogeneous
providers in one schema, preserves per-value QC flags, and maps cleanly to
CF discrete-sampling-geometry (timeSeries) netCDF on export.

The column contract is defined once in :data:`OBSERVATION_COLUMNS` and enforced
by :func:`validate_observation_frame`. Pydantic's :class:`Observation` mirrors
the same contract for single records (APIs, tests, documentation).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from wave_analysis.qc.flags import QCFlag
from wave_analysis.schemas.variables import VARIABLES


class ProcessingLevel(StrEnum):
    """Processing stage of a record (see ``docs/ARCHITECTURE.md``)."""

    RAW = "raw"  # original upstream bytes
    STANDARDIZED = "standardized"  # units/time/names normalised, values unchanged
    DERIVED = "derived"  # computed by this project (e.g. Hm0 from S(f))
    SYNCHRONIZED = "synchronized"  # paired with other modalities


class TimeReference(StrEnum):
    """What instant a timestamp denotes relative to the acquisition window."""

    START = "start"
    END = "end"
    CENTER = "center"
    NOMINAL = "nominal"  # provider-rounded/reporting time; relation to window uncertain


class Observation(BaseModel):
    """A single scalar observation in the standardized layer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observation_id: str
    dataset_id: str
    station_id: str
    platform_id: str | None = None
    sensor_id: str | None = None
    deployment_id: str | None = None

    time_utc: AwareDatetime
    time_reference: TimeReference = TimeReference.NOMINAL
    averaging_period_s: float | None = Field(default=None, ge=0)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    depth_m: float | None = None

    variable: str
    value: float | None
    units: str
    method: str | None = Field(
        default=None,
        description="How the value was produced, e.g. 'provider_reported', 'spectral_hm0', 'zero_upcrossing_h13'.",
    )

    source_qc_flag: str | None = None
    project_qc_flag: QCFlag = QCFlag.NOT_EVALUATED

    source_file: str
    source_url: str | None = None
    download_timestamp: AwareDatetime | None = None
    source_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    processing_level: ProcessingLevel = ProcessingLevel.STANDARDIZED
    processing_version: str

    @field_validator("variable")
    @classmethod
    def _known_variable(cls, v: str) -> str:
        if v not in VARIABLES:
            raise ValueError(f"unknown variable {v!r}; register it in schemas/variables.py")
        return v

    @field_validator("time_utc", "download_timestamp")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.utcoffset() != pd.Timedelta(0):
            raise ValueError("timestamps must be UTC")
        return v


#: Column -> pandas dtype contract for observation tables.
OBSERVATION_COLUMNS: dict[str, str] = {
    "observation_id": "string",
    "dataset_id": "string",
    "station_id": "string",
    "platform_id": "string",
    "sensor_id": "string",
    "deployment_id": "string",
    "time_utc": "datetime64[ns, UTC]",
    "time_reference": "string",
    "averaging_period_s": "float64",
    "latitude": "float64",
    "longitude": "float64",
    "depth_m": "float64",
    "variable": "string",
    "value": "float64",
    "units": "string",
    "method": "string",
    "source_qc_flag": "string",
    "project_qc_flag": "int8",
    "source_file": "string",
    "source_url": "string",
    "download_timestamp": "datetime64[ns, UTC]",
    "source_checksum": "string",
    "processing_level": "string",
    "processing_version": "string",
}

REQUIRED_NON_NULL: tuple[str, ...] = (
    "observation_id",
    "dataset_id",
    "station_id",
    "time_utc",
    "variable",
    "units",
    "source_file",
    "processing_version",
)


class SchemaError(ValueError):
    """Raised when a table violates the observation contract."""


def empty_observation_frame() -> pd.DataFrame:
    """Return an empty frame with the canonical columns and dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in OBSERVATION_COLUMNS.items()})


def coerce_observation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing optional columns, order columns, and cast to canonical dtypes."""
    out = df.copy()
    for col, dtype in OBSERVATION_COLUMNS.items():
        if col not in out.columns:
            if dtype.startswith("datetime64"):
                out[col] = pd.Series(pd.NaT, index=out.index, dtype=dtype)
            elif dtype in ("float64", "int8"):
                out[col] = np.nan
            else:
                out[col] = pd.Series(pd.NA, index=out.index, dtype=dtype)
    for col, dtype in OBSERVATION_COLUMNS.items():
        if dtype.startswith("datetime64"):
            out[col] = pd.to_datetime(out[col], utc=True).dt.as_unit("ns")
        elif dtype == "int8":
            out[col] = out[col].fillna(int(QCFlag.NOT_EVALUATED)).astype("int8")
        else:
            out[col] = out[col].astype(dtype)  # type: ignore[call-overload]
    return out[list(OBSERVATION_COLUMNS)]


def validate_observation_frame(df: pd.DataFrame) -> None:
    """Validate an observation table against the contract.

    Checks: required columns present and non-null, known variables, canonical
    units per variable, UTC timestamps, QC flags in the QARTOD set, and unique
    ``observation_id``.

    Raises
    ------
    SchemaError
        With a message listing every violation found.
    """
    problems: list[str] = []
    missing = [c for c in OBSERVATION_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    for col in REQUIRED_NON_NULL:
        n = int(df[col].isna().sum())
        if n:
            problems.append(f"{col}: {n} null values")
    unknown = sorted(set(df["variable"].dropna()) - set(VARIABLES))
    if unknown:
        problems.append(f"unknown variables: {unknown}")
    for var, grp in df.groupby("variable", observed=True):
        spec = VARIABLES.get(str(var))
        if spec is None:
            continue
        bad_units = sorted(set(grp["units"].dropna()) - {spec.units})
        if bad_units:
            problems.append(f"{var}: units {bad_units} != canonical {spec.units!r}")
        if spec.is_angle:
            vals = grp["value"].dropna()
            if ((vals < 0) | (vals >= 360)).any():
                problems.append(f"{var}: angles outside [0, 360)")
    if str(df["time_utc"].dtype) != "datetime64[ns, UTC]":
        problems.append(f"time_utc dtype {df['time_utc'].dtype} is not datetime64[ns, UTC]")
    valid_flags = {int(f) for f in QCFlag}
    bad_flags = set(np.unique(df["project_qc_flag"].dropna().astype(int))) - valid_flags
    if bad_flags:
        problems.append(f"invalid project_qc_flag values: {sorted(bad_flags)}")
    dup = int(df["observation_id"].duplicated().sum())
    if dup:
        problems.append(f"{dup} duplicate observation_id values")
    if problems:
        raise SchemaError("; ".join(problems))


def make_observation_id(*parts: Any) -> str:
    """Deterministic observation identifier from its natural key.

    ``dataset:station:variable:ISO8601`` - stable across re-ingestion so that
    downstream manifests remain valid when the pipeline is re-run.
    """
    return ":".join(
        p.isoformat().replace("+00:00", "Z") if isinstance(p, datetime | pd.Timestamp) else str(p)
        for p in parts
    )
