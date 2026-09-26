"""Station, platform and deployment records.

A station identifier is **not** an immutable sensor configuration: hulls,
payloads and instruments change between deployments (e.g. NDBC's
``station_table.txt`` lists 41010 as a "2.1-meter ionomer foam buoy" with a
SCOOP payload - a *current* snapshot that need not describe the station's
archive back to 1988). Every physical-measurement claim therefore references a
:class:`Deployment`, and sensors (``schemas.sensor``) are attached to
deployments with validity intervals.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class PlatformType(StrEnum):
    """Kinds of observing platform."""

    MOORED_BUOY = "moored_buoy"
    DRIFTING_BUOY = "drifting_buoy"
    FIXED_STATION = "fixed_station"  # C-MAN, pier, platform
    SEAFLOOR = "seafloor"  # bottom-mounted ADCP / pressure sensor
    SHORE_CAMERA = "shore_camera"
    SHIP = "ship"
    USV = "usv"
    UAS = "uas"
    SATELLITE = "satellite"
    MODEL = "model"  # numerical model grid point (not a physical platform)
    SIMULATION = "simulation"  # digital twin / synthetic


class Station(BaseModel):
    """A named observing location as published by a provider."""

    model_config = ConfigDict(extra="forbid")

    station_id: str
    provider: str
    name: str | None = None
    platform_type: PlatformType
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    water_depth_m: float | None = Field(default=None, gt=0)
    owner: str | None = None
    source_url: str | None = None
    metadata_url: str | None = None


class Deployment(BaseModel):
    """A time interval during which a station had a fixed hull/payload/position."""

    model_config = ConfigDict(extra="forbid")

    deployment_id: str
    station_id: str
    start_utc: AwareDatetime | None = None
    end_utc: AwareDatetime | None = None  # None = ongoing
    hull: str | None = None
    payload: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    water_depth_m: float | None = Field(default=None, gt=0)
    evidence: str | None = Field(
        default=None, description="Where this deployment information came from (URL, document)."
    )

    @model_validator(mode="after")
    def _ordered(self) -> Deployment:
        if self.start_utc and self.end_utc and self.end_utc <= self.start_utc:
            raise ValueError("deployment end must be after start")
        return self
