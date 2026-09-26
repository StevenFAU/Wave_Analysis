"""Sensor records and the modality taxonomy.

The taxonomy distinguishes a sensor's *modality* from its *role* in a dataset.
The same ADCP can be a **reference** (label source, as in Kim et al. 2023) or an
**input** modality (multimodal architecture), and the same IMU can be a
nuisance-compensation input (Yang et al. 2026) or a target. Recording the role
explicitly per dataset prevents accidental label leakage (e.g. using the
reference buoy's own wind sensor as an input to predict its wave height).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class Modality(StrEnum):
    """Sensor modality classes (technical summary §38)."""

    IN_SITU_WAVE = "in_situ_wave"  # Waverider, Spotter, NDBC buoy accelerometer
    ACOUSTIC = "acoustic"  # ADCP / AWAC
    RADAR = "radar"  # X-band marine radar, HF radar
    SAR = "sar"
    RGB = "rgb"
    THERMAL_IR = "thermal_ir"
    MULTISPECTRAL = "multispectral"
    STEREO = "stereo"  # stereo video (e.g. WASS) - geometric surface reconstruction
    IMU = "imu"
    GNSS = "gnss"
    WIND = "wind"
    CURRENT = "current"
    BATHYMETRY = "bathymetry"
    ALTIMETRY = "altimetry"
    NUMERICAL_MODEL = "numerical_model"
    DIGITAL_TWIN = "digital_twin"
    METEOROLOGICAL = "meteorological"


class SensorRole(StrEnum):
    """Role of a sensor within a particular dataset or experiment."""

    REFERENCE = "reference"  # physical ground truth / label source
    INPUT = "input"  # model input
    CONTEXT = "context"  # covariate for stratified evaluation only
    CALIBRATION = "calibration"  # used to calibrate another sensor


class Sensor(BaseModel):
    """A physical (or virtual) sensor attached to a deployment."""

    model_config = ConfigDict(extra="forbid")

    sensor_id: str
    station_id: str
    deployment_id: str | None = None
    modality: Modality
    sensor_model: str | None = None
    manufacturer: str | None = None
    sampling_interval_s: float | None = Field(default=None, gt=0)
    sampling_rate_hz: float | None = Field(default=None, gt=0)
    record_length_s: float | None = Field(default=None, gt=0)
    height_or_depth_m: float | None = None
    variables: list[str] = Field(default_factory=list)
    valid_from_utc: AwareDatetime | None = None
    valid_to_utc: AwareDatetime | None = None
    qc_level: str | None = None
    source_url: str | None = None
