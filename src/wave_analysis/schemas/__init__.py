"""Typed records shared across the pipeline (see ``docs/DATA_MODEL.md``)."""

from wave_analysis.schemas.imagery import (
    CameraExtrinsics,
    CameraIntrinsics,
    ImageRecord,
    SpectralBand,
    ThermalMetadata,
)
from wave_analysis.schemas.observation import (
    OBSERVATION_COLUMNS,
    Observation,
    ProcessingLevel,
    SchemaError,
    TimeReference,
    validate_observation_frame,
)
from wave_analysis.schemas.sample import MultimodalSample
from wave_analysis.schemas.sensor import Modality, Sensor, SensorRole
from wave_analysis.schemas.station import Deployment, PlatformType, Station
from wave_analysis.schemas.variables import VARIABLES, VariableSpec

__all__ = [
    "OBSERVATION_COLUMNS",
    "VARIABLES",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "Deployment",
    "ImageRecord",
    "Modality",
    "MultimodalSample",
    "Observation",
    "PlatformType",
    "ProcessingLevel",
    "SchemaError",
    "Sensor",
    "SensorRole",
    "SpectralBand",
    "Station",
    "ThermalMetadata",
    "TimeReference",
    "VariableSpec",
    "validate_observation_frame",
]
