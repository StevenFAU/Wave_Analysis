"""Synchronised multimodal sample record (technical summary §39).

A :class:`MultimodalSample` is the unit that enters an ML manifest. It is
intentionally richer than a conventional ``(path, label)`` CSV row: it keeps
the physical context needed to understand *why* a model succeeds or fails
(time offsets, separation distance, reference instrument, QC by layer,
environmental conditions, provenance).
"""

from __future__ import annotations

from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from wave_analysis.qc.flags import QCFlag


class SampleTime(BaseModel):
    """Timestamps of the sample and its reference."""

    model_config = ConfigDict(extra="forbid")

    start_utc: AwareDatetime
    end_utc: AwareDatetime | None = None
    reference_utc: AwareDatetime | None = None
    offset_s: float | None = Field(default=None, description="target minus reference, seconds")
    propagation_lag_s: float | None = None


class SampleLocation(BaseModel):
    """Where the sample was observed and how far the reference is."""

    model_config = ConfigDict(extra="forbid")

    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    depth_m: float | None = None
    reference_distance_m: float | None = Field(default=None, ge=0)


class WaveReference(BaseModel):
    """Physical wave labels and where they came from."""

    model_config = ConfigDict(extra="forbid")

    reference_station_id: str
    reference_sensor: str | None = None
    Hs_m: float | None = None
    hs_estimator: str | None = Field(
        default=None, description="e.g. 'spectral_hm0', 'zero_upcrossing_h13'"
    )
    Tp_s: float | None = None
    Tz_s: float | None = None
    direction_deg: float | None = Field(default=None, ge=0, lt=360)
    spectrum_ref: str | None = Field(default=None, description="pointer to preserved S(f) record")
    directional_spectrum_ref: str | None = None


class SampleQuality(BaseModel):
    """QC summary by layer; each is a QARTOD flag."""

    model_config = ConfigDict(extra="forbid")

    provider: QCFlag = QCFlag.NOT_EVALUATED
    physical: QCFlag = QCFlag.NOT_EVALUATED
    imagery: QCFlag = QCFlag.NOT_EVALUATED
    synchronization: QCFlag = QCFlag.NOT_EVALUATED
    ml: QCFlag = QCFlag.NOT_EVALUATED
    image_metrics: dict[str, float] = Field(default_factory=dict)
    illumination: str | None = None


class SampleProvenance(BaseModel):
    """Upstream provenance of every input."""

    model_config = ConfigDict(extra="forbid")

    sources: list[str]
    source_urls: list[str] = Field(default_factory=list)
    source_checksums: list[str] = Field(default_factory=list)
    pipeline_version: str
    git_commit: str | None = None


class MultimodalSample(BaseModel):
    """One synchronised sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    group_keys: dict[str, str] = Field(
        default_factory=dict,
        description="Keys used for leakage-safe splitting: station, deployment, day, event, sequence, region.",
    )
    time: SampleTime
    location: SampleLocation = Field(default_factory=SampleLocation)
    vision: dict[str, Any] = Field(default_factory=dict)
    platform: dict[str, Any] = Field(default_factory=dict)
    wave_reference: WaveReference | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    quality: SampleQuality = Field(default_factory=SampleQuality)
    provenance: SampleProvenance
