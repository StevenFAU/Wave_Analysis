"""Imagery metadata schema (RGB, thermal, multispectral, satellite).

Images carry far more scientific context than a filename and a timestamp.
This schema follows the repository specification §8 and separates:

* **capture** metadata (time window, exposure, frame rate);
* **geometry** (intrinsic / extrinsic calibration, referenced by ID so that
  one calibration can serve many frames and calibration changes are explicit);
* **radiometry** (spectral bands; thermal and multispectral specifics), so that
  RGB, LWIR and satellite products share one acquisition framework *without
  pretending they are equivalent measurements*.

The pinhole model used by Yang et al. (2026) relates a world point to pixels as
``s [u, v, 1]^T = K [R | t] [X, Y, Z, 1]^T`` with ``R = Rz(yaw) Ry(pitch) Rx(roll)``;
:class:`CameraIntrinsics` stores ``K`` (+ distortion) and :class:`CameraExtrinsics`
stores the pose. For moving platforms the pose is time-varying and belongs to
the IMU stream, not to a static extrinsic record.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class RadiometricQuantity(StrEnum):
    """What pixel values represent."""

    DIGITAL_NUMBER = "digital_number"  # uncalibrated (typical webcams)
    RADIANCE = "radiance"
    REFLECTANCE = "reflectance"
    BRIGHTNESS_TEMPERATURE = "brightness_temperature"


class CameraIntrinsics(BaseModel):
    """Intrinsic calibration (OpenCV / CIRN convention)."""

    model_config = ConfigDict(extra="forbid")

    calibration_id: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    distortion_model: str = "brown_conrady"
    distortion_coefficients: list[float] = Field(default_factory=list)
    method: str | None = None
    source: str | None = None


class CameraExtrinsics(BaseModel):
    """Static camera pose in a stated world frame (angles in degrees)."""

    model_config = ConfigDict(extra="forbid")

    calibration_id: str
    world_crs: str = Field(description="e.g. 'EPSG:32617' (UTM 17N) or 'local:<name>'")
    x: float
    y: float
    z: float
    azimuth_deg: float = Field(ge=0, lt=360)
    tilt_deg: float
    roll_deg: float = 0.0
    euler_convention: str = "ZYX"
    method: str | None = Field(default=None, description="e.g. GCP resection, survey")
    valid_from_utc: AwareDatetime | None = None
    valid_to_utc: AwareDatetime | None = None


class SpectralBand(BaseModel):
    """One radiometric band."""

    model_config = ConfigDict(extra="forbid")

    band_name: str
    center_wavelength_nm: float | None = Field(default=None, gt=0)
    bandwidth_nm: float | None = Field(default=None, gt=0)
    radiometric_calibration: str | None = None
    quantity: RadiometricQuantity = RadiometricQuantity.DIGITAL_NUMBER


class ThermalMetadata(BaseModel):
    """Thermal-IR specifics needed to interpret brightness temperature."""

    model_config = ConfigDict(extra="forbid")

    spectral_range_um: tuple[float, float]
    radiometric: bool
    emissivity_assumption: float | None = Field(default=None, gt=0, le=1)
    atmospheric_correction: str | None = None
    sensor_calibration: str | None = None


class ImageRecord(BaseModel):
    """Metadata for one image or video clip."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    camera_id: str
    platform_id: str | None = None
    media_type: str = Field(
        description="'image' | 'video' | 'timestack' | 'timex' | 'variance' | ..."
    )

    capture_start_utc: AwareDatetime
    capture_end_utc: AwareDatetime | None = None
    time_source: str | None = Field(
        default=None, description="e.g. 'exif', 'filename', 'NTP-synced DVR'"
    )
    time_uncertainty_s: float | None = Field(default=None, ge=0)

    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)

    camera_model: str | None = None
    lens_model: str | None = None
    focal_length_mm: float | None = Field(default=None, gt=0)
    sensor_width_mm: float | None = Field(default=None, gt=0)
    sensor_height_mm: float | None = Field(default=None, gt=0)

    image_width: int | None = Field(default=None, gt=0)
    image_height: int | None = Field(default=None, gt=0)
    frame_rate_hz: float | None = Field(default=None, gt=0)
    n_frames: int | None = Field(default=None, gt=0)

    exposure_s: float | None = Field(default=None, gt=0)
    iso: int | None = Field(default=None, gt=0)
    aperture_f: float | None = Field(default=None, gt=0)

    azimuth_deg: float | None = Field(default=None, ge=0, lt=360)
    elevation_deg: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None

    intrinsic_calibration_id: str | None = None
    extrinsic_calibration_id: str | None = None

    bands: list[SpectralBand] = Field(default_factory=list)
    thermal: ThermalMetadata | None = None

    compression: str | None = None
    source_url: str | None = None
    checksum_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    license: str | None = None

    @model_validator(mode="after")
    def _window(self) -> ImageRecord:
        if self.capture_end_utc is not None and self.capture_end_utc < self.capture_start_utc:
            raise ValueError("capture_end_utc precedes capture_start_utc")
        return self
