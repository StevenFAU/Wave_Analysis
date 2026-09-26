# Data Model

## 1. Observations (standardized layer)

One row per *(time, station, product, variable)*. Contract:
`schemas.observation.OBSERVATION_COLUMNS`, enforced by
`validate_observation_frame` (checks required fields, known variables,
canonical units, angle range, UTC dtype, QARTOD flag values, unique IDs).

| Column | Type | Notes |
|---|---|---|
| `observation_id` | string | Deterministic `dataset:station:product:variable:ISO8601Z`; stable across re-runs |
| `dataset_id`, `station_id`, `platform_id`, `sensor_id`, `deployment_id` | string | Deployment and sensor IDs are optional until deployment tables exist |
| `time_utc` | datetime64[ns, UTC] | |
| `time_reference` | string | `start` / `end` / `center` / `nominal` (provider-rounded) |
| `averaging_period_s` | float | e.g. 1200 for a 20-min wave record |
| `latitude`, `longitude`, `depth_m` | float | |
| `variable` | string | Canonical key (see §3) |
| `value` | float | Canonical units |
| `units` | string | Must equal the canonical unit of `variable` |
| `method` | string | Estimator: `provider_reported`, `spectral_hm0`, `zero_upcrossing_h13`, `spectral_tm02`, … |
| `source_qc_flag` | string | Provider flag, verbatim |
| `project_qc_flag` | int8 | QARTOD 1/2/3/4/9 aggregate |
| `source_file`, `source_url`, `source_checksum` | string | Links to the raw manifest |
| `download_timestamp` | datetime64[ns, UTC] | |
| `processing_level` | string | `raw` / `standardized` / `derived` / `synchronized` |
| `processing_version` | string | Package version that wrote the row |

Why long format: heterogeneous providers fit one schema; per-value QC and
provenance are natural; it maps one-to-one onto CF discrete-sampling-geometry
`timeSeries` for netCDF export (ADR 0002).

## 2. Spectra

`processing.spectra.FrequencySpectrum` (in memory) and its long-table form
(`to_long_frame`) in Parquet:

| Column | Units | Notes |
|---|---|---|
| `time_utc` | UTC | |
| `frequency_hz` | Hz | Bin centre |
| `bandwidth_hz` | Hz | Provider bandwidth when published; otherwise midpoint rule (validated against NDBC WVHT) |
| `spectral_density_m2_hz` | m² Hz⁻¹ | S(f) |
| `alpha1`, `alpha2` | degree | Nautical coming-from |
| `r1`, `r2` | 1 | Already un-scaled (NDBC historical ÷ 100) |

Spectral moments: m_n = Σ f_iⁿ S(f_i) Δf_i. Records with a missing in-band bin
return NaN moments rather than a silently low energy.

## 3. Canonical variables

Defined once in `schemas/variables.py`. Wave variables (CF standard name → unit):

| Key | CF standard name | Unit |
|---|---|---|
| `wave_significant_height` | `sea_surface_wave_significant_height` | m |
| `wave_peak_period` | `sea_surface_wave_period_at_variance_spectral_density_maximum` | s |
| `wave_mean_period` | `sea_surface_wave_mean_period` | s |
| `wave_mean_period_tm01` | `sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment` | s |
| `wave_mean_period_tm02` | `sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment` | s |
| `wave_zero_upcrossing_period` | `sea_surface_wave_zero_upcrossing_period` | s |
| `wave_maximum_height` | `sea_surface_wave_maximum_height` | m |
| `wave_from_direction_at_peak` | `sea_surface_wave_from_direction_at_variance_spectral_density_maximum` | degree |
| `wave_mean_from_direction` | `sea_surface_wave_from_direction` | degree |
| `swell_*`, `wind_wave_*` | `sea_surface_swell_wave_*`, `sea_surface_wind_wave_*` | m, s, degree |
| `wave_spectral_density` | `sea_surface_wave_variance_spectral_density` | m2 s |

Meteorological variables (wind, gust, pressure, temperatures, visibility, water
level) are listed in the same module.

## 4. Stations, deployments, sensors

`schemas/station.py`, `schemas/sensor.py`. A **station** is a published
identifier. A **deployment** is an interval with fixed hull, payload, and
position. A **sensor** is attached to a deployment with a validity interval,
a *modality* (what it measures), and a per-dataset *role* (`reference`, `input`,
`context`, `calibration`). Making the role explicit keeps the reference
instrument out of the inputs unless the experiment declares it.

## 5. Imagery

`schemas/imagery.py`: `ImageRecord` (capture window, time source and
uncertainty, camera/lens, exposure, pose, calibration IDs, bands, thermal
metadata, checksum, licence), `CameraIntrinsics` (K, distortion),
`CameraExtrinsics` (pose in a stated CRS, validity interval), `SpectralBand`,
and `ThermalMetadata`. Calibrations are referenced by ID so that a calibration
change is an explicit event, not a silent drift.

## 6. Synchronised samples

`schemas/sample.py::MultimodalSample` implements the record sketched in the
technical summary §39: time (start/end/reference/offset/propagation lag),
location and reference distance, vision, platform, `WaveReference` (with
`hs_estimator` and pointers to preserved spectra), environment, per-layer QC,
`group_keys` for leakage-safe splitting, and provenance.

## 7. Registries and manifests

| File | Content |
|---|---|
| `data/registry/datasets.yaml` | Source catalogue with verification status (schema `registry.DatasetEntry`) |
| `data/registry/ndbc_files.parquet` | Every NDBC file: station, product, period, segment, URL, size, last-modified |
| `data/registry/stations.parquet` | NDBC station metadata (XML and station table merged) |
| `data/registry/ndbc_station_summary.csv` | Per-station capability summary (years per product, directional years, camera) |
| `data/registry/ndbc_cameras_<date>.csv` | Camera snapshot |
| `data/manifests/raw/*.csv` | Every upstream request: URL, SHA-256, size, HTTP metadata, status |
| `data/manifests/ml/<name>/` | ML manifest, exclusion ledger, provenance |
