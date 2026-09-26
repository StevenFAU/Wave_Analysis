# Coastal Data Information Program (CDIP)

Registry id `cdip` · adapter `wave_analysis.sources.cdip.CDIPSource` · **v0.2.0** ·
verified on station 106 (Waimea Bay) on 2026-09-26.

## Provider

Scripps Institution of Oceanography (UC San Diego), with support from USACE and
California State Parks. Operates a network of **Datawell Directional Waverider**
buoys, the reference-class instrument for wave measurement. Many CDIP buoys are
also listed as NDBC stations (e.g. station 41114 off Fort Pierce, FL is CDIP 134;
Waimea Bay, HI is CDIP 106 = NDBC 51201).

## Why CDIP matters here

1. **Reference quality.** Waveriders are the standard against which other wave
   sensors are validated.
2. **Explicit acquisition windows.** `waveTimeBounds` gives start and end of each
   record, so synchronisation can use interval overlap instead of rounded
   timestamps.
3. **Raw displacement.** The per-deployment files include the heave time series
   (`xyzZDisplacement`, typically 1.28 Hz). Both spectral H_m0 and
   zero-crossing H₁/₃, H_max and T_z can then be derived from the *same*
   record, as in the Yang et al. (2026) dual-labelling strategy, but from a
   physical rather than a virtual buoy.
4. **Kamagata-style H_max targets** need the time series, which only a
   displacement record provides.

## Files

THREDDS, no authentication. Whole files are downloaded through `fileServer`, so
the raw layer holds the bytes CDIP serves, with a checksum; `dodsC` gives
OPeNDAP access to the same files.

| Product | URL (below `https://thredds.cdip.ucsd.edu/thredds/fileServer/cdip/`) | Content |
|---|---|---|
| `historic` | `archive/{stn}p1/{stn}p1_historic.nc` | Every publicly released record of the station: bulk parameters, 64-band directional spectra, SST, GPS, buoy health. No displacement. 106: 1.05 GB, 388,533 records |
| `deployment` | `archive/{stn}p1/{stn}p1_d{NN}.nc` | One deployment, **including raw displacement**. 106: 28 files, 18.7 GB in total |
| `realtime` | `realtime/{stn}p1_rt.nc` | The current deployment; stored as a time-stamped snapshot |

`wave-analysis cdip discover <stn>` lists a station's archive files from its
THREDDS catalog (`catalog/cdip/archive/{stn}p1/catalog.xml`).

## Conventions (verified)

Read from the attributes of `106p1_historic.nc` and checked on its 84,893
records of 2009–2013:

| Item | Convention | Check |
|---|---|---|
| Time | `waveTime` is the **UTC start** of the sample ("UTC sample start time"); spectra use 1600 s samples ("Spectral processing performed on 1600-second samples"), reported every 30 min | every sample is 1600 s and `waveTimeBounds[:, 0] == waveTime` |
| H_s | `waveHs` is spectral: 4√m₀ from the published spectrum and bandwidths | 99th percentile of \|H_s − 4√Σ S·Δf\| = 0.0075 m, max 0.046 m |
| Directions | Degrees true, coming-from; CDIP adds the magnetic declination (`additional_processing`) | — |
| Fourier coefficients | `waveA1Value` … `waveB2Value` relative to true north, so α₁ = atan2(b₁, a₁) | α₁ reproduces `waveMeanDirection` exactly |
| Bandwidth | `waveBandwidth` published; equals the width of `waveFrequencyBounds` | exact |
| Periods | `waveTa` = T_m01 and `waveTz` = T_m02, per their CF standard names | — |
| Flags | `waveFlagPrimary`: UNESCO/IOC (2013) `1 good, 2 not_evaluated, 3 questionable, 4 bad, 9 missing`, the same values as the project's QARTOD flags. `waveFlagSecondary` gives one of 20 reasons (e.g. `6 Hs_spike`) | historic file: all 1, because it "excludes all records flagged bad" |
| Deployment | `waveSourceIndex` points into `sourceFilename` and `metaDeploy*` and is **1-based** | index 27 holds 18,356 records, the record count of `106p1_d27.nc` (0-based it would point to `d28`, which has 754) |
| Fill values | `-999.99` (floats), `-127` (flags) | decoded to NaN |

## Mapping to the standardized layer

`CDIPSource.normalize` writes provider-reported bulk observations:

| CDIP | Canonical variable | Units |
|---|---|---|
| `waveHs` | `wave_significant_height` | m |
| `waveTp` | `wave_peak_period` | s |
| `waveTa` | `wave_mean_period_tm01` | s |
| `waveTz` | `wave_mean_period_tm02` | s |
| `waveDp` | `wave_from_direction_at_peak` | degree |

Each row carries `time_reference = start`, `averaging_period_s` from
`waveTimeBounds`, the deployment's position and `deployment_id` (e.g.
`106p1_d10`), and CDIP's flags unchanged in `source_qc_flag` as
`"<primary>/<secondary>"`. `CDIPSource.spectrum` returns the directional
spectra with CDIP's own bandwidths, and `normalize_spectrum` derives bulk
parameters from them with the project's estimators (method `spectral_hm0`,
etc.), as for NDBC. `validate` adds CDIP's primary flag as a `provider` QC
result to the QARTOD and physical checks, with a 30-minute gap test.

## Commands

```bash
uv sync --extra netcdf                                    # xarray + netCDF4
uv run wave-analysis cdip discover 106                    # archive files and sizes
uv run wave-analysis cdip download 106                    # historic file -> data/raw/cdip/106/
uv run wave-analysis cdip download 106 --products deployment --deployments 9-13   # with displacement
uv run wave-analysis cdip standardize 106 --start 2009-01-01 --end 2014-01-01
```

Downloads are recorded in `data/manifests/raw/cdip.csv` (committed). A
re-download reports `unchanged` or `changed` against the previous checksum:
CDIP regenerates historic files (`date_modified`), so this matters for
reproducibility. Standardized tables go to `data/processed/cdip/<stn>/` with a
provenance record.

## Licence and acknowledgement

The files' `license` attribute: "These data may be redistributed and used
without restriction." The `acknowledgment` attribute names the funder and the
station partners; for 106 it reads "CDIP is primarily supported by the U.S.
Army Corps of Engineers (USACE). Station partner: PACIOOS ; Field operator:
PACIOOS/UH". Partners differ by station, so copy the attribute of each file
used.

## Open questions

- Which CDIP buoys lie within a few km of calibrated cameras (CoastCam, WebCOOS,
  USACE FRF Argus)? The proposed pairings for the registered camera sites are in
  `data/registry/camera_sites.yaml`; the dashboard computes their distances.
- The realtime file is not filtered like the historic file (which excludes
  records flagged bad). Its flag distribution should be measured before realtime
  records are used as labels.
