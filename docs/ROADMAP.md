# Roadmap

Semantic versions for the software. Dataset versions are independent
(`docs/REPRODUCIBILITY.md`).

## v0.1.0 — Dataset discovery and NDBC foundation *(this release)*

| Component | Status |
|---|---|
| Repository structure, standards, architecture, ADRs | ✅ |
| Physics core: dispersion, zero-crossing, parametric spectra, virtual-buoy simulation | ✅ tested against analytic results |
| Spectral analysis: moments, bulk parameters, MEM directional distribution, energy-conserving regridding | ✅ |
| NDBC adapter: stdmet (all eras), spec, spectral (realtime + historical, r ÷ 100, grid splits), ADCP | ✅ tested on real files |
| NDBC inventory: 38,908 files, 1,797 stations; spectral, ADCP, and camera availability | ✅ committed in `data/registry/` |
| NDBC buoy-camera prospective archiver | ✅ (`wave-analysis ndbc cameras`) |
| Downloader + manifests + provenance + RO-Crate | ✅ |
| QC: QARTOD tests 9, 15–21; physical and directional checks; imagery annotation; solar geometry | ✅ |
| Synchronisation, leakage-safe splits and audit, manifest builder with exclusion ledger | ✅ |
| Evaluation: regression, circular, ordinal, agreement, block bootstrap, strata | ✅ |
| Baselines + optional reference architectures + physics losses | ✅ |
| Hm0 = WVHT validation at four stations | ✅ |

## v0.2.0 — Registry expansion and first historical paired benchmark

Priorities from the [landscape review](literature/landscape_review.md) §7:

- ✅ CDIP adapter: catalog, whole-file downloads, bulk and directional spectra with
  CDIP's bandwidths, flags, deployment positions (verified on CDIP 106). Still to
  do: displacement → zero-crossing labels. CDIP is the primary label source for
  shore cameras.
- **PacIOOS Waimea benchmark:** `beachcam_003/004` (2009–2013, 17,854 images)
  paired with CDIP 106 (6.3 km). ✅ Archiver and CDIP 106 2009–2013 standardized;
  image download started 2026-09-26. Next: pairing by hour window, splits, baselines.
- **USGS CoastCam adapter:** NIMS S3 snapshots and data-release IO/EO
  calibration. Pair with CDIP 214, 156, 221, and 121, and with the Dorado Spotter
  (five sites for leave-site-out tests).
- **Required non-vision baselines:** WIS/WW3/ERA5 nowcast at the site,
  persistence, and propagated offshore buoy (Q-M6).
- ERDDAP adapter (SECOORA, PacIOOS), discovery by CF standard name.
- WebCOOS access terms and adapter; pairing study for the NC sites used in
  Kamagata's test set.
- NDBC `adcp2` parser plus QARTOD current tests; NDBC deployment table.
- CF/ACDD netCDF export of standardized tables.
- Directional-moment regridding.

## v0.3.0 — First camera + physical ground-truth dataset

- CoastCam (Madeira Beach) adapter, orthorectification from IO/EO.
- `ndbc_buoycam_hs_v0.1.0` dataset from the archiver: caption masking,
  image-quality annotation, pairing with same-station spectra, cross-station
  benchmark with baselines. Zenodo DOI and datasheet.
- First scientific question: *How accurately can H_s be inferred from public
  imagery when entire stations are held out?*

## v0.4.0 — Multimodal

- Wind, current, and IMU/horizon features; Sentinel-1 and HLS context;
  bathymetry. Self-supervised pretraining with audited corpora (KOLOMVERSE,
  WebCOOS, NIMS; V-JEPA 2 freeze-and-probe first).
- Buoy-camera physics features: whitecap fraction and wind (H3/Q-M4), horizon
  tilt (Q-M5).
- Thermal IR: candidate self-collected dataset (Q-D5, CDIP 134 Fort Pierce).

## v0.5.0 — Spectral and spatial products

- S(f) and S(f, θ) targets; spatial wave-field inference on calibrated
  cameras; stereo ground truth (Guimarães et al. 2020 records, CC BY 4.0) to
  validate the spectral pipeline.
