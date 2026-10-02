# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/). Dataset versions are tracked
separately (`docs/REPRODUCIBILITY.md`).

## [Unreleased] — v0.1.0 "Dataset discovery and NDBC foundation"

### Fixed
- Landscape review cited `Quach2020`; the bibliography key is `Quach2021`
  (found by the new catalog consistency test).
- NDBC wave processing is cited as TD 03-01 (Earle 2003). The previously cited
  TD 96-01 (Earle 1996) is superseded, and NDBC says it should no longer be
  used. Bibliography key `Earle1996` is now `Earle2003` (verified V1).

### Added (2026-10-01)
- `processing.synchronize.label_from_candidates`: one label per image from
  `pair_window` candidates with a named rule (`nearest`, `interpolate`,
  `window_mean`, `in_record`), an optional propagation lag, and circular
  handling for directions; returns the label, the spread of the records used,
  their count and the largest offset. ADR 0010 makes the window mean the
  primary label and keeps the others for sensitivity analysis.
- `scripts/waimea_pairs.py` adds the ADR 0010 labels (`Hs_label_m`,
  `Hs_label_spread_m`, `n_label_records`; `Tp_label_s` and `Dp_label_deg` from
  the nearest record, with their window range and a `peak_records_disagree`
  flag) and a `label` block to its summary. Run on the full archive: earlier
  counts unchanged; the two-record H_s spread is a median 3.9 %.
- `scripts/buoy_pair_agreement.py`: hourly H_s agreement of every NDBC/CDIP
  buoy pair within 100 km, split into systematic and random parts. Result for
  2023 (211 pairs) in `data/manifests/processed/`: about 6 % mismatch under
  10 km, a median 15 % at 10–30 km ranging from 4 % on open coasts to 60 %
  across Oahu, and lags that barely matter at hourly resolution.

### Added
- Package `wave_analysis` with layered architecture: `sources`, `ingest`,
  `schemas`, `physics`, `processing`, `qc`, `datasets`, `evaluation`, `models`,
  `registry`, `cli`.
- NDBC adapter: parsers for standard meteorological files (all header eras
  1990s–present, realtime and historical), spectral summary, spectral density
  and directional coefficients (realtime pairs and historical grids; r₁/r₂
  un-scaling; multi-grid splitting), ADCP; normalisation to CF names and
  canonical units; QARTOD validation.
- NDBC availability inventory (station × product × period, including `b`
  segment files), station metadata, camera snapshot; committed registry
  snapshot of 2026-09-26.
- NDBC buoy-camera prospective archiver (`wave-analysis ndbc cameras`), per-view
  splitting of the six-camera strips and caption-band masking (measured on real images).
  The archiver backfills the last 70 h (NDBC keeps superseded images for about 72 h,
  measured), tries every observed minute stamp (the stamp varies per camera),
  writes a dated archive layout, streams a monthly ledger kept with the
  archive, and takes a run lock.
- Continuous collection: systemd user timers (`deploy/systemd/`,
  `scripts/install_collectors.sh`) for the hourly archiver and twice-monthly
  gzip snapshots of realtime spectra for camera stations
  (`ndbc download --camera-stations --realtime --gzip`); an append-only offsite
  copy via rclone (`scripts/sync_offsite.sh`, Cloudflare R2 by default); a
  coverage report (`scripts/buoycam_coverage.py`); ADR 0008.
- CDIP adapter (`wave_analysis.sources.cdip`, `wave-analysis cdip discover |
  download | standardize`): THREDDS catalog listing, whole-file downloads of
  historic, deployment and realtime netCDF files, bulk and directional-spectrum
  normalisation with sample-start times, per-deployment positions and CDIP's
  flags, and QC including CDIP's primary flag. Conventions verified on CDIP 106
  (docs/datasets/cdip.md): H_s = 4√m₀ of the published spectrum, α₁ equal to
  CDIP's band mean direction, 1600 s samples, 1-based deployment index.
- PacIOOS beach-camera archiver (`wave_analysis.sources.pacioos`,
  `wave-analysis pacioos beachcam`): checks each ERDDAP index (UTC vs HST file
  names, duplicates, URLs), downloads the images with a per-request ledger,
  checks every size against the index, and resumes. Started on the Waimea Bay
  pair (17,854 images, 3.3 GB), the first historical benchmark with CDIP 106.
- WebCOOS adapter (`wave_analysis.sources.webcoos`, `wave-analysis webcoos cameras |
  inventory | download`): token from `$WEBCOOS_API_TOKEN` or a mode-600 file, sent
  only to the API host and never logged; camera, inventory and element listings;
  one-minute stills thinned to a time grid; archive with a ledger and resume. The
  90-day rule of the WebCOOS usage guidelines is enforced (`--historical-approved`).
  Optional hourly collector for the five Yin et al. (2025) cameras
  (`install_collectors.sh --webcoos`). Camera sites updated with API positions
  (Oak Island was 7.8 km off) and Jennette's Pier added.
- ERA5 adapter (`wave_analysis.sources.era5`, `wave-analysis era5 download |
  standardize`): 12 wave parameters from the Copernicus CDS, one month per request
  for a box around a camera site, raw files with their requests and a manifest;
  months still ERA5T or incomplete are requested again once they could have changed.
  Normalized at the nearest sea grid point, flagged `era5`/`era5t`. The key stays in
  `~/.cdsapirc` (mode 600), goes only to the CDS host and is never logged. New
  variables `wave_energy_period`, `wind_wave_energy_period`, `swell_energy_period`
  (ERA5's `mwp` is m-1/m0, not Tm01). Optional extra `cds`. Real CDS fixture.
- Waimea Bay audit (`scripts/waimea_extract.py`, `scripts/waimea_timing_audit.py`):
  archive integrity, the capture time read from each image's burned-in caption
  (the file name is up to an hour early; the camera clock was wrong in
  May–July 2013), checked against the sun and CDIP 106.
- Waimea Bay–CDIP 106 pairing table (`scripts/waimea_pairs.py`): every CDIP
  record within 3 h of each trusted capture time, with offsets, and one row per
  indexed image with its exclusion reason; no label rule applied yet. New
  `processing.synchronize.pair_window`.
- Dashboard: a "Data held" panel on the overview with every collection on the
  collector host (NDBC buoy cameras and realtime snapshots, PacIOOS Waimea Bay,
  WebCOOS, ERA5, CDIP): counts and sizes from the files on disk, the period of
  data, freshness, and a check of files against verified ledger rows. Counts
  only; no images are republished. `status.json` gains `collections[]`.
- Shared resumable archiver (`wave_analysis.ingest.archive`) used by the PacIOOS
  and WebCOOS image collections.
- The downloader streams bodies to disk while hashing them, so files larger
  than memory can be fetched; a connection that drops mid-body is retried.
- Public dashboard on GitHub Pages (`dashboard/`, `wave_analysis.dashboard`,
  ADR 0009). Views: overview, cameras map, per-station image viewer with the
  sea state at each image, archive coverage, data sources, and literature.
  - `wave-analysis dashboard catalog|live|build|check`.
  - Hourly live-data publication to a single-commit `dashboard-data` branch
    (`scripts/publish_dashboard.sh`, `install_collectors.sh --dashboard`).
  - Recent sea state from HTTP range requests on NDBC realtime files.
  - A BibTeX reader, and a machine-readable verification index in
    `source_verification.md`.
  - `data/registry/camera_sites.yaml`: camera sites and their reference buoys,
    with distances computed at build time.
  - `site.yml` workflow: builds, checks and deploys the dashboard with the
    MkDocs site under `/docs/`; replaces `docs.yml`.
  - Node unit tests for the front end.
- Byte-preserving downloader with retries, rate limiting, atomic writes,
  SHA-256, and CSV manifests with change detection.
- Physics: linear dispersion (Guo 2002 + Newton), group velocity, steepness,
  Iribarren and Ursell numbers; zero-crossing analysis; Pierson–Moskowitz,
  JONSWAP, cos-2s spreading; random-amplitude sea-surface simulation.
- Spectral analysis: moments, H_m0, T_p (discrete and Young 1995), T_m01, T_m02,
  T_m−10, ν, Q_p, s_p, D_p, D_m, spreads; NDBC ↔ Fourier conversion; Fourier
  and MEM directional distributions; energy-conserving regridding.
- QC: QARTOD tests 9, 15, 16, 17, 18, 19, 20, 21; steepness,
  spectral-consistency, and directional-realisability checks; image-quality
  metrics, dHash; solar position, illumination, glint risk.
- Synchronisation (nearest, window overlap, propagation lag); leakage-safe
  splits (grouped, cross-time with embargo, hold-out, extreme, LOGO) and leakage
  audit; ML manifest builder with exclusion ledger; provenance records and
  RO-Crate.
- Evaluation: regression (named NRMSE normalisers, scatter index, slope),
  Bland–Altman, circular statistics, ordinal classification, block bootstrap,
  probabilistic scores, stratified metrics.
- Models: statistical baselines; optional torch reference implementations of
  Choi 2020, Kim 2023, Yang 2026, and the Kamagata 2026 SlowFast head; physics
  losses.
- Landscape review (`docs/literature/landscape_review.md`): techniques by
  modality, model trends, industry, and public training data with evidence
  levels (V1/V2/S/Q). Includes the camera-to-buoy pairing table (CDIP and
  NDBC), 15 new registry entries (29 total), 20 Crossref-verified bibliography
  entries, hypothesis H3, and questions Q-D6–Q-D8, Q-M4–Q-M6, L-4, L-5.
  arXiv:2606.13302 reviewed in full.
- Documentation: standards, architecture, data model, QC, provenance,
  reproducibility, citation policy, methodology (9 pages), dataset datasheets,
  literature matrix, research gaps, source-verification log, ADRs 0001–0008,
  operations guide for continuous collection and storage.
- Tests on real trimmed NDBC fixtures; live upstream-format tests (`-m network`).
- CI (tests, lint and type-check, docs), pre-commit, issue and PR templates.
