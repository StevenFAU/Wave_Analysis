# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/). Dataset versions are tracked
separately (`docs/REPRODUCIBILITY.md`).

## [Unreleased] — v0.1.0 "Dataset discovery and NDBC foundation"

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
  measured), writes a dated archive layout, streams a monthly ledger kept with
  the archive, and takes a run lock.
- Continuous collection: systemd user timers (`deploy/systemd/`,
  `scripts/install_collectors.sh`) for the hourly archiver and twice-monthly
  gzip snapshots of realtime spectra for camera stations
  (`ndbc download --camera-stations --realtime --gzip`); an append-only offsite
  copy via rclone (`scripts/sync_offsite.sh`, Cloudflare R2 by default); a
  coverage report (`scripts/buoycam_coverage.py`); ADR 0008.
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
- Documentation: standards, architecture, data model, QC, provenance,
  reproducibility, citation policy, methodology (9 pages), dataset datasheets,
  literature matrix, research gaps, source-verification log, ADRs 0001–0008,
  operations guide for continuous collection and storage.
- Tests on real trimmed NDBC fixtures; live upstream-format tests (`-m network`).
- CI (tests, lint and type-check, docs), pre-commit, issue and PR templates.
