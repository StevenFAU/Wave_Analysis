# Wave_Analysis

**A reproducible framework for multimodal ocean-wave data and vision-based wave-state estimation, anchored to traceable physical measurements.**

**[Dashboard](https://stevenfau.github.io/Wave_Analysis/)**: live archive status, buoy-camera images with the sea state at each image, data sources, and literature. **[Documentation](https://stevenfau.github.io/Wave_Analysis/docs/)**

[![tests](https://github.com/StevenFAU/Wave_Analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/StevenFAU/Wave_Analysis/actions/workflows/tests.yml)
[![lint](https://github.com/StevenFAU/Wave_Analysis/actions/workflows/lint.yml/badge.svg)](https://github.com/StevenFAU/Wave_Analysis/actions/workflows/lint.yml)
[![site](https://github.com/StevenFAU/Wave_Analysis/actions/workflows/site.yml/badge.svg)](https://github.com/StevenFAU/Wave_Analysis/actions/workflows/site.yml)
![python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/status-v0.1.0--dev-orange)

---

## Research question

> Can ordinary or low-cost optical sensing infer physically meaningful ocean-wave
> parameters, such as significant wave height, periods, direction, and ultimately the
> wave spectrum, that normally require dedicated wave instruments? Can it do so
> **on cameras and coastlines it has never seen**?

The reviewed literature suggests a sharp answer to the *second* question is not yet
possible. That is not for lack of architectures: CNN/ConvLSTM (Choi et al. 2020),
CNN-BiLSTM with ADCP ground truth (Kim et al. 2023), V-JEPA + SlowFast +
physics loss (Kamagata et al. 2026), and image–IMU attention fusion (Yang et al. 2026)
have all been tried. What is missing is **large, diverse, synchronised, real-world data
whose labels trace back to a physical instrument**
([literature matrix](docs/literature/literature_matrix.md)).

Wave_Analysis therefore builds the **data foundation first**: verified acquisition,
preserved spectra, layered QC, explicit synchronisation, leakage-safe splits, and
provenance for every number. Models come after.

### Working hypotheses

- **H1:** Calibrated spatiotemporal visual sensing can recover meaningful components
  of the wave state, with instrument-grade sensors providing the physical reference
  that supervises, calibrates, and validates it. The goal is *complementary spatial
  coverage*, not replacing buoys.
- **H2:** The largest near-term limitation is data, not architecture. Kamagata et al.
  (2026) reach a held-out H_s R² of only 0.107 with a state-of-the-art model trained on
  six scenes. The [landscape review](docs/literature/landscape_review.md) found no public
  multi-site benchmark pairing imagery with instrument wave spectra.
- **H3:** For single-frame buoy cameras, physically grounded image features (whitecap
  fraction, horizon tilt) combined with the buoy's own wind measurement carry
  sea-state information that end-to-end models should be benchmarked against
  (Q-M4, Q-M5).

---

## What v0.1 provides

| Capability | Highlights |
|---|---|
| **NDBC adapter** | Parsers for every standard-met header era (1990s two-digit years → modern), realtime & historical spectra, directional coefficients (with the historical **r₁/r₂ × 100** un-scaling), mid-year grid changes, ADCP; CF-named, unit-normalised output |
| **Availability inventory** | 38,908 files · 1,797 stations · 127 with complete directional spectra · 168 with ADCP · 91 with a buoy camera, **all 91 with directional spectra** ([data/registry](data/registry)) |
| **Camera archiver** | NDBC lists only the latest buoy-camera image and deletes superseded ones after ~72 h. `wave-analysis ndbc cameras` (hourly systemd timer, 70 h backfill) builds the archive going forward; see [data collection](docs/operations/data_collection.md) |
| **Public dashboard** | [GitHub Pages site](https://stevenfau.github.io/Wave_Analysis/): hourly archive status and coverage, a 72 h image viewer with the buoy's sea state at each image, a map of camera sites and reference buoys, and a browsable registry and bibliography with verification levels. Everything is generated from the repository and the collector's ledger ([ADR 0009](docs/decisions/0009-public-dashboard.md), [operations](docs/operations/dashboard.md)) |
| **Physics** | Linear dispersion, group velocity, zero-crossing statistics, Pierson–Moskowitz/JONSWAP, cos-2s spreading, virtual-buoy and short-crested sea simulation |
| **Spectral analysis** | Moments, H_m0, T_p (discrete & Young 1995), T_m01, T_m02, T_m−10, width, peakedness, steepness, D_m/D_p/spread; **maximum-entropy** directional distribution; **energy-conserving** regridding |
| **Quality control** | IOOS **QARTOD** wave tests 9, 15–21; steepness, spectral-consistency and directional-realisability checks; image-quality, illumination and glint annotations. **Nothing is deleted** |
| **Synchronisation** | Nearest / window-overlap pairing that always stores both timestamps, Δt, distance, and a sync flag; group-velocity propagation lag |
| **ML datasets** | Grouped, cross-time (embargo), cross-station/region, extreme-condition splits; mandatory **leakage audit** (group, temporal, near-duplicate frames); manifests with **exclusion ledgers** |
| **Evaluation** | Named NRMSE normalisers, scatter index, dilution slope, Bland–Altman, **circular** direction metrics, ordinal classification metrics, **block-bootstrap** CIs, stratified reports |
| **Models** | Mean/climatology/persistence/ridge baselines; optional PyTorch reference implementations of all four reviewed architectures plus physics losses |
| **Provenance** | Byte-preserving downloads, SHA-256 manifests with upstream-change detection, per-artefact provenance YAML, RO-Crate |

### Findings made while building v0.1

These are verified against primary sources or real data, with details and
scripts in the linked docs.

1. **NDBC `WVHT` is the spectral H_m0.** NDBC's glossary describes a highest-third
   average, yet 4√m₀ of the published spectrum reproduces WVHT to reporting precision:
   31,944 records at four stations in three basins, |mean Δ| ≤ 0.0014 m, SD ≤ 0.008 m
   ([ndbc.md](docs/datasets/ndbc.md#known-limitations),
   `scripts/validate_ndbc_hm0.py`). Labels must record their H_s estimator,
   because H₁/₃ ≈ 0.95 H_m0.
2. **Kamagata et al. (2026) need three caveats** ([verification log](docs/literature/source_verification.md)):
   their commonly quoted metrics include training data (held-out H_s R² is 0.107);
   the source of their per-clip wave labels is unstated (the cited video dataset has only
   breaking-type labels); and their "dispersion" regulariser T_p² = 6H_s fixes the
   wave steepness at about 0.107, which is unrealistically steep.
3. **NDBC buoycam images are six camera views with a burned-in caption** (station ID,
   UTC time; 30 rows on 47/47 daylight images). A model can read the caption, which
   would invalidate cross-station tests, so it is cropped (`buoycam.split_views`).
   The views show buoy roll and pitch through horizon tilt, which is a possible stand-in
   for the IMU used by Yang et al. (2026).
4. **NDBC has `b`-segment files** (e.g. `41013jb2005.txt.gz`, 118 files) holding a
   second segment of a station-year, usually on a different frequency grid. They
   are indexed separately and never silently merged.
5. Wave timestamps are rounded and payload-dependent, and NDBC's `.spec` and
   `.data_spec` stamp the same record 10 minutes apart. Exact-time joins are unsafe
   ([synchronisation](docs/methodology/synchronization.md)).

---

## Quick start

```bash
git clone git@github.com:StevenFAU/Wave_Analysis.git
cd Wave_Analysis
uv sync --group dev                 # Python ≥ 3.11; add --extra ml for PyTorch models
uv run pytest                       # 130+ offline tests on real, trimmed NDBC files
```

```bash
uv run wave-analysis ndbc inventory                         # station × product × year availability
uv run wave-analysis ndbc download 41010 --years 2023       # raw bytes + SHA-256 manifest
uv run wave-analysis ndbc standardize 41010                 # CF-named Parquet + provenance
uv run wave-analysis ndbc cameras                           # archive new buoy-camera images (+70 h backfill)
scripts/install_collectors.sh                               # run it hourly via systemd user timers
uv run python examples/synchronized_sample/pipeline_demo.py # offline end-to-end demo
```

```python
from wave_analysis.sources.ndbc import parse_spectral, assemble_spectrum
from wave_analysis.processing.spectra import bulk_parameters, directional_distribution

blocks = {
    p: parse_spectral(f"41010{c}2023.txt.gz", p)[0]
    for p, c in {"swden": "w", "swdir": "d", "swdir2": "i", "swr1": "j", "swr2": "k"}.items()
}
spec = assemble_spectrum(blocks)  # S(f) + α1, α2, r1, r2 (r un-scaled)
params = bulk_parameters(spec)  # Hm0, Tp, Tm01, Tm02, Dm, spread, ...
D = directional_distribution(
    spec.alpha1, spec.alpha2, spec.r1, spec.r2, direction_deg=range(0, 360, 5), method="mem"
)
```

---

## Repository layout

```text
src/wave_analysis/
  sources/      provider adapters (ndbc implemented; cdip, erddap, coastcam, webcoos, satellite planned)
  ingest/       downloader, checksums, manifests
  schemas/      observation, variables (CF), station/sensor, imagery, multimodal sample
  physics/      dispersion, zero-crossing, parametric spectra, simulation
  processing/   spectra, synchronisation, geospatial, solar geometry
  qc/           QARTOD, physical checks, imagery annotation, flags
  datasets/     splits + leakage audit, manifest builder, provenance, RO-Crate
  evaluation/   regression, circular, classification, uncertainty, stratified
  models/       baselines, reference architectures (torch), physics losses
configs/        sources/, qc/, experiments/
data/           registry/ + manifests/ (in Git); raw/ interim/ processed/ (not in Git)
docs/           standards, architecture, data model, QC, provenance, methodology,
                datasets (datasheets), literature, decisions (ADRs), operations, roadmap
deploy/         systemd user units for continuous collection
scripts/        inventory, download, camera archiving, ADCP summary, manifest build, validation
tests/          unit/, schemas/, integration/ (+ live checks with -m network), fixtures/ (real, trimmed)
examples/       NDBC walkthrough; offline end-to-end pipeline demo
```

## Documentation

Start at **[docs/index.md](docs/index.md)** (also buildable as a site with
`uv sync --group docs && uv run mkdocs serve`).

| | |
|---|---|
| Design | [Architecture](docs/ARCHITECTURE.md) · [Data model](docs/DATA_MODEL.md) · [ADRs](docs/decisions/README.md) |
| Standards | [FAIR, CF, ACDD, IOOS, QARTOD, RO-Crate, DataCite, CFF](docs/STANDARDS.md) |
| Method | [Wave parameters](docs/methodology/wave_parameters.md) · [Spectra](docs/methodology/spectral_analysis.md) · [Synchronisation](docs/methodology/synchronization.md) · [Splitting](docs/methodology/dataset_splitting.md) · [Evaluation](docs/methodology/evaluation.md) · [Models](docs/methodology/models.md) |
| Data | [Dataset datasheets](docs/datasets/README.md) · [Registry](data/registry/datasets.yaml) · [Camera sites](data/registry/camera_sites.yaml) · [QC](docs/QUALITY_CONTROL.md) · [Provenance](docs/PROVENANCE.md) |
| Literature | [Landscape review](docs/literature/landscape_review.md) · [Matrix](docs/literature/literature_matrix.md) · [Gaps & questions](docs/literature/research_gaps.md) · [Source verification](docs/literature/source_verification.md) |
| Process | [Reproducibility](docs/REPRODUCIBILITY.md) · [Data collection & storage](docs/operations/data_collection.md) · [Dashboard](docs/operations/dashboard.md) · [Roadmap](docs/ROADMAP.md) · [Contributing](CONTRIBUTING.md) · [Citation policy](docs/CITATION_POLICY.md) |

## Principles

> Reproducibility, provenance, uncertainty, physical measurement traceability, and
> environmental diversity are part of the scientific result. The goal is not only a low
> error metric, but knowing what physical information a model learned, which sensor
> established the reference, under what conditions the result holds, and whether it
> generalises beyond the camera and coastline used for training.

1. Preserve raw bytes; **annotate, never delete** difficult observations.
2. Preserve the **highest-information representation** (spectra); derive the rest.
3. Every label names its **reference instrument, estimator, time offset, and distance**.
4. **Split by independent units** and audit for leakage; cross-station is the primary test.
5. Report **baselines, strata, and dependence-aware uncertainty**, not one global RMSE.

## Citing

Use [`CITATION.cff`](CITATION.cff) (GitHub: *Cite this repository*). Datasets built with
this software get their own DOIs. Always cite the upstream providers
([policy](docs/CITATION_POLICY.md)).

## License

Code: [MIT](LICENSE). Upstream data keep their own terms (see each datasheet). NOAA NDBC
data are U.S. Government public domain; some registered datasets (e.g. Yin et al. 2025,
CC BY-NC-ND 4.0) restrict redistribution of derivatives.
