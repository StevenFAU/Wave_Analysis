# Architecture

## Layered data flow

```text
UPSTREAM PROVIDERS (NDBC, CDIP, ERDDAP, CoastCam, WebCOOS, satellites)
        │  sources.<provider>.discover()     availability by station + product + period
        │  sources.<provider>.fetch()        ingest.Downloader: bytes unchanged, SHA-256, manifest row per attempt
        ▼
RAW            data/raw/<source>/...                         (git-ignored; reproducible from manifests)
        │  parse()        transcription only: sentinels → NaN, documented scaling undone
        │  normalize()    CF names, canonical units, UTC, explicit time reference, method tag
        ▼
STANDARDIZED   data/processed/<source>/<station>/*.parquet   observation long table; spectra long table
        │  derive         spectral moments & bulk parameters (tagged as DERIVED)
        │  validate()     QC layers → per-test flag arrays (nothing dropped)
        ▼
SYNCHRONIZED   paired samples: both timestamps, Δt, distance, propagation lag, sync flag
        │  datasets.builder   declarative filters → manifest + exclusion ledger
        │  datasets.splits    leakage-safe partitions + leakage audit
        ▼
ML MANIFEST    data/manifests/ml/<name>/  (+ provenance.yaml)
        │  models (baselines first), evaluation (stratified, dependence-aware CIs)
        ▼
RESULTS        metrics + provenance (git commit, config hash, input checksums)
```

No notebook-only transformation may exist between two stages. Every stage is a
function in `src/wave_analysis` and is reproducible from the previous stage and
a configuration file.

## Package map

| Package | Responsibility | Key objects |
|---|---|---|
| `sources` | Provider adapters implementing the five-stage interface | `DataSource`, `NDBCSource`, `ndbc.run_inventory`, `ndbc.buoycam` |
| `ingest` | Polite, atomic, checksummed downloads; CSV manifests | `Downloader`, `ManifestEntry` |
| `schemas` | Typed contracts (pydantic + frame validators) | `Observation`, `ImageRecord`, `MultimodalSample`, `VARIABLES` |
| `physics` | Linear wave theory, zero-crossing analysis, parametric spectra, simulation | `wavenumber`, `zero_crossing_analysis`, `jonswap`, `simulate_surface_elevation` |
| `processing` | Spectral analysis, synchronisation, geodesy, solar geometry | `FrequencySpectrum`, `bulk_parameters`, `directional_distribution`, `pair_nearest` |
| `qc` | Layered QC with one flag vocabulary | `QCFlag`, `QCTestResult`, `qartod.*`, `waves.*`, `imagery.*` |
| `datasets` | Manifests, splits, leakage audit, provenance, RO-Crate | `build_manifest`, `grouped_split`, `check_leakage`, `build_provenance` |
| `evaluation` | Regression, circular, ordinal classification, agreement, uncertainty, strata | `regression_report`, `circular_report`, `block_bootstrap_ci` |
| `models` | Statistical baselines; optional torch reference architectures and physics losses | `ClimatologyBaseline`, `Choi2020Regressor`, `Yang2026Fusion`, `losses` |
| `registry` | Schema + validation of `data/registry/datasets.yaml` | `DatasetEntry`, `validate_registry` |
| `cli` | `wave-analysis` command | `ndbc inventory/download/standardize/cameras`, `registry validate` |

## Design decisions

Recorded as Architecture Decision Records in [`docs/decisions/`](decisions/):

| ADR | Decision |
|---|---|
| [0001](decisions/0001-preserve-raw-annotate-dont-delete.md) | Preserve raw bytes; annotate difficult data, never delete at ingestion |
| [0002](decisions/0002-long-format-observations.md) | Long ("tidy") observation table with CF names as the standardized form |
| [0003](decisions/0003-preserve-spectra.md) | Preserve spectra and directional moments; derive bulk parameters |
| [0004](decisions/0004-single-qc-flag-vocabulary.md) | One QARTOD flag vocabulary across all QC layers |
| [0005](decisions/0005-group-based-splits.md) | Group- and time-block-based splits with an embargo and leakage audit |
| [0006](decisions/0006-manifests-not-data-in-git.md) | Manifests and small registries in Git; bulk data outside Git |
| [0007](decisions/0007-optional-deep-learning.md) | Deep learning is an optional extra; baselines are mandatory |

## Extension points

- **New provider:** subclass `sources.base.DataSource`, add a registry entry, a
  config, a datasheet, and real trimmed fixtures (`docs/datasets/README.md`).
- **New variable:** add a `VariableSpec` in `schemas/variables.py` (CF standard
  name where one exists). Validation rejects unregistered variables.
- **New QC test:** return a `QCTestResult` with `layer`, parameters, and a
  literature `reference`, and read thresholds from `configs/qc/`.
- **New split protocol:** return a `SplitResult` with its parameters, and add a
  case in `scripts/build_ml_manifest.py`.
