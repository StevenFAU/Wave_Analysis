# Preprocessing (parse → normalize → derive)

## Parse: transcription only

Parsers convert upstream text or binary into typed arrays and do nothing else:

- documented missing-value sentinels → NaN, matched **per variable**
  (NDBC `WDIR = 999` is missing; `WDIR = 99` is a valid direction);
- documented transport scaling undone (NDBC historical r₁, r₂ ÷ 100);
- 2-digit years → 19xx (only present before 1999);
- header aliases (`WD`→`WDIR`, `BAR`→`PRES`);
- units read from the file's units row when present, and otherwise the documented
  defaults for that era, with `units_inferred = True` recorded.

Parsed tables keep **provider names and units**, so they can be diffed against
the raw file in tests.

## Normalize: map to the common schema

- canonical variable keys → CF standard names (`schemas/variables.py`);
- canonical units (`mi`/`nmi` → m, `ft` → m, `degT` → degree mod 360);
- timezone-aware UTC, with `time_reference` and `averaging_period_s`;
- `method = provider_reported`;
- deterministic `observation_id`;
- lineage columns (`source_file`, `source_checksum`, `processing_version`).

Unit conversions are table-driven (`sources/ndbc/source.py::_UNIT_CONVERSIONS`).
An unknown unit raises an error rather than passing through.

## Derive: project-computed quantities

Spectral bulk parameters (`processing.spectra.bulk_parameters`), each with a
distinct `method` tag and `processing_level = derived`. Derived and
provider-reported values coexist. They are never overwritten, so their agreement
is itself a QC signal (`qc.waves.spectral_consistency_test`).

## Split multi-grid files; never merge silently

If a spectral file changes frequency grid mid-period (payload change), the
parser returns one block per grid. Standardisation writes one table per grid
segment (`spectrum_<period>_grid<k>.parquet`). Directional blocks are joined to
the density block only when the grids are identical.

## What preprocessing never does

- Drop rows because they are dark, stormy, or suspect (QC annotates them).
- Interpolate across gaps.
- Resample spectra onto a new grid (that is an explicit, energy-conserving,
  task-level step).
