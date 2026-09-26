# Quality Control

## Principles

1. **Never delete at ingestion.** QC produces flags. Exclusion happens only at
   task construction, through declarative filters whose effect is recorded in an
   exclusion ledger (ADR 0001).
2. **One vocabulary.** Every layer uses the QARTOD/UNESCO primary flags
   (ADR 0004):

   | Flag | Value | Meaning |
   |---|---|---|
   | PASS | 1 | Passed the tests applied |
   | NOT_EVALUATED | 2 | Not tested / no information |
   | SUSPECT | 3 | Suspect **or of high interest**: kept |
   | FAIL | 4 | Failed a critical test |
   | MISSING | 9 | Placeholder for missing data |

   "Or of high interest" is QARTOD's own wording, and it is the reason storms,
   glare, and night frames are flagged rather than discarded.
3. **Per-test results are kept.** A `QCTestResult` holds the test name, layer,
   flag array, parameters, and literature reference, so every exclusion can be
   explained. Aggregation precedence is MISSING > FAIL > SUSPECT > PASS >
   NOT_EVALUATED (`qc.flags.aggregate`).
4. **Thresholds are configuration.** QARTOD thresholds are "operator-defined and
   location-dependent". Global defaults live in `configs/qc/wave_qc.yaml`, and
   overrides are recorded in provenance.

## Layers

| Layer | Examples | Implementation |
|---|---|---|
| Provider | NDBC removes failed values upstream (no per-value flags); CDIP primary/secondary flags; ERDDAP QARTOD flags | preserved verbatim as `source_qc_flag` |
| Physical | gross range; steepness sₚ; spectral consistency | `qartod.gross_range_test`, `waves.steepness_test`, `waves.spectral_consistency_test` |
| Temporal | gaps, duplicates, rate of change, mean ± n·σ | `qartod.gap_test`, `rate_of_change_test`, `mean_std_test` |
| Sensor | flat line; frequency range; low-frequency energy; directional realisability | `qartod.flat_line_test`, `operational_frequency_range_test`, `low_frequency_energy_test`, `waves.directional_moment_test` |
| Visual | darkness, saturation, glare, blur, entropy; illumination; glint geometry | `qc.imagery.image_quality`, `processing.solar` |
| Synchronisation | offset beyond tolerance; unmatched target | `sync_qc_flag` from `processing.synchronize.pair_nearest` |
| ML | near-duplicate frames across splits; group/time leakage | `qc.imagery.difference_hash`, `datasets.splits.check_leakage` |

## QARTOD tests implemented

From *Manual for Real-Time Quality Control of In-Situ Surface Wave Data*,
v2.1 (IOOS 2019, doi:10.25607/OBP-759):

| # | Test | Manual requirement | Flags | Notes |
|---|---|---|---|---|
| 9 | Time-series gap | Strongly recommended | 3 after a gap; 4 duplicate/non-increasing time | Applied to bulk records |
| 15 | Mean and standard deviation | Strongly recommended | 3 | Trailing window excludes the value itself |
| 16 | Flat line | **Required** | 3 at `REP_CNT_SUSPECT`, 4 at `REP_CNT_FAIL` | Zero values never FAIL ("POn ≠ 0"). The manual's prose ("the five most recent observations are equal") and its codable loop (compare with 5 *previous* values) differ by one; the prose is implemented and the count is configurable |
| 17 | Operational frequency range | **Required** | 4 outside instrument range; 3 outside local range | Per spectral bin |
| 18 | Low-frequency energy | **Required** | 3 | Needs broad limits at infragravity-dominated sites (manual note) |
| 19 | Bulk parameter range | **Required** | 4 height out of range (propagates to all bulk parameters); 3 others | `propagate_height_failure` implements "no bulk parameters released" |
| 20 | Rate of change | **Required** | 4 | Optional scaling by actual Δt; not applied to peak period or direction (manual exception) |
| 21 | Neighbor check | Suggested | 3 only (manual: no FAIL) | Caller aligns the neighbour series first |

Not implemented: Tests 1–8 (ADCP signal diagnostics) and 10–14 (short-term
displacement series). They need raw instrument streams, which NDBC does not
distribute. They are planned with CDIP displacement data (v0.2).

## Project physical checks

| Check | Rule | Rationale |
|---|---|---|
| Steepness | SUSPECT if sₚ = 2πH_m0/(gT_p²) > 0.07, FAIL if > 0.142 | Sea-state steepness at the individual-wave breaking limit indicates a corrupt or mis-paired record |
| Spectral consistency | SUSPECT if \|WVHT − 4√m₀\| > max(0.1 m, 10 %) | NDBC WVHT equals spectral H_m0 to about 0.005 m (see `docs/datasets/ndbc.md`), so larger gaps mean mis-pairing or truncation |
| Directional realisability | SUSPECT if \|c₂ − c₁²\| > 1 − \|c₁\|²; FAIL if r ∉ [0, 1] | The four moments must come from a non-negative distribution (positive semi-definite Toeplitz matrix) |

## Visual annotations

`qc.imagery.image_quality` records mean/σ luma, dark fraction, saturation
fraction, glare fraction (bright and nearly achromatic), sharpness (variance of
the Laplacian), and entropy. `processing.solar` adds solar elevation and azimuth,
an illumination category (day / low_sun / civil / nautical twilight / night),
and a sun-in-view glint-risk flag. None of these removes a frame. They become
evaluation strata ("error vs illumination") and optional task filters.
Thresholds: `configs/qc/imagery_qc.yaml`.

## Worked example

NDBC 41010, 2023 (March fixture): `NDBCSource.validate` returns seven
`QCTestResult`s aligned with the H_s series; >90 % pass the gross-range test
(`tests/unit/test_ndbc_inventory_source.py::test_validate_returns_qartod_results`).
