# ADR 0004: One QARTOD flag vocabulary for every QC layer

**Status:** Accepted (2026-09-25)

## Context
QC spans provider flags, physical, temporal, sensor, visual, synchronisation,
and ML-leakage checks. Separate ad-hoc booleans (`good = True`) cannot be
aggregated, explained, or filtered uniformly.

## Decision
All project QC uses QARTOD/UNESCO primary flags (1, 2, 3, 4, 9). Each test
returns a `QCTestResult` (test, layer, flags, parameters, literature reference).
Aggregation precedence is MISSING > FAIL > SUSPECT > PASS > NOT_EVALUATED.
Provider flags are kept verbatim in `source_qc_flag`.

## Consequences
- Filters are uniform across layers (`flag in [1, 3]`).
- Visual and synchronisation checks reuse an oceanographic standard, which is
  unusual but consistent.
