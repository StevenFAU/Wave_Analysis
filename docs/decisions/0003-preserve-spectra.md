# ADR 0003: Preserve spectra and directional moments; derive bulk parameters

**Status:** Accepted (2026-09-25)

## Context
H_s, T_p, and direction are low-dimensional summaries of S(f) and S(f, θ).
Collapsing to them at ingestion discards information needed for spectral targets
(Levels 5–6), for consistent re-derivation with other estimators, and for QC
(spectrum vs reported H_s).

## Decision
Whenever a provider publishes spectra, the full S(f) plus (α₁, α₂, r₁, r₂)
with bandwidths is stored. Bulk parameters are derived with tagged estimators
and stored alongside provider-reported values, never in place of the spectrum.
Frequency grids are never silently changed: files with several grids are split,
and regridding is an explicit, energy-conserving task step.

## Consequences
- Larger standardized tables (47 bins × 5 products per record for NDBC).
- Label definitions can change later without re-downloading.
