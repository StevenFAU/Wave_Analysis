# ADR 0002: Long-format observation table with CF standard names

**Status:** Accepted (2026-09-25)

## Context
Providers publish different variable sets, names, units, and cadences. A wide
table per provider does not compose. Per-value QC and provenance are awkward in
wide form.

## Decision
The standardized layer is a long table (one row per time × station × product ×
variable) with a fixed column contract (`schemas/observation.py`). Variables are
canonical keys mapped once to CF standard names and canonical units
(`schemas/variables.py`), and validation rejects unregistered variables or
non-canonical units.

## Consequences
- Heterogeneous sources concatenate trivially. The table maps onto CF DSG
  timeSeries netCDF.
- Row count grows by the number of variables; pivot for modelling.
- Adding a variable is a deliberate, reviewed change.
