# ADR 0005: Group/time-block splits with embargo and mandatory leakage audit

**Status:** Accepted (2026-09-25)

## Context
Video frames and consecutive wave records are autocorrelated. Random splits
inflate scores. Pretraining on unlabelled video can also leak test conditions.

## Decision
Only group-based (station, deployment, day, event, sequence, region) and
time-block splits with an embargo are supported. Every manifest build runs
`check_leakage` (shared groups, temporal neighbours, near-duplicate frames) and
fails on leakage. The primary benchmark is **cross-station**. Unlabelled
pretraining corpora are audited against the test split.

## Consequences
- Smaller effective training sets and lower, more honest scores.
- Results from different protocols are reported separately, never pooled.
