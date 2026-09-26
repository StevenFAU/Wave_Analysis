# ADR 0006: Manifests and small registries in Git; bulk data outside Git

**Status:** Accepted (2026-09-25)

## Context
Upstream archives are terabytes. Git is for code, metadata, and small examples.

## Decision
`data/raw`, `data/interim`, and `data/processed` are git-ignored. Committed:
source manifests (URL + SHA-256), registries and inventories (small Parquet or
CSV), trimmed real fixtures for tests, and released ML manifests. Large
releases go to Zenodo with DOIs (and optionally DVC or object storage for team
sharing).

## Consequences
- Anyone can rebuild raw data from manifests and verify it byte-for-byte.
- Upstream deletions are a risk; published datasets are archived on Zenodo.
