# Standards

Wave_Analysis adopts established community standards rather than inventing
conventions. This page lists each standard, what it governs, **where in the
repository it is implemented**, and its current compliance status.

| Area | Standard | How Wave_Analysis uses it | Implemented in | Status |
|---|---|---|---|---|
| Data principles | **FAIR** (Wilkinson et al. 2016) | Findable (registry, manifests, DOIs), Accessible (open HTTPS sources), Interoperable (CF names, Parquet/netCDF), Reusable (licences, provenance) | `data/registry/`, `docs/PROVENANCE.md` | in progress |
| Variable names & units | **CF Conventions 1.12** | Every canonical variable maps to a CF standard name and canonical unit; CF-compatible unit strings | `schemas/variables.py` | ✅ for in-situ waves & met |
| Discovery metadata | **ACDD 1.3** | Global attributes on netCDF exports (title, summary, creator, time/geo coverage, license, source) | netCDF export (v0.2) | planned |
| Archival netCDF | **NCEI netCDF templates** (timeSeries DSG) | Observation tables export as CF discrete-sampling-geometry timeSeries | netCDF export (v0.2) | planned |
| Ocean data services | **IOOS DMAC** data-publishing guidance | ERDDAP-compatible outputs; QARTOD flags per value | `docs/datasets/ioos.md` | planned |
| QC | **IOOS QARTOD In-Situ Surface Waves v2.1** (doi:10.25607/OBP-759); flags per **UNESCO/IOC (2013)** | Tests 9, 15–21 implemented with configurable thresholds; one flag vocabulary (1/2/3/4/9) across all QC layers | `qc/qartod.py`, `qc/flags.py`, `configs/qc/` | ✅ bulk-parameter subset |
| Geospatial metadata | **ISO 19115-2** | Collection-level metadata for published imagery datasets | release packaging | planned |
| Research objects | **RO-Crate 1.1** | `ro-crate-metadata.json` generated for released datasets, with checksums and the creating software | `datasets/provenance.py::write_ro_crate` | ✅ minimal crate |
| Dataset citation | **DataCite Metadata Schema 4.x** | DOI metadata for dataset releases (Zenodo) | `docs/CITATION_POLICY.md` | planned at first release |
| Software citation | **Citation File Format 1.2** | `CITATION.cff` at repository root | `CITATION.cff` | ✅ |
| Dataset documentation | **Datasheets for Datasets** (Gebru et al. 2021) | One datasheet per source with a fixed template | `docs/datasets/` | ✅ template; NDBC complete |
| Archiving | **Zenodo + GitHub releases** | Versioned DOI per software release; separate DOIs for datasets | `docs/CITATION_POLICY.md` | planned |
| Versioning | **Semantic Versioning 2.0** (software); independent dataset versions | `pyproject.toml`; `<dataset>_vX.Y.Z` | `docs/REPRODUCIBILITY.md` | ✅ |
| Changelog | **Keep a Changelog 1.1** | `CHANGELOG.md` | root | ✅ |
| Time | **ISO 8601 / UTC** | All internal timestamps are timezone-aware UTC; original timestamps and time references are preserved | `schemas/observation.py` | ✅ |
| Code quality | PEP 8 via **ruff**, static typing via **mypy --strict**, numpy-style docstrings | CI-enforced | `pyproject.toml`, `.github/workflows/` | ✅ |

## Conventions specific to ocean waves

| Topic | Convention adopted | Why |
|---|---|---|
| Direction | Nautical **coming-from**, degrees clockwise from **true** north, in [0, 360) | Matches NDBC `MWD`/`α₁`, CDIP, CF `*_from_direction` |
| H_s | Always tagged with its estimator (`spectral_hm0`, `zero_upcrossing_h13`, `provider_reported`) | H₁/₃ ≈ 0.95 H_m0; mixing them biases labels |
| Spectral density | m² Hz⁻¹ (CF canonical `m2 s`) on explicit bin bandwidths | Needed for exact moments; bandwidths preserved |
| Directional parameters | NDBC (α₁, α₂, r₁, r₂) as storage form; canonical nautical Fourier (a₁, b₁, a₂, b₂) for computation | Lossless conversion both ways (`processing/spectra.py`) |
| Wave definition | Zero-upcrossing by default (NDBC, Yang et al. 2026); zero-downcrossing available (IAHR recommendation) | Both recorded in results |
| QC flags | QARTOD 1 pass / 2 not evaluated / 3 suspect-or-of-interest / 4 fail / 9 missing | One vocabulary for every layer |

## References

- CF Conventions 1.12: <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html>
- CF Standard Name Table: <https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html>
- ACDD 1.3: <https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3>
- NCEI netCDF templates: <https://www.ncei.noaa.gov/netcdf-templates>
- IOOS data publishing: <https://ioos.noaa.gov/data/data-standards/data-publishing/>
- QARTOD waves manual v2.1: <https://doi.org/10.25607/OBP-759>
- RO-Crate 1.1: <https://w3id.org/ro/crate/1.1>
- DataCite schema: <https://schema.datacite.org/>
- Citation File Format: <https://citation-file-format.github.io/>
- Datasheets for Datasets: <https://doi.org/10.1145/3458723>
- FAIR principles: <https://doi.org/10.1038/sdata.2016.18>
