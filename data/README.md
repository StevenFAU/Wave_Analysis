# data/

Bulk data are **not** stored in Git (ADR 0006). This directory holds the
committed metadata that makes the data reproducible, plus git-ignored working
directories.

| Path | In Git? | Content |
|---|---|---|
| `registry/datasets.yaml` | ✅ | Source catalogue with verification status; validate with `wave-analysis registry validate` |
| `registry/camera_sites.yaml` | ✅ | Public camera sites and their proposed wave-reference buoys (distances computed by the dashboard build) |
| `registry/ndbc_files.parquet` | ✅ | NDBC inventory: every file (station, product, period, segment, URL, size, last-modified) |
| `registry/stations.parquet` | ✅ | NDBC station metadata |
| `registry/ndbc_station_summary.csv` | ✅ | Per-station capability summary |
| `registry/ndbc_adcp_availability.csv` | ✅ | ADCP vs spectra vs camera per station |
| `registry/ndbc_cameras_<date>.csv` | ✅ | Camera snapshot |
| `registry/ndbc_inventory.provenance.yaml` | ✅ | Provenance of the inventory run |
| `manifests/raw/*.csv` | ✅ | Every upstream request: URL, SHA-256, status |
| `manifests/processed/` | ✅ | Provenance of standardized tables (small) |
| `manifests/ml/<name>/` | CSV/YAML ✅, Parquet ❌ | ML manifests, exclusion ledgers, provenance |
| `raw/` | ❌ | Upstream bytes, unchanged (offsite copy: `scripts/sync_offsite.sh`) |
| `raw/cdip/<stn>/` | ❌ | CDIP netCDF files as served (`wave-analysis cdip download`); requests in `manifests/raw/cdip.csv` |
| `raw/pacioos/beachcam/` | ❌ | PacIOOS beach-camera images mirroring ERDDAP's `files/` tree, `_index/` snapshots and `_manifests/<dataset>.csv` ledgers (kept with the archive) |
| `raw/era5/<site>/` | ❌ | ERA5 monthly netCDF files as returned by the CDS, each with its `.request.json`; requests in `manifests/raw/era5.csv` |
| `raw/webcoos/` | ❌ | WebCOOS stills by camera and UTC date, `_listings/` of each run's selection, `_manifests/<camera>.csv` ledgers |
| `raw/ndbc/buoycam/` | ❌ | Camera archive: `<station>/<YYYY>/<MM>/*.jpg`, `_listings/`, `_manifests/<YYYY-MM>.<host>.csv` (one ledger per collector host, kept with the archive, ADR 0008) |
| `external/<id>/` | ❌ | Published research datasets as downloaded by `scripts/download_external.py` (`configs/sources/external_datasets.yaml`), each with a `SOURCE.yaml`; requests in `manifests/raw/external.csv`. Not in the offsite sync: the repositories are the backup |
| `interim/` | ❌ | Scratch; also the dashboard's rolling sea-state cache (`interim/dashboard/seastate/`) |
| `processed/` | ❌ | Standardized Parquet tables (`processed/<source>/<station>/`, each with a provenance record) |

Paths can be relocated with `WAVE_ANALYSIS_DATA=/big/disk/wave_data`.

Rebuild: see `docs/REPRODUCIBILITY.md`.
