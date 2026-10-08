# Dataset Documentation

For what the project holds today, what each collection is for, and what
every variable means, see the [data inventory](inventory.md).

Each upstream source has a datasheet in this directory, following one template
adapted from *Datasheets for Datasets* (Gebru et al. 2021). The machine-readable
counterpart is the registry, `data/registry/datasets.yaml`, whose
`verification.status` says how thoroughly each entry was checked.

| Source | Doc | Registry id | Status |
|---|---|---|---|
| NOAA NDBC (met, waves, spectra, cameras) | [ndbc.md](ndbc.md) | `ndbc` | **implemented** v0.1 · verified |
| NDBC ADCP currents | [ndbc_adcp.md](ndbc_adcp.md) | `ndbc` | inventory + parser v0.1 |
| CDIP Waverider network | [cdip.md](cdip.md) | `cdip` | **implemented** v0.2 · verified |
| IOOS / ERDDAP regional servers | [ioos.md](ioos.md) | `ioos_erddap` | planned v0.2 · candidate |
| PacIOOS beach cameras (Waimea Bay) | [pacioos_beachcam.md](pacioos_beachcam.md) | `pacioos_beachcam` | **implemented** v0.2 |
| WebCOOS webcams | [webcoos.md](webcoos.md) | `webcoos` | **implemented** v0.2 · verified |
| ERA5 ocean-wave parameters (model) | [era5.md](era5.md) | `era5_waves` | **implemented** v0.2 · verified |
| USGS CoastCam | [coastcam.md](coastcam.md) | `usgs_coastcam` | planned v0.3 · documented |
| Sentinel-1 SAR | [satellite_sar.md](satellite_sar.md) | `sentinel1_ocn` | planned v0.4 · candidate |
| Multispectral (Sentinel-2/Landsat/HLS) | [multispectral.md](multispectral.md) | `hls` | planned v0.4 · candidate |
| Thermal IR | [thermal_ir.md](thermal_ir.md) | — | research direction |

Other registry entries without a full datasheet yet (USACE FRF, Yin et al. 2025
VWBT, WASS stereo, Copernicus Marine, WAVEWATCH III, NOAA Digital
Coast) are described in `datasets.yaml`. Each gets a datasheet once
implementation starts.

## Template

Copy this skeleton for a new source. Every section is required. Write "Unknown –
see open questions" rather than deleting a section.

```markdown
# <Source name>

## Provider
## Purpose of Dataset
## Geographic Coverage
## Temporal Coverage
## Platforms
## Instruments
## Available Variables
## Sampling Frequency
## Timestamp Convention          <- start/end/centre of acquisition window? rounding?
## Upstream File Formats
## Access Method
## Authentication
## Quality Control               <- provider QC scheme and how it is preserved
## Missing Data Representation
## Known Changes in Instrumentation
## Known Limitations
## Relevant Documentation
## Original Citation
## Terms / License
## Wave_Analysis Ingestion        <- commands
## Wave_Analysis Transformations  <- parse / normalize / derive / QC, each listed
## Validation Tests               <- test files and what they prove
## Suitable ML Tasks
## Unsuitable / Risky Uses
## Open Research Questions
```

Adding a source also requires: a registry entry (validated in CI), an entry in
`data/registry/data_dictionary.yaml` once data is held, a config in
`configs/sources/`, an adapter implementing `sources.base.DataSource`, real
trimmed fixtures in `tests/fixtures/<source>/` with a provenance README, and a
`CHANGELOG.md` entry.
