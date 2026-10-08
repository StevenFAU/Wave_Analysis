# Published Research Datasets (Stereo, Thermal IR)

Registry ids `schwendeman_thomson_2017_ship_stereo`, `carini_duck2016_ir_lidar`,
`alberello_2017_agulhas_wass` · config `configs/sources/external_datasets.yaml`
· downloader `scripts/download_external.py` · added 2026-10-08.

These three datasets fill sensor gaps that no public camera network can
([data source scan](../literature/data_source_scan_2026-10.md#3-open-research-datasets-that-fill-sensor-gaps)):
stereo surface reconstruction from a moving ship with its motion and wave
buoys (G-6, G-9), thermal infrared of the surf zone with lidar wave profiles
(G-5), and stereo of very large waves in sea ice (G-6). They are for method
checks and validation (P6 in the [collection plan](../operations/collection_plan.md)),
not for training: each covers days, not years.

## Storage and checks

```bash
uv run python scripts/download_external.py            # every dataset with URLs; resumable
uv run python scripts/download_external.py --verify   # size and MD5 of what is on disk
```

- Files go to `data/external/<id>/` with a `SOURCE.yaml` copy of the config
  entry. `data/external/` is git-ignored and **not in the offsite sync**: the
  repositories are permanent and cite-able, so they are the backup.
- Each file is checked against the size and, where published, the MD5 from
  the repository's own API. A file that fails is deleted. Every request is a
  row in `data/manifests/raw/external.csv` (SHA-256 of the bytes as served).
- The archives contain MATLAB code. It is kept as published and not run.

## Schwendeman & Thomson (2017): ship-based stereo

<http://hdl.handle.net/1773/38314>, CC BY 3.0 US, article
[10.1175/JPO-D-16-0187.1](https://doi.org/10.1175/JPO-D-16-0187.1). R/V *Thomas
G. Thompson*, North Pacific, winter 2014-2015. 24.8 GB in ten files.

| File | Size | Content |
|---|---|---|
| `StereoXYZData_FiveMin.zip` | 8.2 GB | 52 videos, five minutes each: per minute `XYGrid`, `ZGrid` (elevation and radiance on a 161 × 161 grid at 0.25 m, 5 or 7.5 Hz), `IgmTimeSeries` (camera motion from a Novatel SPAN-IGM IMU), stereo options, disparity and product plots |
| `StereoXYZData_FullCapture.zip` | 9.6 GB | The 10 best videos at full length (20-60 min), same layout |
| `ResultsAndPlots_JPO.zip` | 6.8 GB | The paper's results: whitecap events, stereo spectra, figures |
| `WaveriderSpectra_TGT2015.mat` | 11 MB | 263 records of 30 min from drifting Waveriders 3 and 4, 2014-12-28 to 2015-01-12: `Sf`, `Sftheta` (64 frequencies × 91 directions), moments, position. H_m0 1.05-7.19 m (computed here from `Sf`) |
| `SWIFTdata_TGT2015.mat` | 8 MB | 2,522 SWIFT bursts of 512 s, 2014-12-31 to 2015-01-11: H_s (1.05-7.53 m), T_p, direction, spectra, wind, turbulence |
| `ShipData_TGT2015.mat` | 40 MB | Ship position, speed, heading, wind, air and sea temperature |
| `StereoAnalysisCodes_JPO.zip` | 38 kB | MATLAB analysis code (spectra, zero-crossing, whitecap detection) |

Raw stereo frames are not included; the authors provide them on request.

## Carini et al., Duck 2016: thermal IR and lidar

<http://hdl.handle.net/1773/46586>, CC BY. USACE FRF, 7-8 November 2016.
6.4 GB in two MATLAB files with their readmes:

- `SurfZoneWavesAtOnsetBreaking.mat` (2.8 GB; readme of 2020-03-17) supports
  Carini et al. (2021a, b) in *JGR Oceans*: lidar line scans fused with
  tower- and pier-mounted thermal IR, the IR camera geometry, wave conditions
  from the FRF ADOP in 3.5 m depth (H_s, T_p, direction), the pier tide gauge,
  and bathymetry surveys of 5 and 18 November.
- `lidarIR_sharedData.mat` (3.5 GB; readme of 2018-08-03) supports the
  breaker-type manuscript 2018JC014441: tracked individual waves with lidar
  profiles, IR breaking masks, trough-to-peak height, period, asymmetry,
  skewness, local depth, and manual breaker type (non-breaking, steady,
  spilling, plunging).

## Alberello et al. (2017): WASS on the S.A. Agulhas II

[10.26179/Q9BD-5F74](https://doi.org/10.26179/Q9BD-5F74), CC BY 4.0. Antarctic
marginal ice zone, 4 July 2017, during an explosive polar cyclone. Six NetCDF
files of reconstructed surface elevation (mm, 1 m grid, 2 Hz) plus README,
LICENSE and metadata: 1.59 GB.

The AADC releases files only after an email address is submitted on its
[download page](https://data.aad.gov.au/dataset/571cdc2c-c810-4996-ae82-b79dc25a894e/download);
for a dataset under 2 GB the zip download starts and a one-time link is
emailed. Then:

```bash
uv run python scripts/download_external.py alberello_2017_agulhas_wass --from-zip ~/Downloads/<file>.zip
```

extracts the nine files by name, checks their sizes, and records them in the
ledger.
