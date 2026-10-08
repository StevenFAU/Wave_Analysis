# WIS hindcast at buoy sites (USACE)

Registry id `usace_wis` · module `wave_analysis.sources.wis` · **v0.3.0** ·
access verified and downloaded 2026-10-08.

## Why it matters

The landscape review asks for a wave-model nowcast at every camera site as a
required baseline. A camera model has value only where it beats what a model
already says
([landscape review §1, item 8](../literature/landscape_review.md)). ERA5 was
the only one held, and its nearest sea point is 40 km from the Waimea
cameras on a 0.5° grid. The U.S. Army Corps of Engineers' Wave Information
Study (WIS) writes its hindcast out **at the position of each buoy it is
checked against**, so there is a model value at CDIP 106 itself, 6 km off
Waimea. Before 2020 its winds also differ from ERA5's, so the two models make
partly independent errors.

Like ERA5 it is **context and a non-vision baseline, never ground truth**.

## Access

THREDDS at `chldata.erdc.dren.mil` (the older `chlthredds.erdc.dren.mil`
redirects there), with no login. Each buoy site has two kinds of file:

| File | Content | Size |
|---|---|---|
| `WIS_buoy_sites/<basin>/ST<stn>/ST<stn>.nc4` | Bulk parameters, hourly, the whole record | 0.6–23 MB |
| `WIS_buoy_sites/<basin>/ST<stn>/<yyyy>/WIS-ocean_waves_ST<stn>_<yyyymm>.nc` | The same bulk parameters plus the 2-D spectrum, one month | ~1.8 MB |

`<stn>` is the NDBC/WMO id (CDIP 106 is `51201`). The basins are `Pacific`,
`Atlantic`, `GulfOfMexico`, `GreatLakes` and `Alaska`. A buoy is listed only
for the years it has been on WIS's evaluation list: 51201, 41113 and 44056 go
back to 1980, while 46274 starts in 2022. A buoy can be listed in two basins
(51003: Pacific and Alaska); the adapter takes the first in that order.

```bash
uv run wave-analysis wis discover @sites
```

```bash
uv run wave-analysis wis download @sites
```

```bash
uv run wave-analysis wis standardize 51201 --start 2009-01-01 --end 2014-01-01 --spectra
```

`@sites` means every station in `configs/sources/wis.yaml`: CDIP 106 and the
offshore buoy 51003 for Waimea, and the reference buoy of every camera site in
`data/registry/camera_sites.yaml` and of the WebCOOS near-buoy cameras. The
config also asks for Waimea's monthly files from 2009 to 2013. Each request
goes into `data/manifests/raw/wis.csv` with its SHA-256. WIS re-runs and
replaces its archive from time to time, so a later download records each
file as `unchanged` or `changed`.

Held since 2026-10-08: 21 bulk files and 60 monthly files, 343 MB. WIS ends
on 2025-12-31, so it covers the Waimea archive (2009–2013), the USGS CoastCam
data releases and any historical WebCOOS stills, but none of the 2026
collection. For 2026 onward the matching model is NOAA's operational
WAVEWATCH III, whose point output at every buoy is kept on AWS (registry
`noaa_ww3`).

## What the files contain (checked on ST51201, 2026-10-08)

| Property | Value |
|---|---|
| Model | WAVEWATCH III 4.16 in the Atlantic and Pacific, WAM 4.5.1C in the Gulf of Mexico (global attribute `model`) |
| Time | Hourly, instantaneous, `time` in seconds since 1970-01-01 UTC. A monthly file runs from 01:00 on the 1st to 00:00 on the 1st of the next month |
| Position | The buoy's: 21.6705 N, 158.1171 W for 51201 |
| Record | 402,468 hours, 1980-02-01 to 2025-12-31, with 35 two-hour steps and no missing `waveHs` |
| Winds | Pacific and Atlantic: NCEP/NCAR Reanalysis-1 adapted by Oceanweather (scatterometer adjustment, tropical cyclones, major storms reanalysed) before 2020, ERA5 from 2020 |
| Observations | Buoys and altimeters are used to evaluate the hindcast, not assimilated into it |
| Licence | Free to redistribute and use for scholarly work, acknowledging the USACE Wave Information Study; commercial use needs written permission |

The winds and the evaluation-only use of buoys come from WIS's own overview
([RD26 presentation](https://wis.erdc.dren.mil/files/RD26_Presentation_USACE_WIS.pdf)).
WIS tunes and checks its hindcast against buoys such as CDIP 106 without
assimilating their data. It is independent of the buoy in the sense that
matters for a baseline: no buoy value at a given hour enters the model at
that hour.

### Parameters

| WIS name | Definition | Canonical variable | `method` |
|---|---|---|---|
| `waveHs` | 4 √m0 | `wave_significant_height` | `spectral_hm0` |
| `waveTpPeak` | Period of the peak band | `wave_peak_period` | `spectral_tp_band` |
| `waveTp` | Parabolic fit around the peak (like ERA5 `pp1d`) | not mapped | |
| `waveTm` | m-1/m0, the energy period; WIS calls it "mean spectral period" | `wave_energy_period` | `spectral_tm_10` |
| `waveTm1` | m0/m1 | `wave_mean_period_tm01` | `spectral_tm01` |
| `waveTm2` | √(m0/m2) | `wave_mean_period_tm02` | `spectral_tm02` |
| `waveMeanDirection` | Mean direction, degrees true, coming from | `wave_mean_from_direction` | `spectral_mean_direction` |
| `waveHsWindsea`, `waveTmWindsea`, `waveMeanDirectionWindsea` | The same for WIS's wind sea | `wind_wave_*` | as above |
| `waveHsSwell`, `waveTmSwell`, `waveMeanDirectionSwell` | The same for the swell | `swell_*` | as above |
| `windSpeed`, `windDirection` | 10 m wind driving the model (from) | `wind_speed`, `wind_from_direction` | `model_10m` |

The files also hold the peak periods and directional spreads of each
partition, friction velocity, drag coefficient and wave stress, which stay in
the raw file.

### The 2-D spectrum

`directionalWaveEnergyDensity` [m² s rad⁻¹] has 29 frequencies, 0.035–0.505
Hz, each 1.1 times the last, and 72 directions of 5°. Its directions are
where the waves go **toward**, the opposite of the bulk directions. The
adapter turns each bin to coming-from before computing the directional
moments, so `alpha1` and `alpha2` follow the same convention as NDBC and CDIP.
With WAVEWATCH III's bandwidths (Δf = f (r − 1/r)/2, r = 1.1), January 2012
at 51201 gives:

| Check | Result |
|---|---|
| H_m0 from the spectrum / `waveHs` | 0.999 (median) |
| Mean direction from the spectrum − `waveMeanDirection` | 0.01° median, 0.12° at the 95th percentile |
| T_m01, T_m02 from the spectrum / WIS's | 1.015, 1.062 |

The periods come out longer because the model adds a high-frequency tail
beyond the last bin, and higher moments weigh that tail more. For periods,
use WIS's own bulk values.

### Quirks

- **Longitude 360° off.** In some months the per-record `longitude` is stored
  360° too far west: −518.117 for −158.117 at 51201, in all of 1988–2017 and
  parts of 1980 and 1987. The scalar `lon` of the `.nc4` file is NaN. The
  adapter wraps positions to [−180, 180).
- **Titles.** ST46274 sits under `Pacific` but its title says "WIS Atlantic
  Hindcast". Standardized tables take each file's URL from the ledger, not
  from the title.
- **Station lists vs files.** ST46236's catalog lists years from 2014, but its
  `.nc4` file starts on 2017-12-01.
- **"Under evaluation".** The README beside the buoy-site files says the
  netCDF files are under evaluation and to use them at your own risk.

## Standardized output

`wave-analysis wis standardize` writes `data/processed/wis/<stn>/`:

- `bulk_<start>_<end>.parquet`: the parameters above as observations,
  `time_reference = center`, `averaging_period_s = 0`, `platform_id =
  wis_ST<stn>`, and `source_qc_flag` set to the model named in the file.
- With `--spectra`, from the monthly files: `spectrum_<start>_<end>.parquet`
  (S(f) with `alpha1`, `alpha2`, `r1`, `r2`) and
  `spectral_bulk_<start>_<end>.parquet`, the project's own bulk parameters
  computed from it.

For 51201, 2009-01-01 to 2014-01-01: 43,824 hours, 613,536 bulk rows and
43,823 spectra. The first hour, 2009-01-01 00:00, is in the December 2008
file.

## Use in the Waimea benchmark

Benchmark v1 (`configs/experiments/waimea_hs_v1.yaml`) adds WIS to the
nowcast baselines, alone and together with ERA5, on the same splits as v0.
The results are in
[PacIOOS beach cameras](pacioos_beachcam.md#benchmark-v1-a-second-wave-model-2026-10-08).

## Citation

Acknowledge "USACE Coastal and Hydraulics Laboratory, WIS team" and cite the
WIS website with the access date:

> USACE Wave Information Study (WIS), U.S. Army Engineer Research and
> Development Center, Coastal and Hydraulics Laboratory,
> https://wis.erdc.dren.mil (accessed 2026-10-08).
