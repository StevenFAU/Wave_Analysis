# ERA5 ocean-wave parameters (ECMWF / Copernicus)

Registry id `era5_waves` · module `wave_analysis.sources.era5` · **v0.2.0** ·
access verified 2026-09-26.

## Why it matters

ERA5 is ECMWF's global reanalysis. Its coupled wave model gives an hourly,
gap-free sea state at every camera site, including those without a nearby buoy
or whose buoy is offline (CDIP 200 at Oak Island has had no data since
2026-07-20). The project uses it as **context and as a non-vision baseline**
("what a model says at the site", Q-M6), never as ground truth: it is a
0.5 degree model field, not a measurement at the camera.

## Access

The Copernicus Climate Data Store (CDS) API, dataset
[`reanalysis-era5-single-levels`](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels),
through the `cdsapi` client (`uv sync --extra cds`).

1. Register at the CDS and accept the dataset's licence (CC-BY 4.0) once, at
   the bottom of the dataset's **Download** tab. A minimal valid selection is
   needed before the licence box shows; nothing needs to be submitted.
2. Save the personal access token (from your CDS profile) in `~/.cdsapirc`
   without it appearing on screen or in the shell history:

   ```bash
   (umask 077 && read -rsp "CDS token: " k && k="${k#key:}" && k="${k//[[:space:]]/}" && [ -n "$k" ] && printf 'url: https://cds.climate.copernicus.eu/api\nkey: %s\n' "$k" > ~/.cdsapirc && chmod 600 ~/.cdsapirc && echo " saved" || echo " nothing saved")
   ```

`$CDSAPI_URL` and `$CDSAPI_KEY` override the file. The code refuses a config
file other users can read, refuses a URL whose host is not
`cds.climate.copernicus.eu`, discards the client's debug output (which prints
the key), and removes the key from any error it records.

A request is limited to 121,000 fields (variables × days × hours; the area
does not count). One month of the 12 parameters below is at most 8,928 fields.
A request took 30 s to 5 min in the queue on 2026-09-26. The public catalogue
(`/api/catalogue/v1/collections/reanalysis-era5-single-levels`, no key) gives
the dataset's current temporal extent.

## What the files contain (verified on returned files, 2026-09-26)

| Property | Value |
|---|---|
| Grid | Wave model on a 0.36 deg reduced grid, served interpolated to a regular **0.5 deg** grid; the requested area is snapped to it |
| Land | NaN at every time (e.g. 21.5 N 158.0 W, on Oahu, next to the Waimea camera) |
| Time | `valid_time`, hourly, UTC; every wave parameter is **instantaneous** (`GRIB_stepType = instant`) |
| ERA5 vs ERA5T | `expver` coordinate along `valid_time`: `0001` final ERA5, `0005` preliminary ERA5T (the last ~3 months; June 2026 was final and July 2026 still ERA5T on 2026-09-26) |
| Latest data | About 5 days behind real time; a request past the latest hour returns only the hours that exist (asked for 18–30 Sept, got to 21 Sept 18:00) |
| Checks | Each variable carries `GRIB_paramId`, checked against the table below when parsing |

## Parameters

Definitions from Bidlot, *Ocean wave model output parameters* (ECMWF, 2020),
which the ERA5 documentation refers to.

| CDS name | Short | paramId | Definition | Canonical variable | `method` |
|---|---|---|---|---|---|
| significant_height_of_combined_wind_waves_and_swell | swh | 140229 | 4 √m0 | `wave_significant_height` | `spectral_hm0` |
| peak_wave_period | pp1d | 140231 | 1/f_p, parabolic fit around the discrete spectral peak | `wave_peak_period` | `spectral_tp_parabolic_fit` |
| mean_wave_period | mwp | 140232 | **m₋₁/m₀** (energy period), *not* Tm01 | `wave_energy_period` | `spectral_tm_10` |
| mean_wave_period_based_on_first_moment | mp1 | 140220 | m₀/m₁ | `wave_mean_period_tm01` | `spectral_tm01` |
| mean_zero_crossing_wave_period | mp2 | 140221 | √(m₀/m₂) | `wave_mean_period_tm02` | `spectral_tm02` |
| mean_wave_direction | mwd | 140230 | mean direction, degrees true, coming from | `wave_mean_from_direction` | `spectral_mean_direction` |
| significant_height_of_wind_waves | shww | 140234 | as swh, wind-sea part | `wind_wave_significant_height` | `spectral_hm0` |
| mean_period_of_wind_waves | mpww | 140236 | m₋₁/m₀, wind sea | `wind_wave_energy_period` | `spectral_tm_10` |
| mean_direction_of_wind_waves | mdww | 140235 | mean direction, wind sea | `wind_wave_from_direction` | `spectral_mean_direction` |
| significant_height_of_total_swell | shts | 140237 | as swh, total swell | `swell_significant_height` | `spectral_hm0` |
| mean_period_of_total_swell | mpts | 140239 | m₋₁/m₀, total swell | `swell_energy_period` | `spectral_tm_10` |
| mean_direction_of_total_swell | mdts | 140238 | mean direction, total swell | `swell_from_direction` | `spectral_mean_direction` |

The wind-sea/swell split is ERA5's own (by wind forcing), not NDBC's
separation frequency; NDBC's swell and wind-wave values are a different
estimator under the same variable names, which `dataset_id` and `method`
distinguish. On the fixture file, √(Hs_ww² + Hs_swell²) is 0–2.3 % below the
total Hs, never above it.

## Collecting

```bash
uv run wave-analysis era5 download pacioos_waimea --start 2009-01 --end 2013-12 --list-only
uv run wave-analysis era5 download webcoos_oak_island webcoos_currituck webcoos_jennettes_pier --start 2026-07
uv run wave-analysis era5 standardize pacioos_waimea --start 2009-01 --end 2013-12
```

Sites are `site_id`s from `data/registry/camera_sites.yaml`. Each request is
one calendar month for a box of ±1 deg (`--box`) around the site, snapped to
the grid. Re-running is safe: final months are skipped; months that can still
change upstream are requested again once they could have: incomplete ones
after a day, ERA5T ones after 30 days (`--refresh` requests everything). The
CDS serves an identical repeated request from its cache.

| Path | Content |
|---|---|
| `raw/era5/<site>/era5_waves_<site>_<YYYY-MM>.nc` | The file as returned by the CDS (whole box) |
| `raw/era5/<site>/era5_waves_<site>_<YYYY-MM>.request.json` | The exact request and its SHA-256 |
| `manifests/raw/era5.csv` | Every request: file SHA-256 and size, CDS request id, request hash, and the month's state (final / preliminary / incomplete) |
| `processed/era5/<site>/bulk_<start>_<end>.parquet` | Observation table at the nearest sea grid point, with provenance |

## Standardized layer

`normalize` takes the grid point nearest the site among those with sea data
(its position and distance are in the provenance record, and `platform_id`
names it, e.g. `era5_grid_22.00_-158.00`). `station_id` is the site.
`time_reference` is `center` with `averaging_period_s = 0` (instantaneous).
`source_qc_flag` is `era5` or `era5t`. `validate` runs the QARTOD gap,
gross-range and rate-of-change tests and the steepness test on Hs; with model
output these catch a broken request or file rather than bad values.

| Site | Nearest sea point | Distance |
|---|---|---|
| `pacioos_waimea` | 22.0 N, 158.0 W | 40.2 km (21.5 N 158.0 W, 17.3 km, is land) |
| `webcoos_oak_island` | 34.0 N, 78.0 W | 22.9 km |

## Licence and citation

CC-BY 4.0 (the CDS catalogue's `license` link, `cc-by`). Attribute and cite:

> Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz
> Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D.,
> Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2023): ERA5 hourly data on
> single levels from 1940 to present. Copernicus Climate Change Service (C3S)
> Climate Data Store (CDS), DOI: 10.24381/cds.adbb2d47 (accessed <date>).

and for the reanalysis itself Hersbach et al. (2020), *Q. J. R. Meteorol.
Soc.* 146, 1999–2049, doi:10.1002/qj.3803.

## Limitations

- 0.5 deg (~50 km) resolution: nearshore processes (refraction, shoaling,
  sheltering by islands and headlands) are not resolved. The nearest sea point
  for Waimea is 40 km away, north of Oahu, and sees swell that the bay may not.
- Model output with known biases (e.g. peak storm heights); use it as a prior
  or baseline, not a label.
- ERA5T values can change when ERA5 replaces them; rows carry `era5t` until
  then.
- Point-to-point nearest-neighbour only; no interpolation, and no wind yet
  (10 m wind is on the 0.25 deg atmospheric grid and needs a separate request).

## Open questions

- Whether to add 10 m wind (u10, v10) as a second product.
- A fixed-grid alternative: the CDS time-series dataset
  (`reanalysis-era5-single-levels-timeseries`) is faster for points but has
  only swh, mwp and mwd among the wave parameters (no peak period).
