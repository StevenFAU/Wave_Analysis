# B10 · A third wave model in the corrector: raise the bar the camera must clear

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: waits for
a Copernicus Marine account (Steven's to create).

## Question

The camera's gain is measured against a corrected forecast built from ERA5 and
GFS-Wave. A stronger forecast would shrink the gain, and a reviewer can fairly ask by
how much. Copernicus MFWAM is a third, independent global wave model with wind-sea and
swell partitions. How much does adding it improve the no-camera comparator, and how
much of the camera's 8% gain in wind-wave height survives?

## Data

MFWAM analysis and forecast fields at the 79 buoys for 2026-09-23 to 2026-11-14:
total, wind-sea and primary and secondary swell heights and periods. Check the
product's licence and whether its analysis assimilates data from these buoys (wave
models commonly assimilate altimeter heights, and some assimilate buoy spectra). If it
does, say so beside every number: such a comparator has seen the truth.

## Arms

The registered corrector and nested fusion, re-run unchanged except for the feature
set, on the rows where all three models exist:

| Arm | Forecast features |
|---|---|
| F2 | ERA5 + GFS-Wave + location and depth + measured wind (today's comparator) |
| F3 | F2 + MFWAM |
| F2 + camera, F3 + camera | The fixed 50/50 blend and the registered gate on each |

## Pass marks

- **Comparator.** Wind-wave and total-height RMSE, F3 minus F2: reported with
  intervals. No pass or fail; it is a measurement of the bar.
- **The camera against the higher bar.** Wind-wave RMSE, (F3 + camera, fixed blend)
  minus F3: upper end below zero. Met means the camera's contribution survives a
  three-model forecast.
- Reported: the share of F2's camera gain that remains against F3.

## Cost and where

Desktop CPU. The download is small (point extractions). One connection, polite pacing,
each file in the ledger, a datasheet and a data-dictionary entry.

## Build

A source adapter `src/wave_analysis/sources/mfwam.py`, a feature flag in
`scripts/sensing_forecast_corrector.py`. Outputs to `runs/next-b10-<stamp>/`.
