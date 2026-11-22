# S5 · Shore camera against a nearshore forecast

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: waits for
data (30 local dates).

## Question

The buoy results mean something because the camera was judged against the strongest
forecast available. The shore results have not had that test: they are compared with
typical conditions or an offshore hindcast. NOAA's nearshore wave forecast (NWPS) is
the strongest shore comparator, and it has been collected daily since 2026-11-15. Does
a shore camera add to it?

## Data

- NWPS spectra and wave-system files for the offices covering the paired cameras, in
  `data/raw/nwps/<yyyymmdd>/<office>/`.
- Stills and clips at Currituck (CDIP 433) and Jennette's Pier (CDIP 243) from
  2026-11-15 on. Oak Island has no label: CDIP 200 has sent no data since 2026-07-20.
- **Trigger.** Score once 30 local dates exist with six or more daylight stills at both
  cameras and both buoys reporting. Not earlier, and not extended after a look.

## Arms

| Arm | What it is |
|---|---|
| N forecast | NWPS significant height at the output point nearest the buoy, at the shortest lead available at image time, with a line fitted on the first 10 dates |
| C camera | The shore model frozen from [S1](s1_shore_stills.md) (arm A, held-out-year fit), applied with nothing refitted. If S1 has not produced one, the frozen probe |
| N+C | The fixed 50/50 mean of N and C |
| Typical | The median height of each buoy over the first 10 dates |

The first 10 dates fit the two simple comparators and are not scored. The remaining 20
or more dates are scored.

## Pass mark

- **Primary.** Total-height RMSE, N+C minus N: upper end below zero, intervals over
  whole local dates.
- Reported: C against N alone; peak period for N against the brightness spectrum;
  results per camera. Because the camera model is frozen, this is also the first score
  of a shore model on dates collected after it was trained.

## Prepare now (no score involved)

- An extractor from the NWPS files to an hourly table of height and peak period at the
  point nearest each buoy, with a unit test on one stored day.
- A daily count of eligible dates, so the trigger is visible.
- Check after any reboot that the NWPS timer is still running; NOMADS keeps about five
  days.

## Cost and where

Desktop CPU.

## Build

`scripts/nwps_points.py`, `scripts/sensing_shore_forecast.py`. Outputs to
`runs/next-s5-<stamp>/`.
