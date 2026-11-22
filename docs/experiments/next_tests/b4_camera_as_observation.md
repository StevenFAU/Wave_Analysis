# B4 · The camera as an observation: a variance-based update of the forecast

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: not
started. **Deadline: freeze by 2026-11-27 24:00 UTC.**

## Question

Fusion so far swaps or blends the camera's wind sea into the forecast. In retrospect a
fixed blend (20.10 cm) and a hand-built conditional rule (19.92 cm) both beat the
registered gate (20.79 cm) on wind-wave height. Wave forecasting has a standard way to
combine a forecast with an observation: weight each by its expected error. Does that
update, with error estimates that depend on the sea state and on the image, beat the
fixed blend?

## Method

For each strip, in squared heights (proportional to energy):

- background x_b = (forecast wind-wave height)², observation y = (camera wind-wave
  height)², the camera first corrected by the bias line the fusion already fits;
- analysis x_a = x_b + K (y − x_b), with K = s_b ÷ (s_b + s_o) clipped to 0 to 1;
- wind-wave height = √max(x_a, 0); total height = √(x_a + forecast swell²).

s_b and s_o are the expected squared errors of forecast and camera. Each is a
log-linear model fitted on calibration rows where the truth is known:

- s_b from the log of the forecast wind-wave height, the measured wind speed, the
  number of GFS-Wave systems and whether the leading system is flagged wind-generated;
- s_o from the log of the camera's wind-wave height, the camera's own predicted spread
  (`sigma_wind_wave_log`), the sun's elevation and the illumination class.

**Nested fitting, as in the registered fusion.** For an outer fold, every fitted piece
uses only calibration rows predicted by the inner runs that held that fold out
(`runs/buoy-pair*-seed*-nested-gentle`, `runs/forecast-baseline-inner-v1.parquet`). It
is applied once to the outer fold's test buoys.

## Arms, on the identical 34,723 nested rows

Forecast, camera, the registered gate, the fixed 50/50 blend, the variance-weighted
blend, the conditional rule of `scripts/wind_wave_reliability.py`, and this update.

## Pass mark

- **On the package (a finding).** Wind-wave RMSE, update minus fixed blend: upper end
  below zero.
- **On later strips (the confirmation).** The same contrast on the strips of the
  14-day replication (2026-11-28 to 2026-12-11), with the parameters fitted on the
  package and frozen. Commit the parameter files and their SHA-256 by 2026-11-27 24:00
  UTC, before that window opens. This is B4's own contrast. It does not change the
  replication's registered endpoint or verdict, and it is scored in the same run.
- Reported: total height; the update against the conditional rule; results by group;
  and whether the analysis spread √((1 − K) s_b) gives 90% intervals that cover 90%.
- If the freeze date is missed, the confirmation moves to window W3.

## Why it matters either way

Met: the camera can be described as an observation with a stated error, which is the
form a forecasting centre can use, and [B6](b6_neighbour_buoys.md) extends it to
neighbouring buoys. Not met: a fixed blend is the honest recommendation, and the paper
says the simple rule is enough.

## Control

Buoy 41040 is the known failure: the forecast is already within 7.5 cm there and the
camera reads low, so adding it raised error to 16.2 cm. Report K and the error there.
An update that works should trust the camera less at that buoy.

## Cost and where

Desktop CPU, minutes. Uses the `forecast` dependency group.

## Build

`scripts/sensing_observation_update.py` (fit, freeze, score), reusing the row
alignment of `scripts/sensing_nested_fusion.py`; unit tests on a synthetic case where
the best K is known. Outputs to `runs/next-b4-<stamp>/`; frozen parameters under
`data/manifests/ml/` with a `.sha256` beside them.
