# X1 · Camera plus a real IMU: a pilot on the ship stereo dataset

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: not started.

## Question

Yang's fusion of an image with platform attitude was tested only in simulation, and our
own version used tilt read from the image, which added nothing. NOAA does not transmit
buoy motion, so the buoy archive cannot settle it. One dataset on disk has imagery, a
real inertial unit and wave truth together. Does real motion data add to imagery, and
does imagery add to motion?

## Data

`data/external/schwendeman_thomson_2017_ship_stereo` (CC BY 3.0 US; R/V Thomas G.
Thompson, North Pacific, December 2014 to January 2015):

- gridded sea-surface **elevation** and **radiance** from a ship-mounted stereo pair,
  in one-minute bursts (40 m × 40 m at 0.25 m, 5 or 7.5 Hz), 52 five-minute captures
  and 10 full captures;
- the ship's **motion** from a Novatel SPAN-IGM inertial unit, per capture;
- SWIFT drifter and Waverider buoy spectra nearby.

Earlier preparation (`scripts/measurement_stereo.py`, `runs/prep-v1/stereo*.csv`) found
53 usable bursts.

**Two limits, stated up front.** The radiance grids are not raw frames: they were
already projected to a level grid using the same inertial unit, so the "image" here is
motion-stabilised. And 53 bursts on a handful of days is a pilot, not a benchmark.

## Targets

- **In view:** significant height from the stereo elevation of the same burst, 4 × its
  standard deviation. Truth measured inside the picture.
- **Nearby buoy:** Waverider or SWIFT significant height within one hour, reported
  with the distance.

## Arms

A ridge (penalty from 1, 10, 100, chosen inside the training folds) on:

| Arm | Inputs |
|---|---|
| I image | Frozen public DINOv3 features of the radiance frames, averaged over the burst |
| M motion | Standard deviations and spectral peaks of heave, pitch and roll over the burst |
| I+M | Both |
| Typical | The training bursts' mean |

Folds leave out one **day** at a time, so bursts of one capture never straddle train
and test.

## Pass mark

- **Primary (Yang's claim on real data).** In-view height RMSE, I+M minus I: upper end
  below zero, intervals over whole days.
- **The other direction.** I+M minus M: reported with its interval. A ship's motion is
  itself a wave measurement, so M alone may be strong; say so if it is.
- With this sample, a wide interval is the expected outcome. Report it as a pilot
  whichever way it falls, and do not claim a result from a point estimate.

## What follows

The dataset's landing page says raw frames are available on request. A request for
them is drafted for Steven at `.claude/drafts/ship_stereo_raw_frames_request_email.md`
(local, not in the repository). With raw frames the test can use unstabilised imagery,
which is the honest version of the question.

## Cost and where

Desktop CPU. The `.mat` files need an HDF5 or MATLAB reader: use
`uv run --no-project --with h5py --with scipy` if the locked environment lacks one.

## Build

`scripts/sensing_stereo_imu.py`, extending the readers of
`scripts/measurement_stereo.py`. Outputs to `runs/next-x1-<stamp>/`.
