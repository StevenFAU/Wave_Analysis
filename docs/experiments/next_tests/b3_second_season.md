# B3 · A second season: do the results hold into winter, and does more data help?

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: waits for
the archive to reach 2026-12-31 (run about 2027-01-07).

## Question

Everything so far is 52 days of one autumn. Winter brings larger seas, lower sun and
more swell. Two things need answering. Do the autumn models still beat the forecast's
wind sea in December? And does training on roughly twice the data lower error, in
particular the 0.70 m total-height error of the West Coast fold?

## Data

- **Package P0**: the current 41,091 strips, 2026-09-23 to 2026-11-14.
- **Package P1**: every usable strip from 2026-09-23 to 2026-12-31 24:00 UTC at the
  same buoys, built with the same light rules and label rule, and the **same station
  groups and folds**. A buoy that first appears after P0 joins the group and fold of
  its nearest P0 buoy within 300 km, or is left out. The archive adds roughly 800
  usable strips a day, so P1 should be about twice P0.
- **December rows**: P1's strips from 2026-12-01 on.

B3 runs after the W3 confirmations are scored, because P1 contains W3.

## Part a · Season shift (no training)

The P0 fold models, frozen, on the December rows at their held-out buoys.

- **Pass mark.** Wind-wave RMSE, camera minus the ERA5 line (as fitted on P0): upper
  end below zero, and the camera's error at most 85% of ERA5's. On P0 the ratio was
  0.76 (0.260 against 0.343 m).
- Reported by ISO week from November to December: camera, ERA5 and corrected-forecast
  error, the mean sea state and the mean sun elevation, so a drift can be told from a
  change in the sea.

## Part b · Learning curve (retraining)

Retrain the reference recipe (DINOv3, gentle, five folds, three seeds) on P1.

- **Primary.** Total-height RMSE on the **P0 strips**, out of fold, P1-trained minus
  P0-trained: upper end below zero and a gain of at least 0.016 m (the seed spread).
  The test strips are identical; only the training data grew.
- **West Coast.** The same contrast on fold 0 alone, intervals over whole buoys.
- **December.** On the December rows, P1-trained (out of fold) minus P0-trained.
- Reported: swell and wind-wave height; a three-point curve if a half-way package
  (through 2026-12-05) is also trained, which is optional.

## What follows

Part a not met: every claim is limited to autumn in the paper, and the fusion is
re-fitted by season. Part b met: keep collecting and retrain each quarter; the grant
case for a multi-year archive has a measured slope. Part b not met: more of the same
data does not help, and effort moves to time context (B1) and new inputs.

## Prepare now (no score involved)

- Backfill GFS-Wave from 2026-11-15 (`scripts/gfs_wave_archive.py`).
- Make `scripts/buoycam_package.py` and `scripts/buoycam_cv_folds.py` build P1 with
  P0's groups and folds from an end date, and test it on a P0 rebuild: the checksums
  of P0's label file must reproduce.
- Keep the collectors healthy; a lost week of winter strips cannot be fetched again
  (NOAA keeps images about 72 hours).

## Cost and where

Part a: desktop CPU inference, as in the fresh-strip confirmation. Part b: 15 runs on
about twice the data, 20 to 25 GPU-hours. P1's new images travel to the server through
object storage or `deploy/dgx/run.sh sync`, not as a fresh upload of everything.

## Build

Options in the two packaging scripts, a config `ndbc_buoycam_heights_v3cv.yaml`.
Outputs to `runs/next-b3-<stamp>/`.
