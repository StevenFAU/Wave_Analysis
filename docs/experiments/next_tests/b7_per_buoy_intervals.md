# B7 · Per-buoy prediction intervals: when can the camera be trusted?

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: not started.

## Question

The model's nominal 90% ranges cover 90% of total-height readings overall but fewer
than 80% at 11 of the 79 buoys. An operator needs ranges that hold at their own site.
Separately, 30 hours of buoy readings at a new site cut total-height error from 56.7 to
50.7 cm. Do the first ten labelled daylight hours at a buoy make its ranges honest, and
at what cost in width?

## Data

Out-of-fold predictions with their log-space mean and spread for total, wind-wave and
swell height (`runs/buoycam-cv-stage1-20261119/oof_dinov3-gentle.csv`). For each buoy,
the **calibration set** is its first k labelled strips in time, k = 10, 30 or 100. The
**evaluation set** is its strips at least 24 hours after the last calibration strip. A
buoy is evaluated when it has 100 or more evaluation strips.

## Arms

| Arm | What is fitted at the buoy |
|---|---|
| G global | Nothing (the model's own ranges) |
| R global rescale | Nothing at the buoy; one spread multiplier per fold from the training buoys |
| C scale | A split-conformal multiplier of the spread from the k calibration strips |
| CS shift and scale | A log-space offset and a conformal multiplier from the same strips |

## Pass mark

- **Primary, total height, k = 10.** With arm CS, the 90% ranges cover between 85% and
  95% of evaluation readings at 90% or more of the evaluated buoys. The mean width may
  not exceed arm G's by more than 10%.
- Reported: the same for wind-wave and swell height; k = 30 and 100; arm C against CS;
  the buoys that still fail and why; the change in RMSE from the offset.
- The count of buoys is the unit. No bootstrap is needed for the primary; give the
  exact count and list the failing buoys.

## Confirmation

Freeze the procedure (not the per-buoy numbers, which are refitted from each buoy's
first k strips of the window) and replay it on window W3.

## Cost and where

Desktop CPU, minutes.

## Build

`scripts/buoycam_site_intervals.py`, borrowing the few-shot row selection of
`scripts/sensing_site_adaptation.py`; unit tests that no evaluation strip is earlier
than 24 hours after calibration. Outputs to `runs/next-b7-<stamp>/`.
