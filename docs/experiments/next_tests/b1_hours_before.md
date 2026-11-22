# B1 · Hours before: does a buoy's recent image history recover swell?

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: not started.

## Question

The camera explains most of the wind sea from one strip but only about half of the
swell (swell-height RMSE 0.72 m, total 0.532 m out of fold). Swell changes over many
hours. Does giving the model the strips of the previous day from the same buoy lower
the swell and total-height error at buoys it never saw?

## Data

- Cached features of the camera reference (README rule 2), all five folds, three seeds.
- Labels: `Hs_label_m`, `wind_wave_label_m`, `swell_label_m` in
  `data/processed/packages/ndbc_buoycam_heights_v2cvn/labels.csv`.
- **History of a strip** at buoy b and time t: every strip of buoy b in the 24 hours
  before t, at most the 24 latest. No wave or wind reading of the past is an input.
  Measured on the package: 99% of strips have three or more earlier strips in 24 hours
  (mean 10.8), and 72% have three or more in 6 hours. Cameras send no image at night.
- All 41,091 strips are scored. A strip with no history is predicted from itself.

## Arms

All arms are heads on the cached features, trained on a fold's train rows, with early
stopping on its validation buoys' total-height RMSE (patience 5, at most 50 epochs).
Loss: Huber on the logs of the three heights. AdamW, learning rate 1e-3, weight decay
0.01, batch 256. These settings are fixed; no search.

| Arm | Input |
|---|---|
| S0 single | The strip's own features into a two-layer head. The reference |
| S1 mean | Own features, the mean of the history's features, and the history count |
| S2 sequence | A two-layer transformer (width 256, 4 heads) over the history and the strip, with an encoding of hours before t; read out at the strip |
| S3 order control | S2 trained and tested with the history shuffled and no time encoding |
| S4 smoothing | No training: the mean of S0's log predictions over the history and the strip |
| S5 other-buoy control | S2 with each test strip's history replaced by the history of a buoy from another station group at the same hours |

## Pass mark

- **Primary.** Swell-height RMSE, S2 minus S0, all strips. Met when the interval's
  upper end is below zero and the gain is at least 0.036 m (5% of 0.72 m).
- **Secondary.** Total height, S2 minus S0: upper end below zero and a gain of at least
  0.016 m (the seed spread). Wind-wave height must not get worse: upper end of S2 minus
  S0 below +0.005 m.
- **What the other arms mean.** S2 against S1 says whether order matters. S2 against
  S4 says whether the gain is more than averaging noisy predictions. S5 must lose the
  gain; if it keeps it, the gain is a regional or time-of-day cue, and the bar is not
  met.
- Reported, not a bar: results by number of earlier strips (0, 1 to 2, 3 to 6, 7 or
  more), by fold, and windows of 6 and 48 hours. Also the fixed 50/50 blend of S2's
  wind-wave height with the forecast on the nested rows, beside the same blend with the
  single-strip camera.

## What follows

- Met: freeze the S2 heads (checksums committed) and confirm on W3. Then propose, in a
  new bar, fine-tuning the network on sequences on the GPU server.
- Not met, but S4 gains: report that time context helps only as smoothing.
- Not met at all: the limit is what the images contain, and [B2](b2_swell_cue.md) and
  [X2](x2_render_swell.md) carry the question.

## Cost and where

Desktop CPU (run with `nice`; never the desktop GPU), minutes per head. If a GPU is
used, the shared server, under 5 GPU-hours.

## Build

`src/wave_analysis/models/strip_sequence.py`, `scripts/buoycam_sequence.py` (build
histories, train, predict out of fold, score), unit tests for the history builder (no
strip later than t, none from another buoy), a config under `configs/experiments/`.
Outputs to `runs/next-b1-<stamp>/`.
