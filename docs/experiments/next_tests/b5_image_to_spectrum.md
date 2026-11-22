# B5 · Image to spectrum: what does the camera add, frequency by frequency?

Fixed 2026-11-21, before any run. Shared rules: [README](README.md). Status: not started.

## Question

Frozen features explain 86% of the variation in the 3.9 to 6.5 s band and 47% in the
10.5 to 33 s band. That says what the camera sees. It does not say what the camera
**adds** once a forecast and the wind are known, and nobody has predicted a full buoy
spectrum from an image with a forecast-based comparator. Does the camera lower the
error of the predicted spectrum beyond forecast and wind, and in which bins?

## Data

- The absolute energy in the 46 frequency bins per strip, from
  `scripts/measurement_labels.py` (prepared under `runs/prep-v1/buoys-bins`). A strip
  with a zero-total spectrum is missing.
- Target per bin: the equivalent height h_i = 4 √(E_i Δf_i) in metres. This avoids the
  logarithm of empty bins and adds up in energy to the total height.
- Camera features: README rule 2. Forecast features: the corrector's inputs (ERA5 bulk
  and partition fields, GFS-Wave systems, location, depth, measured wind).

## Arms

Stage 1, CPU, a ridge per bin (penalty chosen once on the validation buoys from 1, 10,
100), five folds:

| Arm | Inputs |
|---|---|
| K typical | The training buoys' mean spectrum |
| N no camera | Forecast features |
| P camera | Camera features |
| N+P | Both |

Stage 2, GPU, only if stage 1's primary is met:

| Arm | Model |
|---|---|
| H | The network fine-tuned with a 46-output head (Huber on h_i plus a predicted spread per bin), its own trunk, gentle recipe, three seeds, five folds |
| N+H | Ridge of forecast features and H's outputs, fitted with nested calibration |

A contrastive image-and-spectrum pretraining arm is proposed in a new bar only if H
beats P.

## Pass marks

**Spectral RMSE** is the square root of the mean squared error of h_i over bins and
strips.

- **Primary (stage 1).** Spectral RMSE, N+P minus N: upper end below zero.
- **Stage 2.** H minus P: upper end below zero and a gain of at least 5%. And N+H minus
  N+P likewise.
- **The figure that is the result either way.** Per bin, the error of N, P and N+P and
  the share of N's error that the camera removes, with intervals. This is the camera's
  added value against wave period.
- Reported: coverage of H's 90% ranges per bin; total height recovered from the
  predicted bins against the height head (it must not be worse than 0.532 m by more
  than 0.016 m).

## Controls

Permute the camera features among strips of the same buoy and forecast-height tercile:
the gain of N+P must vanish. Keep spectral targets on their own head or trunk: trained
jointly with height they cost 0.086 m before.

## Cost and where

Stage 1 on the desktop CPU. Stage 2: 15 runs of 45 to 55 minutes, 12 to 14 GPU-hours.

## Build

`scripts/buoycam_spectrum_bins.py` (stage 1 and scoring), an option in
`scripts/train_wave_spectral.py` for the 46 absolute bins (stage 2). Outputs to
`runs/next-b5-<stamp>/`.
