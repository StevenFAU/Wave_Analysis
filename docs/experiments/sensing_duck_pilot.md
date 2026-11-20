# Duck thermal imagery and lidar pilot

Use the two existing days at Duck to test whether optical breaking masks carry information about the height of an already tracked wave. The package contains 989 tracks from 25 runs on November 7 and 793 tracks from 28 runs on November 8, 2016. Use each entire day once for training and the other for evaluation. All frames of one wave remain together. Precomputed masks follow instrument-assisted tracks, so conclusions apply only given a tracked wave.

Use six fixed features from the normalized 64 × 64 mask. Use its whole-image mean, standard deviation, and four quadrant means. Standardize on the training day and fit ridge (penalty 10, untransformed targets). Compare against the training-day mean, on the same rows, for track-median individual height (metres) and trough-to-trough period (seconds). For this deterministic pilot require lower RMSE on each held day separately. Record per-day counts, the pooled errors and per-day errors. Two dates are insufficient for a date-bootstrap claim. Period measures spacing between waves. Testing crest arrival times requires an independent time series and matching observed and detected crest events.

Separately audit the example thermal recording for time-resolved alignment. Assess the crossing times of a fixed optical pixel independently of the lidar labels only when the image clock and spatial mapping are established. The provider also releases a projected breaking mask already co-registered along the lidar line; a result using that mask cannot establish independent pixel-to-crest timing.

## First run

| Held date | Tracks | Height RMSE, mean → mask (m) | Period RMSE, mean → mask (s) |
|---|---:|---:|---:|
| November 7 | 989 | 0.504 → 0.558 | 1.967 → 2.177 |
| November 8 | 793 | 0.816 → 0.718 | 2.782 → 2.717 |
| Pooled | 1,782 | 0.662 → 0.634 | 2.365 → 2.432 |

Both endpoints missed the precommitted two-date criterion. A breaking mask improved height on the second date only. This limits the case for using the mask alone. A raw thermal video model would be a different experiment.

The source's example tower movie has 1,501 frames, the pier movie has 2,968 and the lidar time vector has 1,551 scans. The file documents a per-scan projected breaking mask and geometry, but the example movie group provides no explicit per-frame UTC vector. Array indices alone cannot establish sub-wave alignment. Independent arrival-time evaluation remains open pending a verified image clock or frame-to-scan map from the existing release.

Run `scripts/sensing_duck_mask.py PACKAGE --out runs/duck-mask-pilot.csv` using the existing measurement worktree's `runs/prep-v1/duck` package.
