# Wave measurements beyond height: slope, breakers and the rest

**Question.** The buoy-camera study found the camera to be a wind-sea instrument: it
reads the wind-wave height better than ERA5 and adds little to the total
([sea-state study](ndbc_buoycam_sea_state.md), stage 2 and stage 3). Surface slope is a
wind-sea quantity, so the next measurement is the slope the buoy itself records. Codex
prepared this and other measurement experiments on 2026-11-19 (branch
`codex/wave-measurement-experiments`, code `0674b9c`); this doc fixes their bars and
records the review of that code (branch `claude/wave-measurements-review`).

Launch and rebuild guide: [measurement_checkin.md](measurement_checkin.md). Rationale and
published comparisons: [Wave measurements to learn next](../../reports/Wave%20measurements%20to%20learn%20next.md).

## Bar M-S1: surface slope from the buoy cameras (fixed 2026-11-19, before any run)

**Target: the buoy's RMS surface slope, dimensionless.** For the strip's own buoy,

    s = sqrt( sum_i E(f_i) df_i k_i^2 ),   k_i = (2 pi f_i)^2 / g   (deep water)

over NDBC's 46 realtime spectral bands from 0.030 to 0.495 Hz (centres 0.0325 to
0.485 Hz; nominal WPM widths 0.005, 0.01 and 0.02 Hz). E is the measured spectral
density (m²/Hz) of the records within 30 min of the strip, averaged as the heights are
(ADR 0010); k is in rad/m, so k² E df is a slope variance and s is a slope (rise over
run): s = 0.056 is a typical slope of about 3.2°. s² is the mean-square slope of the
waves the buoy resolves ([Longuet-Higgins 1957](https://doi.org/10.1098/rsta.1957.0002),
the second moment of the wavenumber spectrum; band table: [NDBC](https://www.ndbc.noaa.gov/faq/wavespectra.shtml),
[Earle 2003](https://www.ndbc.noaa.gov/wavemeas.pdf)). The upper edge, 0.495 Hz, is a
wavelength of about 6.4 m, so this is **not** the optical slope of sun-glitter studies
([Cox & Munk 1954](https://doi.org/10.1364/JOSA.44.000838)), which is dominated by
shorter waves and ripples. Built by `scripts/measurement_labels.py` (column `slope_rms`)
for all 41,091 strips: mean 0.056, standard deviation 0.021. With the nominal widths,
4 sqrt(m0) reproduces NDBC's WVHT to 0.976-0.988 at all 79 buoys (the
stage-3 check, now with the right widths). Beside it, not scored by the bar: the same
slope with finite-depth dispersion (`slope_finite_depth`; the 79 depths from NDBC's
station pages; 0.05 % from deep water on average, up to 8 % at the shallowest buoys)
and without the bands above 0.35 Hz (`slope_below_035`), for the noisy high-frequency
tail. **50 strips have a spectrum with no energy at all** (all at buoy 46081, WVHT 0.0):
dropouts, not calm seas; they get no slope and are left out of its fit and score.

**Folds and checkpoints.** The grouped cross-validation of bar B2-CV: five folds over
the 43 station groups (`split_fold0` .. `split_fold4`; the measurement package's
columns are identical to `ndbc_buoycam_heights_v2cv`'s on all 41,091 strips, checked
2026-11-19). Fold k is scored only on its test buoys with the seed-0 DINOv3-gentle model
of fold k, which trained on that fold's training buoys and chose its epoch on its
validation buoys. The probe checks each checkpoint's fold and that its own test strips
are exactly the fold's test strips, and records its SHA-256:

| Fold | Checkpoint run | SHA-256 (first 12) |
|---|---|---|
| 0 | `buoy-fold0-20261118t2234-seed0-cv-dinov3-gentle` | `ea0a3ff2ed74` |
| 1 | `buoy-fold1-20261118t2342-seed0-cv-dinov3-gentle` | `a7edaec7bbe8` |
| 2 | `buoy-fold2-20261118t2346-seed0-cv-dinov3-gentle` | `a272a4e2ea52` |
| 3 | `buoy-fold3-20261119t0038-seed0-cv-dinov3-gentle` | `41d2e02fa7dc` |
| 4 | `buoy-fold4-20261119t0130-seed0-cv-dinov3-gentle` | `d4a593b9d7c8` |

**Probe** (`scripts/measurement_buoy_probe.py`, one run per fold; nothing trained on the
GPU). The camera's features are the frozen model's pooled view-mixer output (256
numbers per strip, the masked mean over usable views). Every arm is a ridge regression
(penalty 10 on standardised inputs); the standardisation and the fit use the fold's
**training strips only**, and the test buoys are only predicted.

- **Comparator, wind + geometry:** the buoy's wind speed and gust, their squares and
  product; and the horizon over at least three clear views: the strip's raw roll RMS and
  the spread of its horizon row, plus the roll and pitch RMS after removing each camera's
  mounting (its median angle and row over its own strips, from images only, as the
  [motion study](ndbc_buoycam_sea_state.md#the-camera-as-a-motion-sensor-the-horizon-follows-the-buoys-slope-2026-11-18)).
- **Candidate, all inputs:** the same plus the 256 camera features.

Both arms score the same strips: the test strips with a slope, wind and at least three
clear horizons (the "eligible" strips; their count and buoys are reported).

**Bar M-S1.** All inputs must lower the out-of-fold RMSE of the slope against wind +
geometry, with the 95 % interval of the difference (all inputs minus wind + geometry)
below zero: paired bootstrap over the 43 station groups (2,000 draws, seed 0), over the
eligible strips of all five folds pooled. Groups, not buoys, are the unit, because
buoys of one group share weather; this is the more cautious of the two intervals B2-CV
reported.

Reported beside the bar, not part of it: the difference with every group weighted
equally; coverage; the difference per fold; the camera alone, wind alone and geometry
alone; the same arms for the wind-wave height (a check that the camera still sees the
wind sea); the other probe targets; and the **validation-fit sensitivity**: the same two
arms fitted on the fold's validation strips instead of its training strips.

**Why the sensitivity.** On the training strips the camera's features are in-sample:
the encoder was fitted to their heights, so they read those strips better than they will
read a new buoy, and a ridge fitted there leans on them too much. That can only make the
camera look worse at the test buoys. The validation strips were seen only to choose the
epoch, so their features behave like a new buoy's; their cost is fewer buoys (11-14 per
fold) to fit on.

**What justifies the slope fine-tune** (`configs/experiments/buoy_slope_v1.yaml`: ten
epochs, slope and wind-wave heads, started from the same fold's encoder and mixer): bar
M-S1 met, **or** the validation-fit sensitivity meeting the same condition (its interval
below zero), because the fine-tune trains end to end and does not share the in-sample
problem. If neither interval lies below zero, the frozen camera adds nothing to wind and
geometry at unseen buoys and the fine-tune is not run. If it runs, its own bar is fixed
here before it does.

**Expected in advance.** Wind alone is a strong comparator: across all strips the slope's
rank correlation is 0.87 with the wind speed and 0.90 with the wind-wave height, and
Codex's CPU pilot put a wind-only ridge at about 0.0116 against the slope's spread of
0.021. The camera reads the wind sea better than ERA5, so a small gain over wind is
plausible; a large one is not.

**Not part of this bar, run if time allows:** the same probe with only the water below
each horizon (`--water-only`), the control that the camera does not read the slope from
the horizon line alone; and the confirmation of a pass on strips archived after the
package (from 2026-11-14 16:00 UTC), as stage 2's.
