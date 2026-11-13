# Sea state from the Currituck camera (`currituck_sea_state_v0`)

**Question.** From one still of the WebCOOS camera `currituck_hampton_inn`
(Corolla, North Carolina), can a network tell the sea state in four WMO
classes as well as, or better than, a wave hindcast that never sees the image?
The answer is a confusion matrix on held-out days and on a held-out year,
beside baselines that use no image (ADR 0007). Decided with Steven on
2026-11-06; package built and baselines scored on 2026-11-10; training on the
H200 is next ([cluster guide](../operations/cluster.md)).

Config: `configs/experiments/currituck_sea_state_v0.yaml`. Package:
`scripts/sea_state_package.py` (desktop). Training and evaluation:
`scripts/train_sea_state.py` (GPU), `wave_analysis.models.sea_state`,
`wave_analysis.evaluation.sea_state`.

## Ground truth

CDIP 433, Duck FRF 17 m, 9.2 km from the camera. NDBC publishes the same buoy
as **44056**, so this is the national buoy record; it is taken from CDIP for
the sample start and end times, the QC flags and the displacement records.
Both compute Hs as 4√m0. The label is the ADR 0010 window mean of the records
centred within ±30 min of the still (`Hs_label_m`), from the pairing in
[WebCOOS](../datasets/webcoos.md).

## Stills

Daylight and low-sun stills that the pairing labelled: **37,155 stills on
1,606 days**, January 2022 to October 2026. Left out: 1,952 stills without a
buoy record within 30 min or with an unreadable image, and 1,206 taken in
civil or nautical twilight or at night (the camera turns monochrome). Each
still goes into the package resized to 900 × 509 px (JPEG quality 90): 3.3 GB.

## Classes

WMO-No. 306, code table 3700 ("state of the sea"). Its note assigns an exact
bounding height to the lower code figure ("a height of 4 m is coded as 5"), so
the bands are closed on the right. That matters here: 338 labels are exactly
0.5, 1.25 or 2.5 m.

| Class | WMO codes | Hs | Stills | Days | Share |
|---|---|---|---|---|---|
| smooth | 0–2 | ≤ 0.5 m | 5,450 | 399 | 14.7 % |
| slight | 3 | 0.5–1.25 m | 22,977 | 1,267 | 61.8 % |
| moderate | 4 | 1.25–2.5 m | 7,909 | 523 | 21.3 % |
| rough or more | 5–9 | > 2.5 m | 819 | 67 | 2.2 % |

By local year (smooth / slight / moderate / rough or more): 2022 889 / 3,937 /
1,081 / 197; 2023 1,123 / 5,053 / 1,952 / 127; 2024 1,100 / 5,303 / 2,205 /
133; 2025 1,346 / 4,780 / 1,779 / 243; 2026 992 / 3,904 / 892 / 119.

What follows for every result:

- Always answering "slight" is right 58–62 % of the time, so **balanced
  accuracy and macro-F1 lead**, not accuracy. A class never predicted counts
  as F1 = 0 in macro-F1 (fixed on 2026-11-10; before, such classes were
  skipped, and always answering "slight" scored 0.76).
- **9.9 %** of the stills lie within 4 % of a class boundary, about the
  sampling error of a 30-minute buoy Hs. There even a perfect model disagrees
  with the label half the time, so the report also gives every metric without
  those stills, and within-one accuracy.
- "Rough or more" is 819 stills on 67 days. Its row of the confusion matrix is
  noisy; counts are given beside every rate.
- Within-one accuracy says little with four classes: even the majority answer
  is within one class of 97–99 % of the labels.

## Splits

Never by random still: stills of one day share their sea state.

| Split | Train | Val | Test | Later test |
|---|---|---|---|---|
| **day**: whole local days, 70/15/15 (seed 0) | 25,987 stills, 1,120 days | 5,577, 241 | 5,591, 245 | — |
| **year**: train 2022–2024, test 2025 | 19,630, 831 | 3,470, 149 (15 % of the weeks of 2022–2024) | 8,091, 352 (2025) | 5,907, 271 (2026) |

The year split leaves out the first three days of 2025 (72 h embargo, 57
stills) so that one swell cannot link training and test. The leakage audit
finds no shared day and no train–test pair closer than 6 h in either split.
The day split is the optimistic one: neighbouring days share swells. The
held-out year is the honest test, and 2025 has the most rough stills (243).
In the day split, the test partition has only 62 rough stills.

## Baselines (no image)

Fitted on each split's training stills, scored on its test stills with 95 %
intervals from a bootstrap over whole days:

| Split / test | Baseline | Stills (coverage) | Balanced acc. | Macro-F1 | QW kappa | Accuracy |
|---|---|---|---|---|---|---|
| year / 2025 | majority ("slight") | 8,091 (100 %) | 0.25 | 0.18 [0.18, 0.19] | 0.00 | 0.58 [0.54, 0.63] |
| year / 2025 | monthly climatology | 8,091 (100 %) | 0.25 | 0.18 | 0.00 | 0.58 |
| year / 2025 | **WIS 44056, corrected** | 8,091 (100 %) | **0.77 [0.70, 0.82]** | **0.80 [0.74, 0.84]** | **0.82 [0.78, 0.85]** | 0.83 [0.81, 0.86] |
| day / test | majority | 5,591 (100 %) | 0.25 | 0.19 [0.18, 0.20] | 0.00 | 0.61 [0.56, 0.66] |
| year / 2025 | ERA5, corrected | 8,091 (100 %) | 0.73 [0.67, 0.78] | 0.76 [0.70, 0.80] | 0.78 [0.74, 0.82] | 0.80 [0.77, 0.83] |
| day / test | WIS 44056, corrected | 4,138 (74 %) | 0.75 [0.55, 0.82] | 0.78 [0.56, 0.83] | 0.78 [0.71, 0.83] | 0.82 [0.78, 0.86] |
| day / test | ERA5, corrected | 5,591 (100 %) | 0.69 [0.57, 0.78] | 0.73 [0.58, 0.79] | 0.74 [0.68, 0.78] | 0.79 [0.76, 0.83] |
| year / 2026 | majority | 5,907 (100 %) | 0.25 | 0.20 [0.19, 0.21] | 0.00 | 0.66 [0.62, 0.71] |
| year / 2026 | **ERA5, corrected** | 5,907 (100 %) | **0.75 [0.67, 0.80]** | **0.77 [0.70, 0.81]** | **0.76 [0.71, 0.81]** | 0.82 [0.79, 0.85] |

- **The monthly climatology is the majority class**: "slight" is the most
  frequent class in every calendar month, so season alone gives nothing at
  class level.
- **WIS is a strong baseline.** The USACE Wave Information Study hindcast at
  44056, interpolated to the still's time and corrected with
  Hs = 0.10 + 1.14 · Hs_WIS (fitted on the training stills of the year split),
  recalls 77 % of each class on average in 2025, and all its errors are
  within one class. Confusion matrix on 2025 (rows true,
  columns predicted; smooth, slight, moderate, rough):
  `[790, 548, 0, 0] / [328, 4232, 171, 0] / [3, 207, 1547, 22] / [0, 0, 66, 177]`.
  Its weakest class is "smooth" (59 % recall): calm days are hard to hindcast.
- **WIS has no 2022 at this site** (the THREDDS catalog of ST44056 has no
  2022 folder; 2021 and 2023 have all twelve months; checked 2026-11-10), and
  it ends 2025-12-31. So its correction in the year split is fitted on 2023–2024
  only, it covers 74 % of the day split's test stills, and none of 2026.
- **ERA5 covers every still and is weaker than WIS.** ERA5 at the nearest
  grid point with sea data (36.5 N 76.0 W, 33 km from the camera), corrected
  with Hs = −0.06 + 2.32 · Hs_ERA5 (fitted on the year split's training
  stills). That point lies on the coast by Currituck Sound, and its values are
  damped: over the 58 months its mean Hs is 41 % of the open-ocean point
  east of it (36.5 N 75.5 W), hence the slope. But of the 30 sea points in the
  downloaded box it tracks the buoy best: r = 0.94 on the training stills,
  against 0.91–0.92 for the ocean points at 75.5 W. Tried on 2026-11-13, the
  ocean point scored lower on every test (balanced accuracy 0.66 on 2025), so
  the nearest-sea-point rule stays. On 2025 ERA5 is below WIS (0.73 against
  0.77), and its weakest class is again "smooth" (52 % recall). It is the
  **only physics baseline for 2026**. Confusion matrices (rows true, columns
  predicted): 2025 `[694, 644, 0, 0] / [380, 4144, 207, 0] / [0, 316, 1443, 20] / [0, 0, 71, 172]`;
  2026 `[600, 392, 0, 0] / [265, 3492, 147, 0] / [0, 220, 657, 15] / [0, 0, 29, 90]`.
  ERA5's last two months (September and October 2026) are still the
  preliminary ERA5T, which ECMWF finalizes about three months later.
- The model is always compared with a baseline **on the same stills**
  (`metrics.json`: `model_same_stills`).

**The bar for the camera:** on the held-out year, beat balanced accuracy 0.77
and macro-F1 0.80 by more than the intervals, and above all improve on the
smooth and rough rows. On 2026, where WIS has no data, the bar is ERA5:
balanced accuracy 0.75, macro-F1 0.77.

## Model

`convnext_tiny` (torchvision, ImageNet weights), a new four-class head,
class-weighted cross-entropy (inverse frequency, mean 1), AdamW with a one-epoch
warm-up and cosine decay, 15 epochs, bf16 on the GPU. The epoch is chosen by
macro-F1 on val; the test partitions are predicted once, with that epoch.

**Input: the sea band only.** Rows 27–60 % of the frame (900 × 168 px): the
horizon sits at about a third of the height in every year 2022–2026, and the
band keeps the sea and the surf zone while leaving out the beach. Two
confounds are visible on the beach and on the lens rather than in the waves:

- People, umbrellas and tents mark summer, which goes with calm seas. The
  full frame is run as an ablation (`INPUT=full_frame`) to show how much the
  beach adds.
- Rain, spray and haze on the lens go with storms (for example 2024-09-16,
  Hs 2.3 m, a blurred frame). The crop does not remove this, so a model may
  partly learn "weather" rather than "waves"; the per-class errors on hazy and
  clear days are worth a look before any claim.

The stills carry only a constant "Hampton Inn" caption in the lower left (no
date or time), outside the crop. **No horizontal flips:** the camera is fixed,
and a mirrored scene is not one it can see. Augmentation is a 0.85–1.0 scale
jitter, ±2° rotation and mild colour changes.

## Status

- [x] Package built and checked (`SHA256SUMS`), baselines scored (2026-11-10).
- [x] CPU smoke test of the training script (2026-11-10, commit `72fc917`, a
      clean worktree): 100 stills per partition, one epoch, 70 s end to end;
      every output file written, provenance `dirty: false`. Its numbers mean
      nothing (100 training stills).
- [ ] Runs on the H200: year split, day split, full-frame ablation
      ([cluster guide](../operations/cluster.md)).
- [x] ERA5 baseline (2026-11-13): standardized for 2022-01 to 2026-10, package
      rebuilt from a clean tree (commit `fdff589`, `dirty: false`, `SHA256SUMS`
      checked).
