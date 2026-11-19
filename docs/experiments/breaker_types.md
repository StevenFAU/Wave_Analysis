# Breaker type at unseen beaches: Yin et al.'s VWBT_9000

**Question.** Can a camera tell spilling, plunging and surging breakers apart at a beach it
was never trained on, and does it need the motion in the clip to do so? This replaces
the breaker trainer the [measurement review](wave_measurements.md#review-of-the-prepared-experiments-2026-11-19)
rejected (frozen encoders re-decoded every epoch; one frame every 1.9 s; no bar). Branch
`claude/yin-breakers`.

## Data

**VWBT_9000, Set A** ([figshare v5](https://doi.org/10.6084/m9.figshare.28814993.v5),
CC BY 4.0; data paper [Yin et al. 2025](https://doi.org/10.1038/s41597-025-06005-5)):
9,000 clips of 30 s, 3,000 per class, from 15 cameras at six beaches (A-D WebCOOS
cameras in the USA, E-F the authors' own in Fujian). Each clip is a 224 x 224 crop around
a breaking event, labelled by its most prominent breaker; collapsing breakers count as
surging. Three researchers labelled each clip by consensus and two further experts
re-labelled it blind; only clips on which both stages agreed were kept. Set B
holds the same clips as 60 JPEG frames at 2 Hz, the frames Yin et al. trained on. All four
files are checked against the record's MD5s (`deploy/dgx/breakers.sh fetch`).

| Site | Beach (tan β, Table 1) | Clips | Spilling | Plunging | Surging | Site-days |
|---|---|---|---|---|---|---|
| A | Northern Outer Banks, intermediate (0.10) | 1,999 | 940 | 614 | 445 | 122 |
| B | Oak Island, intermediate (0.10) | 2,558 | 528 | 1,142 | 888 | 122 |
| C | Port Aransas, dissipative (0.02) | 642 | 588 | 43 | 11 | 46 |
| D | Point Reyes, reflective (0.15) | 466 | 6 | 0 | 460 | 37 |
| E | Huangcuo, intermediate (0.09) | 694 | 215 | 281 | 198 | 21 |
| F | Xisha Bay, intermediate (0.06) | 2,641 | 723 | 920 | 998 | 45 |

A site-day is one beach on one local date (standard time of the site). **357 of the 393
site-days hold a single breaker type**: within a site the label is mostly a property of
the day. That is why a random split is easy (below) and why the interval resamples
site-days.

**The split is the official leave-one-site-out file** of v5
(`loso_split_by_database_filename.csv`, six folds). Fold k tests on site k only. Its
validation clips are the latest 20 % by time of each site and class of the five training
sites (the rule in the authors' [BreakTypeNet code](https://github.com/comyo/BreakTypeNet)).
The test site is never used for training or for choosing an epoch. Note that validation
shares 17-21 site-days with training in every fold, so validation scores run high. That
can only make the chosen epoch worse at the test site, and it does so for every arm alike.

## What is published, and the comparator

Yin et al. report one result: a CRNN reaching **96.1 % mean validation accuracy over a
random stratified 5-fold split** of all 9,000 clips, with the epoch chosen on the same fold
it is scored on ([paper](https://europepmc.org/article/PMC/PMC12575733), [code](https://github.com/comyo/Nearshore-Wave-Breaking-Type-Classification)).
In a random split, clips of the same site-day sit on both sides. **There is no published
site-held-out score.** The authors' later BreakTypeNet repository has the LOSO pipeline
that produced the official split, but no numbers. So the comparator is **Yin et al.'s own
CRNN, re-run on the official split**:

- frozen ImageNet ResNet-18 (torchvision V1 weights) on each of their Set B frames 1-59
  (`range(1, 60)` in their notebook; frames are numbered 1-60);
- per frame FC 1024-BN-ReLU, FC 768-BN-ReLU, dropout 0.3, FC 512; a three-layer LSTM of
  512; the last step through FC 256-ReLU, dropout 0.3, FC 3;
- Adam, learning rate 0.001, batch 50, at most 50 epochs, early stopping after more than
  5 epochs without a better validation accuracy, the epoch with the best validation
  accuracy kept.

Deviations, all forced by caching or harmless: the frozen ResNet runs with eval-mode batch
norm (ImageNet statistics; in their loop it ran in training mode, so its statistics drifted
to the training frames); the head's batch norm normalises over all frames of a batch at
once rather than over each time step in turn. The same CRNN under their own random 5-fold
protocol is reported beside, as a check that the re-implementation is not handicapped:
it should come close to 96.1 %.

## Temporal sampling

Every Set A clip is MPEG-4 at **5 fps, 150 frames** (60 clips checked across all 15
cameras before the design; the feature pass records every clip's count and rate). The
source cameras ran at 5-20 fps (Table 2), so Yin et al. resampled every clip to 5 fps,
the rate of the slowest camera. A breaking event lasts about one to two seconds (the
review's estimate). At 5 fps that is 5-10 frames, at Yin et al.'s 2 Hz it is 2-4, and at
the old trainer's one frame per 1.9 s it is at most one. **All 150 frames are used**: 5 fps
is all the clips carry. The temporal head's kernel spans 5 frames (1 s), and three dilated
layers (1, 2, 4) give it 29 frames (5.8 s) of context, about one wave period (ERA5 peak
period median 5.3 s over the clips).

## Features: each clip decoded once

`scripts/breaker_features.py` reads every clip from the zips once on the lab server
(OpenCV's bundled FFmpeg; the server image has no ffmpeg). It writes frozen per-frame
features as fp16 arrays:

- `a_dinov3`: DINOv3 ViT-B/16 (`vit_base_patch16_dinov3.lvd1689m`), every Set A frame
  at 224 x 224, the class token and the mean patch token (1,536 numbers);
- `a_resnet18`: Yin et al.'s ResNet-18 on every Set A frame (512 numbers);
- `b_resnet18`: the same ResNet-18 on Set B's 60 frames.

Every head trains on this cache (`scripts/train_breaker_heads.py`), so the 6 x 3 x 4
neural runs decode nothing.

## Arms

| Arm | Frames | Model |
|---|---|---|
| `yin_crnn_b2hz` (**comparator**) | Set B, 2 Hz, frames 1-59 | Yin et al.'s CRNN on frozen ResNet-18 |
| `yin_crnn_a5fps` | Set A, 5 fps, 150 | the same CRNN |
| `dinov3_temporal` (**candidate**) | Set A, 5 fps, 150 | frozen DINOv3; Linear 256, three residual temporal convolutions (kernel 5, dilations 1, 2, 4), mean and max over time, Linear 3 |
| `dinov3_orderfree` (motion control) | Set A, 5 fps, 150 | the same head with kernel 1: each frame alone, then pooled; frame order and so motion cannot be used |
| `iribarren_fixed` | none | offshore surf-similarity ξ₀ = tan β / sqrt(H_s / L₀), L₀ = g T_p² / 2π; spilling below 0.5, plunging 0.5-3.3, surging above ([Battjes 1974](https://doi.org/10.9753/icce.v14.26)) |
| `iribarren_fit` | none | multinomial logistic regression on log ξ₀, log H_s, log T_p, log tan β and wind speed, fitted on the training sites |

H_s, T_p and wind speed are the ERA5 values Yin et al. matched to each clip
(`ERA5_offshore_wave_wind_by_clip.csv`), and tan β is each site's slope from their Table 1.

Every neural arm uses Yin et al.'s training loop: batch 50, at most 50 epochs, learning
rate 0.001, the same early stopping and the same epoch choice. The DINOv3 heads use
AdamW (weight decay 0.05) on features standardised with the fold's training clips. Seeds
0, 1 and 2 for every neural arm and fold; the physics arms are deterministic.

## Metric

- **Macro-F1 over present classes:** F1 of each breaker type, 2 TP / (2 TP + FP + FN),
  averaged over the types that occur among the true labels scored. It runs from 0 to 1;
  always guessing the most common type scores at most 0.5 on a site with two types.
- **Pooled out-of-site macro-F1** (primary): over all 9,000 clips, each predicted by the
  model of the fold that held its site out. All three types are present.
- **Equal-site macro-F1:** the mean of the six sites' own macro-F1 (each over its present
  types), so site D's 466 clips weigh as much as F's 2,641.
- **Site-by-class confusion:** for each site, true type against predicted type.

## Bars (fixed 2026-11-19, before any run that scores them)

Each statistic is the mean over seeds 0-2 of each arm's pooled macro-F1. The 95 % interval
of a difference comes from a paired bootstrap over site-days: each site's days are
resampled with replacement and the site keeps its number of days (2,000 draws, seed 0).
All arms are recomputed on the same draws. Site-days are the unit because labels and
conditions are shared within a day.

**Bar Y-B1 (the camera against the published model).** `dinov3_temporal` must raise the
pooled out-of-site macro-F1 over `yin_crnn_b2hz`, with the 95 % interval of the difference
(candidate minus comparator) above zero.

**Bar Y-B2 (motion).** `dinov3_temporal` must raise the pooled out-of-site macro-F1 over
`dinov3_orderfree`, with the interval above zero. If it is not met, these features and this
head do not use the clip's motion at unseen beaches; it does not show that motion carries
no information.

Reported beside, not part of the bars: the equal-site differences with their intervals;
each seed's difference; accuracy; each arm's per-site macro-F1 and site-by-class confusion;
`dinov3_temporal` and `yin_crnn_b2hz` against `iribarren_fit` (does the camera beat
offshore waves and beach slope); `yin_crnn_a5fps` against `yin_crnn_b2hz` (does Yin et
al.'s model gain from 5 fps); `dinov3_orderfree` against `yin_crnn_b2hz` (the encoder
alone); and Yin et al.'s random 5-fold protocol for `yin_crnn_b2hz` and `dinov3_temporal`
(mean over folds of the accuracy at the best validation epoch, the quantity they report).

**Expected in advance.** Far below 96 % for every arm. A new beach changes the camera,
light, crop scale and wave climate together, and within a site the label is mostly the
day. Site D, the only reflective beach (almost all surging), and site C, almost all
spilling, should be the hardest folds, because no training site resembles them. Battjes'
thresholds should rarely predict surging: offshore ξ₀ above 3.3 needs very long, low swell
on a steep beach. The fitted physics arm may be a strong comparator, because the day's
waves drive both the label and ξ₀.

## Runs

`deploy/dgx/breakers.sh` (fetch, push, decoder, features, heads, results) on the lab GPU
server, one container at a time on one GPU, each launch inside the project's shared launch
lock. A feature smoke test on 24 clips comes first and is not scored.
