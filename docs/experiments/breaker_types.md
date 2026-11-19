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

## Result (2026-11-19): Y-B1 met, Y-B2 not met

Three runs on the lab server, all on commit `0cedfefee`: the feature pass over the 9,000
clips (`video-breakers-20261119t2216-features-v1`, 11 minutes, one GPU), the heads
(`video-breakers-20261119t2254-heads-v1`, 30 minutes for 72 neural fits, both physics arms
and the score) and a 24-clip smoke test before them, which was not scored. 8,998 clips
decoded to 150 frames at 5 fps. Two Huangcuo clips have 148 frames (4.93 fps), and their
last frame is repeated twice. Those are the same two clips whose Set B folders hold a 61st
frame. Every clip is scored once per arm and seed, by the model of the fold that held its
site out (393 site-days).

| Arm | Pooled macro-F1 | Equal-site macro-F1 | Accuracy |
|---|---|---|---|
| DINOv3, order-free head | **0.850** | 0.828 | 0.849 |
| DINOv3, temporal head (candidate) | 0.840 | 0.818 | 0.840 |
| Yin et al.'s CRNN, 5 fps | 0.658 | 0.554 | 0.657 |
| Yin et al.'s CRNN, their 2 Hz frames (comparator) | 0.653 | 0.564 | 0.651 |
| Battjes' thresholds on ERA5 ξ₀ | 0.344 | 0.248 | 0.432 |
| Logistic regression on ERA5 waves, wind and slope | 0.337 | 0.268 | 0.339 |

Chance is about 0.33: three balanced classes guessed at random. Each neural row is the
mean of seeds 0-2.

- **Bar Y-B1 is met:** DINOv3 temporal minus Yin et al.'s CRNN is **+0.187 [+0.162,
  +0.214]** (95 % interval over site-days). Equal-site: +0.254 [+0.198, +0.285]. It holds at
  every seed (+0.16 to +0.20) and every site.
- **Bar Y-B2 is not met:** temporal minus order-free is **−0.010 [−0.018, −0.002]**, so the
  head that cannot see frame order is slightly *better*. Each seed agrees (−0.003 to
  −0.014). Equal-site: −0.009 [−0.027, +0.004].
- **The frame rate does not help Yin et al.'s model either:** their CRNN at 5 fps minus at
  their 2 Hz is +0.005 [−0.006, +0.015].
- **Their protocol, re-run as a check:** under a random 5-fold split our CRNN reaches 94.0 %
  accuracy, against the 96.1 % they report, so the re-implementation is close. DINOv3
  reaches 97.8 %. Holding the site out takes Yin et al.'s model from 94 % to 65 % and
  DINOv3's from 98 % to 84 %.
- **Offshore physics is at chance at a new beach.** Both ERA5 arms score about 0.34.
  Battjes' thresholds never predict surging (no clip has ξ₀ above 3.3), and the fitted
  regression does not transfer between beaches. The camera beats it by +0.50 [+0.44, +0.57].
- Validation accuracy (training sites, sharing days with training) was 0.90 for both DINOv3
  heads and 0.86 for the CRNN, against 0.84 and 0.65 at the held-out site.

Per site (macro-F1 over the types present there, mean of the seeds):

| Site | DINOv3 temporal | DINOv3 order-free | Yin CRNN 2 Hz | ERA5 regression |
|---|---|---|---|---|
| A, Outer Banks | 0.82 | 0.82 | 0.67 | 0.22 |
| B, Oak Island | 0.74 | 0.76 | 0.61 | 0.30 |
| C, Port Aransas (588 spilling, 43 plunging, 11 surging) | 0.71 | 0.73 | 0.24 | 0.04 |
| D, Point Reyes (6 spilling, 460 surging) | 0.90 | 0.90 | 0.45 | 0.19 |
| E, Huangcuo | 0.85 | 0.84 | 0.65 | 0.46 |
| F, Xisha Bay | 0.90 | 0.91 | 0.74 | 0.39 |

Site-by-class recall of the candidate (share of each true type it labels correctly):

| Site | Spilling | Plunging | Surging |
|---|---|---|---|
| A | 0.91 | 0.74 | 0.82 |
| B | 0.95 | 0.73 | 0.62 |
| C | 0.95 | 0.76 | 0.48 (11 clips) |
| D | 0.94 (6 clips) | none | 0.94 |
| E | 0.78 | 0.91 | 0.82 |
| F | 0.84 | 0.88 | 0.97 |

Pooled over all sites, most errors are between plunging and surging: per seed, about 430
surging clips are called plunging and 350 plunging clips are called surging. Spilling is
rarely confused with surging (about 30 and 90 clips). Yin et al.'s CRNN fails most at the
two beaches unlike any training site. At dissipative Port Aransas it calls most spilling
clips plunging or surging (spilling recall 0.27); at reflective Point Reyes it finds only
54 % of the surging clips.

**What it means.** At a beach it was never trained on, a frozen DINOv3 encoder classifies
breaker type at macro-F1 0.84, against 0.65 for the published CRNN re-run the same way.
The gain comes from the encoder, not from motion: neither the frames' order (Y-B2) nor
the frame rate (2 Hz against 5 fps, for the CRNN) improves the score. What the frames show
(foam, the wave face, run-up) is enough at these crops. This does not show that motion
carries no information. The per-frame features are global summaries (class token and mean
patch), which may not keep the movement of an overturning lip, and a video encoder or a
spatial temporal head could still use it. Two limits apply to every number here. The clips
are curated crops around a visible event with unanimous labels, so on continuous video a
detector must first find the event. And within a site the label is mostly the day, so the
357 single-type site-days carry most of the evidence.

Not done: a video encoder (V-JEPA 2 is cached on the server), a spatial temporal head,
and a fine-tuned encoder. Each needs a bar of its own.

## Bar Y-B3: does a frozen video encoder use the breaking motion? (fixed 2026-11-19, before any run)

Y-B2 left open whether motion helps once the encoder itself sees it: DINOv3 looks at one
frame at a time, so its features hold no motion and only the head could have used the
frames' order. **V-JEPA 2** encodes 64 frames at once, in space-time patches, and was
trained to predict masked parts of videos, so its tokens can carry motion. Y-B3 asks
whether they do, at unseen beaches, by feeding the same encoder the same clip in order and
with its frames shuffled. Branch `claude/yin-breakers-vjepa`.

**Features (`scripts/breaker_features.py --encoders vjepa2`).** Each Set A clip is decoded
once (all 150 frames at 5 fps, as in Y-B1). Three versions of the clip are built from the
decoded frames:

- `video`: the 150 frames in order;
- `shuffled`: the same 150 frames in a fixed random order (one permutation per clip,
  seeded by its file name): the same pixels, no coherent motion;
- `still`: in each window below, the window's middle frame repeated 64 times: no motion at
  all, and the appearance of three frames only.

Each version goes through frozen **V-JEPA 2 ViT-L** (`facebook/vjepa2-vitl-fpc64-256`,
24 blocks, 1,024 numbers per token, from the server's Hugging Face cache) in **three
windows of 64 frames** (12.8 s each, the model's clip length) starting at frames 0, 42 and
86, which together cover all 150 frames. Frames are resized from 224 to 256 pixels (the
model's input size; the whole crop is kept) and normalised with ImageNet's mean and
standard deviation, as V-JEPA 2's own preprocessing does. The model cuts each window into
tubelets of 2 frames x 16 x 16 pixels, so it returns 32 x 16 x 16 tokens per window. The
cache keeps the encoder's output (`last_hidden_state`, after its final norm: the tokens
V-JEPA 2's own frozen classifiers read), **averaged over the 16 x 16 spatial positions**:
one token per tubelet, that is per 0.4 s. The three windows overlap, so each of the clip's
75 tubelets is taken from the window whose centre is nearest (ties to the earlier window):
tubelets 1-27 from the first window, 28-48 from the second, 49-75 from the third. Each
clip and version is then a sequence of 75 tokens of 1,024 numbers that covers its 30 s
once (about 1.4 GB per version for the 9,000 clips).

**Arms.** Every V-JEPA 2 arm uses Y-B1's head and training loop unchanged (standardised
features, Linear 256, three residual temporal convolutions, mean and max over time,
Linear 3; AdamW, weight decay 0.05, learning rate 0.001, batch 50, at most 50 epochs, the
same early stopping and epoch choice; seeds 0, 1 and 2; the six official folds). On tokens
0.4 s apart, kernel 5 spans 2 s, and the three layers see 29 tokens (11.6 s).

| Arm | Frames into the encoder | Head |
|---|---|---|
| `vjepa2_video` (**candidate**) | the clip in order | temporal (kernel 5) |
| `vjepa2_shuffled` (**comparator**) | the same frames, shuffled | temporal (kernel 5) |
| `vjepa2_still` | each window's middle frame, repeated | temporal (kernel 5) |
| `vjepa2_video_orderfree` | the clip in order | order-free (kernel 1) |
| `vjepa2_shuffled_orderfree` | the same frames, shuffled | order-free (kernel 1) |

With the order-free head the order of the 75 tokens cannot be used, so any gap between the
two order-free arms comes from inside the encoder's windows.

**Metric and interval.** As for Y-B1 and Y-B2: pooled out-of-site macro-F1 over present
classes (each clip predicted by the model of the fold that held its site out; F1 of each
breaker type averaged over the types present), the mean over seeds 0-2, and the 95 %
interval of a difference from the paired bootstrap over site-days stratified by site
(2,000 draws, seed 0). The Y-B1 run's predictions (`video-breakers-20261119t2254-heads-v1`,
unchanged) are scored on the same draws, so every V-JEPA 2 arm is also compared with
DINOv3 and with Yin et al.'s CRNN.

**Bar Y-B3 (motion in a video encoder).** `vjepa2_video` must raise the pooled
out-of-site macro-F1 over `vjepa2_shuffled`, with the 95 % interval of the difference
(in order minus shuffled) above zero.

Reported beside, not part of the bar: the equal-site difference and each seed's
difference; `vjepa2_video` against `vjepa2_still`; `vjepa2_video_orderfree` against
`vjepa2_shuffled_orderfree`; `vjepa2_video` against Y-B1's `dinov3_temporal` (0.840
pooled), `dinov3_orderfree` (0.850) and `yin_crnn_b2hz` (0.653); per-site macro-F1 and
the site-by-class confusion of each arm.

**How it will be read, fixed in advance.** Shuffling keeps every pixel but makes each
window an abrupt flicker that V-JEPA 2 never saw in training, so it can lower the score
for that reason alone. The still version has no such flicker but shows only three frames.
So: if Y-B3 is met **and** `vjepa2_video` also beats `vjepa2_still`, the encoder uses the
clip's motion. If Y-B3 is met but `vjepa2_still` scores about as well as `vjepa2_video`,
the shuffled arm's loss is more likely damage from scrambled input than lost motion. If
Y-B3 is not met, these frozen, spatially averaged V-JEPA 2 tokens do not use the motion at
unseen beaches; that still does not show that motion carries no information (an attentive
probe over all tokens, V-JEPA 2's usual read-out, keeps where things move, and fine-tuning
could do more).

**Expected in advance.** A small gain of order over shuffle (V-JEPA 2's frozen tokens are
known to carry motion, Something-Something v2) but no large one, because Y-B2 found the
order unneeded at these crops and averaging over the crop blurs where the lip overturns.
V-JEPA 2 is weaker than DINOv3 at still-image recognition, so `vjepa2_video` may well
score below DINOv3's 0.84.

**Runs.** A 24-clip smoke test of the feature pass first (not scored), then the feature
pass over the 9,000 clips and the heads, each one container on one GPU with at least 40 GB
free, launched inside the project's shared launch lock.
