# Tests 2 and 3 (video half): per-wave shore labels and the papers' video networks

Status: **labels built and counted 2026-11-19 (development dates only); bars T2 and T3-V
fixed 2026-11-19, before any arm was scored on these labels.** Test 2 needs no training
and is scored right after this commit. Test 3's two networks need the GPU server and
wait their turn in the queue (see "Order and compute").

Code: `scripts/sensing_cdip_waves.py` (labels), `scripts/sensing_cdip_inventory.py`
(counts), `scripts/sensing_shore_test2.py` (test 2), `scripts/sensing_shore_test3.py`
(test 3, video half). Registered video baseline: [webcoos_video.md](webcoos_video.md).

## Labels: CDIP zero-crossing waves per clip

CDIP's zero-crossing files list every half-wave of each buoy record: its time bounds, its
peak elevation (positive for a crest, negative for a trough), a height, a period and a
primary quality flag. For each shore clip on a development date, the label is the buoy
record that contains the clip's midpoint (a good-flagged record in the station's
historic file, 1,600 s long), and from that record:

- **mean individual wave height** (Kim et al. 2023's target),
- **H_1/3 and H_1/10**: the mean of the largest ceil(N/3) and ceil(N/10) heights,
- **T_z**: the mean zero-crossing period,
- **N**, the wave count, kept only as the label's sample size: the periods of the N
  complete waves add up to the record's length, so N is about 1,600 s / T_z and carries
  no information beyond T_z.

A wave counts only when it is complete inside the record (an upcrossing crest followed
by its trough, the two half-waves contiguous) and at least 20 such waves remain. The
buoys stand 6-29 km off the cameras, so a label describes the incident sea of the
record; it is paired with a clip statistically, never wave by wave.

**Convention checked on the data.** The file defines a wave's height and period on its
crest row. On all 1,145 records, the crest row's height equals the crest minus the
*following* trough to rounding (99th percentile of the difference 1e-7 m; against the
preceding trough the median difference is 0.14 m), and its period equals the span of
the crest and following trough rows (99th percentile 3e-7 s; 1.1 s median otherwise).
So each wave is an upcrossing wave, read once from its crest row, as the script does.
Codex's convention note (cbdd6523b) is right.

**Quality tiers.** `good`: both half-waves of a wave carry primary flag 1 (the primary
endpoint). `incl_unevaluated`: flag 1 or 2 (not evaluated), the sensitivity tier. Flag 2
never enters the primary tier.

### Inventory (development dates, finished 2026-11-19 15:35 EST)

`runs/cdip-development-v1.csv` (SHA-256 `d0d5af8a...`), counted by
`scripts/sensing_cdip_inventory.py`. "Interim cohort" is the 992 clips the tests below
use (next section).

| | All development clips | Interim cohort |
|---|---:|---:|
| Dates with labels | 17 of 18 | 17 of 18 |
| Clips labelled | 1,444 of 1,523 | 922 of 992 |
| Independent buoy records | 1,145 (CDIP 433: 389, 243: 375, 200: 381) | 791 |
| Complete waves per record | median 315, fewest 165 | median 295, fewest 165 |
| Clips in Kim's range, 0.75-3.25 m | 311 on 8 dates | 186 on 6 dates |
| 0.75-1.25 m | 241 clips, 8 dates | 119 clips, 6 dates |
| 1.25-1.75 m | 65 clips, 3 dates | 62 clips, 2 dates |
| 1.75-2.25 m | 5 clips, 1 date | 5 clips, 1 date |
| 2.25-3.25 m | none | none |
| Below 0.75 m | 1,133 clips | 736 clips |

- **The missing date** is 2026-09-24 (25 Currituck clips): CDIP 433's historic file ends
  before it. 79 other clips fall in the gap between two records (each record covers
  1,600 s of its 1,800-s slot).
- **Records shared by clips:** 847 records hold one clip, 297 two, one three (mostly
  the two Oak Island views of one record). The bootstrap resamples whole dates, so
  shared records never sit on both sides.
- **Flag sensitivity is empty.** Over all 1,145 records, 748,073 half-waves carry flag 1
  and 2,424 (0.3 %) flag 4 (bad); none carries flag 2. The provider evaluated every
  wave, so the `incl_unevaluated` tier equals the `good` tier wave for wave (0 records
  differ). The "not evaluated" flags met earlier belong to the raw displacement
  variables, not to these per-wave rows.
- **Kim's range is one storm.** Above 1.25 m there are only 2024-04-11 and 04-12 (plus
  two clips on 2024-07-12 in the full set). The class endpoints on these dates would
  measure one storm, not a classifier (coverage rule below).
- **Label checks against the bulk records** (interim cohort, medians): mean height /
  H_s = 0.583 (interquartile 0.561-0.601), not Kim's 0.631; H_1/3 / H_s = 0.92;
  H_1/10 / H_1/3 = 1.25 (Rayleigh 1.27); T_z / T_m02 = 1.05; T_z / T_p = 0.65. The interim
  run's proxy, 0.631 H_s, therefore overstated mean height by about 8 %: it put 222 clips
  in Kim's range against 186 measured, and gave 5.4 % of clips a different class.

## Cohort and folds (both tests)

- **Clips:** the 992 clips of the shore video's interim run (18 development dates; the
  clip list whose hash that run recorded); for the per-wave targets, the 922 of them
  with a label. Their frozen V-JEPA 2 and DINOv3 window features are already cached on
  the server, and their out-of-fold estimates were fixed before that run scored
  anything. The other
  development-date clips have no features yet; computing them belongs to the video
  session's final run.
- **Folds:** that run's five folds of whole local dates (`folds_by_date`, seed 0); inside
  each training fold, one training date picks the epoch (`_val_split`), as there.
- **Bootstrap:** paired, over whole local dates, 2,000 draws, seed 0, 95 % percentile
  interval (`paired_block_bootstrap_ci`).
- **Kim's classes:** five 0.5-m classes of measured mean height from 0.75 to 3.25 m;
  truth exists only for clips inside that range, and a prediction outside it counts as
  the nearest end class.
- **Development only.** Only the 18 dates in `configs/experiments/webcoos_video_dev_dates.txt`
  are labelled. Labels of every other date stay unopened until these bars and the
  recipes below are committed; see "Confirmation".

## Test 2 bar: per-wave targets

**Question.** With Kim's real target (measured mean height, not 0.631 H_s) and
time-domain periods, does the shore video's whole-clip estimate beat the training
fold's majority class and a single frame on unseen dates?

**Arms.** No network is trained. The interim run's out-of-fold H_s estimates are carried
to each height target by one ratio per training fold, exp(mean log(target / H_s)) over
that fold's training clips, computed from labels only (no prediction enters it, so
nothing leaks from the test fold).

| Arm | Estimate |
|---|---|
| `clip` | V-JEPA 2 window probe, energy mean of all K windows (the 0.513 m read-out) x ratio |
| `window` | the same probe, midpoint 16-s window only, x ratio |
| `snapshot` | DINOv3 probe on one frame (the midpoint window's middle frame) x ratio |
| `majority` | the most common class among the training fold's in-range clips (ties to the lower) |
| `climatology` | the training fold's geometric-mean label |
| `spectrum` (T_z) | the registered pixel-spectrum ridge (recipe v1, ten features, alpha 1) refitted on log T_z |
| `clip` (T_z) | the `clip` T_p estimate x the training fold's T_z / T_p ratio |

**Bars** (tier `good`; each met when the interval lies wholly on the stated side of zero):

- **T2-A:** five-class accuracy, `clip` minus `majority`, lower end above zero.
- **T2-B:** five-class accuracy, `clip` minus `snapshot`, lower end above zero.
- **T2-C:** RMSE of T_z, `spectrum` minus `climatology`, upper end below zero.

**Coverage rule (set after counting the labels, before any score).** A class endpoint
(T2-A, T2-B, T3-K) is assessable only when the in-range clips cover at least 8 local
dates and at least two classes each appear on at least 3 dates; otherwise it is recorded
as unassessable, not as a miss. Fewer dates make a date-bootstrap interval unreliable,
and a class seen on one or two dates is one sea state. On the development cohort (6
dates; the second class on 2) the class endpoints are **unassessable**: they are
computed and reported as development numbers, and decided only on the confirmation
dates. T2-C (922 clips, 17 dates) is decided on the development dates.

**Reported beside them, not bars:** RMSE of mean height, H_1/3 and H_1/10 for `clip`,
`window`, `snapshot` and `climatology` with paired differences; within-one-class
accuracy; accuracy per camera; class and date counts; the median wave count; everything
again on the `incl_unevaluated` tier (the flag sensitivity).

## Test 3 bar, video half: Kim's and Kamagata's networks on the same clips

Each paper's network is trained from scratch here on the cohort's folds, five seeds
(0-4, the project read-out's ensemble size), predictions averaged, and compared on
identical clips with the project's `clip` read-out from the interim run.

### Kim et al. 2023: `Kim2023CNNBiLSTM`

- **Input:** the central 180 s of the clip as 360 grayscale frames at 2 fps (every
  second frame of the 4-fps window frames, 256 px high, Currituck's label blacked out as
  for every arm), each centre-cropped to a square and resized to 224 px, grey copied to
  three channels, ImageNet normalisation. Clips need K >= 12 windows (192 s).
- **Network:** torchvision GoogLeNet with ImageNet weights, frozen, 1,024-d per frame
  (cached once); bidirectional LSTM, 2,000 hidden units; dropout 0.5; a linear layer to
  the five classes, read from the last time step (the repository's class, unchanged).
- **Training:** cross-entropy on the in-range training clips; Adam, learning rate 1e-4,
  batch 16, gradient norm clipped at 2, at most 60 epochs; the inner validation date
  (drawn among training dates that hold in-range clips) picks the epoch by validation
  cross-entropy. The paper does not state these settings; they are fixed here. No
  augmentation (the paper's screen splitting is not reproduced).
- **Score:** five-class accuracy on the eligible in-range clips; class probabilities
  averaged over the five seeds.

**Bar T3-K:** accuracy of the project's `clip` classes (test 2's arm) minus Kim's, on the
same eligible clips. Project ahead if the interval's lower end is above zero; Kim ahead
if its upper end is below zero; otherwise no difference resolved. T3-K counts only under
the coverage rule above, so it is decided on the confirmation dates. On the development
cohort Kim's network would learn one storm from 4-5 training dates per fold, so it is
**not trained there**; it runs only as a smoke test (a few clips, two epochs) to prove the
pipeline. Reported beside T3-K: Kim minus `majority`, within-one accuracy, per camera.

### Kamagata et al. 2026: `SlowFastWaveHead` with `kamagata_physics_loss`

- **Input:** for each 16-s window, the cached frozen V-JEPA 2 ViT-L tokens of the last
  layer (layer 23), averaged over the 4 x 4 spatial positions and the two tiles at each
  of the eight time positions (2 s each): eight 1,024-d steps per window. A learned
  linear layer maps 1,024 to 384 so the paper's head runs at its own width.
- **Substitutions, stated:** V-JEPA 2 ViT-L at 4 fps instead of V-JEPA ViT-S at 10 fps
  (only the cached encoder is used); eight 2-s steps per 16-s window instead of 16
  frames of 1.6 s (the slow path, every fourth step, sees two); no optical-flow crop.
  The head itself is the paper's: SlowFast transformers of 2 layers, width 384, dropout
  0.1, 4 and 8 heads; MLP 384-128-5; 5 Softplus.
- **Clips:** all 992 clips of the interim cohort, so the comparison is with the 0.513 m
  read-out on exactly its clips.
- **Targets:** H_s and T_p (the interim run's CDIP labels, the same H_s as the 0.513 m
  read-out), T_z (test 2's `good` tier, missing on 70 clips) and the peak direction
  (CDIP, degrees, a plain scalar as in the paper). H_max has no label here: its output is
  not supervised. Each target is scaled to [0, 1] by its training-fold range; a missing
  label drops out of the loss.
- **Loss and training:** mean squared error over the available targets plus 0.1 x
  |log(T_p^2 + eps) - log(6 H_s + eps)| on the unscaled H_s (m) and T_p (s); AdamW,
  learning rate 1e-4, weight decay 0.05, batch 128 windows (the paper states none), at
  most 50 epochs (the paper's), every window of a training clip carrying its clip's
  labels; the inner validation date picks the epoch by its loss.
- **Read-out:** as the project's: a clip's H_s is the energy mean over its K windows,
  periods the geometric mean; seeds averaged in log space.

**Bar T3-S:** RMSE of H_s, the project's `clip` (0.513 m on the interim run) minus
Kamagata's `clip`, on the same 992 clips. Project ahead if the upper end is below zero;
Kamagata ahead if the lower end is above zero; otherwise no difference resolved. Decided
on the development cohort (both recipes are fixed and neither is tuned there), repeated
on the confirmation dates. Reported beside it: Kamagata read from the midpoint window only (input-duration
ablation), the same network with physics weight 0 (does the loss help?), its T_p RMSE
beside the project's and its T_z RMSE beside the pixel spectrum's and climatology's,
and the paper's own measures (RMSE, R², Pearson) on the Oak Island clips beside its
held-out 0.586 m.

## Order and compute

Test 2 runs on the desktop CPU (seconds), after this commit. T3-S needs one GPU for
under an hour (one read of the last layer of the 74 GB feature cache, which is not
modified, then 50 small heads: five folds, five seeds, with and without the physics
loss), plus a Kim smoke test of minutes. Both come after the paper strip runs (Kim
snapshot, Yang with and without attitude) in the GPU queue, one GPU, at most five of the
project's containers. Outputs go to the run's own folder and a separate GoogLeNet cache;
the video session's windows, cache and pixel-spectrum code are read, never written.

## Confirmation

The class endpoints (T2-A, T2-B, T3-K) are decided on the confirmation dates: every
packaged date outside the development list, in the video session's final run. Once
that run's out-of-fold estimates exist, the label script is pointed at those dates (a
`--dates` option added then; no label of theirs is fetched before), test 2 is scored on
the final run's estimates with this commit's scorer, and Kim's network is trained in the
final run's folds and scored on the confirmation dates' clips, once. T2-C and T3-S are
repeated there. If the confirmation dates also fail the coverage rule, the class
endpoints are reported as unassessable for this sample.

## What changed from the first draft (Codex, 0d331c01f) and why

- **Cohort fixed to the interim run's 992 clips and folds**, so every comparison is
  paired with the 0.513 m read-out on identical clips, without new feature extraction in
  the video session's cache.
- **Test 2 needs no training.** The draft fitted new linear classification heads; here
  the registered estimates are carried to the new targets by a ratio from training-fold
  labels, which keeps the arms exactly those whose design was fixed before scoring. The
  draft's comparison with displacement-derived labels and its leave-site-out run are
  dropped from this bar (new provider queries, retraining); they can follow.
- **T2-A and T2-B are separate bars**: the report's criterion (beat the majority class)
  and the hypothesis (beat a snapshot) can fail separately.
- **Seeds:** five for each paper network, matching the project read-out's ensemble.
- **Coverage rule** stated as numbers (8 dates, two classes on 3 dates each), set from
  the label counts above before any arm was scored; it moves the class endpoints to the
  confirmation dates.
- **Flag sensitivity:** kept in the scorer, but empty on these dates (no flag 2).
- **Labels verified** (convention, flags, ratios to the bulk records), and the inventory
  counted; Codex's label script itself is unchanged.
