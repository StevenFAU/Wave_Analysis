# Shore-camera video vs a single image (WebCOOS, CDIP labels)

Status: **design fixed 2026-11-18, before any clip was scored**; whole-clip implementation
and pixel-spectrum recipe fixed 2026-11-18, before any clip was scored; **first scored
(interim) run 2026-11-19: M3 and M4 met, M1, M2 and M5 not met** (Results). Config
`configs/experiments/webcoos_video_v1.yaml`; code `scripts/webcoos_video_clips.py`
(clips, frames, windows, package, shipping) and `scripts/train_video_probe.py` (features,
heads, scores).

## Question

Does motion help? Kim et al. 2023 report 78.3 % from one snapshot and 93.2 % from 180 s
of video on one fixed camera, but on a random split of 30 one-hour sessions (test frames
from the same hour as training frames). Kamagata et al. 2026 use V-JEPA + SlowFast on
1.6-s windows of shore video and reach H_s RMSE 0.586 m, R² 0.107, Pearson 0.436 on five
held-out videos (Oak Island E/W, Xishawan), with labels that cannot be traced to an
instrument. Neither isolates the motion: no published camera work compares video and a
single frame **of the same clips, through the same encoder, at unseen days**, against an
instrument. This test does.

## Data

- **Clips:** the WebCOOS video sample (`scripts/webcoos_video_sample.sh`, approved by
  WebCOOS): Currituck (Hampton Inn), Jennette's Pier (south), Oak Island east and west,
  one clip per 30-min CDIP record in daylight. Test days first, then six sample weeks
  2024-04 to 2026-01.
- **Labels:** CDIP buoys (433 Currituck, 243 Jennette, 200 Oak Island): H_s the mean of
  the records centred within 30 min of the clip's midpoint (ADR 0010), T_p from the
  nearest record. The buoys lie 20-30 km offshore in 13-21 m of water, so the label is the
  incident sea, not the breakers the camera sees.
- **Frames:** 16 s around the midpoint at 4 fps (64 frames; V-JEPA 2's pretraining rate
  and length), 256 px high, aspect kept; Currituck's burned-in "Hampton Inn" label blacked
  out first. Clips with the sun below the horizon at their midpoint are left out.
- **Why whole frames, not surf-zone crops:** the cameras do not hold a fixed view. A
  monthly montage of stills (2022-2026) shows Jennette's pointing changing from month to
  month and Oak Island east replaced in mid-2024 (2048 x 1536 to 2560 x 1440); Currituck's
  framing holds but its lens blurs from 2025. A fixed crop box would cut different scenes
  on different days. Each frame is fed as two squares (its left and right edge), which
  together cover it without stretching (V-JEPA 2's position code needs squares).

## Arms (identical pixels)

| Arm | Encoder (frozen) | Input |
|---|---|---|
| `video` | V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`, Apache-2.0/MIT) | the 64 frames in order |
| `still` | the same | the middle frame repeated 64 times: same encoder, same token count, no motion |
| `shuffled` | the same | the 64 frames in a fixed random order: appearance kept, motion destroyed |
| `image` | DINOv3 ViT-L/16 (`vit_large_patch16_dinov3.lvd1689m`, DINOv3 licence) | the middle frame |

Tokens of four layers (50, 67, 83 and 100 % depth) are average-pooled (V-JEPA 8 x 4 x 4
per tile, DINOv3 4 x 4) and cached. Backbone choice and the controls follow the
backbone review (`reports/Vision backbones for wave cameras.md`): V-JEPA 2 is the only
family pretrained on a span that holds a wave period, and the probing literature
recommends the shuffled and single-frame controls.

## Heads and folds

- **Primary:** an attentive probe (one self-attention block, a learned query, d = 256);
  beside it an MLP on each layer's mean token. Targets log H_s and log T_p, standardised
  on each training fold; Huber loss; AdamW, 60 epochs, one-cycle schedule.
- **Folds:** 5 folds of whole local dates (all cameras of a date together, so one storm
  never sits on both sides). Inside each training fold one date picks the epoch.
  5 seeds, predictions averaged in log space. Every clip is predicted once, out of fold.

## Bars (fixed before scoring)

- **M1:** `video` minus `still`, RMSE of H_s, 95 % interval below zero (paired bootstrap
  over local dates, 2000 resamples).
- **M2:** the same for T_p.

Reported beside them, not bars: `video` minus `shuffled` (if the shuffled clips do as
well, the gain is appearance, not motion), `video` minus `image` (DINOv3), each camera
separately, Kamagata's measures (RMSE, R², Pearson of H_s; RMSE of T_p) on the Oak Island
clips, and Kim's five classes of mean wave height (0.631 H_s, edges 0.75-3.25 m).

**When:** the first scored run once at least 15 local dates are framed (the test days
and the first sample weeks); it is interim until the sample download finishes, and the
final run uses the same code and config.

## Whole-clip bars (fixed 2026-11-18, before any clip was scored)

**Why.** The arms above read one 16-s window per clip, which holds one to three waves.
The buoy's H_s is a statistic of a 30-min record: for a 17-min record its 90 % sampling
interval is about ±12 % (Donelan & Pierson 1983), and the error grows as the record
shortens, to roughly ±50 % (one standard deviation) for 16 s. A 16-s window also resolves
frequency only in steps of 1/16 Hz, so it cannot measure T_p from the motion alone. Most
clips last 10 min, i.e. 37 windows, of which the arms above read one. Kim et al. 2023 gained 12 points going from one frame
to 180 s. These bars ask whether reading the whole clip helps, with the encoder and probe
held fixed (`reports/Wave camera architectures beyond backbones.md`, rank 2).

**Windows.** Each clip is cut into K = floor(duration / 16 s) non-overlapping 16-s
windows from its start, 64 frames each at 4 fps, 256 px high, same masks and two squares
as above (K = 37 for a 600-s clip). The **midpoint window** is the one that contains the
clip's midpoint. Of the 401 clips packed so far, 43 are shorter than 240 s (down to 21 s,
K = 1, where `clip` equals `window`): every usable clip is scored, and the bars are also
reported on the clips with K ≥ 15.

**Arms (one trained model, two ways of reading a clip).** Frozen V-JEPA 2 ViT-L tokens
per window, as above. The attentive probe is trained on every window of the training
clips, each window carrying its clip's label; folds, inner validation, seeds and
hyperparameters are those of the arms above.

| Arm | Test-time input | Clip estimate |
|---|---|---|
| `window` | the midpoint window only | the probe's output |
| `clip` | all K windows | H_s: energy mean, sqrt(mean(H_s,w²)); T_p: exp(mean(log T_p,w)) |

**Bars.**

- **M3:** `clip` minus `window`, RMSE of H_s, 95 % interval below zero.
- **M4:** the same for T_p.
- **M5:** `clip` + pixel spectrum minus `clip`, RMSE of T_p, 95 % interval below zero.
  The pixel spectrum is a label-free frequency spectrum of pixel brightness over the
  whole clip (cBathy-style: FFT of brightness time series at pixels sampled on the
  water, averaged; peak frequency and band energies as scalar features), computed from
  the same 4-fps frames. Its exact recipe is committed with its code before the first
  scored run. It enters as extra inputs to a ridge head beside the `clip` prediction,
  fitted within the same folds.

Same paired bootstrap over local dates (2,000 resamples) and same first-scored-run rule
(at least 15 local dates) as M1 and M2.

**Reported beside them, not bars:**
- `clip` with the mean of log H_s instead of the energy mean.
- `pooled`: a learned record pooler (one attention block over the K window embeddings
  with a learned query, EchoJEPA-style) trained per clip. It is kept only if it beats
  the `clip` average.
- `clip-image`: frozen DINOv3 on the middle frame of every window, averaged the same
  way. If it gains as much as `clip`, the gain comes from more looks at the sea, not from
  motion.
- The pixel spectrum alone (ridge head), the physics floor for T_p, beside Kamagata's
  held-out T_p RMSE of 1.42 s.
- Kim's five classes from one window, from the windows covering the 176 s centred on
  the midpoint (11 windows, Kim's 180-s span), and from the whole clip.

### Whole-clip implementation (fixed 2026-11-18, before any clip was scored)

Code: `scripts/webcoos_video_clips.py windows | pack | ship`, `scripts/train_video_probe.py
--windows`, `wave_analysis.processing.pixel_spectrum`, `wave_analysis.models.video_probe`.

**Where the windows are cut: on the desktop, shipped through R2.** Measured on
2026-11-18: cutting a 10-min clip whole takes 130-200 CPU-seconds and gives 29-39 MB of
frames (Currituck, Oak Island east, Jennette's; about 35 MB on average), so the 457 clips
paired today are about 16 GB and the whole sample (about 4,800 clips) about 170 GB. Three
routes to the GPU server:

| Route | Rate (measured) | Today's clips | Whole sample | Verdict |
|---|---|---|---|---|
| SSH through FAU's bastion (`run.sh sync`) | 1.46 Mbit/s, about 16 GB a day | about 1 day | about 11 days | too slow |
| Download the archive's MP4s on the server, cut there | server internet about 38 MB/s | — | — | not allowed now (below) |
| Desktop to the project's R2 bucket to the server (`ship`) | 36 Mbit/s up, 20 MB/s down; 22 Mbit/s end to end with a 3 MiB/s cap, about 240 GB a day | about 2 hours | under a day | **used** |

The server route was preferred because it would also put raw clips where continued V-JEPA
2 pretraining could use them, but it is not workable now. The pace agreed with WebCOOS is
one request at a time (`docs/datasets/webcoos.md`; a lock enforces it), and the desktop's
sample download runs for about two more days, so a server download now would be a second
connection. After that, it would fetch the same clips a second time: about 0.6 TB by the
measured clip sizes (Jennette's are about 290 MB), about three times the 200 GB the
drafted request to WebCOOS named. Raw clips on the server need WebCOOS's consent first. The
desktop cuts about one clip every 16 s with four workers (about 2 h for today's clips,
about 20 h for the whole sample), beside the download. R2 credentials stay on the
desktop: the server fetches each zip through a 12-hour signed link (in a mode-600 file,
never on a command line), checks its SHA-256 and lists it in `VERIFIED`; the staged copy
is then deleted.

**Windows.** Each usable clip is decoded whole (`fps=4`, `scale=-2:256`, masks first) and
its first 64 K frames kept, in one stored zip per clip; window w is frames 64 w to
64 w + 63, i.e. seconds 16 w to 16 w + 16 from the clip's start. K is floor(duration /
16 s), or fewer when the stream holds fewer frames than its stated length (counted and
reported). The midpoint window holds second duration / 2.

**Features.** Per window, V-JEPA 2 ViT-L on the 64 frames in order (`video`) and DINOv3
ViT-L on frame 32 (the window's middle frame, as for `image`), with the tiling, layers and
pooling of the arms above, cached per clip in float16 (about 2.1 MB and 0.26 MB a
window) on the server's data disk, reused by later runs.

**Probes.** For each encoder, the 25 (fold, seed) attentive probes train in lock step as
one stacked model (`StackedProbes`, `torch.func`): each epoch the cache is read once in
batches of 256 windows (order shuffled, shared), each probe takes its own training windows
from every batch in its own order (seeded by its seed), and every probe holding a full
mini-batch of 32 steps in one vmapped forward and backward; the epoch's remainder is its
last, partial mini-batch. Each probe keeps its own AdamW state and its own one-cycle
schedule (a real `OneCycleLR` over epochs x ceil(windows / 32) steps), so the maths equals
training it alone (a unit test checks this); the run stops if a probe did not use each of
its training windows exactly once per epoch. Stepping the probes one by one cost about
20 h per encoder at the interim size on the busy shared GPU (smoke test 1); lock step
is estimated at about 3 h. Matmuls run in TF32 for these probes (the 64-frame heads stay in FP32).
Targets are standardised on the training clips; each window carries its clip's label; the
validation date's windows pick the epoch (mean squared error over windows). Each window
is predicted by the five probes of its clip's test fold, averaged in log space; then the
clip is read.

**Readouts.** `window`: the midpoint window. `clip`: H_s energy mean, T_p geometric mean
over the K windows. `clip_meanlog`: geometric mean of H_s. `span11`: the 11 windows
centred on the midpoint window (shifted to stay inside the clip; all when K < 11), for
Kim's 180 s. The same two readouts of the DINOv3 probe: `window_image`, `clip_image`.

**`pooled`.** A record pooler (`RecordPooler`: per-layer projection, learned window-position
and tile embeddings, one self-attention block, a learned query) over each window's mean
V-JEPA 2 token per tile and layer (16 KB a window), trained per clip with the same folds,
inner validation, seeds and hyperparameters. It reads the frozen encoder, not the window
probe, so it carries no in-sample leak; it also sees less spatial detail than the probe.

**Pixel spectrum, recipe v1** (`wave_analysis.processing.pixel_spectrum`, computed on the
desktop from the zipped frames, stored per clip and in the package's
`pixel_spectrum.csv`):

1. Each frame to 8-bit grey (ITU-R 601), cropped to a multiple of 4 px wide, averaged over
   4 x 4 px blocks (a 64-row grid).
2. For each block's brightness series, Welch's spectrum: 4 Hz, Hann window, 512-sample
   (128-s) segments (the whole series if shorter), half overlapping, linear detrend.
3. Water, without labels: blocks ranked by the wave band's (0.04-0.33 Hz, 3-25 s) share of
   the brightness-rate spectrum (2 pi f)^2 P(f) above 0.04 Hz. Slow drift (clouds,
   exposure: red) and pixel noise (white) score low; wave motion high. Blocks with no
   power above 0.04 Hz (masks, saturation) are dropped; the top 20 % are the water.
4. S(f): the mean over water blocks of each block's band spectrum divided by its band
   power.
5. Features (10, the ridge inputs): log T_pix, the period of the largest excess of S over a
   power law fitted to it in log-log (parabola-refined); log T_m01 and log T_m02 from the
   moments of S; the share of S in 25-16, 16-12, 12-9, 9-6 and 6-3 s; log10 of the water
   blocks' median band power; their mean rate share.

The recipe was set on six clips (one or two per camera) by looking at their spectra and
water masks. The first version chose water by the band's share of the plain spectrum and
took the plain maximum: on the Currituck and Jennette views it put the peak at the band's
slow edge, under a red background with a visible wave peak above it, and it counted
cloud blocks as water. Both fixes follow from those plots, not from labels; the CDIP T_p
of three Currituck clips was printed once during the check and nothing was tuned to it. On a stormy, wet Oak Island
west view the spectrum has no peak, and T_pix is then noise.

**Ridge heads (T_p).** Target log T_p; inputs standardised on each training fold (a missing
value becomes the training mean); alpha = 1; intercept not penalised; the same five
folds. `spectrum`: the 10 spectrum features. `clip_spectrum` (bar M5): log of the `clip`
T_p plus the 10 features. `clip_ridge`: log of the `clip` T_p alone, a control showing
how much of any M5 gain is recalibration of the clip estimate. The training clips' `clip`
T_p is their own out-of-fold prediction (stacking); the probes behind those predictions
were trained with the test fold's clips, a second-order leak accepted here because
nested cross-fitting would quadruple the probe training. `pixel_raw`: T_pix itself, no
training at all, the physics floor beside Kamagata's 1.42 s.

**Scoring.** All arms on the same clips (those with windows and the 64-frame cut), M1-M5
by the paired bootstrap over local dates (2,000 resamples). The K >= 15 report scores the
same predictions on those clips only. The 64-frame arms and the whole-clip arms may run
as two containers on the same GPU (`--clip-arms-only`, `--whole-clip-only`); both select
clips the same way and record a hash of the clip list, which must agree.

## Development and confirmation dates (fixed 2026-11-19, before any v2 development)

The 18 local dates of the interim run are development data from now on:
`configs/experiments/webcoos_video_dev_dates.txt` (SHA-256 `64685e3602fac031...`,
2024-04-08 to 2026-09-24, the 992 interim clips). Their labels may be looked at to choose
between variants of later features, and every such look is recorded here. Every local
date outside this list that the final run adds is a **confirmation date**: its labels
and any score of a new feature on it stay unopened until that feature's recipe and bars
are committed. M1-M5 and spectrum recipe v1 are unchanged and are scored on all final
dates as planned.

## Progress

- **Smoke test (2026-11-18, `video-days-20261118t1655-smoke-smoke`):** 24 clips from 4
  dates, 3 epochs; the whole path runs on the H200 (features, eight heads, scores).
  Feature extraction for all four arms takes about 1.5 s per clip, so the full sample
  is about 1.5 h of GPU time. Its numbers mean nothing (four dates, each a single sea).
  Two fixes it found: smoke samples now span dates, and the containers run with
  `TORCH_DISABLE_NATIVE_JIT=1` (PyTorch 2.14 sent the one-query attention's backward to
  a Triton kernel that needs a C compiler).
- **Whole-clip smoke tests (2026-11-18, `video-days-20261118t2321-wholeclip-smoke`,
  `...t2344-wholeclip2-smoke`):** 24 clips, 697 windows, 3 epochs; the whole path runs
  (windows, features, both window probes, readouts, pooler, ridge heads, scores). Window
  features take about 0.23 s a window on the busy GPU (about 2.3 h for 36,000 windows).
  The first test stepped the 25 probes one by one (about 10 s an epoch for 700 windows),
  hence the lock-step training above. Numbers mean nothing (two dates).
- **Data:** the archive serves a few clips as empty files (HTTP 200, 0 bytes); they are
  excluded as unreadable. Jennette's clips are about 290 MB each (2592 x 1944), so the
  sample takes about two days at the archive's pace; the first scored run follows the
  first two sample weeks.
- **Scored interim run (2026-11-19, commit e0bdd4c5c):** started once 17 local dates had
  at least 10 framed clips each. `video-days-20261119t0628-m12` (the 64-frame arms, 1 h)
  and `video-days-20261119t0628-wholeclip` (window features 1.8 h, video window probes
  3.2 h, image window probes 0.8 h, about 34 GB of GPU memory) ran side by side on one
  shared H200, on the same 992 clips (the clip-list hashes agree). The download goes on;
  the final run uses the same code and config.

## Results

### Interim run, 2026-11-19: 992 clips, 18 local dates

422 clips from Currituck, 234 from Jennette's, 208 from Oak Island west and 128 from
Oak Island east; 33,115 windows; 914 clips have K ≥ 15. Attentive probes throughout.

How to read the numbers:

- **RMSE:** the typical size of the error, in m (H_s) or s (T_p); lower is better.
- **Difference:** the first arm's RMSE minus the second's, on the same clips; below zero
  means the first is better. The interval is the 95 % range when whole dates are
  resampled (2,000 times); a bar is met when the whole interval is below zero.
- **Climatology** (a reference added for reading the results, not a design arm): every
  clip given the average label of the other folds' dates. It scores 0.740 m for H_s and
  2.72 s for T_p; an estimate that does not beat it has not learned that quantity.

| Bar | Compared (RMSE) | Difference [95 %] | Met |
|---|---|---|---|
| M1 | H_s: `video` 0.533 m vs `still` 0.535 m | −0.002 m [−0.079, +0.075] | no |
| M2 | T_p: `video` 2.86 s vs `still` 2.80 s | +0.06 s [−0.03, +0.20] | no |
| M3 | H_s: `clip` 0.513 m vs `window` 0.524 m | −0.011 m [−0.018, −0.004] | yes |
| M4 | T_p: `clip` 2.86 s vs `window` 2.88 s | −0.022 s [−0.036, −0.006] | yes |
| M5 | T_p: `clip` + spectrum 2.47 s vs `clip` 2.86 s | −0.39 s [−0.74, +0.02] | no |

What it means:

- **Motion inside 16 s adds nothing for H_s.** The same encoder given the 64 frames in
  order, one frame repeated, or the frames shuffled gives the same H_s error
  (0.533-0.535 m); DINOv3 on one frame gives 0.546 m. All cut the error by 26-28 % from
  climatology.
- **Reading the whole clip helps H_s, slightly.** 0.524 to 0.513 m (2 %; R² 0.47 to
  0.49). Averaging DINOv3 over the same windows gains nothing (0.527 to 0.527 m), so the
  gain is not simply more looks; but `clip` and `clip-image` cannot be told apart
  (+0.014 m [−0.009, +0.044]). The mean of log H_s does as well as the energy mean
  (+0.001 m), and the 11 central windows give most of the gain (−0.008 m).
- **No attentive probe learned T_p, so M4 is met but worth nothing.** None beats
  climatology (2.73-2.94 s against 2.72 s; of the mean-token heads only `shuffled` does,
  2.58 s). DINOv3 averaged the same way gains the
  same 0.02 s, so M4's gain comes from more looks, not motion. The probes do learn
  something at Currituck (`clip` 2.04 s against 2.69 s) but are far off at Oak Island
  (east 4.33 s against 2.98 s).
- **The pixel spectrum is the only T_p estimate that beats climatology.** Alone (ridge on
  its 10 features): 2.46 s, R² 0.13, Pearson 0.40. M5's interval just crosses zero on all
  clips and lies below zero on the clips with K ≥ 15 (−0.45 s [−0.81, −0.03]). About half
  of M5's gain is recalibration of a poor `clip` estimate: `clip` refitted alone gives
  2.67 s (−0.19 s [−0.42, +0.07]). With no training at all, the pixel period T_pix
  follows the buoy's T_p (Pearson 0.43; 0.53 at Currituck, 0.58 at Oak Island east) but
  reads 0.8 s long on average (RMSE 3.19 s).
- **The learned record pooler is worse than the plain average** (H_s 0.584 m,
  +0.071 m [+0.001, +0.131] against `clip`), so it is dropped, as the design says.
- **Kim's five classes: no skill at unseen dates, whatever the span.** One window
  58.4 %, the 11 central windows 58.0 %, the whole clip 58.4 %; always naming the
  average class gives 58.8 % (154 of the 262 clips in Kim's range are in the lowest
  class). Kim's 78 to 93 % came from a random split.
- **Kamagata's measures on Oak Island (336 clips):** H_s RMSE 0.661 m, R² 0.28, Pearson
  0.55 for `clip` (0.62-0.68 m for the 64-frame attentive arms; Kamagata's held-out 0.586 m, 0.107,
  0.436; climatology here 0.805 m). T_p 2.79 s for `clip` + spectrum (Kamagata 1.42 s;
  climatology here 2.97 s). The comparison is loose: their test is five videos with
  labels that cannot be traced to an instrument, ours 336 clips of unseen dates against
  CDIP 200.
- **Clips with K ≥ 15 (914):** the same picture; M3 −0.011 m [−0.019, −0.004], M4
  −0.020 s [−0.032, −0.006].

Full tables, per camera and per arm: each run's `report.md` and `metrics.json`.
