# Domain pretraining: DINOv3 ViT-B continued on buoy-camera views (rank 6)

**Question.** Does continuing DINOv3 ViT-B/16's self-supervised training on our own
unlabelled buoy-camera views give better frozen features for wave height at buoys
it has never seen than the off-the-shelf weights do? This is rank 6 of
`reports/Wave camera architectures beyond backbones.md` ("Domain pretraining is
affordable but has slipped to sixth"; evidence in
`research_notes/Wave camera architectures beyond backbones/domain_pretraining_and_compute.md`).
One clean frozen-vs-continued pair, not a programme. Fixed 2026-11-18, before any
pretraining. Runs approved by Steven on 2026-11-18 ("proceed with all of your
recommendations").

Why it is worth one test: B3 found frozen DINOv3 ViT-B features 0.111 m behind arm A
in H_s RMSE (`ndbc_buoycam_sea_state.md`, Bar B3), so the frozen-feature gap is real.
In FINO, continued pretraining raised frozen-probe scores most where images differ
from web photos (microscopy, chest X-ray) and little where web photos already cover
the domain (camera traps). Buoy views are low-resolution, tilted and often hazy, so
they sit somewhere in between. No study covers sea-surface imagery.

## Data: unlabelled views of the 55 training buoys

- **Source:** the images of `ndbc_buoycam_sea_state_v0` (41,091 strips, caption cut;
  byte-identical to the v2 heights package used for B1-B3). No label, height, wind or
  ERA5 value is read.
- **Buoys:** the 55 **training** buoys of the station split only: 28,597 strips.
  The 12 test buoys are excluded, so the test stays sealed. **The 12 validation
  buoys are excluded too.** The plan expected train and validation buoys together
  (about 200k views); I keep validation out because it makes two choices below (the
  pretraining checkpoint and P1's epoch), and those choices are only fair if the
  validation buoys are as unseen as the test buoys, for both arms. What remains
  still clears the roughly 100k-image level where continued pretraining started to
  pay off in the chest X-ray scaling study.
- **Views:** every usable view (the package rule: mean grey ≥ 20 and standard
  deviation ≥ 4) of those strips, one image each, at the stored size (270 × 480;
  216 × 384 at 44008): **165,021 views**. Measured from the package on 2026-11-18.
- **Caveat for later:** the buoy-strip session is building grouped cross-validation,
  in which training buoys become test buoys in turn. Weights pretrained on these 55
  buoys are **not** clean under those folds; a fold-wise re-check would need
  pretraining that leaves each fold's buoys out.

## Recipe: FINO, not ExPLoRA

Read both (FINO: arXiv:2606.05107 and the `FINO` branch of
`facebookresearch/dinov3`, commit `81aae9a`; ExPLoRA: arXiv:2406.10973).

**Chosen: FINO**, the domain-adaptation recipe of DINOv3's own authors, run with
their released code. Reasons, most important first:

1. It was built and tested on DINOv3 itself, with DINOv3's objective (DINO + iBOT,
   SIGReg in place of KoLeo), and reports **frozen-probe** gains, including on
   out-of-distribution splits. P1 is exactly a frozen probe at unseen buoys.
2. ExPLoRA's evidence is on DINOv2 and MAE. Running it on DINOv3 means writing LoRA
   into DINOv3's trainer: an unpublished combination, more code of our own, more
   ways to be wrong in a one-shot test.
3. Forgetting, the main risk in the evidence (the chest X-ray model forgot an external
   hospital after long training), is handled by FINO's frozen first stage, its low
   learning rate and keeping the best validation checkpoint (below).
4. Cost does not decide it: a full-unfreeze ViT-B fits one GPU with room to spare,
   so LoRA's savings are not needed.

**Configuration:** `configs/pretraining/dinov3_vitb16_buoycam_fino.yaml`, FINO's FMoW
reference recipe (`vitl16_fmow_guided.yaml`) with every change marked `# ours:`:

- **Start:** the public DINOv3 ViT-B/16 weights (LVD-1689M) as timm distributes them
  (`timm/vit_base_patch16_dinov3.lvd1689m`, the B3 weights). Converted to DINOv3's own
  format by `scripts/dinov3_weights.py`; the converted model gives **the same patch
  features as timm's to the last bit** (checked on CPU, 2026-11-18). timm's ViT-B
  has no query/key/value bias (its maintainers found the released ones all zero), so
  the run keeps none: the continued weights load into exactly the B3 model. The public weights
  carry no iBOT mask token, so that token is trained from zero rather than frozen.
- **Objective:** DINO + iBOT + SIGReg (FINO's reference), DINO and iBOT heads new
  (65,536 prototypes each), teacher an exponential moving average of the student.
- **Metadata guide (label-free):** one guide, **station, adversarial** (gradient
  reversal; prototypical head; FINO's weight 0.03 and settings of its "year" guide),
  so the features do not learn to tell the 55 buoys apart. FINO suppresses the
  factor that should not matter and separates training from use: year for FMoW,
  time of day for camera traps. For buoys that factor is the buoy itself (its
  camera, mount and hull), and P1 is a test at other buoys. Not used: **time of
  day** (the sun's glitter on the sea measures wave slope, so suppressing
  illumination could suppress a wave signal), **view index** (no reason to expect
  it to matter for height), **season** (the package spans 7 weeks of autumn, one
  season). No auxiliary guide: every allowed field is a nuisance, and pulling
  features together by a nuisance has no expected benefit. No wave value of any
  kind enters the run.
- **Crops and augmentation:** FINO's reference unchanged (2 global crops 224 px,
  scale 0.4-1; 8 local crops 96 px, scale 0.05-0.4; flips, colour jitter, blur,
  solarisation as in DINOv3). Changing them would add choices this test cannot
  check.
- **Schedule:** two stages as FINO: backbone frozen (except the patch embedding) for
  the first 5,000 steps while the new heads learn, then unfrozen with a 1,000-step
  ramp; peak learning rate 5e-5 scaled by DINOv3's square-root rule to the batch;
  teacher momentum 0.994 → 1. **Batch 256 views per step if it fits in 70 GB on the
  shared GPU, else 128; 7.68 million view-crops in all** (30,000 steps at 256 or
  60,000 at 128): about 47 passes over the 165,021 views. FINO's authors saw long
  adaptation erode pretrained features and capped one domain at about 12 epochs;
  they searched up to 50. The checkpoint choice below covers that whole range.
- **Teacher checkpoints** every 2,500 steps (every 5,000 at batch 128): 12 in all.

## Budget and timing rule

One GPU of the lab GPU server, at most about **150 GPU-hours** for the pretraining
(hours the GPU is held, busy or not). A smoke run of about 15 minutes measures views
per second; the full run is projected from it. If the projection exceeds 120 hours,
the number of steps is cut to fit (the first stage stays 5,000 steps), and the change
is written here before the full run starts. The run is stopped and reported if the
loss diverges (not a number, or rising for 2,000 steps after the backbone unfreezes)
or the features collapse (the spread of the probe's features across views falling
towards zero at a checkpoint).

### Smoke runs and timing (2026-11-18, before the full run)

Two short runs on GPU 3 of the lab server (shared: another user's job kept it busy
throughout), 700 steps with the stages shortened (frozen to step 200, two checkpoints):

| Batch 256 | Peak memory | Seconds per step | Views per second |
|---|---|---|---|
| activation checkpointing off (as first configured) | 84 GB | 0.84 | 305 |
| **activation checkpointing on (FINO's own setting)** | **25 GB** | **1.44** | **178** |

- Both stages, the unfreezing and its ramp, and the teacher checkpoints work; the data
  loader keeps up (under 1 ms of waiting per step). The run starts from the public
  weights: after 150 unfrozen steps no backbone tensor had moved more than 0.7 %.
- **One change, by the rule above:** batch 256 without activation checkpointing needs
  84 GB, above the 70 GB limit. Rather than fall back to batch 128, the full run turns
  activation checkpointing back on, as FINO's reference has it (my config had turned it
  off for speed). The batch stays 256, the recipe is FINO's in one more respect, and
  only the speed changes.
- **Projection:** 30,000 steps × 1.44 s ≈ **12 GPU-hours**, far under the 120-hour
  limit, so no steps are cut. At the end of the smoke runs the DINO loss was still near
  its starting value (11.06 against log 65,536 ≈ 11.09): the new heads had only begun
  to learn, as expected after a few hundred steps.

### Full run: checks while it trains

Run `pretrain-vitb16-fino-20261118t2317`, GPU 3, started 2026-11-18 23:18 UTC with the
config above unchanged.

- **Divergence check at step 8,000** (2,000 steps after the 1,000-step ramp that
  follows the unfreezing at step 5,000): no step not a number, and every loss still
  falling. Medians over the logging window as the log prints them, step 5,000 → 8,000: total 15.36 → 13.23,
  DINO global 10.37 → 8.76, DINO local 10.38 → 8.97, iBOT 4.98 → 4.28, SIGReg 0.035 →
  0.031. Before the unfreezing the heads alone had taken the DINO loss from 11.09
  (log 65,536, its starting value) to 10.38. The station guide's accuracy (its head
  learns the buoy while gradient reversal pushes the backbone to hide it) stayed under
  1 % while the backbone was frozen and rose to 4 % by step 8,000. Chance among 55
  buoys is 1.8 %, so the features carry a little more buoy identity than before. This
  is watched, not a stop rule.
- **Early probe of the step-7,500 teacher** (an early signal only; the checkpoint is
  chosen when the run ends). The rule below, 11 validation buoys:

  | Weights | H_s RMSE | Wind-wave RMSE | H_s, all 12 buoys | Feature spread |
  |---|---|---|---|---|
  | Public (off the shelf) | 0.642 m | 0.371 m | 1.412 m | 0.091 |
  | Step 7,500 | 0.582 m | 0.344 m | 0.722 m | 0.150 |

  The feature spread rose rather than fell, so there is no collapse. The probe shared
  the GPU with the training run and took 41 minutes, against 12 alone.
- **Killed at step 10,120 and resumed (2026-11-19, written before the restart).** At
  02:28 UTC the training process got SIGKILL: the container reached its 160 GB memory
  limit while the server still had over 1 TB free. Nothing in the run diverged; its
  last full checkpoint is step 10,000 (saved 2 minutes earlier). It is restarted under
  the same name and resumes there: weights, optimiser, step and schedules (the backbone
  stays unfrozen; the freeze is set by the step number). Two changes, both outside the
  recipe: the container gets 256 GB, and glibc's malloc arenas are capped at two
  (`MALLOC_ARENA_MAX=2`), a usual cause of data-loader workers that grow. Its memory is
  logged every 5 minutes. The resumed loader shuffles with a new seed (DINOv3 seeds it
  with the start step), so the order of views after step 10,000 differs from an
  unbroken run's. The recipe and the data are unchanged.

## Choosing the checkpoint (validation buoys only)

`scripts/dinov3_probe.py`, for the off-the-shelf weights and each teacher checkpoint:
frozen features of every usable view of the training and validation strips (the B3
input: 270 × 480 padded to 272, ImageNet normalisation, the patch tokens averaged;
a strip is the mean of its usable views), a ridge regression of log H_s and log
wind-wave height fitted on the 55 training buoys (alpha by 5-fold cross-validation
grouped by buoy), RMSE in metres at the 12 validation buoys. **The checkpoint with
the lowest validation H_s RMSE goes to P1**, whatever it scores against the
off-the-shelf weights. The test buoys are never read by the probe. These ridge
numbers are the early signal, reported beside P1, not a bar.

**Change (2026-11-18, after probing only the public weights, before any continued
checkpoint was probed):** the score leaves out validation buoy **44008**. Its views
are 216 × 384 (scaled up to 270 × 480), a format no training buoy has, and the linear
probe extrapolates without bound there: with the public weights it predicts 15-23 m
for seas of 4-5 m at 44008 (bias +3.7 m), which alone lifts the validation RMSE to
1.41 m, while the other 11 buoys score 0.33-0.97 m each. A checkpoint chosen on that
number would be chosen on one buoy's format. **The checkpoint with the lowest H_s RMSE
at the 11 other validation buoys goes to P1**; the RMSE at all 12, at 44008 alone and
of the log heights are reported beside it. P1 itself is unchanged (its heads are
trained, and it is scored at the test buoys, none of which has that format).

## Bar P1: does the continued ViT-B give better frozen features? (fixed before pretraining)

- **Continued arm:** B3's frozen recipe unchanged (`scripts/train_wave_heights.py`,
  `configs/experiments/ndbc_buoycam_heights_v2.yaml`, `--split station --backbone
  timm:vit_base_patch16_dinov3.lvd1689m --lp-epochs 10`, images only, 10 epochs, the
  epoch chosen on validation RMSE of H_s), with the chosen checkpoint loaded by the
  new `--backbone-weights`; seeds 0, 1, 2, averaged by mean log height.
- **Off-the-shelf arm:** B3's three frozen runs (seeds 0, 1, 2, commit `bb1a568`). The
  training code (`train_wave_heights.py`, `models/wave_heights.py`) is unchanged from
  `bb1a568` to this experiment apart from `--backbone-weights`, which does nothing
  when not given, and the test strips are the same 6,183, so the runs pair strip by
  strip and are not rerun.
- **P1:** the continued arm must lower the test RMSE of **H_s and of the wind-wave
  height** against the off-the-shelf arm, **each 95 % interval of the difference
  below zero** (paired bootstrap over the 12 test buoys, 1,000 draws,
  `scripts/compare_papers.py --arm vitb-frozen <B3> --arm continued-frozen <new>`).
  For scale: off-the-shelf frozen is 0.560 m (H_s) and 0.257 m (wind-wave), arm A
  0.450 and 0.219 m, the best model 0.440 m.
- **Seed spread** (for the follow-up rule): the standard deviation of the three
  single-seed test H_s RMSEs, the larger of the two arms'. B3 frozen: 0.542, 0.558,
  0.606 m, so at least 0.033 m.

**Reported, not a bar:** the same comparison with B3's gentle recipe
(`--lp-epochs 2 --backbone-lr-scale 0.1`, three seeds) against B3's gentle runs
(0.455 m), and both continued arms against arm A and every paper's measure.

**Follow-up, only if P1 is met with an H_s gain larger than the seed spread:**
distil the continued ViT-B into ConvNeXt-Tiny with DINOv3's public distillation code
and re-check under the grouped cross-validation (with pretraining that leaves each
fold's test buoys out). That step gets its own bar, fixed when it is reached.

## Code

- `src/wave_analysis/pretraining/buoy_views.py`: the unlabelled views as a dataset
  for DINOv3's trainer (`BuoyViews`), returning each view with its station index as
  the only metadata.
- `scripts/dinov3_pretrain.py`: registers `BuoyViews` with DINOv3's data loader and
  calls DINOv3's `train.py`; DINOv3's checkout stays unmodified.
- `scripts/dinov3_weights.py`: timm ⇄ DINOv3 weight conversion with the equivalence
  check.
- `scripts/dinov3_probe.py`: the validation ridge probe above.
- `deploy/dgx/pretrain.sh` and `deploy/container/pretrain.sh`: DINOv3's launcher
  assumes Slurm, which the server lacks; these run it with `torchrun` on one GPU in a
  Docker container, beside `run.sh`, with the DINOv3 checkout pinned to `81aae9a`.
- `scripts/train_wave_heights.py --backbone-weights`: loads a timm state dict into
  the backbone after it is built (strict).

DINOv3 is under Meta's DINOv3 License (research use; publications must acknowledge
it). The continued weights inherit it.

## Status

- [x] Pre-registration (this document and the config), 2026-11-18, before any
      pretraining.
- [x] Smoke runs and timing (2026-11-18): about 12 GPU-hours projected; activation
      checkpointing on, batch 256.
- [ ] Full pretraining run.
- [ ] Checkpoint choice on the validation buoys.
- [ ] P1 and the gentle comparison.
