# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/). Dataset versions are tracked
separately (`docs/REPRODUCIBILITY.md`).

## [Unreleased] — v0.1.0 "Dataset discovery and NDBC foundation"

### Added (2026-11-20, test 6: label-free site adaptation)
- **Test 6 not met** (bar in `docs/experiments/sensing_site_adaptation.md`, fixed before
  any run). Each camera's cached DINOv3-gentle features were standardised against the
  training cameras at the same ERA5 total-height quintile, and the heads refitted.
  - Total height: 0.532 → 0.500 m out of fold, but −0.031 m [−0.069, +0.019] over station
    groups, not beating the head-matched control.
  - Wind-wave height: worse, 0.260 → 0.377 m.
  - The total-height gain is the US West Coast fold's (0.699 → 0.586 m).
  - Nine of the eleven under-covered buoys rise above 0.80 coverage, but five others fall
    below it.
- **Few-shot curve, new for a wave camera.** A shrunk per-buoy log bias from k labelled
  hours in a buoy's first 14 days lowers total-height RMSE by 0.022 m at k = 1 and 0.060 m
  at k = 30 (0.567 → 0.507 m on later strips), both intervals below zero. Wind-wave gains
  only 0.011 m at k = 30.
- **`scripts/sensing_site_adaptation.py`.** CPU only, on the measurement probe's cached
  features (`--save-features`); no new GPU work.

### Added (2026-11-20, breaker type: does a video encoder use the motion?)
- Bar Y-B3, fixed before any run (`docs/experiments/breaker_types.md`): frozen V-JEPA 2
  ViT-L on each VWBT_9000 clip in order, against the same frames shuffled. Each clip is read
  as three 64-frame windows over its 30 s at 5 fps, and the tokens are averaged over space
  (75 per clip). The head and folds are Y-B1's, with seeds 0-2. Still-frame and order-free
  arms are reported beside, along with Y-B1's DINOv3 and CRNN predictions on the same
  bootstrap draws. `scripts/breaker_features.py --encoders vjepa2` decodes each clip once
  for all three versions. `scripts/train_breaker_heads.py` gains five `vjepa2_*` arms and
  `score --with` (another run's predictions scored beside), and `deploy/dgx/breakers.sh`
  gains `WITH=<run>`.
- **Bar Y-B3 met:** in order minus shuffled is +0.038 [+0.024, +0.052] pooled macro-F1
  (0.833 against 0.795) at the held-out beach, at every seed. In order also beats each
  window's middle frame repeated (+0.081 [+0.066, +0.098]), and the gap holds with heads
  that cannot read token order (+0.035). So the encoder uses the breaking motion.
- But motion only brings V-JEPA 2 level with DINOv3 on single frames: −0.007
  [−0.026, +0.012] against DINOv3's temporal head, and 0.848 against 0.850 for the best arm
  of each.

### Added (2026-11-20, test 1 with nested calibration)
- **Test 1 met** (camera added to a GFS-Wave + ERA5 forecast with the buoy's measured
  wind; bar and amendment in `docs/experiments/sensing_model_bar.md`). The gate was refitted
  per outer fold on inner predictions only. On 34,723 strips at 71 buoys in 41 groups:
  - wind-wave height RMSE 0.2258 → 0.2079 m (−0.0179 [−0.0253, −0.0088], station groups);
  - total height RMSE 0.2670 → 0.2623 m (−0.0047 [−0.0083, −0.0007]).
- **Supporting results.**
  - Permuting the camera within buoy and forecast-height quartile makes the fusion worse
    than the forecast alone, so the gain comes from the individual images.
  - The secondary variance-weighted arm does better still: 0.2578 m total and 0.2020 m
    wind-wave.
  - With the anemometer withheld, the swap's total-height gain does not clear zero; its
    wind-wave gain does.
- **The historical diagnostic is superseded** (0.2635 / 0.2095 m; its calibration
  predictions came from models that trained or validated on the outer test groups, per the
  inventory in the bar doc).
- **S2 check.** The same nested calibration applied to bar S2's stage-2 fusion gives
  0.2667 against 0.2883 m total and 0.2407 against 0.2928 m wind-wave. S2 and its
  2026-11-27 confirmation are unchanged.
- **New scripts.**
  - `scripts/sensing_nested_splits.py`: leave-two-folds-out pair splits, package
    `ndbc_buoycam_heights_v2cvn`. They drove 30 inner DINOv3-gentle runs
    (10 pairs × seeds 0–2, 21.1 GPU-hours on the shared server).
  - `scripts/sensing_nested_inventory.py`: which groups each cached model saw.
  - `scripts/sensing_nested_fusion.py`: the nested scorer and the S2 check; fed the outer
    predictions, it reproduces the historical numbers exactly.
- **Changed scripts and config.**
  - `scripts/sensing_forecast_corrector.py --splits`.
  - Pair-split bootstrap keys in `configs/experiments/ndbc_buoycam_heights_v2cv.yaml`.
  - Results in `docs/experiments/sensing_forecast_results.md`.

### Added (2026-11-19, breaker type at unseen beaches)
- Breaker-type classification on Yin et al.'s VWBT_9000 (9,000 clips, six beaches) on the
  official leave-one-site-out split, with bars Y-B1 and Y-B2 fixed before any run
  (`docs/experiments/breaker_types.md`). It replaces the rejected breaker trainer: each clip
  is decoded once into frozen ResNet-18 and DINOv3 features at the clips' full 5 fps
  (`scripts/breaker_features.py`), and the heads train on that cache
  (`scripts/train_breaker_heads.py`, `deploy/dgx/breakers.sh`). Yin et al. publish only a
  random 5-fold accuracy, so the comparator is their own CRNN re-run (94.0 % under their
  protocol, against their 96.1 %).
- **Bar Y-B1 met:** frozen DINOv3 frames reach pooled macro-F1 0.840 at the held-out beach,
  against 0.653 for Yin et al.'s CRNN, +0.187 [+0.162, +0.214], at every site and seed.
- **Bar Y-B2 not met:** the temporal head is no better than the same features with frame
  order removed (−0.010 [−0.018, −0.002]); 5 fps does not help the CRNN either. Offshore
  ERA5 waves and beach slope are at chance (0.34).

### Added (2026-11-19, wave measurements beyond height)
- Review of Codex's prepared measurement experiments and bars M-S1 and M-S2, fixed before
  their runs, in `docs/experiments/wave_measurements.md`. Fixed before any run: the slope
  probe's comparator gains the de-mounted horizon roll and pitch, every ridge fits on the
  fold's training buoys by name (plus a validation-fit sensitivity), the checkpoint's test
  strips must equal the fold's, and 50 all-zero spectra get no slope. The breaker trainer
  is not run (re-decodes every clip every epoch for frozen encoders; one frame per 1.9 s).
- `deploy/dgx/measure.sh`: frozen buoy probes on the GPU server in the `run.sh` pattern.
- **Bar M-S1 met** (five folds over the 43 station groups, 38,387 strips at 77 buoys):
  the buoy's RMS slope is predicted with RMSE 0.0089 from camera + wind + horizon
  geometry against 0.0102 from wind + geometry, −0.0013 [−0.0017, −0.0008]; the camera
  alone (0.0093) also beats wind + geometry. With only the water below each horizon the
  gain halves but holds (−0.0007 [−0.0010, −0.0005]).
- **Bar M-S2 not met:** the ten-epoch slope fine-tune is no better than the frozen probe
  (0.0086 against 0.0084, +0.0002 [+0.0001, +0.0004]). Every fold kept a heads-only
  epoch: four of five runs diverged once the backbone unfroze, so end-to-end slope
  training is still untested.
- **Bar M-S3 not met:** with a stable recipe (`buoy_slope_v2lr`: the backbone's learning
  rate re-warms at unfreezing, gradient clipping, lr 1e-4; `fit_heights(rewarm_epochs=,
  grad_clip=)`, off by default) no fold diverges, but the fine-tune is level with the
  frozen probe (0.0084 each, +0.00004 [−0.00012, +0.00017]).
- **Test 5 (bar T5):** ERA5 10-m wind and gust at the buoys (`scripts/buoycam_era5_wind.py`)
  and the 46 bins' energies (`measurement_labels.py`). **T5-W met:** one camera strip gives
  the buoy's wind speed with RMSE 0.96 m/s against 1.27 m/s for ERA5 (−0.31 [−0.44,
  −0.10]). The camera adds to wind at every spectral bin from 0.0525 Hz (19 s) up, so its
  skill is not confined above the sea-swell separation.

### Added (2026-11-19, the papers' own networks on the sealed buoys)
- Test 3, strip half (`docs/experiments/sensing_paper_bars.md`, three seeds each, 12 sealed
  buoys, retrospective): DINOv3-gentle beats every paper network trained on our strips. Choi
  (VGG16, calm or rough) 99.2 against 94.7 % (+4.6 points [+2.1, +8.6]); Kim (snapshot CNN,
  five classes) 70.2 against 50.1 % (+20.1 [+12.8, +26.2]), and against 46.5 % with Kim's own
  training options; Yang (ResNet-18 + attitude transformer, H_s) 0.418 against 0.514 m
  (−0.096 m [−0.122, −0.063]). Yang's horizon attitude makes it worse (+0.020 m [+0.007, +0.031]).
- Choi's controls: the DINOv3 backbone inside Choi's own pipeline (gentle schedule) is no
  better than VGG16 (94.1 against 94.7 %, −0.6 points [−1.2, −0.2]), and the six-view project
  model beats it by +5.2 points [+2.4, +9.6]: the gain is the project's input and design, not
  the backbone. One view alone gives VGG16 89-92 %. With the shared recipe the backbone
  collapsed to calling every strip calm; the gentle rerun was registered before it ran.
- The same arms over the five grouped folds (all 79 buoys out of fold, 75 runs, station-group
  intervals) agree: Choi +2.9 points [+1.6, +4.0], Kim +13.9 [+10.8, +18.4] (+18.7 with Kim's
  own options), Yang −0.066 m [−0.085, −0.046]. Attitude: −0.022 m [−0.041, +0.004] there,
  so across both splits it has no consistent effect on Yang.
- `Kim2023SnapshotCNN` now follows the paper's Fig. 5 (4 × 4 pooling after blocks 1-3, none
  after block 4, flattened map into one dense layer); the earlier class pooled 2 × 2 after every
  block and averaged globally. Fixed before any Kim run.
- `scripts/train_sensing_papers.py`: per-view test probabilities, arm `choi-dinov3` (the
  project's backbone in Choi's pipeline), Adam and a gentle schedule as recipe options;
  `scripts/predict_sensing_papers.py` (per-view probabilities from a finished run's chosen
  epoch); `scripts/score_sensing_papers.py`: registered arm pairs (`--control`), each view
  position (`--views`), and the five grouped folds (15 runs per arm, station-group intervals).

### Added (2026-11-18/19, domain pretraining)
- Rank 6 of the architecture review, domain pretraining
  (`docs/experiments/domain_pretraining.md`, fixed before any pretraining): DINOv3
  ViT-B/16 continued with FINO's recipe for 30,000 steps on unlabelled buoy-camera views
  of the 55 training buoys. The sealed test buoys were left out, and the only metadata
  was the station. Step 7,500 was chosen by a ridge probe at the validation buoys.
- Bar P1 not met. With B3's frozen recipe, at the 12 test buoys, H_s moved by
  −0.010 m [−0.027, +0.002] and wind-wave height by −0.012 m [−0.018, −0.006], against
  the public weights. The distillation follow-up is not triggered. With the gentle
  recipe the continued weights are worse (H_s +0.028 m [+0.012, +0.040]). About 19
  GPU-hours used.
- Code: `src/wave_analysis/pretraining/buoy_views.py` (the views as a dataset for
  DINOv3's trainer); `scripts/dinov3_pretrain.py`, `scripts/dinov3_weights.py` (timm ⇄
  DINOv3 conversion) and `scripts/dinov3_probe.py`; `deploy/dgx/pretrain.sh` and
  `deploy/container/pretrain.sh` (single-GPU `torchrun` in Docker, DINOv3 pinned to
  `81aae9a`, memory limit `MEM`); `scripts/train_wave_heights.py --backbone-weights`.
  The continued weights fall under Meta's DINOv3 License (research use, acknowledgement
  required).

### Added (2026-11-19, shore video: sealed storm set)
- `configs/experiments/webcoos_video_storm_v1.yaml`: the storm-date estimate run for the shore-labels storm bars
  (T2-A, T2-B, T3-K). It is webcoos_video_v1's whole-clip recipe in five folds of storm dates, with no physics-v2
  block. Its M3-M5 lines are estimates there, not bars. It starts on its own once the storm package is on the server.
- Storm storage: `WEBCOOS_VIDEO_SET=storm` gives `scripts/webcoos_video_clips.py` the storm set's own raw root,
  output folder and package (`webcoos_video_storm_v1`). The new `offload` step moves cut storm clips to the GPU
  server, checked against their ledger SHA-256, and frees the local copies. `ship` and `offload` share one R2 helper.
  The storm driver now cuts and offloads each date before fetching the next, so the desktop keeps 40 GB free; its
  budget counts fetched bytes from the ledgers, and dates already done are skipped on a rerun.
- `scripts/webcoos_storm_dates.py`: storm dates chosen from CDIP records alone. A storm date
  has a daylight mean H_s of 2.0 m or more at the paired buoy. Development and sample dates
  are excluded, and only one date is kept per storm and site. The dates are ordered over
  site × H_s band (2.0-2.8 m, ≥ 2.8 m) × season.
  `configs/experiments/webcoos_video_storm_dates.csv` lists 72 dates (17 with H_s ≥ 2.8 m).
- `scripts/webcoos_storm_sample.sh` downloads them in that order: hourly clips, until 24
  dates have video or 120 GB is reached (about 78 GB expected). It runs after the video
  sample, under the same lock and pace, into its own root `data/raw/webcoos_video_storm`.
  The sample's pairing never reads that root, so the set stays out of the final run.
- The storm set is sealed: no per-clip label, feature or score of a storm date is opened
  until the bars that use it are committed (`docs/experiments/webcoos_video.md`, "Storm set").

### Results (2026-11-19, test 9: time-exposure and brightest images)
- T9-A and T9-B not met: on the held-out year 2025 one still estimates H_s better than the
  ten-minute time-exposure or brightest-pixel image of the same minutes (0.265 m against
  0.308 and 0.311 m; +0.043 m [+0.026, +0.061] and +0.045 m [+0.035, +0.056]), and the three
  together are no better than the still (0.281 m). The same in 2026 and at every camera.
  Write-up: `docs/experiments/sensing_timex_bars.md`.

### Results (2026-11-19, test 3 video half: Kamagata's network)
- T3-S: the project's whole-clip read-out (H_s RMSE 0.513 m) is ahead of Kamagata's
  SlowFast head with its physics loss (1.140 m) on the same 992 clips and date folds,
  -0.627 m [-0.924, -0.364]. The physics term (T_p^2 = 6 H_s) collapses the head to about
  2.0 m for every clip; without it the head reaches 0.653 m and the project is still ahead,
  -0.140 m [-0.207, -0.061]. Write-up: `docs/experiments/sensing_per_wave_bars.md`.
- Kim's network: smoke test only (GoogLeNet loading fixed); the full run waits for the
  sealed storm set, where the class endpoints are now decided (bars fixed before any storm
  label was opened).

### Added (2026-11-19, test 9: time-exposure and brightest images)
- Bar T9 fixed before any run (`docs/experiments/sensing_timex_bars.md`): a still, a
  `timex` and a `brt` image of the same ten minutes at the four CDIP-paired cameras,
  frozen DINOv3 probes, trained on 2024 and scored on 2025 (97.5 % day-bootstrap
  intervals for each input against the still). Package of 30,931 triples
  (`scripts/sensing_timex_package.py`; 5 truncated source images excluded and listed),
  probes `scripts/sensing_timex_probe.py`, config `configs/experiments/sensing_timex_v1.yaml`.

### Added (2026-11-19, shore per-wave labels: tests 2 and 3, video half)
- CDIP per-wave labels for the shore clips on the 18 development dates
  (`scripts/sensing_cdip_waves.py`, Codex's, checked): mean individual height,
  H_1/3, H_1/10, T_z and wave count from the buoy record holding each clip's midpoint.
  1,444 clips, 1,145 records, 17 dates (2026-09-24 is past CDIP 433's archive). The
  provider evaluated every wave (no flag 2), so the flag sensitivity is empty. Counts:
  `scripts/sensing_cdip_inventory.py`; write-up `docs/experiments/sensing_per_wave_bars.md`.
- Measured mean height is 0.58 H_s at these buoys, not Kim's 0.631; the interim run's
  proxy overstated it by about 8 %.
- Bars T2 (per-wave targets) and T3-V (Kim's CNN-BiLSTM and Kamagata's SlowFast head with
  its physics loss, on the interim run's 992 clips and date folds) fixed before any run;
  Kim's training options corrected to the paper's Table 6 before any Kim run. Scorer
  `scripts/sensing_shore_test2.py`, trainer `scripts/sensing_shore_test3.py`, config
  `configs/experiments/sensing_shore_v1.yaml`.

### Results (2026-11-19, test 2 on the development dates)
- Kim's classes are unassessable on these dates (186 clips in range, 6 dates, mostly one
  storm); every arm names the lowest class for every clip (64.0 %, the majority's), since
  no estimate reaches 1.25 m. They are decided on the confirmation dates.
- T2-C not met: the pixel spectrum's T_z error is 1.29 s against climatology's 1.37 s,
  -0.08 s [-0.20, +0.08].
- Test 3's video arms are queued behind the paper strip runs; Kim's network is not
  trained on the development dates.

### Added (2026-11-19, shore video: physics features v2 and bars M6-M7)
- The 18 interim dates are now development data (`configs/experiments/webcoos_video_dev_dates.txt`,
  committed with its SHA-256 before any development); every date the final run adds is a
  confirmation date, on which alone bars M6-M7 are scored (`docs/experiments/webcoos_video.md`,
  "Physics features v2").
- Spectrum recipe v2 (`wave_analysis.processing.pixel_spectrum.physics_v2`; v1 unchanged):
  the common mode taken out of each water block with its own gain, an onset spectrum of
  max(dI/dt, 0), a cBathy-style coherence (lambda_1 / sum(lambda) of the cross-spectral
  matrix), a peak-prominence measure; four features added to v1's ten. Five label-free
  breaking features for H_s (foam fraction and its Otsu separability, foam arrivals per
  minute, foam-band width, foam persistence). CDIP energy period Tm-1,0 as a second
  period label (`Te_label_s`).
- `scripts/webcoos_video_clips.py physics` (features beside each windows zip, packed as
  `physics_v2.csv`); `scripts/train_video_probe.py` adds the M6-M7 ridge arms and scores
  them on confirmation dates only, refusing if the development-date file changed.
- Development results (chosen on, so optimistic): M6 −0.037 s [−0.109, +0.050] on T_p;
  M7 +0.032 m [+0.017, +0.051] (the breaking features make H_s worse). With about 28
  confirmation dates, M6 resolves only differences of about 0.07-0.09 s, so it is
  underpowered for its development estimate. Coherence weighting, as first planned,
  lowered the period's correlation with T_p and was dropped as a weight (development log
  in the doc).
- Bibliography: Otsu 1979, Welch 1967, Donelan & Pierson 1983, Lippmann & Holman 1989
  (DOIs checked against Crossref).

### Results (2026-11-19, shore video interim run: bars M1-M5)
- First scored run of the WebCOOS video test (`docs/experiments/webcoos_video.md`,
  Results): 992 clips of 18 local dates, folds of whole dates, CDIP labels. **M1 and M2
  not met**: video, a repeated frame and shuffled frames give the same H_s error
  (0.533-0.535 m; climatology 0.740 m). **M3 met**: reading all windows of a clip cuts
  H_s RMSE from 0.524 to 0.513 m. **M4 met but empty**: no attentive probe's T_p beats
  climatology (`clip` 2.86 s against 2.72 s). **M5 not met** (−0.39 s [−0.74, +0.02]; met on
  clips with K ≥ 15); the pixel spectrum alone is the only T_p estimate that beats
  climatology (2.46 s). Kim's five classes: no skill at unseen dates (58 %, the same as
  naming the average class). The record pooler is worse than the plain average and is
  dropped.

### Added (2026-11-19, grouped cross-validation, partition fusion and spectral targets)
- `scripts/buoycam_cv_folds.py` and package `ndbc_buoycam_heights_v2cv`: v2's strips in
  five folds over the 43 station groups, balanced by strips, each with validation groups
  drawn from its own training folds to match their swell share. Every buoy is predicted
  once, out of fold.
- A distribution head beside the point head (`model.distribution`, beta-NLL with
  beta = 0.5, Seitzer 2022); `predictions.csv` gains each height's log mean and spread.
- `wave_analysis.evaluation.uncertainty`: log-normal CRPS, merging seeds' Gaussians, and
  buoy-level split-conformal thresholds (Lee, Barber & Willett 2023).
- `scripts/buoycam_oof.py`: out-of-fold scores of a model's five folds x seeds, with
  conformal 90 % intervals (coverage per buoy and per sea state), CRPS, every paper's
  measure and a paired bar between two arms.
- **Bar B2-CV not met:** DINOv3-gentle lowers out-of-fold H_s RMSE against ImageNet arm A
  by 0.019 m [−0.028, −0.011] at all 79 buoys (0.532 against 0.550 m), but the wind sea
  only by 0.002 m [−0.008, +0.004]. Conformal intervals cover 0.90 on average; 11 buoys
  fall below 0.80.
- `scripts/buoycam_partition_check.py`: WVHT² = SwH² + WWH² holds within rounding on all
  295,547 records at the camera buoys. `scripts/buoycam_partition_fusion.py`: a gated
  swap of ERA5 + wind's wind sea for the camera's, beside ridge, gradient boosting and
  the raw partition rule, fitted fold by fold.
- **Bar S2 met out of fold:** the gated swap lowers H_s RMSE from 0.288 to 0.267 m
  (−0.021 m [−0.031, −0.012]) and wind-wave RMSE from 0.293 to 0.240 m against
  ERA5 + wind at 77 buoys. To be confirmed on strips archived after the package
  (about 2026-11-26).
- `buoycam_era5.py`: ERA5 periods and directions (`pp1d`, `mwp`, `mp2`, `mwd`, `mpww`,
  `mdww`, `mpts`) at each buoy's wave grid point. The parquet is now written whole
  before it replaces the old one.
- `scripts/buoycam_spectral_labels.py` and package `ndbc_buoycam_spectral_v3cv`: peak,
  mean (Tm-1,0), average and wind-sea periods, the wind-sea part of the 1-D spectrum
  and the mean direction (relative to the sun's azimuth: the caption's heading is a
  fixed "90°") from each buoy's own spectral records.
  `wave_analysis.models.wave_spectral` and `scripts/train_wave_spectral.py` train
  these outputs.
- **Bar S3 not met**, as expected: ERA5 wins peak period (1.70 against 3.44 s), energy period
  (0.67 against 2.40 s) and mean direction (14° against 76°). The camera beats ERA5 on the wind
  sea's period (Tm-1,0 0.55 against 0.75 s, −0.21 s [−0.24, −0.17]); trained jointly, the
  extra outputs raise H_s RMSE by 0.086 m, so heights stay with stage 1's model.
- `deploy/dgx/run.sh`: split names may contain digits (`fold0`).

### Added (2026-11-19, bar S2 on new weeks: the tool)
- `scripts/buoycam_fusion_confirm.py` (`build`, `predict`, `score`): the pre-registered
  confirmation of bar S2 on strips archived after the package. The package's light rules
  and labels, each strip predicted by its own buoy's fold's three seed models (CPU,
  float32), and each fold's ERA5 lines and fitted combiners applied without refitting.
  Before scoring it checks that its camera, ERA5 lines and combiners reproduce the
  out-of-fold tables on the package's strips; a dry run counts strips but scores none.
  `scripts/buoycam_pairs.py --start` pairs only strips taken from a given time on.

### Added (2026-11-18, whole-clip video: windows, pixel spectrum, shipping)
- Shore video, whole-clip bars M3-M5 implemented and their recipe fixed before scoring
  (`docs/experiments/webcoos_video.md`, "Whole-clip implementation"):
  `scripts/webcoos_video_clips.py windows` cuts every clip whole at 4 fps into its
  K = floor(duration / 16 s) windows (one stored zip per clip, about 35 MB a 10-min clip)
  and computes its pixel spectrum (`wave_analysis.processing.pixel_spectrum`, recipe v1:
  4 x 4 px blocks, Welch spectra, water chosen without labels by the wave band's share of
  the brightness-rate spectrum, peak above a fitted power-law background);
  `scripts/train_video_probe.py --windows` caches V-JEPA 2 and DINOv3 features per
  window, trains the 25 fold/seed window probes in lock step (`StackedProbes`, checked
  against separate training), reads each clip by its midpoint window and by all its
  windows, and adds the record pooler (`RecordPooler`), ridge heads on the spectrum and
  the scores of M3-M5 (also on clips with K >= 15).
- `scripts/webcoos_video_clips.py ship`: bulk data to the GPU server through the
  project's R2 bucket (36 Mbit/s up, 20 MB/s down, signed links, SHA-256 checks), since
  the SSH route through FAU's bastion carried 1.46 Mbit/s. The server does not download
  from WebCOOS itself: the agreed pace is one request at a time, and the desktop's sample
  download holds it.

### Added (2026-11-18, architecture review and whole-clip bars)
- `reports/Wave camera architectures beyond backbones.md` (notes in `research_notes/`
  under the same name): what to train beyond the backbone choice, ranked by value for
  this data, with H200 cost and the deciding experiment for each. In order: grouped
  cross-validation over the 43 buoy groups with seed ensembles and calibrated intervals;
  reading whole shore clips instead of one 16-s window; fusion that takes the wind sea
  from the camera and the swell from ERA5; a pixel spectrum for wave period; period,
  direction and spectrum outputs on the buoy strips; continued self-supervised
  pretraining; larger video encoders only if whole-clip reading plateaus.
- Shore video: whole-clip bars M3-M5 fixed before any clip is scored
  (`docs/experiments/webcoos_video.md`, "Whole-clip bars"; config `whole_clip`): one
  probe trained on every 16-s window; the whole-clip average against the midpoint window
  for H_s (M3) and T_p (M4), and a pixel-brightness spectrum added for T_p (M5).

### Added (2026-11-18, results and shore video)
- Backbone bars B1-B3 (three seeds each, 12 unseen buoys, each fixed before its runs):
  B1 not met (DINOv3 ConvNeXt-Tiny with arm A's full fine-tuning: worse, +0.039 m); B2
  not met by a hair (DINOv3 with the gentle recipe, `--lp-epochs 2 --backbone-lr-scale
  0.1`: 0.440 against 0.450 m, interval reaching zero; now the best model); B3 not met
  (DINOv3 ViT-B/16 gentle level with arm A, frozen far worse). Backbones differ by about
  0.01 m once each has a recipe that suits it. `--backbone-lr-scale` in
  `train_wave_heights.py`.
- Protocol check R (`scripts/buoycam_random_split.py`, package `ndbc_buoycam_heights_v2r`):
  Choi's and Kim's random split on our strips gives H_s 0.433 m and Kim's five classes
  68.8 %, the same as at unseen buoys; the gap to Kim's 78 % is not the split.
- `deploy/dgx/run.sh`: `TORCH_DISABLE_NATIVE_JIT=1` in the containers; `push` skips
  packing when the server already has the package.
- v2 results (three seeds per arm, 12 unseen buoys): the camera alone gives H_s RMSE
  0.450 m, R² 0.81, Pearson 0.915 (Kamagata 2026: 0.586 m, 0.107, 0.436) and 99.1 % on
  Choi 2020's calm/rough task (84 %), but 69 % on Kim 2023's five classes (78 %). V2-1
  not met: the horizon attitude makes H_s worse (+0.038 m). V2-2: the camera beats ERA5
  on the wind-wave height (0.215 against 0.341 m), not on the total.
- Test 3 not met: camera + ERA5 + wind against ERA5 + wind, H_s 0.258 against 0.259 m;
  wind-wave height 0.199 against 0.286 m (−0.087 [−0.114, −0.060]).
- Backbone review `reports/Vision backbones for wave cameras.md` (notes in
  `research_notes/`): DINOv3 for the buoy strips, V-JEPA 2 ViT-L (64 frames at 4 fps) for
  shore video, frozen probes first.
- Shore video vs single image (`docs/experiments/webcoos_video.md`, design fixed before
  scoring): frames now 64 at 4 fps, 256 px high, overlays masked; frozen V-JEPA 2 (video,
  still, shuffled) and DINOv3 (image) arms, attentive and mean probes, folds of dates
  (`scripts/train_video_probe.py`, `wave_analysis.models.video_probe`); `run.sh sync` for
  growing packages, `KIND=video`; `transformers` in the `ml` extra.

### Added (2026-11-18, v2)
- Wave heights in metres: `wave_analysis.models.wave_heights` (`MultiViewRegressor`: one
  backbone per view, optional per-view horizon roll and pitch, a one-layer transformer over
  the views, log total, wind-wave and swell heights) and `scripts/train_wave_heights.py`.
  Package `ndbc_buoycam_heights_v2` = v0's strips, partitions and labels plus wind-wave and
  swell heights, the buoy's wind and each view's horizon motion (`views.csv`). Bars V2-1
  (does the motion help?) and V2-2 (camera vs ERA5 in metres) and the published reference
  points (Kamagata 2026, Choi 2020, Kim 2023, Yang 2026) fixed before training.
- `deploy/dgx/run.sh train`: `TRAIN_SCRIPT` picks the trainer, `TAG` names the run.
- Choi 2020, Kim 2023 and Yang 2026 read in full (PDFs supplied by Steven): Choi used one
  view per sample with the same buoys and days in train and test (84.05 %); Kim's label is
  the *average* wave height (0.631 x H_s), one value for each of 30 one-hour sessions, with a
  random frame split; Yang is simulation only (attitude cut RMSE 0.098 -> 0.073 m).
  `compare_papers.py` takes Kim's classes on 0.631 x H_s.
- WebCOOS video: `webcoos historical --product video` (the archive's mp4 tree) and
  `scripts/webcoos_video_sample.sh` (a test set of rough and typical days, then six sample
  weeks, one clip per 30-min CDIP record in daylight); CDIP 243 for Jennette.
- Backbones from timm in the heights model (`timm:<name>`, e.g. DINOv3's ConvNeXt-Tiny and
  DINOv2/DINOv3 ViTs; `timm` added to the `ml` extra), `--backbone` and `--lp-epochs`
  (LP-FT) in `train_wave_heights.py`; Bar B1 (DINOv3 ConvNeXt-Tiny vs ImageNet) fixed
  before its runs. Shore video: `scripts/webcoos_video_clips.py` pairs clips with CDIP
  and cuts 64 frames at 2 fps (448 x 336, unstretched).
- `scripts/buoycam_combined.py` (Test 3, fixed before v2 was scored): camera + ERA5 + the
  buoy's wind against ERA5 + wind, ridge fitted on the val buoys.
- `scripts/compare_papers.py`: every published measure (Kamagata 2026 RMSE/R²/Pearson,
  Choi 2020 calm/rough, Kim 2023 bins, WMO classes) and bars V2-1/V2-2 from the v2 runs.
- `docs/literature/benchmark_targets.md`: every comparable published result (camera,
  radar, SAR) with its labels and split, the test of ours that matches it, and the models
  the sources suggest.
- Oak Island (Kamagata 2026's test views): CDIP 200 downloaded and standardized (76,963
  records, 2013 to 2026-07-20), both cameras in `webcoos_pairs.py`; the WebCOOS video
  archive inventoried (clips back to 2024-03, 20 fps at Currituck).

### Added (2026-11-18)
- Buoy-camera v1, the camera for the wind sea: `ndbc_buoycam_windsea_v1` labels each strip
  with NDBC's wind-wave height (window mean, ADR 0010) and keeps v0's strips, partitions
  and image bytes (`split.station.partitions_from`); baselines ERA5 wind-wave height and the
  buoy's own wind through a power law; no rotation augmentation; three seeds judged by
  their average. Bars for Test 1 (camera vs ERA5 wind sea) and Test 2 (camera + ERA5 vs
  ERA5 for the total sea state) fixed in the experiment doc before training.
- **v1 results:** Test 1 met: at the 12 unseen test buoys the camera (three seeds averaged)
  reads the wind-sea class better than ERA5's wind-wave height (balanced accuracy 0.795 vs
  0.694, gain +0.10 [+0.05, +0.16]; macro-F1 0.807 vs 0.709, +0.10 [+0.06, +0.15]) and far
  better than the buoy's own wind (+0.31). Test 2 not met: camera + ERA5 vs ERA5 alone for
  the total sea state, balanced accuracy +0.056 [+0.034, +0.095], macro-F1 −0.012
  [−0.025, +0.045] (more smooth false alarms cancel gains in the other classes).
- Test 2b (fixed before it was run, after Test 2): camera + ERA5 without class weights, not
  met. Both gains positive but intervals reach below zero (balanced accuracy +0.053
  [−0.007, +0.082], macro-F1 +0.044 [−0.024, +0.060]); the gain sits in calm seas, all of
  whose test strips are at two buoys. No further combiner is tried on this test set.
- `wave_analysis.processing.horizon.detect_horizon` and `scripts/buoycam_horizon.py`: the
  horizon in each view as a motion sensor. Its roll and pitch follow the buoy's own
  spectral slope within each buoy (Spearman 0.56 [0.54, 0.58], positive at 75-76 of 76
  buoys) and the wind waves, not the swell; the six views are separate moments.
- `buoycam_pairs.py`: window-mean wind-wave and swell labels and the buoy's own wind
  (`datasets.buoycam.stdmet_records`) for every strip; earlier columns unchanged.
- `buoycam_era5.py --param shww --param shts`: ERA5's wind-sea and swell heights at every
  buoy; `extract` reproduces the earlier table exactly when only `swh` is present.
- `scripts/ensemble_runs.py` (average seeds), `scripts/buoycam_fusion.py` (Test 2);
  `buoycam_bar.py` takes the bar's baselines from the config (`evaluation.bar`, `report`).
- `deploy/dgx/run.sh push`: `BASE_PKG=<package>` builds a package on the server from an
  earlier package's identical images and sends only the other files; a package's content
  is named by the SHA-256 of its `SHA256SUMS`.

### Added (2026-11-17)
- `deploy/dgx/run.sh`: training on a shared GPU server over SSH in a Docker
  container, the way FAU's CAAI lab uses its H200 server (the lab does not run
  jobs through Rancher). Same pieces as `deploy/k8s` (code of one commit, package
  tar checked by SHA-256, environment from uv.lock), on the server's data disk;
  one GPU per run, picked as the least busy with enough free memory. Uploads go
  by rsync, which resumes a broken transfer: the route through FAU's bastion
  carried about 2 Mbit/s on 2026-11-17 (heavy packet loss; the home line does 34).
  `deploy/container/{setup,train}.sh` run inside the container for both routes
  (`train.sh` was `deploy/k8s/train_in_pod.sh`).

- `scripts/buoycam_bar.py`: scores a buoy-camera run against the bar fixed before
  training (paired bootstrap of the gain over ERA5, over whole buoys or buoy-days)
  and writes `bar.json` into the run.
- First buoy-camera result (station split, one H200): the camera does not beat ERA5
  on the test buoys (balanced accuracy 0.70 against 0.74, gain −0.03 [−0.15, +0.01];
  macro-F1 0.65 against 0.77, −0.12 [−0.39, −0.03]); it mostly calls the sea one
  class too calm. `docs/experiments/ndbc_buoycam_sea_state.md`.
- Three more buoy-camera runs in parallel (2026-11-18): a second seed scores macro-F1
  0.12 lower on the test buoys (−0.142 to −0.079) and leans further calm; without
  class weights, no difference; the time split beats each buoy's majority class
  (+0.32 balanced accuracy) but not ERA5 on the 41 % of test strips it covers.

- `scripts/buoycam_swell_check.py` (research question Q-M8): the buoy camera sees the
  wind sea, not the swell. Within each buoy and class its too-calm share rises by
  0.20 [0.15, 0.25] from the least- to the most-swell third of strips (ERA5: flat);
  its calls follow wind-wave height more than swell height; over 24 held-out
  buoys, swell share predicts the under-call (Spearman −0.89).

### Added (2026-11-14)
- NDBC buoy-camera sea-state dataset (`ndbc_buoycam_sea_state_v0`): every archived
  strip paired with its own buoy's wave height (`scripts/buoycam_pairs.py`), and a
  package of 41,091 twilight and daylight strips at 79 buoys with the caption cut,
  a split by unseen buoys at least 300 km apart and a split by later weeks
  (`scripts/buoycam_package.py`, `configs/experiments/ndbc_buoycam_sea_state_v0.yaml`).
  Method in `docs/experiments/ndbc_buoycam_sea_state.md`.
- First results of `currituck_sea_state_v0` (desktop GPU): on the held-out year
  2025 the camera alone scores balanced accuracy 0.73, below WIS (0.77) and level
  with ERA5; `scripts/sea_state_fusion.py` shows it adds macro-F1 +0.04 to +0.06
  to either hindcast (intervals above zero), without clearing WIS on both
  metrics. Details in `docs/experiments/currituck_sea_state.md`.
- Full-frame ablation: alone, below the sea band on 2025 (balanced accuracy
  −0.07 [−0.10, −0.04]) and level on 2026; with a hindcast, level or slightly
  better (macro-F1 +0.02, not planned beforehand).
- `scripts/sea_state_haze_check.py`: sea-band sharpness of every still and a fog
  test of a trained model. The Currituck camera has grown hazier each year (93 % of
  2026 stills below the blurriest third of 2022-2024); haze of that size lowers the
  model's calm calls only from 0.73 to 0.67-0.70, against 0.49 seen in 2025.

- Six-view training for the buoy cameras: `model.input: views` reads each strip's
  six views with one backbone and averages the features of the usable ones
  (`MultiViewNet`, `StripDataset`); `train_sea_state.py` takes any split of a
  package, bootstraps over the config's group column and has `--class-weights none`.
- `scripts/buoycam_era5.py`: ERA5 wave height at every camera buoy from one global
  request per month (`ERA5WaveSource.fetch_area`), and the package's ERA5 baseline.

- `deploy/k8s/`: training on a Kubernetes cluster with NVIDIA GPUs (FAU's H200
  machine through Rancher), driven from the desktop by `run.sh` (storage volume,
  helper pod, upload with checksums, environment from uv.lock, one-GPU job).
  `git_state` reads the commit of a `git archive` tree from its `GIT_COMMIT` file.

### Fixed (2026-11-14)
- `sources.ndbc.buoycam.caption_band_rows`: the caption is a fixed 30-row band in
  both strip formats (measured); it was scaled to 25 rows for 2304 × 246 strips.
- `scripts/train_sea_state.py` records the git commit at the start of a run, not
  at the end (`build_provenance(git=...)`): the full-frame run's provenance named a
  commit made while it ran.

### Added (2026-11-13)
- Training on the desktop's AMD GPU: `deploy/local/setup_rocm_env.sh` (PyTorch for
  ROCm in `.venv-rocm`), `deploy/local/train_sea_state.sh` (the three runs in a row),
  `--micro-batch` (a batch computed in pieces, same update) and `--amp` in
  `scripts/train_sea_state.py`; cuDNN benchmark mode stays off on ROCm, where MIOpen's
  run-time kernel compile fails. Timings in `docs/operations/cluster.md`.
- ERA5 baseline in the `currituck_sea_state_v0` package (2022-01 to 2026-10,
  every still): balanced accuracy 0.73 on 2025 (WIS 0.77) and 0.75 on 2026,
  where it is the only physics baseline. The nearest sea grid point is a damped
  coastal point by Currituck Sound but tracks the buoy best of the 30 in the
  box (r = 0.94); an open-ocean point scored lower.

### Added (2026-11-10)
- `wave-analysis archive verify`: checks every raw file (following the links
  to other drives) against its ledger's SHA-256 and size, and reports files
  that are empty, the wrong size, the right size with other bytes (a power cut
  can leave a file full of NUL bytes), missing, without a ledger row, or left
  half-written; ledgers containing NUL bytes are reported too. `--since
  last-exchange` checks only what was written since the last offsite exchange.
  It records each file's MD5, the ETag R2 reports, so
  `wave-analysis archive compare-remote` compares the offsite copy without
  downloading it. `wave-analysis archive repair` settles a file whose local and
  offsite copies differ: whichever copy matches the ledger wins, and with
  `--refetch` the source is asked again; every decision is logged.
- Runbook "After an unclean shutdown" and the archive statuses in
  `docs/operations/data_collection.md`.

### Fixed (2026-11-10)
- Downloads, ledger rows, run listings and ERA5 files are synced to disk
  (fsync) before they count as written, and their folders after the rename.
  Three resets of the desktop (2026-10-18, 2026-11-03 twice) had left renamed
  files whose data never reached the disk: empty stills, one still the right
  size but all NUL bytes, and ledgers ending in NUL bytes.
- The buoy-camera archiver treats an empty image as missing and fetches it
  again while NDBC keeps it; WebCOOS listings without a size no longer count
  an empty still as archived.
- The offsite exchange no longer uploads damaged files and no longer stops on
  the first conflict. It first checks every file written since the last
  successful exchange against its ledger and holds back damaged files and
  ledgers. A file that differs from its offsite copy is listed in
  `data/interim/offsite/conflicts.txt`; the other files are still pushed and
  pulled, and the exchange fails at the end with the `archive repair` command
  to run. Before this, 28 empty copies uploaded after the reset of 2026-11-03
  stopped every exchange for six days (watchdog issue #15).
- R2: 29 damaged objects (28 empty, 1 all NUL bytes, from that reset) were
  overwritten with local copies checked against their ledgers. The whole local
  archive (629,917 files, 294 GB) matches its ledgers; 38 WebCOOS stills are
  empty at the source (HTTP 200 with 0 bytes) and stay as evidence.

### Added (2026-10-13)
- First WebCOOS camera paired with a wave buoy: `currituck_hampton_inn` with
  CDIP 433 (Duck FRF 17 m, 9.15 km away). `scripts/webcoos_extract.py` checks
  every still against its ledger and extracts brightness, colour saturation
  and thumbnails; `scripts/webcoos_pairs.py` keeps every buoy record within
  ±3 h of every still and adds the ADR 0010 labels: 38,361 of 40,313 stills
  are labelled, on 1,609 days. No still is dropped and the archive is only
  read. Method, checks and results: `docs/datasets/webcoos.md`.
- `cdip standardize --product realtime` and support for the layout of CDIP's
  realtime files (one deployment position, transmissions as source files),
  which hold the current deployment until it is archived. Their records are
  named after the deployment's start date, e.g. `433p1_rt20250725`.

### Fixed
- The position of the camera `currituck_hampton_inn` was the WebCOOS API's
  asset location (36.3779, −75.8258), which is 13.7 km north of the hotel. It
  is now the GPS position in the EXIF of the camera's stills (36.2593,
  −75.7852), 80 m from the hotel in OpenStreetMap. The reference buoy is
  9.2 km away, not 22.2 km. For the night filter of the historical download
  the sun's elevation changes by at most 0.12° (36 of 1,118,919 listed stills
  change sides of −6°).
- The offsite exchange failed whenever a collector was writing its ledger
  during the upload: rclone rejects a file that changes in transit. The
  WebCOOS historical download adds a row every two seconds, so the exchange of
  2026-10-13 stopped after the images: the ledgers' step failed, other hosts'
  files were not pulled and `.offsite_last_sync` was not written.
  `scripts/sync_offsite.sh` now uploads the ledgers from a copy made just
  before; the next exchange uploads the rows written since
  (`tests/integration/test_sync_offsite.py`, which needs rclone).
- Ledgers recorded an absolute path on the collector host for files under a
  collection that is a symlink to another drive (`portable_path` followed the
  link out of the repository). The path is now tried as written first, so
  such files are recorded as `data/raw/...` like all others (found before the
  WebCOOS folders were linked to a second drive, 2026-10-12).
- Landscape review cited `Quach2020`; the bibliography key is `Quach2021`
  (found by the new catalog consistency test).
- NDBC wave processing is cited as TD 03-01 (Earle 2003). The previously cited
  TD 96-01 (Earle 1996) is superseded, and NDBC says it should no longer be
  used. Bibliography key `Earle1996` is now `Earle2003` (verified V1).
- The FAU Harbor Branch entry in `camera_sites.yaml` described its coordinate
  as the Fort Pierce Inlet shoreline; it is the approximate campus position on
  the lagoon.

### Added (2026-10-12)
- `webcoos historical` keeps only images taken with the sun at or above −6°
  at the camera (`--min-sun-elevation`; `--all-hours` keeps night images too).
  Some cameras recorded around the clock in earlier years (54 % of
  `currituck_hampton_inn`'s January 2022 stills are night frames), which the
  size estimate given to WebCOOS did not allow for. Camera positions are the
  WebCOOS API's (`ONPREM_POSITIONS`); the saved month listings still name
  every image.
- The WebCOOS historical download started on 2026-10-13 at 01:15 UTC on the
  desktop, onto a second drive behind symlinks
  (`docs/operations/data_collection.md`).

### Added (2026-10-09)
- `wave-analysis webcoos historical`: past years of WebCOOS stills from the
  on-premise archive WebCOOS pointed us to (email of 2026-10-09), thinned to
  one per 30 min like `download`, at the pace they asked for (`Downloader`
  gains `min_gap_s` and `max_bytes_per_s`, wget's `--wait` and
  `--limit-rate`). Skips stills already archived, saves each past month's
  listing once, and writes a per-host ledger (`_manifests/<camera>.<host>.csv`).
  `--product timex|brt` fetches WebCOOS's 10-minute time exposures and
  brightest-pixel images (to `raw/webcoos_timex/`, `raw/webcoos_brt/`), and
  `--every all` keeps every image. `collector_host` moved to `ingest.archive`.
- `scripts/webcoos_historical.sh`: the agreed WebCOOS download (stills, then
  `timex`, then `brt`, every 30 min, eight cameras) in one resumable run;
  `WEBCOOS_PRODUCTS` limits the products. `webcoos historical` stops when the
  archive's drive has less than `--min-free-gb` free (default 50).
- Dashboard collections `webcoos_timex` and `webcoos_brt`; WebCOOS cameras
  fetched only from the archive are not reported as stalled.
- `WAVE_ANALYSIS_SYNC_PULL_SKIP`: top-level collections a host leaves out of
  the pull (the laptop skips the WebCOOS history).

### Changed (2026-10-09)
- `scripts/sync_offsite.sh` skips lock and partial files at any depth,
  including the top level, selects ledgers with ordered `--filter` rules, and
  follows symlinks (`--copy-links`), so a collection can live on another drive.

### Added (2026-10-08)
- The USACE Wave Information Study hindcast at buoy sites: adapter
  `wave_analysis.sources.wis` and `wave-analysis wis discover | download |
  standardize`, stations in `configs/sources/wis.yaml` (CDIP 106 and the
  reference buoy of every camera site), ledger `data/manifests/raw/wis.csv`
  (21 whole-record files and Waimea's 2009-2013 monthly spectra, 358 MB),
  datasheet [docs/datasets/wis.md](docs/datasets/wis.md), and a `wis`
  collection on the dashboard and in the data inventory.
- Waimea benchmark v1 (`configs/experiments/waimea_hs_v1.yaml`): v0 plus WIS
  as a nowcast, alone and with ERA5. ERA5 + WIS is 0.06 m better than ERA5 on
  held-out days and weeks; against it the image ridge adds 0.012 m at most
  ([results](docs/datasets/pacioos_beachcam.md#benchmark-v1-a-second-wave-model-2026-10-08)).
  `scripts/waimea_benchmark.py` takes the config as an argument and any
  number of nowcast models; v0 reproduces byte for byte.
- Every registry `candidate` was checked against its provider
  ([data source scan §8](docs/literature/data_source_scan_2026-10.md#8-registry-candidates-checked-2026-10-08)):
  `usace_wis`, `noaa_ww3` and `socib_cala_millor` are now verified, the other
  eight documented.
- A data inventory: `data/registry/data_dictionary.yaml` says what each
  collection is for (pictures, labels, context, model, our tables,
  bookkeeping) and what every field and variable in it means, with a
  cross-reference of the same quantity under NDBC, CDIP, ERA5 and label names.
  It is validated (`wave_analysis.data_dictionary`; canonical keys tested
  against the NDBC, CDIP and ERA5 adapters) and rendered twice from the one
  file: the dashboard's new **Data** view (`#/data`, with live counts) and
  [docs/datasets/inventory.md](docs/datasets/inventory.md) (MkDocs hook
  `scripts/mkdocs_data_dictionary.py`), which also explains where NDBC wave
  numbers are computed and what that means for pairing with cameras.
- The dashboard's collection list now includes the NDBC historical files
  (`ndbc_history`) and the published research datasets (`external`), each
  checked against its ledger.
- A second collector host. Buoy-camera ledgers are named per host
  (`_manifests/<YYYY-MM>.<host>.csv`, `buoycam.collector_host`,
  `WAVE_ANALYSIS_HOST`), so two hosts can archive the same images.
  `scripts/sync_offsite.sh` now pulls other hosts' files after pushing,
  compares raw files by size, and copies ledgers with `--update`.
  `scripts/install_collectors.sh --secondary` installs the archiver without
  the single-writer jobs ([data collection](docs/operations/data_collection.md#a-second-collector-host)).
- A collector watchdog (`.github/workflows/collector-watchdog.yml`,
  `dashboard.watchdog`): hourly on GitHub, it opens an issue labelled
  `collector-alert` when the published status shows no archiver run or status
  push for 3 h or no offsite sync for 30 h, says when images start to be lost
  for good, and closes the issue on recovery ([alerts](docs/operations/data_collection.md#alerts)).

- The first Waimea Bay benchmark (`scripts/waimea_benchmark.py`,
  `configs/experiments/waimea_hs_v0.yaml`): a manifest of the 16,404 paired
  images with splits by HST day, ISO week and year, a leakage audit, and
  climatology, persistence, ERA5 and image-ridge baselines under the
  window-mean and nearest labels, with day-bootstrap intervals and paired
  comparisons ([results](docs/datasets/pacioos_beachcam.md#benchmark-v0-splits-and-baselines-2026-10-08)).

- A data source scan ([docs/literature/data_source_scan_2026-10.md](docs/literature/data_source_scan_2026-10.md)):
  WebCOOS now lists 86 cameras, 41 within 25 km of a directional wave buoy,
  with stills inventories for twelve; three open datasets for the sensor gaps
  (ship-based stereo with ship motion and wave buoys, surf-zone thermal IR
  with lidar, stereo of extreme waves in sea ice); camera networks with
  archives to request; label-only sources; and searches that found nothing.

- The WebCOOS collector also archives eleven beachfront cameras within 25 km
  of a directional wave buoy (`NEAR_BUOY_CAMERAS`); `webcoos download` accepts
  camera sets as `@yin2025` and `@near_buoy`.

- `scripts/download_external.py` and `configs/sources/external_datasets.yaml`:
  published research datasets go to `data/external/<id>/`, checked against
  the repository's sizes and MD5s and recorded in
  `data/manifests/raw/external.csv`; `--from-zip` for repositories that hand
  out zips by email. Registry entries and a doc
  ([research datasets](docs/datasets/external_research_datasets.md)) for the
  ship-based stereo, Duck thermal IR + lidar, and Agulhas II WASS datasets.
  The two UW datasets (31.2 GB) are downloaded and match the repository's
  MD5s; their contents are described from inspection, including that the
  five-minute stereo archive has 55 bursts where its readme says 52. The
  Agulhas II WASS files (1.6 GB) came as the AADC zip and are extracted with
  `--from-zip`. Their `datenum` variable is wrong and their missing value is
  0, not NaN; the doc says to take times from the file names.

### Fixed (2026-10-08)
- `buoycam.known_missing_urls` read the last three ledger files rather than
  the last three months, which would skip months once several hosts write
  ledgers.
- The offsite-sync unit finds an rclone installed in `~/.local/bin`.
- The buoy-camera backfill fetches the oldest missing hours first across all
  cameras, instead of each camera newest-first. After the 76 h outage of
  2026-10-04, two recovery runs cut short by a reboot never reached 92
  camera-hours that were still on the server.
- `scripts/buoycam_coverage.py` counted an hour as never published when one
  minute stamp returned 404 even though the image was archived at another
  (765 reported against 161 real over 8 days). Its gap count was correct.

### Added (2026-10-01)
- `processing.synchronize.label_from_candidates`: one label per image from
  `pair_window` candidates with a named rule (`nearest`, `interpolate`,
  `window_mean`, `in_record`), an optional propagation lag, and circular
  handling for directions; returns the label, the spread of the records used,
  their count and the largest offset. ADR 0010 makes the window mean the
  primary label and keeps the others for sensitivity analysis.
- `scripts/waimea_pairs.py` adds the ADR 0010 labels (`Hs_label_m`,
  `Hs_label_spread_m`, `n_label_records`; `Tp_label_s` and `Dp_label_deg` from
  the nearest record, with their window range and a `peak_records_disagree`
  flag) and a `label` block to its summary. Run on the full archive: earlier
  counts unchanged; the two-record H_s spread is a median 3.9 %.
- `scripts/buoy_pair_agreement.py`: hourly H_s agreement of every NDBC/CDIP
  buoy pair within 100 km, split into systematic and random parts. Result for
  2023 (211 pairs) in `data/manifests/processed/`: about 6 % mismatch under
  10 km, a median 15 % at 10–30 km ranging from 4 % on open coasts to 60 %
  across Oahu, and lags that barely matter at hourly resolution.
- Documentation: a data collection plan (gaps, streams, priorities,
  self-collected site specification; `docs/operations/collection_plan.md`); a
  common-baseline policy for H_s labels (`wave_parameters.md` §5); measured
  reference distances and label rules (`synchronization.md`); the WebCOOS
  video pilot; TD 03-01 facts in the NDBC datasheet; research gaps G-9 (no
  real image + attitude dataset) and G-10 (reference representativeness), and
  questions Q-D9 and Q-M7; an incident log for the collector.

### Added
- Package `wave_analysis` with layered architecture: `sources`, `ingest`,
  `schemas`, `physics`, `processing`, `qc`, `datasets`, `evaluation`, `models`,
  `registry`, `cli`.
- NDBC adapter: parsers for standard meteorological files (all header eras
  1990s–present, realtime and historical), spectral summary, spectral density
  and directional coefficients (realtime pairs and historical grids; r₁/r₂
  un-scaling; multi-grid splitting), ADCP; normalisation to CF names and
  canonical units; QARTOD validation.
- NDBC availability inventory (station × product × period, including `b`
  segment files), station metadata, camera snapshot; committed registry
  snapshot of 2026-09-26.
- NDBC buoy-camera prospective archiver (`wave-analysis ndbc cameras`), per-view
  splitting of the six-camera strips and caption-band masking (measured on real images).
  The archiver backfills the last 70 h (NDBC keeps superseded images for about 72 h,
  measured), tries every observed minute stamp (the stamp varies per camera),
  writes a dated archive layout, streams a monthly ledger kept with the
  archive, and takes a run lock.
- Continuous collection: systemd user timers (`deploy/systemd/`,
  `scripts/install_collectors.sh`) for the hourly archiver and twice-monthly
  gzip snapshots of realtime spectra for camera stations
  (`ndbc download --camera-stations --realtime --gzip`); an append-only offsite
  copy via rclone (`scripts/sync_offsite.sh`, Cloudflare R2 by default); a
  coverage report (`scripts/buoycam_coverage.py`); ADR 0008.
- CDIP adapter (`wave_analysis.sources.cdip`, `wave-analysis cdip discover |
  download | standardize`): THREDDS catalog listing, whole-file downloads of
  historic, deployment and realtime netCDF files, bulk and directional-spectrum
  normalisation with sample-start times, per-deployment positions and CDIP's
  flags, and QC including CDIP's primary flag. Conventions verified on CDIP 106
  (docs/datasets/cdip.md): H_s = 4√m₀ of the published spectrum, α₁ equal to
  CDIP's band mean direction, 1600 s samples, 1-based deployment index.
- PacIOOS beach-camera archiver (`wave_analysis.sources.pacioos`,
  `wave-analysis pacioos beachcam`): checks each ERDDAP index (UTC vs HST file
  names, duplicates, URLs), downloads the images with a per-request ledger,
  checks every size against the index, and resumes. Started on the Waimea Bay
  pair (17,854 images, 3.3 GB), the first historical benchmark with CDIP 106.
- WebCOOS adapter (`wave_analysis.sources.webcoos`, `wave-analysis webcoos cameras |
  inventory | download`): token from `$WEBCOOS_API_TOKEN` or a mode-600 file, sent
  only to the API host and never logged; camera, inventory and element listings;
  one-minute stills thinned to a time grid; archive with a ledger and resume. The
  90-day rule of the WebCOOS usage guidelines is enforced (`--historical-approved`).
  Optional hourly collector for the five Yin et al. (2025) cameras
  (`install_collectors.sh --webcoos`). Camera sites updated with API positions
  (Oak Island was 7.8 km off) and Jennette's Pier added.
- ERA5 adapter (`wave_analysis.sources.era5`, `wave-analysis era5 download |
  standardize`): 12 wave parameters from the Copernicus CDS, one month per request
  for a box around a camera site, raw files with their requests and a manifest;
  months still ERA5T or incomplete are requested again once they could have changed.
  Normalized at the nearest sea grid point, flagged `era5`/`era5t`. The key stays in
  `~/.cdsapirc` (mode 600), goes only to the CDS host and is never logged. New
  variables `wave_energy_period`, `wind_wave_energy_period`, `swell_energy_period`
  (ERA5's `mwp` is m-1/m0, not Tm01). Optional extra `cds`. Real CDS fixture.
- Waimea Bay audit (`scripts/waimea_extract.py`, `scripts/waimea_timing_audit.py`):
  archive integrity, the capture time read from each image's burned-in caption
  (the file name is up to an hour early; the camera clock was wrong in
  May–July 2013), checked against the sun and CDIP 106.
- Waimea Bay–CDIP 106 pairing table (`scripts/waimea_pairs.py`): every CDIP
  record within 3 h of each trusted capture time, with offsets, and one row per
  indexed image with its exclusion reason; no label rule applied yet. New
  `processing.synchronize.pair_window`.
- Dashboard: a "Data held" panel on the overview with every collection on the
  collector host (NDBC buoy cameras and realtime snapshots, PacIOOS Waimea Bay,
  WebCOOS, ERA5, CDIP): counts and sizes from the files on disk, the period of
  data, freshness, and a check of files against verified ledger rows. Counts
  only; no images are republished. `status.json` gains `collections[]`.
- Shared resumable archiver (`wave_analysis.ingest.archive`) used by the PacIOOS
  and WebCOOS image collections.
- The downloader streams bodies to disk while hashing them, so files larger
  than memory can be fetched; a connection that drops mid-body is retried.
- Public dashboard on GitHub Pages (`dashboard/`, `wave_analysis.dashboard`,
  ADR 0009). Views: overview, cameras map, per-station image viewer with the
  sea state at each image, archive coverage, data sources, and literature.
  - `wave-analysis dashboard catalog|live|build|check`.
  - Hourly live-data publication to a single-commit `dashboard-data` branch
    (`scripts/publish_dashboard.sh`, `install_collectors.sh --dashboard`).
  - Recent sea state from HTTP range requests on NDBC realtime files.
  - A BibTeX reader, and a machine-readable verification index in
    `source_verification.md`.
  - `data/registry/camera_sites.yaml`: camera sites and their reference buoys,
    with distances computed at build time.
  - `site.yml` workflow: builds, checks and deploys the dashboard with the
    MkDocs site under `/docs/`; replaces `docs.yml`.
  - Node unit tests for the front end.
- Byte-preserving downloader with retries, rate limiting, atomic writes,
  SHA-256, and CSV manifests with change detection.
- Physics: linear dispersion (Guo 2002 + Newton), group velocity, steepness,
  Iribarren and Ursell numbers; zero-crossing analysis; Pierson–Moskowitz,
  JONSWAP, cos-2s spreading; random-amplitude sea-surface simulation.
- Spectral analysis: moments, H_m0, T_p (discrete and Young 1995), T_m01, T_m02,
  T_m−10, ν, Q_p, s_p, D_p, D_m, spreads; NDBC ↔ Fourier conversion; Fourier
  and MEM directional distributions; energy-conserving regridding.
- QC: QARTOD tests 9, 15, 16, 17, 18, 19, 20, 21; steepness,
  spectral-consistency, and directional-realisability checks; image-quality
  metrics, dHash; solar position, illumination, glint risk.
- Synchronisation (nearest, window overlap, propagation lag); leakage-safe
  splits (grouped, cross-time with embargo, hold-out, extreme, LOGO) and leakage
  audit; ML manifest builder with exclusion ledger; provenance records and
  RO-Crate.
- Evaluation: regression (named NRMSE normalisers, scatter index, slope),
  Bland–Altman, circular statistics, ordinal classification, block bootstrap,
  probabilistic scores, stratified metrics.
- Models: statistical baselines; optional torch reference implementations of
  Choi 2020, Kim 2023, Yang 2026, and the Kamagata 2026 SlowFast head; physics
  losses.
- Landscape review (`docs/literature/landscape_review.md`): techniques by
  modality, model trends, industry, and public training data with evidence
  levels (V1/V2/S/Q). Includes the camera-to-buoy pairing table (CDIP and
  NDBC), 15 new registry entries (29 total), 20 Crossref-verified bibliography
  entries, hypothesis H3, and questions Q-D6–Q-D8, Q-M4–Q-M6, L-4, L-5.
  arXiv:2606.13302 reviewed in full.
- Documentation: standards, architecture, data model, QC, provenance,
  reproducibility, citation policy, methodology (9 pages), dataset datasheets,
  literature matrix, research gaps, source-verification log, ADRs 0001–0008,
  operations guide for continuous collection and storage.
- Tests on real trimmed NDBC fixtures; live upstream-format tests (`-m network`).
- CI (tests, lint and type-check, docs), pre-commit, issue and PR templates.
