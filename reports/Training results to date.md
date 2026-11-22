# Training results to date

Every model the project has trained up to 2026-11-21: what worked, what did not, the best
scores, and where we stand against the published work. Each experiment's design was
fixed in its doc before it was scored ("bars"); this report adds no new analysis.

The buoy results are on `main` at `944fa87ff` (November 21, 2026), including the two HPC
follow-ups. The final shore-video run (M1-M7) is on `claude/video-whole-clip-491c90`,
not yet merged. The storm-set class bars are scored once the storm re-run lands. The
fresh-strip protocol and scorer are on `claude/fresh-strip-confirmation`. The table at
the end gives each result's branch.

## Bottom line

**The camera is a wind-sea instrument.** At buoys it never saw, one hourly six-view strip
from an NDBC buoy camera:
- **Wind-wave height:** better than ERA5's wind-sea estimate (0.2558 against 0.3432 m on
  36,857 paired out-of-fold strips; 0.215 against 0.341 m on the sealed buoys).
  - It does **not** beat a stronger forecast on its own: ERA5 + GFS-Wave, corrected with
    location, depth and the buoy's own wind, reaches 0.226 m against the camera's
    0.254 m.
  - Added to that forecast, it gives the best estimate (0.208 m).
- **Wind speed:** a quarter less error than ERA5's corrected 10-m wind (0.96 against
  1.27 m/s), with the buoy's anemometer as the reference. It can stand in for a missing
  anemometer.
- **Surface slope:** better than the buoy's wind and the camera geometry can predict
  (0.0093 against 0.0102).

It cannot see most of the swell, so for **total** wave height the wave model stays ahead
(ERA5 0.267 m against the camera's 0.434 m). Combined, they beat either alone:
- swapping the camera's wind sea into ERA5 lowers total-height error from 0.288 to
  0.267 m at all 79 buoys;
- it lowers wind-wave error from 0.293 to 0.240 m.

Against the stronger forecast (ERA5 + GFS-Wave + buoy wind) the total-height gain is
thin: 0.2670 to 0.2623 m (1.8 %). Its interval clears zero by under a millimetre, and it
is not significant when every station group is weighted equally. The wind-wave gain over
that forecast is clear (0.226 to 0.208 m).

This gain holds under a nested check, in which the fusion step never sees predictions
from models trained on the test buoys. It also holds when the camera and the forecast
correction train only on strips before October 29 and are scored from October 31 at
unseen buoys: a plain 50/50 average of the two lowers wind-wave error from 0.260 to
0.230 m (−3.1 cm [−5.0, −1.8]), and wins in all five folds. The last confirmation, on weeks archived after the
package, is expected around November 27, once ERA5 waves, wind and gust cover November 21 at 16:00 UTC.

**The best single model** is DINOv3 ConvNeXt-Tiny with the "gentle" recipe: the backbone
frozen for 2 epochs, then trained at a tenth of the head's learning rate.
- On the 12 sealed test buoys: H_s error 0.440 m, R² 0.82.
- At all 79 buoys, each predicted by a model that never saw it or any buoy within
  300 km: 0.532 m.

**Against the four target papers,** re-running each paper's own network on our data,
our model wins every like-for-like comparison, on the sealed buoys and again at all 79
buoys out of fold:
- Choi: +4.6 points of accuracy;
- Kim: +20 points;
- Yang: 0.096 m lower error;
- Kamagata: 0.14 m lower error.

Where only published numbers can be compared (their data, not ours), we are ahead of
Choi and Kamagata on height and behind Kim's snapshot accuracy and Kamagata's shore
period. The gaps to Kim come from his protocol (one camera, random frames), not from his
model.

**What did not work:**
- a bigger or newer backbone, beyond about 0.01 m;
- continued self-supervised pretraining on our own images;
- adding the buoy's measured tilt as an input;
- fine-tuning beyond frozen features for slope;
- motion inside a 16-s shore clip, for wave height;
- ten-minute summary images in place of a single still;
- any camera model for total period or wave direction.

## How to read the numbers

- **RMSE:** the typical size of an error, in metres (heights), seconds (periods) or m/s
  (wind). Lower is better.
- **R²:** the share of the variation in the truth that the estimate explains. 1 is
  perfect; 0 is no better than always guessing the average.
- **Accuracy / macro-F1:** the share of class calls that are right / the average over
  classes of how well each class is found (chance on three balanced classes is about
  0.33).
- **Difference [95 % interval]:** the first arm minus the second, on the same rows. The
  interval comes from resampling whole buoys, station groups, dates or site-days
  (whichever the test names) 2,000 times. A bar is met when the whole interval lies on
  the winning side of zero.
- **Sealed 12 buoys:** the original held-out test buoys (6,183 strips). They have now
  informed several decisions, so results on them look back and do not confirm.
- **Out of fold:** five folds over the 43 station groups (79 buoys), each buoy predicted
  by models that saw neither it nor any buoy within 300 km. This is now the protocol for
  choosing models. Its numbers are higher, mostly because of the US West Coast fold
  (0.70 m), which its models never trained on.
- **Seeds:** every trained arm is the average of at least three seeds unless it says
  otherwise.

## Data and compute

| Data | What | Labels |
|---|---|---|
| NDBC buoy cameras | 41,091 six-view strips at 79 buoys (43 station groups), 2026-09-23 on | the same buoy's spectrum: H_s, wind-wave and swell height, periods, direction, slope; the buoy's wind |
| WebCOOS Currituck stills | about 37,000 stills 2022-2026, one camera | CDIP 433 H_s, four WMO classes |
| WebCOOS shore video | 992 clips on 18 development dates (four cameras), more downloading; a sealed 24-date storm set | CDIP 433, 243, 200: H_s, T_p, and per-wave heights and periods |
| WebCOOS `timex` / `brt` | about 217,000 ten-minute summary images | CDIP H_s |
| VWBT_9000 (Yin et al. 2025) | 9,000 breaker clips at six beaches | spilling / plunging / surging, expert consensus |
| ERA5, WIS, GFS-Wave, NWPS | wave-model baselines at each buoy | none (they are the comparators) |

**Compute.**
- **Where:** all neural training ran on FAU's shared H200 server (141 GB per GPU),
  except the first Currituck model, which ran on the desktop GPU before it overheated.
- **Runs:** a buoy strip run takes 40-55 min; probes and the Kim and Yang networks take
  7-15 min; the continued pretraining took about 9 GPU-hours.
- **Sharing:** the project now holds at most 7 of the 8 shared GPUs, only where at least
  40 GB is free.
- **Against the papers:** the four papers' models are at most 0.14 B parameters
  (Kamagata trained on 8 × A100 40 GB). Nothing here needed more than one H200 per run.

## The papers we compare with

| Paper | What they did | Their headline | Like-for-like test here |
|---|---|---|---|
| **Choi et al. 2020**, *Ocean Eng.* 201 | VGG16 on one NDBC buoy-camera view (lower half, 224²), 20 buoys; same buoys and days in train and test | 84.05 % calm (< 1.52 m) vs rough (> 3.05 m) | their task on our unseen buoys; their VGG16 re-run on our strips |
| **Kim et al. 2023**, *Ocean Eng.* 288 | GoogLeNet (snapshot) and GoogLeNet + BiLSTM (video) on one fixed camera; 30 one-hour sessions, random frames; seafloor AWAC average height | 78.3 % snapshot, 93.2 % with 180 s of video, five 0.5-m classes | five classes on our buoy strips; snapshot CNN re-run on strips; CNN-BiLSTM on shore video (waits for the storm set) |
| **Yang et al. 2026**, *Ocean Eng.* 352 | ResNet-18 + IMU attitude attention fusion, ship camera, **Unity simulation** | RMSE 0.098 m image only → 0.073 m with attitude (simulated seas) | horizon attitude as input on real buoys; Yang's network re-run with and without attitude |
| **Kamagata et al. 2026**, arXiv:2607.11998 | V-JEPA + SlowFast on 16 frames of shore video; labels traceable to no instrument; five held-out videos (Oak Island E/W, Xishawan) | H_s RMSE 0.586 m, R² 0.107, Pearson 0.436; T_p 1.42 s; direction 37° | shore video at Oak Island against CDIP 200; SlowFast head with their physics loss re-run on our clips |

Also compared or used as context:
- **Breakers:** Yin et al. 2025 (VWBT_9000, CRNN, 96.1 % on a random split).
- **Shore period:** Kamagata et al. 2026b (ViViT, T_p 0.80 s on four scenes).
- **Single-site shore cameras:** Xu et al. 2024 (one camera, random frames, MAE 0.04 m
  with weather inputs) and Buscombe et al. 2020 (RMSE 0.11-0.14 m).
- **Other sensors:** X-band radar (RMSE 0.36-0.39 m) and SAR (about 0.3 m), from
  `docs/literature/benchmark_targets.md`.

## Leaderboard: the best score for each task

| Task | Best | Score | Comparator |
|---|---|---|---|
| Total H_s, camera alone (sealed 12) | DINOv3-gentle | **0.440 m**, R² 0.82 | ERA5 0.267 m |
| Total H_s, camera alone (out of fold, 79 buoys) | DINOv3-gentle | **0.532 m**, R² 0.79 | ImageNet ConvNeXt 0.550 m |
| Total H_s, camera + ERA5 + wind (sealed 12) | Test 3 ridge | **0.258 m** | ERA5 + wind 0.259 m (level) |
| Total H_s, fusion (out of fold) | gated partition swap | **0.267 m** | ERA5 + wind 0.288 m |
| Total H_s, camera + forecasts + wind (nested, 34,723 strips) | gated partition swap | **0.262 m** | ERA5 + GFS-Wave + wind 0.267 m |
| Wind-wave height, camera + forecasts + wind (nested) | gated partition swap | **0.208 m** | ERA5 + GFS-Wave + wind 0.226 m |
| Wind-wave height, trained before Oct 29, scored after Oct 31 | 50/50 camera + forecast | **0.230 m** | ERA5 + GFS-Wave + wind 0.260 m |
| Wind-wave height, camera alone (sealed 12) | DINOv3-gentle | **0.214 m** | ERA5 0.341 m |
| Wind-wave height, camera + ERA5 + wind (sealed 12) | Test 3 ridge | **0.199 m** | ERA5 + wind 0.286 m |
| Wind-sea period Tm-1,0 (out of fold) | spectral head | **0.55 s** | ERA5 0.75 s |
| Buoy wind speed (out of fold) | frozen probe | **0.96 m/s** | ERA5 1.27 m/s |
| Surface slope (out of fold) | frozen probe + wind + geometry | **0.0089** | wind + geometry 0.0102 |
| Calm vs rough (Choi's task, sealed 12) | DINOv3-gentle | **99.2 %** | Choi's VGG16 re-run 94.7 % |
| Kim's five classes, strips (sealed 12) | DINOv3-gentle | **70.2 %** | Kim's CNN re-run 50.1 % |
| Shore H_s, single still (held-out year 2025) | frozen DINOv3 probe | **0.265 m** | climatology 0.613 m |
| Shore H_s, video (46 dates, final run) | DINOv3 averaged over the whole clip | **0.293 m**, R² 0.70 | climatology 0.545 m; V-JEPA 2 whole clip 0.305 m |
| Shore H_s, video (18 dates, interim) | V-JEPA 2, whole clip | **0.513 m** | Kamagata's head re-run 0.653 m |
| Shore period T_p (46 dates, final run) | whole clip + pixel spectrum | **2.59 s** | climatology 3.24 s |
| Breaker type at unseen beaches | DINOv3 frames | **macro-F1 0.85** | Yin's CRNN 0.65 |

The shore still (0.265 m, one camera, held-out year) and the shore video (0.293 m, four
cameras, 46 dates) are different experiments, not a test of still against video. The
interim video number (0.513 m on 18 dates) fell to 0.29 m with 46 dates, so most of the
earlier gap was the amount of data, not the medium.

## Results by track

### 1. Currituck shore stills, the first model (2026-11-14)

- **Model:** ConvNeXt-Tiny (ImageNet) on the sea band of one Currituck still, four WMO
  classes; tested on a held-out year (2025) and on held-out days.
- **Camera alone: did not beat the hindcast.** Balanced accuracy 0.73 and macro-F1 0.73
  in 2025, against 0.77 and 0.80 for the WIS hindcast. It ties ERA5. It calls rough seas
  too often and does not carry well between years.
- **Camera with the hindcast: helps a little, consistently.** Macro-F1 rises by 0.038
  [0.007, 0.065] with WIS (2025) and 0.052-0.057 with ERA5 (2025, 2026). The bar set
  beforehand (beat WIS on both measures) was not cleared.
- **The full frame (beach included) is no better alone.** It adds a little on top of the
  hindcast, plausibly by reading the season from the beach.

### 2. NDBC buoy cameras: classes to heights (2026-11-17 and -18)

| Step | Question | Result |
|---|---|---|
| v0, station split | do four WMO classes from the camera beat ERA5? | **No.** Balanced accuracy 0.70 against 0.74; macro-F1 0.65 against 0.77. It calls the sea one class too calm. |
| Why | what does it miss? | It sees the wind sea, not the swell: its calls follow the wind-wave height, not the swell. |
| v1, wind-sea classes | does it beat ERA5 on the wind sea? | **Yes.** Balanced accuracy 0.80 against 0.69 (+0.10 [0.05, 0.16]); the buoy's own anemometer reaches 0.49. |
| v2, heights in metres | camera alone, at unseen buoys | **H_s 0.450 m, R² 0.81, Pearson 0.915; wind-wave 0.219 m (R² 0.93); swell 0.597 m (R² 0.54).** |
| V2-1 | does the buoy's tilt (horizon) as an input help? | **No, it hurts.** H_s +0.038 m [0.018, 0.057]. |
| V2-2 | camera against ERA5 | **Wind sea yes** (0.215 against 0.341 m); **total no** (0.434 against 0.267 m). |
| Test 3 | camera + ERA5 + buoy wind against ERA5 + wind | **Wind-wave 0.199 against 0.286 m (30 % less error); total level** (0.258 against 0.259 m). |
| Horizon as a motion sensor | does horizon roll follow the buoy's slope? | **Yes**, Spearman 0.56 within buoys (positive at 75 of 76). It follows the wind sea, not the swell. |

### 3. Backbones and recipes (2026-11-18 to -19)

| Bar | Compared | Result |
|---|---|---|
| B1 | DINOv3 ConvNeXt-Tiny against ImageNet, same recipe | **Worse**, H_s +0.039 m. The recipe did not suit the new weights. |
| B2 | DINOv3 with the gentle recipe against ImageNet arm A | **Not met by a hair** (−0.010 m [−0.024, +0.001]), but the **best model so far**: 0.440 m, R² 0.82, wind-wave 0.214 m, swell 0.572 m. |
| B3 | DINOv3 ViT-B/16, gentle and frozen | Gentle level with arm A (+0.005 m); frozen far worse (+0.111 m). |
| Protocol check R | does Kim's random split explain the gap to his 78 %? | **No.** A random split of our strips gives 68.8 % (H_s 0.433 m). The gap is his single site and session labels. |
| B2-CV (out of fold) | DINOv3-gentle against ImageNet at all 79 buoys | **Not met.** H_s −0.019 m [−0.028, −0.011], lower at all 79 buoys, but wind-wave level (−0.002 m). |
| Uncertainty | buoy-level conformal 90 % intervals | Cover 0.90 overall, but 11 buoys fall below 0.80. |
| P1 (rank 6) | DINOv3 pretraining continued on our buoy views (about 9 GPU-hours) | **Not met.** Frozen: H_s −0.010 m [−0.027, +0.002], wind-wave −0.012 m [−0.018, −0.006]. With the gentle recipe the continued weights are **worse** (+0.028 m). Closed. |

**Test 6, adapting each camera from its own unlabelled images (2026-11-20): not met.**
Each camera's features were standardised against the training cameras at the same ERA5
total height, out of fold.
- **Total height:** 0.532 to 0.500 m, but the interval over station groups crosses zero
  (−0.069 to +0.019). Nearly all of the gain is the US West Coast fold (0.699 to 0.586 m);
  with groups weighted equally there is no gain.
- **Wind-wave height:** much worse (0.260 to 0.377 m), at every fold. Removing a camera's
  offset also removes its local wind-sea climate.
- **Few labelled hours at a new buoy** (reported beside the bar): 10-30 labelled hours cut
  total-height error by 9-11 %. A short calibration period is the practical route to a
  new site.

**Reading:** once each backbone has a recipe that suits it, backbones differ by about
0.01 m, less than the spread between seeds and buoys. The data and the protocol matter
more than the encoder.

### 4. Fusion with the wave model, and periods (2026-11-19)

| Bar | Compared (out of fold) | Result |
|---|---|---|
| S2 | camera wind-sea energy swapped into ERA5, gated, against ERA5 + wind | **Met.** Total H_s 0.267 against 0.288 m (−0.021 [−0.031, −0.012]); wind-wave 0.240 against 0.293 m. |
| S3 | camera periods and direction against ERA5 | **Not met** for total period, energy period and direction (peak period 3.44 against 1.70 s; direction 76° against 14°). |
| S3, wind sea | the wind sea's own period | **The camera wins:** wind-wave peak period 0.69 against 0.93 s; Tm-1,0 0.55 against 0.75 s. |
| Joint heads | period heads trained with the height heads | Cost H_s +0.086 m, so new targets get their own heads. |

**Test 1, the camera against short-range forecasts (nested, 2026-11-20): met on both
measures.**
- **Setup:** 34,723 strips at 71 buoys. The stronger comparator corrects ERA5 and
  GFS-Wave (212 cycles) with location, depth and the buoy's own wind.
- **Wind-wave height:** adding the camera lowers error from 0.2258 to 0.2079 m
  [−0.0253, −0.0088].
- **Total height:** from 0.2670 to 0.2623 m [−0.0083, −0.0007].

The first versions of test 1 and of S2 fitted the fusion step on camera predictions from
models that had trained on the outer test groups. They were redone with 30 inner camera
runs, so that each fold's fusion step is fitted only on predictions from models that
never saw that fold. Both hold: S2 under the same check is 0.2667 against 0.2883 m.

**HPC follow-ups (2026-11-21; `docs/experiments/hpc_buoy_followup.md`):**

| Test | Compared | Result |
|---|---|---|
| A. Early training, later scoring | camera and forecast correction trained only on strips before Oct 29; 50/50 average against the forecast, wind-wave height, 7,855 strips at 67 unseen buoys from Oct 31 | **Met.** 0.230 against 0.260 m (−3.07 cm [−5.03, −1.80]); better in 32 of 39 groups and in every fold; −4.49 cm with groups weighted equally. The camera alone is 0.293 m. |
| B. Swell from the water below the horizon | water crop + wind against wind alone, swell-band height (0.030–0.095 Hz) | **Not met.** +0.24 cm [−2.23, +2.02]. The whole view + wind is 18 cm better than the crop + wind (0.704 against 0.886 m). |

A is retrospective: ERA5 is a reanalysis, and the scoring window had been looked at
before. It shows that the camera's added value is not from training on dates inside the
scoring window; it is not a forecast claim. B shows that these encoders, trained on full
frames, keep little swell signal in the cropped water. It does not show that the water
pixels lack it.

### 5. New measurements from the buoy cameras (2026-11-19 to -20)

| Bar | Compared (out of fold, 38,000-40,600 strips) | Result |
|---|---|---|
| M-S1, surface slope | frozen camera + wind + geometry against wind + geometry | **Met.** 0.0089 against 0.0102 (13 % less error), lower at 36 of 43 groups. The camera alone (0.0093) also beats wind + geometry. |
| Water-only control | the same probe with only the water below the horizon | Keeps about half the gain: half the signal is water texture, half the horizon's motion. |
| M-S2, slope fine-tune | fine-tuned against frozen | **Not met.** 0.0086 against 0.0084; the recipe never really trained the backbone. |
| M-S3, stable fine-tune | a recipe that does train the backbone | **Not met.** Level with frozen (0.0084 each). Frozen features are enough. |
| T5-W, virtual anemometer | camera wind speed against ERA5 10-m wind | **Met.** 0.96 against 1.27 m/s (−0.31 [−0.44, −0.10]); gust 1.14 against 1.57 m/s. |
| T5-V, visibility curve | R² by frequency band | The camera sees every band from 16 s up and adds to wind from 19 s up: swell band R² 0.47, 6.5-10.5 s 0.78, 3.9-6.5 s 0.86. **The camera sees further into the swell than expected.** |

### 6. The papers' own networks on our buoy strips (sealed 12 buoys)

| Comparison | Ours | Paper's network | Difference [95 %] | Bar |
|---|---|---|---|---|
| Choi, calm vs rough (3,185 strips) | 99.22 % | VGG16 94.66 % | +4.55 points [2.08, 8.61] | met |
| Kim, five classes (4,791 strips) | 70.17 % | snapshot CNN 50.09 % (46.48 % with Kim's own options) | +20.1 points [12.8, 26.2] | met |
| Yang, H_s (3,419 strips, 11 buoys) | 0.418 m | 0.514 m | −0.096 m [−0.122, −0.063] | met |
| Yang, with minus without attitude | | 0.514 against 0.495 m | +0.020 m [0.007, 0.031] | attitude does not help |

- **Choi's controls show the gain comes from our design, not our backbone.** DINOv3
  inside Choi's pipeline scores 94.1 %, level with VGG16. The gain comes from whole views
  at 270 × 480 px instead of lower halves at 224², one model across six views (worth
  about 3-5 points), and training on every sea state with wind-wave and swell heights.
- **Kim's snapshot CNN barely leaves the lowest class.** Always naming that class scores
  46.3 %.
**The same comparisons over the five grouped folds** (all 79 buoys, out of fold; 2026-11-20):

| Comparison | Strips, groups | Ours | Paper's network | Difference [95 %] | Bar |
|---|---|---|---|---|---|
| Choi, calm vs rough | 23,864, 43 | 98.47 % | VGG16 95.53 % | +2.94 points [1.55, 4.00] | met |
| Kim, five classes | 27,523, 43 | 66.61 % | 52.72 % (47.87 % with Kim's options) | +13.9 points [10.8, 18.4] | met |
| Yang, H_s | 21,150, 41 | 0.498 m | 0.564 m | −0.066 m [−0.085, −0.046] | met |
| Yang, with minus without attitude | 21,150, 41 | | 0.564 against 0.587 m | −0.022 m [−0.041, +0.004] | not shown to help |

- **The folds confirm every sealed-buoy win,** by somewhat smaller margins. No paper's
  recipe was tuned on the folds.
- **Attitude has no consistent effect on Yang's network.** It was 0.020 m worse on the
  sealed buoys and 0.022 m better (interval reaching zero) on the folds.

### 7. Shore video (WebCOOS, CDIP labels; final run, 3,685 clips, 46 dates)

Four cameras (Currituck, Jennette's, Oak Island east and west), folds of whole dates.
M1-M5 are scored on all 46 dates; M6-M7 only on the 28 confirmation dates, which no
development choice saw. Climatology (the other folds' average label) scores H_s 0.545 m
and T_p 3.24 s.

| Bar | Compared | Result |
|---|---|---|
| M1 | V-JEPA 2 on 64 ordered frames against the middle frame repeated, H_s | **Met.** 0.331 against 0.368 m (−0.036 [−0.062, −0.009]). |
| M2 | the same for T_p | **Met.** 2.79 against 2.96 s (−0.17 [−0.31, −0.05]). |
| M3 | all 16-s windows of the clip averaged against the midpoint window, H_s | **Met, small.** 0.305 against 0.313 m (−0.008 [−0.011, −0.005]). |
| M4 | the same for T_p | **Met, small.** 2.76 against 2.79 s (−0.027 [−0.046, −0.010]). |
| M5 | whole clip + pixel brightness spectrum against whole clip, T_p | **Met.** 2.59 against 2.76 s (−0.18 [−0.32, −0.03]); the gain is the spectrum, not recalibration. |
| M6 | spectrum v2 (14 features) against v1 (10), T_p | **Met, small.** 2.85 against 2.91 s (−0.059 [−0.098, −0.008]). |
| M7 | breaking features added for H_s | **Not met.** +0.002 m. |

- **M1 and M2 are met, but the gain is more frames, not motion.** The same 64 frames
  shuffled do as well as in order (H_s −0.000 m [−0.024, +0.020]), and DINOv3 on the one
  middle frame is level with video or ahead (0.312 against 0.331 m). Motion inside 16 s
  adds nothing measurable to H_s or T_p. (For breaker type the order does matter: Y-B3.)
- **Best shore height:** DINOv3 averaged over all windows of the clip, 0.293 m (R² 0.70);
  V-JEPA 2 over the clip 0.305 m. Interim run on 18 dates: 0.513 m, so the data were
  the limit.
- **Best shore period:** the clip plus all 14 spectrum features, on the confirmation
  dates T_p 2.70 s (R² 0.38) and T_e 1.43 s (R² 0.45), against climatology's 3.52 s and
  1.98 s.
- **Kim's five classes at unseen dates:** still no skill on ordinary days. 777 of 937
  in-range clips are in the lowest class, so always naming it scores 82.9 %; `clip`
  scores 83.4 %. The storm set decides this (T2-A, T2-B, T3-K).
- **Kamagata's measures at Oak Island (1,770 clips, 44 unseen dates):**
  - H_s: `clip` 0.290 m, R² 0.69, Pearson 0.84, against their published 0.586 m, 0.107
    and 0.436 on their own five videos. About half their error.
  - T_p: 3.17 s (spectrum alone 2.70 s) against their 1.42 s, so behind. Their test
    set's T_p spread is not reported, so the period gap is hard to read.
- **Storm days:** a first storm run trained on storm days only, against the bars'
  rules, and is superseded. The corrected run, with all 46 ordinary dates in every
  training fold, lands on 2026-11-21 late evening; its estimates feed T2-A, T2-B and
  T3-K.

Interim run (2026-11-19, 992 clips, 18 dates), for the record: M1, M2 and M5 not met,
M3 and M4 met; H_s 0.513 m; pixel spectrum alone 2.46 s against climatology's 2.72 s.

**Per-wave labels and the papers' video networks** (CDIP individual waves):

| Bar | Compared | Result |
|---|---|---|
| T2-A, T2-B | Kim's classes on mean individual height | **Unassessable.** The sample is too calm: 1,133 of 1,444 clips are below 0.75 m, and only 5 reach 1.75 m. A sealed 24-date storm set (H_s ≥ 2 m) is downloading for this. |
| T2-C | zero-crossing period from the pixel spectrum against climatology | **Not met.** 1.29 against 1.37 s (−0.08 [−0.20, +0.08]). |
| T3-S | our whole-clip read-out against Kamagata's SlowFast head with their physics loss | **Ours ahead.** 0.513 against 1.140 m. But the physics loss, at our target scaling, makes their head predict about 2 m for every clip. **The fair number is without it: 0.513 against 0.653 m (−0.140 [−0.207, −0.061]).** |
| T3-K | Kim's CNN-BiLSTM on shore video | Trained (five folds, five seeds, 430 storm clips on 22 dates); scored with T2-A and T2-B once the storm re-run lands. |

**Ten-minute summary images (test 9, frozen DINOv3 probes; 30,931 still / `timex` / `brt` triples, scored on the held-out year 2025, 10,942 triples on 362 days):**

| Bar | Compared | Result |
|---|---|---|
| T9-A | `timex` (time-exposure) against one still | **Not met.** 0.308 against 0.265 m: the still is better. |
| T9-B | `brt` (brightest pixel) against one still | **Not met.** 0.311 against 0.265 m. |

Averaging over ten minutes removes the texture of individual waves, which the image
encoder reads best. Adding them to the still does not help either.

### 8. Breaker type at unseen beaches (VWBT_9000, Yin et al. 2025)

| Bar | Compared (pooled macro-F1, each beach held out) | Result |
|---|---|---|
| Y-B1 | frozen DINOv3 frames + temporal head against Yin's CRNN | **Met.** 0.840 against 0.653 (+0.187 [0.162, 0.214]), at every beach. |
| Y-B2 | temporal head against an order-free head (DINOv3) | **Not met.** The order-free head is slightly better (0.850). |
| Y-B3 | frozen V-JEPA 2, frames in order against shuffled | **Met.** 0.833 against 0.795 (+0.038 [0.024, 0.052]). The video encoder does use breaking motion, but only ties DINOv3 on single frames (−0.007). |

- **On Yin's own random split** our CRNN re-run reaches 94.0 % (they report 96.1 %) and
  DINOv3 97.8 %. Holding the beach out drops their model to 65 % and ours to 84 %.
- **Offshore physics** (Battjes' thresholds, or a regression on ERA5 waves and beach
  slope) is at chance at a new beach (about 0.34).

### 9. Other signals (Codex's track)

- **Seismic swell (Waimea, 91 held-out dates in 2013): stopped.** Seismic alone 0.535 m
  against climatology's 0.455 m; the registered stop rule closed the camera + seismic
  extension.
- **Duck thermal-mask pilot: mixed.** It helped height on one date (0.816 → 0.718 m) and
  hurt on the other (0.504 → 0.558 m). Period missed too. The FRF data request is on
  hold.

## What worked

- **The six-view strip design:** whole views at full width, one shared encoder,
  usable-view masking, heights in metres with wind-wave and swell heads. This, more than
  any backbone, is what beats the papers.
- **DINOv3 weights with the gentle recipe** (freeze 2 epochs, backbone at 0.1 × learning
  rate).
- **Treating the camera as a wind-sea sensor:** wind-wave height, wind-sea period,
  slope, wind speed and gust all beat ERA5 or the buoy's own instruments at unseen
  buoys.
- **Partition-aware fusion:** camera wind sea plus the wave model's swell lowers
  total-height error, against ERA5 and against ERA5 + GFS-Wave. It holds under the nested
  check.
- **Frozen features with small heads:** they matched fine-tuning for slope and served
  breakers, wind and the visibility curve.
- **Reading the whole shore clip** rather than one 16-s window (small, M3-M4), and more
  shore data: 46 dates halved the interim height error.
- **The pixel brightness spectrum** for shore period (M5, M6): it beats climatology and
  adds to the clip estimate.
- **Training the camera only on earlier dates** still adds to the forecast at later
  dates (HPC follow-up A).

## What did not work

- **Camera alone for total wave height or total period.** The swell is mostly invisible;
  ERA5 wins.
- **Wave direction from the camera.** The strips carry no compass heading (the caption
  is a fixed "90°").
- **The buoy's tilt as an input.** It hurts our model (+0.038 m) and has no consistent
  effect on Yang's (+0.020 m sealed, −0.022 m over the folds). Yang's simulated gain does
  not carry over to real buoys.
- **Larger or newer backbones** (ViT-B/16, frozen encoders): at best level once the
  recipe fits.
- **Continued pretraining on our own images:** not better frozen, worse fine-tuned.
- **Label-free site adaptation (test 6):** helps only the US West Coast fold, and badly
  harms wind-wave height. A few labelled hours at the new buoy work instead.
- **Fine-tuning for slope:** level with frozen features, even with a stable recipe.
- **Training period heads jointly with height:** cost 0.086 m in H_s.
- **Motion inside 16 s of shore video, for height or period:** video beats a repeated
  still only by seeing more frames; shuffled frames do as well. Also a learned record
  pooler (worse than a plain average), breaking features for shore height (M7), and
  temporal heads for breakers (order-free heads do as well).
- **The water below the horizon, cropped, for swell** (HPC follow-up B): no gain over wind.
- **Ten-minute `timex` and `brt` images:** worse than one still.
- **Kim's five classes at shore cameras:** no skill at unseen dates, and too few rough
  days to test fairly (the storm set addresses this).
- **Seismic alone at Waimea:** below climatology.

## Open, and what would change the picture

- **Fresh strips (expected November 27).** The historical package check passed. Score S2
  and test 1 on the frozen window once ERA5 coverage reaches the registered cutoff.
- **Storm set (22 daylight dates, 512 clips; corrected estimate run lands 2026-11-21
  late evening):** T2-A, T2-B and T3-K (Kim's classes and Kim's CNN-BiLSTM, already
  trained) are scored once on it.
- **Merge:** the final shore-video write-up and the storm-set scoring branch merge into
  `main` after the storm scores.
- **Sealed 12 buoys:** reused for many decisions. Headline claims should move to the
  out-of-fold numbers and to fresh strips.
- **Licences:** the Yin clips are CC BY 4.0 on figshare but recorded as CC BY-NC-ND in
  the registry. Settle this before any redistribution.

## Where each result lives

| Work | Branch | Experiment doc |
|---|---|---|
| Currituck stills | `main` | `docs/experiments/currituck_sea_state.md` |
| Buoy strips v0-v2, backbones, grouped CV, fusion, periods | `claude/strips-grouped-cv-085fd2`, `claude/trusting-lalande-4a4e8b` (adds the fresh-strip check) | `docs/experiments/ndbc_buoycam_sea_state.md`, `docs/literature/benchmark_targets.md` |
| Continued pretraining (P1) | `claude/dinov3-continued-pretraining-fc2b1a` | `docs/experiments/domain_pretraining.md` |
| Slope, wind and visibility curve (M-S1 to M-S3, T5) | `claude/wave-measurements-review` | `docs/experiments/wave_measurements.md` |
| Breakers (Y-B1, Y-B2; Y-B3) | `claude/yin-breakers`; `claude/yin-breakers-vjepa` | `docs/experiments/breaker_types.md` |
| Papers' networks on strips | `claude/sensing-strips-papers` (from `codex/sensing-strips`) | `docs/experiments/sensing_paper_bars.md` |
| Forecast fusion, nested | `claude/sensing-model-nested` (from `codex/sensing-model`) | `docs/experiments/sensing_model_bar.md` |
| Site adaptation (test 6) | `claude/site-adaptation` | `docs/experiments/sensing_site_adaptation.md` |
| Fresh-strip confirmation (to be scored) | `claude/fresh-strip-confirmation` | `docs/experiments/fresh_strip_confirmation.md` |
| Shore video M1-M7, storm set | `claude/video-whole-clip-491c90` (final run; not yet merged) | `docs/experiments/webcoos_video.md` |
| Per-wave labels, Kamagata and Kim video, `timex`/`brt` | `claude/sensing-shore-labels` (from `codex/sensing-shore`); storm-set scoring on `claude/sensing-shore-storm` (not yet merged) | `docs/experiments/sensing_per_wave_bars.md`, `docs/experiments/sensing_timex_bars.md` |
| HPC follow-ups: early training, water crop | `main` | `docs/experiments/hpc_buoy_followup.md` |
| Seismic, NWPS | `codex/sensing-signals` | `docs/experiments/sensing_seismic_bar.md` |
