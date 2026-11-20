# Paper outline: the buoy camera as a wind-sea sensor

Drafted November 20, 2026. The retrospective results are on `main` at `a1149f412`.
The fresh-strip protocol and scorer are on `claude/fresh-strip-confirmation` at
`cfda28b22`. [Registered comparison](../docs/experiments/sensing_model_bar.md) · [Reproduce the headline table](../docs/experiments/sensing_forecast_results.md#reproduce) · [Scoring inputs](../data/manifests/ml/forecast-fusion-nested-v1.sha256)

**Recommendation.** Write one paper about the NDBC buoy cameras, built on the
out-of-fold numbers. Keep the shore cameras as one secondary section; it carries the
Kamagata comparison and two negative results. Leave the breaker study (Yin et al.'s
clips) for a short paper of its own. **Deciding principle:** every section must bear on
one claim, and that claim must hold at buoys no model saw.

## How numbers are tagged

Every number below carries its unit, its 95 % interval where the source gives one, and
one of these protocol tags.

| Tag | Protocol | Interval unit |
|---|---|---|
| **S12** | the 12 sealed test buoys (6,183 strips), station split, groups at least 300 km apart. **Retrospective:** these buoys have informed earlier choices | whole buoys |
| **OOF-b / OOF-g** | out of fold: five folds over the 43 station groups, all 79 buoys; each buoy predicted by models that saw neither it nor any buoy within 300 km | whole buoys (b, 1,000 draws) or station groups (g, 2,000 draws) |
| **NEST** | out of fold with nested calibration: the fusion step for fold k is fitted only on predictions from models that never saw fold k. 34,723 strips, 71 buoys, 41 groups | station groups, 2,000 draws |
| **DEV** | shore video: 992 clips on 18 development dates at four WebCOOS cameras, five folds of whole dates | whole dates, 2,000 draws |
| **Y25** | shore stills: train 2024, test on the held-out year 2025 (10,942 triples, 362 days) | whole days, 97.5 % interval |

"On our data" marks every comparison that is not like-for-like: a published number from
other data, labels or splits set beside ours.

## Claim check: what the evidence supports

**Lead claim.** Adding the buoy camera to ERA5, GFS-Wave and measured wind reduced
wind-wave height RMSE by 8% at held-out buoys in the retrospective comparison. Total
height improved by 4.7 mm under the registered weighting. Its equal-group interval
crossed zero. The prospective test will assess both endpoints on later strips.

| Piece of the claim | Evidence | Verdict | Wording to use |
|---|---|---|---|
| Better than the wave model for the wind sea | Wind-wave height: camera 0.2558 m against ERA5 0.3432 m on 36,857 paired strips, camera better by 0.087 m [0.067, 0.110] (OOF-b). The quoted 0.215 against 0.341 m is S12. **Against ERA5 + GFS-Wave corrected with location, depth and the buoy's wind, the camera alone loses:** 0.2544 against 0.2258 m (NEST rows). Against the same forecast without the buoy's wind it is ahead, 0.2544 against 0.2740 m, with a station-group interval of −0.0346 to +0.0038 m for the difference | **Holds against ERA5's wind-sea partition, not against a corrected forecast** | "estimates wind-wave height better than ERA5's wind-sea partition" |
| Better than the buoy's own anemometer | Wind-sea class (four WMO bands): balanced accuracy 0.795 against 0.486 for the anemometer through a fitted power law, +0.31 [+0.27, +0.34] (S12, v1). Slope: camera alone 0.0093 against wind + horizon geometry 0.0102, −0.0009 [−0.0013, −0.0004] (OOF-g, M-S1) | **Holds for wind-sea state and slope** | "reads the wind-sea state and the wave slope better than the buoy's anemometer does" |
| Wind speed 0.96 against 1.27 m/s | The anemometer is the **reference** here; 1.27 m/s is ERA5's 10-m wind, line-corrected (OOF-g, T5-W: −0.31 m/s [−0.44, −0.10]) | **Does not support "better than the anemometer"** | "can stand in for the anemometer, with a quarter less error than ERA5's wind" |
| Fusion improves total wave height | Against ERA5 + buoy wind: 0.288 → 0.267 m, −0.021 m [−0.031, −0.012] (OOF-b, S2); the nested check gives 0.2883 → 0.2667 m. Against ERA5 + GFS-Wave + buoy wind: 0.2670 → 0.2623 m, −0.0047 m [−0.0083, −0.0007] (NEST, test 1). That interval clears zero by 0.7 mm. With equal group weights it reaches +0.0027 m. **Without the buoy's wind the total-height gain is not shown:** −0.0038 m [−0.0082, +0.0042]. On S12, Test 3's ridge left the total level: 0.258 against 0.259 m, −0.001 [−0.052, +0.049]. Fresh strips are not scored yet | **Holds against ERA5 + wind (7 %). Thin against the stronger forecast (1.8 %). Not yet confirmed on fresh strips** | "lowers total-height error by 7 % against ERA5 corrected with the buoy's wind; against a stronger forecast the gain is 1.8 %, small but above zero on the registered unit" |
| The four papers' networks beaten on the sealed buoys and the five folds | Choi, Kim (snapshot) and Yang: yes, on both S12 and OOF-g. **Kamagata's network was re-run only on shore video (DEV), not on buoy strips.** Kim's headline video network (CNN-BiLSTM, 93.2 %) has not been run yet. Kim's snapshot CNN barely beats always naming the lowest class (46.3 %) | **Three papers on buoys; Kamagata on shore development dates only** | "re-running three papers' networks on our strips and Kamagata's on our shore video" |
| "At buoys it has never seen" | True of training for S12 and OOF. But DINOv3-gentle was chosen with S12 among the evidence, and all strips are daylight or civil twilight (sun at most 6° below the horizon) in autumn 2026 | Holds, with those limits stated | Lead with OOF; "in daylight and twilight" |

## Title options

1. **(Recommended)** *The buoy camera as a wind-sea sensor: evaluation at unseen NDBC
   stations and partition-aware fusion with wave models*
2. *What a buoy camera sees: wind-sea height, period, slope and wind from NDBC imagery
   at unseen stations*
3. *Six views of the wind sea: buoy-camera strips benchmarked against wave models and
   published networks*
4. *Replacing the modelled wind sea: fusing buoy-camera estimates with ERA5 and
   GFS-Wave*

Option 1 names the instrument and both tests, and it makes no "better than" claim. That
claim waits for the fresh strips. Option 4 fits only if the fresh strips confirm the
fusion.

## Abstract skeleton

1. Problem: wave models carry the swell well and the local wind sea less well. Buoy
   cameras photograph the sea hourly at 79 NDBC buoys, and NDBC keeps the images only
   about 72 h.
2. Data: 41,091 six-view strips from autumn 2026, each labelled by the spectrum of the
   buoy that carries the camera.
3. Protocol: five folds over 43 station groups, each buoy predicted by models that saw
   neither it nor any buoy within 300 km. Bars were fixed before scoring.
4. Primary result. Across 34,723 eligible strips at 71 buoys in 41 groups, the camera
   reduced wind-wave height RMSE from 0.2258 to 0.2079 m against the forecast with
   measured wind. The paired station-group interval for the change was −0.0253 to −0.0088 m.
5. Total significant height RMSE fell from 0.2670 to 0.2623 m. With equal weight per
   geographic group, the interval for the change included zero. Report the frozen-model
   fresh-strip result alongside these retrospective estimates when it is available.
6. Result 3, published networks: re-run on the same strips, three papers' networks
   score below ours on their own measures. The gain comes from the input design, not
   the backbone.

## Contributions

1. **A same-hull benchmark and an unseen-station protocol.** 41,091 strips from 79 NDBC
   buoys in 43 station groups, 2026-09-23 to 2026-11-14. Each strip is labelled by its
   own buoy's spectrum. There is a sealed 12-buoy test and five grouped folds; every bar
   was committed before its run. *Test:* protocol check R trained the same recipe on a
   random split of the same strips, the protocol of Choi and Kim. H_s RMSE moved only
   from 0.450 to 0.433 m [0.378, 0.497]. Kim's five-class accuracy did not move: 68.8 %
   [65.7, 71.8] against 69.4 % (S12). So the split explains little of the gap to the
   published numbers.
2. **The camera measures the wind sea at unseen buoys.** All from DINOv3 ConvNeXt-Tiny
   with the gentle recipe. *Tests:*
   - wind-wave height 0.2558 against ERA5 0.3432 m on 36,857 paired strips, −0.087 m [−0.110, −0.067] (OOF-b);
   - wind-sea energy period Tm-1,0 0.55 against 0.75 s, −0.21 s [−0.24, −0.17] (OOF-b;
     reported beside bar S3, so found, not confirmed);
   - RMS slope 0.0089 with wind + geometry against 0.0102 without the camera,
     −0.0013 [−0.0017, −0.0008] (OOF-g, bar M-S1);
   - wind speed 0.96 against ERA5's 1.27 m/s, −0.31 m/s [−0.44, −0.10] (OOF-g, bar
     T5-W).
   The same model loses total H_s to ERA5: 0.514 against 0.292 m, +0.22 m [0.18, 0.26]
   (OOF-b).
3. **Partition-aware fusion with wave models.** NDBC's partitions add as energies
   (WVHT² = swell² + wind-wave², within rounding on all 295,547 records). A gated swap
   therefore replaces the model's wind sea with the camera's and keeps the model's
   swell. *Tests:*
   - bar S2 against ERA5 + buoy wind: H_s 0.288 → 0.267 m, −0.021 m [−0.031, −0.012];
     wind-wave 0.293 → 0.240 m, −0.053 m [−0.067, −0.039] (OOF-b). The nested check
     agrees: 0.2883 → 0.2667 m, −0.0215 m [−0.0306, −0.0128] (whole buoys).
   - test 1 against ERA5 + GFS-Wave + buoy wind: wind-wave 0.2258 → 0.2079 m, −0.0179 m
     [−0.0253, −0.0088]; H_s 0.2670 → 0.2623 m, −0.0047 m [−0.0083, −0.0007] (NEST).
   - permuting the camera's outputs within buoy and forecast quartile removes the gain:
     H_s −0.0190 m [−0.0303, −0.0104], intact minus degraded (NEST). So the gain comes
     from what each image shows that hour.
4. **Published networks re-run on the same strips and folds.** *Tests (OOF-g):*
   - Choi's VGG16: +2.94 points [+1.55, +4.00] on calm against rough;
   - Kim's snapshot CNN: +13.89 points [+10.83, +18.40] on five classes;
   - Yang's ResNet-18 + attitude network: −0.066 m [−0.085, −0.046] on H_s RMSE.
   Choi's controls (S12) locate the gain in the input design, not the backbone. DINOv3
   inside Choi's pipeline scores −0.60 points [−1.24, −0.16] against VGG16. Our six-view
   design adds +5.15 points [+2.40, +9.63] over that. Yang's question on real buoys:
   horizon attitude has no consistent effect, +0.020 m [+0.007, +0.031] (S12) and
   −0.022 m [−0.041, +0.004] (OOF-g). Kamagata's SlowFast head on shore video, without
   its physics loss: 0.653 against our 0.513 m, −0.140 m [−0.207, −0.061] (DEV).
5. **Registered negative results.**
   - Once each backbone has a recipe that suits it, backbones differ by about 0.01 m.
   - Continued self-supervised pretraining on our own views does not help.
   - Fine-tuning for slope adds nothing over frozen features: +0.00004 [−0.00012,
     +0.00017] (OOF-g).
   - Motion inside 16 s of shore video adds nothing for H_s: −0.002 m [−0.079, +0.075]
     (DEV).
   - Ten-minute `timex` images are worse than one still: +0.043 m [+0.026, +0.061]
     (Y25).

## Section structure

### 1. Introduction
- Wave models carry swell from far away. The local wind sea depends on local wind, which
  models and reanalyses miss at short scales. A camera on the hull sees the short, steep
  waves.
- The gap in camera studies: no instrument-referenced, multi-station test at held-out
  stations against a wave-model baseline among the papers reviewed. The landscape review
  (§1.8) found these papers lack non-vision baselines. This is a statement about the
  reviewed set, not a priority claim (see "Priority claims").
- Motivation from our own shore pilot (Currituck, one paragraph). A camera judged
  against a buoy 9.2 km away scored below the WIS hindcast: balanced accuracy 0.73
  against 0.77 in 2025. With the hindcast it added macro-F1 +0.038 [0.007, 0.065]. This
  led to labels from the same hull.
- Contributions 1-5.

### 2. Related work
- **The four targets** (full texts read): Choi 2020 (NDBC buoy cameras, same buoys and
  days in train and test); Kim 2023 (one fixed camera, 30 one-hour sessions, random
  frames, seafloor AWAC mean height); Yang 2026 (Unity simulation, ship camera + IMU);
  Kamagata 2026 (arXiv:2607.11998; shore video, labels traceable to no instrument).
- **The five 2026 *Ocean Engineering* sensing papers Steven supplied**
  (`docs/experiments/sensing_paper_findings.md` on `codex/sensing-signals`). Each is
  placed by sensor and split; none is a like-for-like comparator:
  - Y. Liu, Xu et al.: coastal camera, SWAN labels, 0.27 m;
  - Ahammed and Dang: laboratory polarimetry;
  - Li et al.: stereo;
  - G. Zhang et al.: Beaufort classes;
  - H. Liu, Qin and Ti: bridge-site video against a wave gauge.
- Single-site shore cameras: Xu et al. 2024 and Buscombe et al. 2020.
- Other sensors as context: X-band radar 0.358 m (Kwon et al. 2023), Sentinel-1 about
  0.3 m (Quach et al. 2021), and the SAR self-supervision results that motivated frozen
  foundation backbones.
- Uncertainty: buoy-level split conformal (Lee, Barber and Willett 2023), beta-NLL
  (Seitzer et al. 2022).

### 3. Data and labels
- **Strips.** 94,187 hourly strips archived at 80 buoys, 2026-09-23 to 2026-11-14.
  Each strip is six views of 480 × 300 px with a burned-in caption, cut before
  packaging (the bottom 30 rows). Light rules: sun at most 6° below the horizon; at
  least 3 usable views (mean grey ≥ 20, standard deviation ≥ 4). 41,091 strips at 79
  buoys are kept; 45,944 were too dark.
- **Labels from the same hull.** NDBC realtime `.spec` WVHT, taken as the mean of the
  records centred within 30 min of the strip (ADR 0010). One record's 90 % interval is
  about −10 % to +15 % of H_m0 (Earle 2003).
  - Wind-wave and swell heights come from NDBC's separation frequency.
  - Periods, direction (Kuik et al. 1988) and RMS slope (46 bands, deep water) come from
    the realtime spectral files.
  - Wind and gust are the buoy's anemometer values.
  - These are realtime values, not the quality-controlled annual files.
- **The partition check.** WVHT² = SwH² + WWH² within rounding on 295,547 records at
  the 79 buoys. ERA5's three heights add the same way within 1 % at 91 % of buoy-hours.
- **Comparators.**
  - ERA5: the nearest sea grid point (median 19 km, at most 44 km), line-corrected on
    each fold's training strips; still the preliminary ERA5T.
  - GFS-Wave: station bulletins, 212 cycles, the latest nominal issue at or before the
    strip and the nearest valid hour within 30 min (total height on 38,093 of 41,091
    rows).
  - ERA5 10-m wind and gust.
- **Shore data (section 6.4 only).** The WebCOOS video sample at four cameras with
  CDIP 433, 243 and 200 labels; those buoys sit 20-30 km offshore in 13-21 m of water.
  WebCOOS still / `timex` / `brt` triples at the same four cameras.

### 4. The strip model
- `MultiViewRegressor`: one shared DINOv3 ConvNeXt-Tiny reads each view at 270 × 480
  px. A one-layer transformer runs over the usable views, followed by a masked mean.
  Three log-height outputs (total, wind-wave, swell), Huber loss on standardised logs.
  A distribution head (beta-NLL, β = 0.5) gives the predicted spread.
- The gentle recipe: the backbone is frozen for 2 epochs, then trained at 0.1 × the
  head's learning rate. 10 epochs, no crop or rotation (the horizon keeps the buoy's
  real tilt). The epoch is chosen on validation H_s. Three seeds are averaged in log
  space.
- Frozen-feature probes: ridge regressions on the 256 pooled features, for slope, wind,
  gust and the energy in each spectral band.
- Fusion: the gated partition swap, H_s² = T² + k·g·(b·C² − W²). A learned gate g on
  five visibility features falls back to the comparator where the camera cannot see.
  The variance-weighted arm is reported as secondary.
- Uncertainty: buoy-level split conformal 90 % intervals.

### 5. Protocols
- **Sealed 12 buoys.** The closest test-train pair is 322 km apart. These buoys have
  served Tests 2, 2b and 3 and bars B1-B3, so they now look back and do not confirm.
- **Five grouped folds.** The 43 groups are balanced by strip count. The US West Coast
  group (18 buoys, 9,440 strips) forms fold 0 on its own. Each fold's validation groups
  are drawn from its training folds to match their swell share. This is the protocol
  for choosing and for headline numbers.
- **Nested fusion.** 30 inner runs: one per pair of outer folds, trained on the other
  three, seeds 0-2, 21.1 GPU-hours. The inventory shows each calibration prediction
  comes from a model that neither trained nor validated on the outer fold.
- **Fresh strips.** Strips archived after 2026-11-14 16:00 UTC, with ERA5 to
  2026-11-21 16:00 UTC; about 6,000 strips at about 70 buoys are expected. Each strip is
  predicted by its own buoy's fold models and fused by that fold's combiners, with no
  refit. C4a retains the S2 test. C4b repeats the stronger forecast comparison with
  its registered station-group intervals.
- **Bars.** Each test names its comparator, metric, unit of resampling and pass mark
  before any run. The doc and commit are given in each table caption. Seeds: three per
  trained arm.

### 6. Results
- **6.1 What the camera sees alone** (Table 3, Figures 4-5): wind-sea height, period,
  slope and wind against ERA5 and the anemometer; total height, swell, period and
  direction, where ERA5 wins.
  - The visibility curve (T5-V): the camera adds to wind at every band from 0.0525 Hz
    (19 s) up. Swell band (10.5-33 s) R² 0.47 [0.33, 0.61] (OOF-g). The mechanism is not
    separated: the encoder was trained on swell height too.
  - The horizon as a motion sensor: roll follows the buoy's slope within buoys,
    Spearman 0.56 [0.54, 0.58], positive at 75 of 76 buoys.
- **6.2 Fusion** (Table 4, Figure 6): S2, the nested check, test 1 with its
  sensitivities, the degraded-camera control, and the result without the anemometer.
- **6.3 Published networks on our strips** (Table 5): S12 and OOF-g side by side,
  Choi's controls, Yang with and without attitude.
- **6.4 Shore cameras** (Table 8):
  - single still at four cameras: 0.265 m against climatology's 0.613 m (Y25);
  - whole-clip video: 0.513 m (DEV);
  - Kamagata's head re-run;
  - the pixel brightness spectrum for T_p: 2.46 s against climatology's 2.72 s (DEV;
    no interval reported for this difference).

### 7. Comparison with published work
Two kinds of comparison, never mixed:
- **Like-for-like:** their network, our data, our protocol (Table 5).
- **Side-by-side, on our data:** their published number against ours (Table 6).
  - Ahead of Choi on our data: 98.4 % calm against rough (OOF-b) against their 84.05 %
    on the same buoys and days.
  - Behind Kim's snapshot accuracy on our data: 66.7 % [63.9, 69.6] (OOF-b) against
    78.3 %. Protocol check R shows the split is not the gap.
  - No counterpart yet for Kim's 93.2 % video result.
  - No Yang number transfers: simulation only.
  - Against Kamagata at Oak Island (DEV, 336 clips, CDIP 200 labels), RMSE 0.661 m
    against their 0.586 m: behind. R² 0.28 against 0.107 and Pearson 0.55 against 0.436:
    ahead. T_p 2.79 s against 1.42 s: behind. These numbers have no intervals.

### 8. Negative results worth reporting
| Question | Bar | Result (protocol) | Source |
|---|---|---|---|
| Does a bigger or newer backbone help? | B1, B2, B3, B2-CV | DINOv3 with ImageNet's recipe: +0.039 m (S12). Gentle DINOv3 against ImageNet: −0.010 m [−0.024, +0.001] (S12); −0.019 m [−0.028, −0.011] on H_s but −0.002 m [−0.008, +0.004] on wind-wave (OOF-b). ViT-B/16 gentle: +0.005 m [−0.009, +0.020] (S12) | `ndbc_buoycam_sea_state.md` |
| Does continued pretraining on our views help? | P1 | Frozen: H_s −0.010 m [−0.027, +0.002]. Gentle: +0.028 m [+0.012, +0.040], worse (S12). About 9 GPU-hours | `domain_pretraining.md` |
| Does measured attitude help (Yang's question)? | V2-1, Yang rows | Our model: +0.038 m [+0.018, +0.057], worse (S12). Yang's network: +0.020 m [+0.007, +0.031] (S12); −0.022 m [−0.041, +0.004] (OOF-g) | `ndbc_buoycam_sea_state.md`, `sensing_paper_bars.md` |
| Does fine-tuning beat frozen features for slope? | M-S2, M-S3 | Unstable recipe: +0.0002 [+0.0001, +0.0004]. Stable recipe: +0.00004 [−0.00012, +0.00017] (OOF-g) | `wave_measurements.md` |
| Do period heads trained with height help? | S3 harm check | H_s +0.086 m [+0.068, +0.110], worse (OOF-b) | `ndbc_buoycam_sea_state.md` |
| Does motion inside 16 s of shore video help H_s? | M1 | 0.533 against 0.535 m, −0.002 m [−0.079, +0.075] (DEV) | `webcoos_video.md` |
| Do ten-minute `timex` / `brt` images beat one still? | T9-A, T9-B | +0.043 m [+0.026, +0.061] and +0.045 m [+0.035, +0.056], both worse (Y25) | `sensing_timex_bars.md` |
| Can the camera give total period or direction? | S3 P1-P3 | T_p 3.44 against 1.70 s, +1.75 s [+1.41, +2.07]; direction 75.8° against 13.8°, +62.0° [+58.5, +65.6] (OOF-b) | `ndbc_buoycam_sea_state.md` |

### 9. Limits
- **Swell.**
  - Out-of-fold swell RMSE 0.717 m (R² 0.53).
  - Total H_s camera alone: 0.514 against ERA5's 0.292 m (OOF-b).
  - The camera under-reads swell-dominated seas.
- **Direction.** No strip carries a compass heading: the caption's "90°" is a fixed
  label. Sun-relative direction gives 75.8° error against ERA5's 13.8°.
- **The West Coast fold.** H_s 0.70 m. Its models never trained on that swell climate
  (swell share 0.79 against 0.57 in training).
- **Conformal coverage at 11 buoys.**
  - The 90 % intervals cover 0.904 of H_s over all strips, mean width 1.83 m (OOF).
  - 11 of 79 buoys fall below 0.80, none below 0.50. Six are West Coast buoys the
    camera under-reads by 0.6-0.7 m; five it over-reads by about 0.4 m.
  - The guarantee is an average over new buoys, not a promise at each one.
- **Light and season.** Daylight and civil twilight only; 7.5 autumn weeks; realtime
  labels.
- **Reuse of the sealed buoys.** S12 informed model choice; OOF and the fresh strips
  carry the claims.
- **Forecast timing.** GFS-Wave is matched at nominal issue time, so this is a
  retrospective comparison; an operational claim needs publication time enforced.
- **Fusion depends on the anemometer.** Without the buoy's wind, the total-height gain
  against the stronger forecast is not shown.
- **Proxies in the paper re-runs.**
  - Kim's classes on strips use 0.631 × H_s for mean height. The measured ratio at the
    CDIP shore buoys is 0.583.
  - Yang's attitude here is the image horizon, not an IMU.
  - Kamagata's physics loss collapses at our target scaling; the fair number is
    without it.
- **Shore.** Interim: 18 development dates, with CDIP buoys 20-30 km offshore.

### 10. Conclusion
The camera covers the wind sea and the wave model covers the swell, so fuse by
partition. Close on what would extend it: night, other seasons, a compass heading in the
caption, and quality-controlled labels.

## Planned tables and figures

| # | What it shows | Source (doc on branch; run) |
|---|---|---|
| T1 | Data: strips, buoys, groups, period, light rules, label definitions, comparators and their coverage | `ndbc_buoycam_sea_state.md` (Ground truth, Strips, Splits) on `claude/trusting-lalande-4a4e8b`; package `ndbc_buoycam_heights_v2cv` |
| T2 | Protocols: S12, five folds (per-fold buoys, strips, swell share), nested, fresh strips; resampling unit per test | same doc (Grouped cross-validation, Stage 2); `sensing_model_bar.md` on `claude/sensing-model-nested` |
| T3 | Camera alone against ERA5 and the anemometer, per quantity, OOF with S12 beside: H_s, wind-wave, swell, WWP, wind-sea Tm-1,0, T_p, Tm-1,0, direction, wind, gust, slope | B2-CV and S3 tables (`runs/buoycam-cv-stage1-20261119`, `runs/buoycam-cv-spectral-20261119`); `wave_measurements.md` on `claude/wave-measurements-review` (M-S1, T5; `runs/measurement-t5/dinov3-s012/`) |
| T4 | Fusion: S2 (OOF-b and -g), the nested S2 check, test 1 (NEST) with whole-buoy and equal-group sensitivities, degraded camera, no anemometer, variance-weighted arm | `ndbc_buoycam_sea_state.md` (Stage 2; `runs/buoycam-cv-fusion-20261119`); `sensing_forecast_results.md` on `claude/sensing-model-nested` (`runs/forecast-fusion-nested-v1/`) |
| T5 | Published networks re-run: Choi, Kim (both recipes), Yang (with and without attitude), S12 and OOF-g; Choi's two controls | `sensing_paper_bars.md` on `claude/sensing-strips-papers` (`runs/*-station-three-seed-comparison.json`, `runs/*-folds-comparison.json`) |
| T6 | Published numbers beside ours, each row marked "on our data", with the protocol difference | `benchmark_targets.md` on `claude/trusting-lalande-4a4e8b`; `docs/literature/papers/*.md` |
| T7 | Negative results (section 8 table) | as cited per row |
| T8 | Shore cameras: M1, M3, M5, T3-S (with and without the physics loss), Kamagata's measures at Oak Island, T9-A/B | `webcoos_video.md` on `claude/video-whole-clip-491c90`; `sensing_per_wave_bars.md`, `sensing_timex_bars.md` on `claude/sensing-shore-labels`. **Replace with the final run's numbers** |
| F1 | Example strips across the four WMO classes and light conditions (NDBC images are public domain) | package `ndbc_buoycam_heights_v2cv` |
| F2 | Map: 79 buoys coloured by fold, 43 groups, the sealed 12 marked | `folds.csv` of `ndbc_buoycam_heights_v2cv` |
| F3 | Model and fusion diagram: shared encoder, view transformer, masked mean, three heads; the partition swap with its gate | `wave_analysis.models.wave_heights`; `scripts/buoycam_partition_fusion.py` |
| F4 | Per-buoy RMSE, camera against ERA5, for wind-wave and total height (dumbbells, OOF) | `oof_dinov3-gentle.csv` with each fold's ERA5 line |
| F5 | Visibility curve: R² per spectral bin for camera, wind, and camera + wind, with intervals | `runs/measurement-t5/dinov3-s012/` |
| F6 | [Historical pooled and group effects](figures/buoycam-fusion/forecast-fusion.svg). Supporting results by fold: comparator, gated swap, variance-weighted (NEST), with the S2 check | `runs/forecast-fusion-nested-v1/rows.parquet` |
| F7 | Conformal coverage per buoy, the 11 buoys below 0.80 marked, with mean bias | B2-CV outputs in `runs/buoycam-cv-stage1-20261119` |
| F8 (supplement) | Swell share against under-calls (Q-M8); horizon roll against slope | `scripts/buoycam_swell_check.py`, `scripts/buoycam_horizon.py` outputs |

WebCOOS frames go in no figure until WebCOOS agrees to their publication (see below).

### Historical figure and buoy 41040 image review

**Draft figure caption.** Adding buoy-camera estimates to ERA5, GFS-Wave and measured wind reduced pooled wind-wave height RMSE from 22.58 to 20.79 cm (7.9%) across 34,723 strips at 71 buoys. Panel A shows RMSE reductions of 1.79 cm for wind-wave height (95% interval 0.88 to 2.53 cm) and 0.47 cm for total significant height (0.07 to 0.83 cm). The intervals come from 2,000 paired resamples of whole geographic groups. With equal weight per group the total-height interval included zero. Each bar in panel B is one group's wind-wave reduction. Of 41 groups, 27 improved and 14 worsened. The median group improved by 1.57 cm. Panel C is reserved for the frozen fresh-strip test. [Vector figure](figures/buoycam-fusion/forecast-fusion.svg) · [Raster figure](figures/buoycam-fusion/forecast-fusion.png) · [Group-level data](figures/buoycam-fusion/station-groups.csv)

**Why the worst group matters.** At buoy 41040, 631 eligible strips spanned 51 UTC dates from September 23 to November 12. The forecast alone had the lowest wind-wave RMSE of all 41 groups (7.49 cm, group median 20.65 cm). The camera alone had an RMSE of 20.91 cm, close to the group median of 21.65 cm, and underestimated height by 14.82 cm on average. Fusion lowered the forecast estimate on 569 strips and increased RMSE to 16.18 cm. The largest group improvement was 8.48 cm at g46084 (483 strips).

| Measured wind at 41040 (m/s) | Strips | UTC dates | Forecast RMSE (cm) | Fusion RMSE (cm) | Fusion bias (cm) |
|---|---:|---:|---:|---:|---:|
| ≤4.43 | 164 | 23 | 5.45 | 5.01 | −1.57 |
| >4.43 to 6.00 | 189 | 34 | 7.16 | 13.08 | −10.51 |
| >6.00 to 7.00 | 136 | 31 | 7.70 | 17.36 | −14.09 |
| >7.00 | 142 | 23 | 9.47 | 24.85 | −20.50 |

Bounds are station-specific sample quartiles, displayed rounded. Tied wind readings share a bin. A date can appear in several bins. Bias is prediction minus the buoy reference. These are exploratory summaries for a station selected for its large loss.

**What the strips show.** I compared the three largest fusion underestimates on different UTC dates with three accurate same-buoy strips. Each control had absolute fusion error ≤10 cm, came from a different day than its case, and had the same illumination class. Controls were selected without reuse, first by closest observed wind-wave height, then by closest sun elevation. On October 4 and October 24 the buoy measured 1.80 m in both strips. Fusion estimated 1.18 and 1.80 m respectively. The October 4 strip shows open water and a visible horizon in most views, with one hazy sector. The accurate October 24 strip includes strong sun glitter and a conspicuously tilted horizon. The October 25 underestimate also has glare. All six strips met the image-quality threshold in all six views. [Six original panoramas with measurements and predictions](figures/buoycam-fusion/41040-strips.png)

Across the full station sample, the 63 largest underestimates had a median observed wind-wave height of 1.40 m, versus 0.43 m for the 63 smallest absolute errors. Sixty of the 63 large underestimates were in daylight. At winds above 7 m/s, the rank correlation of fusion error with median absolute roll was −0.01 and with median absolute vertical horizon displacement was −0.02. Mean view brightness still had a correlation of −0.43. The orientation measures use valid horizon fits relative to each view's usual mounting. Brightness also varies with illumination and sea state, so the images alone cannot establish a cause. [Station measurements, image IDs, and selections](figures/buoycam-fusion/41040-strips.csv)

**Check at other buoys.** Applying the same >7 m/s cutoff to the other 70 buoys gave 13,034 strips in 40 groups. In this subset, adding the camera reduced pooled RMSE from 33.81 to 31.28 cm and improved 26 groups. A blanket wind-speed cutoff would discard useful information. This cutoff came from the selected station's wind distribution, so this cross-site check is exploratory. [Other groups above 7 m/s](figures/buoycam-fusion/other-groups-above-7mps.csv)

**Hypothesis for a later test.** The camera gets too much weight where the local forecast is unusually accurate and the camera reads low. Estimate each arm's reliability from training groups using forecast height, measured wind and camera–forecast disagreement. Select any weighting change inside the training folds and test on independent stations or dates. Keep the registered confirmation fixed. [Figure and image-selection script](../scripts/buoycam_fusion_review.py)

**Historical follow-up.** Fitting that rule on geographically separate calibration strips reduced pooled RMSE to 19.92 cm. A constant camera weight scored 20.10 cm, the variance blend 20.20 cm and the registered gate 20.79 cm. Adding a week of local wave measurements produced no overall improvement. This historical comparison was motivated by a station already inspected. [Protocol, intervals and buoy 41040 results](../docs/experiments/wind_wave_reliability.md) · [Fold and group scores](wind-wave-reliability/scores.csv)

## Still to land before submission

| Item | Date | State on 2026-11-20 | What it decides |
|---|---|---|---|
| **Fresh-strip confirmation** | Once ERA5 wave fields, wind and gust cover November 21 at 16:00 UTC, expected around November 27 | Protocol and scorer fixed on `claude/fresh-strip-confirmation` (`cfda28b22`). C4a tests ERA5 + wind. C4b tests ERA5 + GFS-Wave + measured wind. | Report the registered endpoints on the same frozen time window |
| **Final shore-video run** | Starts on its own after the download checks; about **2026-11-21 to -22** | The sample download ended 2026-11-20 10:19 UTC with the log line "video sample done (rc=1)". Check the exit code and the date count before reading results | M1-M7 on all dates; M6-M7 only on confirmation dates. Replaces every DEV number in T8 |
| **Storm-set Kim comparison** | **No date yet** | Per the video session's note (09:20 UTC), the storm download fetched 1 of the 24 planned dates with video (1.6 GB). The storm-date estimate run exited with code 1. **Not on track; needs the video session** | T2-A, T2-B and T3-K: Kim's five classes and his CNN-BiLSTM (his 93.2 % headline) on shore video. Until then, section 7 says Kim's video result has no counterpart |
| **Citable code and data** | Before submission | Retrospective code and results are on `main`. The fresh-strip tools are on their own branch. | Tag the version used for the final results and archive the release with a DOI |
| Priority-claim search | Before the first full draft | Not done (see below) | Whether any "first" sentence can be written |
| Yin licence | Only if the breaker study joins | Open: figshare says CC BY 4.0; the project registry records CC BY-NC-ND 4.0 | Redistribution of derived clips or features |
| Quality-controlled relabel | When NDBC's annual files appear (2027) | Not possible yet | A revision-stage robustness check; name it as a limit now |

## Priority claims: what may and may not be said

Checked against the four target papers' notes, the five supplied 2026 *Ocean
Engineering* papers (summaries in `sensing_paper_findings.md`), `benchmark_targets.md`
and the landscape review.
- **May say:** "Among the camera studies we reviewed, none tests at stations held out
  of training against a wave-model baseline." The landscape review (§1.8) supports this
  for its set. Of the five 2026 papers, Y. Liu, Xu et al. train *against* SWAN, so the
  model is their label, not their baseline. Confirm against the full texts before
  writing it.
- **May say:** "We re-run Choi's, Kim's and Yang's networks on our strips and folds."
  This is a description, not a priority claim.
- **May not say without a dedicated search:** "first", "novel", "the only buoy-camera
  archive" (the landscape review's §6 wording), or "first fusion of camera imagery with
  wave models". Y. Liu, Xu et al. combine a camera with a wave model in another way.
  The shore-video doc's sentence "no published camera work compares video and a single
  frame of the same clips" is also unchecked.
- **May not say:** "better than the anemometer" for wind speed, or "better than the wave
  model" for the wind sea without naming ERA5's partition (see the claim check).

## Venues

1. **(Recommended) *Ocean Engineering*.** Choi 2020, Kim 2023 and Yang 2026 appeared
   there, as did the five 2026 sensing papers. The like-for-like re-runs answer that
   literature directly.
2. ***Journal of Atmospheric and Oceanic Technology* (AMS).** Its focus is
   instruments and observing methods. It suits the paper framed as a measurement: same
   hull, conformal intervals, ERA5 and GFS-Wave as reference systems.
3. ***IEEE Transactions on Geoscience and Remote Sensing*.** It covers remote sensing of
   sea state; the SAR wave-height comparator, Quach et al. 2021, appeared there. It fits
   the fusion-with-models angle, but expects more methodological depth than the paper
   comparison.

## Data and code availability

- **Code.** The project's public GitHub repository, MIT licence. Tag the version used for the
  retrospective and fresh-strip scores and archive the release with a DOI. Give the
  repository URL at submission.
- **NDBC buoy-camera strips and spectra.** U.S. Government works, public domain in the
  U.S. NDBC keeps the images about 72 h, so the project's archive is the source of these
  strips. Release the 41,091 packaged strips with their labels, folds and checksums.
  State that the labels are realtime values.
- **ERA5.** CC BY 4.0 (Copernicus Climate Data Store). Cite Hersbach et al. 2020 and the
  dataset. State that the period used is the preliminary ERA5T.
- **GFS-Wave.** NOAA/NCEP station bulletins (212 cycles matched), a U.S. Government
  product. Confirm NCEP's attribution wording. Release the matched rows, not the
  bulletins.
- **CDIP (shore labels).** The files' licence attribute allows redistribution and use
  without restriction. Copy each station's acknowledgment attribute (433, 243, 200).
- **WebCOOS.** Use the citation and acknowledgment wording in
  `docs/datasets/webcoos.md`. The terms do not address redistribution of images, so ask
  WebCOOS before any frame appears in a figure or a released dataset. The large
  historical video download had WebCOOS's approval.
- **Yin et al. VWBT_9000.** The licence question is open (CC BY 4.0 on figshare
  against CC BY-NC-ND 4.0 in the project registry). It matters only if the breaker
  study joins this paper, which the recommendation above rules out.
- **Model weights.** Checkpoints fine-tuned from DINOv3 fall under the DINOv3 licence;
  read its terms before releasing them. Release predictions and fold assignments in
  any case.
