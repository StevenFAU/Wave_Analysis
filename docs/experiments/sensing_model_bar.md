# Test 1 · Does the camera add information to a stronger wave model?

Freeze this protocol before fitting. Use the existing five outer geographic folds, 43 station groups and 79 buoys. The two co-primary endpoints are total significant height and wind-wave significant height, in metres. The stage-2 anchors are ERA5 + anemometer (0.288 and 0.293 m out of fold). Their published errors are context. Refit and score every arm on identical eligible rows for each endpoint.

**Inputs.** Join ERA5 bulk and partition fields, GFS-Wave bulletin total height, system count, up to six ranked system heights, periods, circular directions and wind-generation flags, plus depth, latitude and longitude. Match numeric station ID and UTC. For this retrospective experiment use the latest nominal issue cycle at or before the strip and the nearest valid forecast hour within 30 minutes. Log issue, lead, actual valid time, offsets, missing stations and cases by fold. This emulates short-lead guidance using nominal issue time. Operational claims require separately enforcing actual publication time. Do not impute an absent station bulletin as zero wave energy. For a present bulletin encode absent systems as masked zeros. A distinct secondary experiment may add MFWAM once accessible.

**Comparators.** Compare ridge (standardize within training, penalty 10) and histogram gradient boosting (squared error, learning rate 0.05, 200 iterations, 15 leaves, no early stopping). Keep the two feature sets separate: model fields alone, and the same fields plus coincident buoy speed/gust. Exclude buoy wave history, identifiers and image information from both. Select the lower validation RMSE per target and input set on the designated fold validation groups. Input preprocessing and parameter fitting use training groups only. Require a forecast-coverage table including excluded stations, and report the existing ERA5 baseline on the same rows.

**Fusion.** Primary camera arm uses the existing gated wind-sea partition swap with DINOv3-gentle and the selected model + anemometer baseline. Use fold-matched camera predictions. Fit swap parameters and gate using predictions independent of the fitting labels (inner held-group predictions on outer training groups); report any reuse of the historical stage-2 cross-fitting arrangement separately. Secondary arm combines model and camera wind-sea squared heights using inverse residual variance estimated on training groups within training-defined model-height quartiles (pool bins with fewer than 30 cases). Retain the corrected model swell energy. Repeat with the anemometer withheld. No tuning on outer test results.

**Pass.** For seeds 0, 1 and 2, average predictions per strip before scoring. For each of the two co-primary endpoints require the upper endpoint of the paired 95% percentile interval for RMSE(camera fusion) − RMSE(selected model + anemometer) below zero. Draw 2,000 resamples of whole station groups with replacement, RNG seed 0. Give by-buoy and equal-group sensitivity. Report failure of either endpoint separately. Controls use the same cases: joint inputs, camera alone, model alone, and camera degraded by permuting the complete camera prediction vector within buoy and training-defined model-height quartile (seeds 0–2, include movable fraction). Report the intact-minus-permuted delta. Attribute incremental information to individual images only if this interval also excludes zero in the expected direction. Keep the separate fresh-strip confirmation on its original schedule. [Existing fusion design](ndbc_buoycam_sea_state.md)

## Amendment, 2026-11-19: nested calibration (fixed before any nested run)

**Why.** The historical diagnostic in [the results](sensing_forecast_results.md) fitted the gate for outer fold k on the other folds' out-of-fold predictions. The table below records, for each outer fold k and each calibration fold j, how many of fold k's station groups the cached fold-j models trained on, picked their epoch on, or predicted (train/val/test). It was produced by `scripts/sensing_nested_inventory.py`, which checks each run's own validation and test strips against the package's split columns. The forecast corrector of test 1 uses the same `split_fold<j>` columns, so the same counts apply to it.

| Outer fold k | Groups (buoys, strips) | Calibration fold j | Cached fold-j models: fold-k groups train/val/test |
|---|---|---|---|
| 0 | 1 (18, 9,440) | 1 | 1/0/0 |
| | | 2 | 1/0/0 |
| | | 3 | 1/0/0 |
| | | 4 | 1/0/0 |
| 1 | 9 (17, 7,823) | 0 | 7/2/0 |
| | | 2 | 4/5/0 |
| | | 3 | 5/4/0 |
| | | 4 | 6/3/0 |
| 2 | 10 (15, 8,039) | 0 | 9/1/0 |
| | | 1 | 6/4/0 |
| | | 3 | 6/4/0 |
| | | 4 | 4/6/0 |
| 3 | 12 (15, 8,063) | 0 | 11/1/0 |
| | | 1 | 6/6/0 |
| | | 2 | 7/5/0 |
| | | 4 | 8/4/0 |
| 4 | 11 (14, 7,726) | 0 | 9/2/0 |
| | | 1 | 9/2/0 |
| | | 2 | 9/2/0 |
| | | 3 | 8/3/0 |

Every calibration prediction came from a model that had trained or validated on all of the outer test fold's groups. This holds for all 15 cached DINOv3-gentle fold runs (seeds 0–2) and for the forecast corrector. The sealed station-split DINOv3-gentle models trained on groups of every fold, so they cannot serve either. No cached camera model is reusable for calibration. **The historical diagnostic is superseded as test 1's result.** It lowered total height RMSE from 0.2670 to 0.2635 m and wind-wave height RMSE from 0.2258 to 0.2095 m. The nested score below replaces it. The historical numbers stay in the results only as the historical arrangement, reported separately as this bar already required.

**Inner models.** For every unordered pair {a, b} of the five outer folds, one inner model trains on the other three folds and predicts a and b. For outer fold k, the calibration prediction on a strip of fold j comes from the {j, k} model, which never saw j or k. That gives ten pairs instead of twenty fold-by-fold inner models.
- **Validation.** Each pair picks its epoch on validation groups drawn from its three training folds by the outer folds' own rule (`buoycam_cv_folds.draw_val`). The target is 0.1875 of the pool's strips, which keeps the outer folds' 0.15/0.80 ratio. The draws use RNG seed 100 + 10a + b.
- **Package.** `scripts/sensing_nested_splits.py` writes package `ndbc_buoycam_heights_v2cvn`: v2cv's images, plus ten `split_pair<ab>` columns and `pairs.csv`.
- **Camera recipe.** Unchanged from the outer folds: DINOv3 ConvNeXt-Tiny, gentle (`--lp-epochs 2 --backbone-lr-scale 0.1`), same config, seeds 0, 1 and 2. Seeds are merged as in stage 1: the point is the mean of the seeds' log heights, and the log-normal is merged by its moments.
- **Forecast corrector.** The inner corrector uses the same arms, features and validation selection on each pair's split columns.
- **Cost.** 30 GPU runs, estimated at 18–20 GPU-hours. The outer fold runs took 39–64 min, mean 47. An inner run trains on about 20,000 strips instead of 25,000 and scores about 16,000 test strips once. At most two GPUs are used at once. If the measured cost points above about 22 GPU-hours, stop and report before continuing.

**Calibration and application.** For outer fold k, the calibration rows are the strips of the four other folds that have every input and both targets. Each carries its pair model's camera ensemble and its pair corrector's forecast. On these rows fit:
- the gated swap, `PartitionFusion(gate, comparator, transfer)` as in the historical arm;
- the comparator-total quartile cuts;
- the variance-weighting bins.

Then apply all three once to fold k, using the cached outer predictions: the fold-k camera ensemble in `oof_dinov3-gentle.csv` and the fold-k corrector in `forecast-baseline-oof-v1.parquet`. No setting is chosen after seeing an outer result.

**Arms, on identical rows.** The rows are the outer-fold strips with every input and both targets (the historical 34,723).
- *No camera:* the selected forecast + measured wind (the bar's comparator), and the forecast without wind.
- *Camera only:* the camera ensemble's own heights.
- *Gated swap:* the primary arm.
- *Degraded camera:* the gated swap with the camera degraded.
- *Variance-weighted:* defined below.

Each fusion arm is fitted and scored twice, with and without the anemometer in its comparator.

- **Degraded camera.** Permute the complete camera vector within buoy × quartile of the comparator's total height, with the quartiles cut on the calibration rows. The vector has eight columns: the three heights, the two spreads, usable views, mean grey and contrast. Use permutation seeds 0, 1 and 2 and average the three fused predictions. Report the movable fraction: the share of strips whose camera vector comes from another strip.
- **Variance-weighted.** In each comparator-total quartile of the calibration rows, compute v_c = mean((C² − Y²)²) and v_m = mean((W² − Y²)²). Here C is the camera wind-wave height, W the comparator's, and Y the observed wind-wave height. "Residual variance" is this mean squared residual of squared heights. Bins with fewer than 30 calibration rows use the pooled values.
  - Wind-sea energy: E = (C²/v_c + W²/v_m) / (1/v_c + 1/v_m).
  - Wind-wave height: √E.
  - Total height: √(max(T² − W², 0) + E). T² − W² is the corrected model's swell energy, retained as registered.

**Pass, unchanged.** For each co-primary endpoint, the upper end of the paired 95% percentile interval for RMSE(gated swap) − RMSE(selected forecast + measured wind) must be below zero. The interval comes from 2,000 resamples of whole station groups, RNG seed 0. Report each endpoint's outcome separately. Sensitivities:
- whole buoys as the resampling unit;
- equal-group weighting, where each group's mean squared error counts once.

The intact-minus-degraded delta is reported with the same interval.

**S2 check, not a new bar.** The same inner camera predictions calibrate the stage-2 combiners of `scripts/buoycam_partition_fusion.py` (`fit_fold`) for each outer fold, applied once to fold k. Report the gated swap against ERA5 + wind beside S2's scored 0.267 vs 0.288 m. Intervals use S2's own unit (whole buoys) and station groups, with 1,000 resamples and seed 0, as S2's script does. Bar S2, its scored result and its fresh-strip confirmation on 2026-11-27 are unchanged.

**Fresh strips.** The separate fresh-strip confirmation keeps its date, 2026-11-27, and its hypotheses.

**Result, 2026-11-20: met on both co-primary endpoints.** With the nested gate, adding the camera lowered forecast + measured wind RMSE:
- wind-wave height from 0.2258 to 0.2079 m, interval −0.0253 to −0.0088;
- total height from 0.2670 to 0.2623 m, interval −0.0083 to −0.0007.

The S2 check holds at 0.2667 against 0.2883 m. Details are in [the results](sensing_forecast_results.md#test-1-result-with-nested-calibration-november-20).
