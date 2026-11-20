# Test 1 · Forecast baseline and camera diagnostic, November 19

The measured-wind and forecast combination narrows the remaining opportunity for cameras. At the 71 eligible stations (41 geographic groups), choose ridge or histogram boosting independently on each outer fold's validation groups. Heights are in metres. Predictions use the fold's training groups, and all arms in each row share the same evaluation strips.

| Instrument target | Strips | Existing ERA5 + measured wind | ERA5 + GFS + location and depth | Same inputs plus measured wind |
|---|---:|---:|---:|---:|
| Significant height | 36,819 | 0.2927 | 0.2665 | 0.2667 |
| Wind-wave height | 34,723 | 0.2972 | 0.2740 | 0.2258 |

The distributed GFS station bulletins supplied usable total heights on 38,093 of the 41,091 package rows. Additional required source fields and target availability determine the sizes above. The GFS systems and ERA5 directions use circular features. Fit the historical ERA5 + wind log-height ridge on the same rows with its historical penalty. The new ridge and boosted models fit heights directly. Deterministic fits use the same predictions for each seed. These are retrospective nominal issue-time comparisons.

## Camera diagnostic using the historical stage-2 procedure (superseded)

Superseded on November 20 by the [nested result](#test-1-result-with-nested-calibration-november-20), the registered primary analysis. The numbers below stay as the historical arrangement.

Combine the forecast with the three-seed mean of the held-fold DINOv3 camera outputs using the existing learned wind-sea gate. The table uses the 34,723 strips with both reference targets and inputs. Below-zero differences favor fusion. The intervals resample 41 whole geographic groups, 2,000 times.

| Comparator | Target | Forecast alone | Camera alone | Combined | 95% interval, combined minus forecast |
|---|---|---:|---:|---:|---:|
| Forecast + measured wind | Significant height | 0.2670 | 0.5193 | 0.2635 | −0.0054 to −0.0019 |
| Forecast + measured wind | Wind-wave height | 0.2258 | 0.2544 | 0.2095 | −0.0239 to −0.0063 |
| Forecast without measured wind | Significant height | 0.2670 | 0.5193 | 0.2606 | −0.0091 to −0.0034 |
| Forecast without measured wind | Wind-wave height | 0.2740 | 0.2544 | 0.2266 | −0.0624 to −0.0241 |

Shuffling the camera outputs and image-quality inputs jointly within buoy and training-defined forecast-height bin raises combined RMSE to 0.2711 m for total and 0.3974 m for wind-wave height with measured wind, averaging the three shuffled predictions. Those contrasts describe disruption of the inputs.

**Interpretation.** The camera has its clearest incremental value for wind waves. The total-height difference with measured wind is about 3.5 millimetres of aggregate RMSE. The gate was trained on other folds' out-of-fold predictions using the historical stage-2 design. Some base predictors producing those calibration rows saw geographic groups held out from the gate. The registered primary analysis requires inner cross-fitting confined to each outer training fold. This diagnostic therefore does not declare a pass on test 1. The pre-existing fresh-strip confirmation keeps its separate date and hypotheses.

## Test 1 result with nested calibration, November 20

**Test 1 is met on both co-primary endpoints.** The gate was refitted on each outer fold's training partition, using only predictions from models that never saw that fold. Adding the camera then lowered forecast + measured wind errors:
- wind-wave height RMSE from 0.2258 to 0.2079 m, 95% interval of the difference −0.0253 to −0.0088 m;
- total height RMSE from 0.2670 to 0.2623 m, interval −0.0083 to −0.0007 m.

The bar and its [nested amendment](sensing_model_bar.md#amendment-2026-11-19-nested-calibration-fixed-before-any-nested-run) were committed before any run. This result replaces the historical diagnostic above.

**What is compared.** All arms use the same 34,723 strips at 71 buoys in 41 groups, with the same forecast comparators as above.
- For each outer fold, the gate, its quartile cuts and the variance weights are fitted on the strips of the four other folds.
- Those strips carry predictions from inner models that saw neither their own fold nor the outer fold. The camera side is 30 DINOv3-gentle runs: one per pair of outer folds, trained on the remaining three, seeds 0–2. The forecast side is the same corrector refitted on the same pair splits.
- The fitted combiners are applied once to the outer fold, with its cached predictions.

The inner models are slightly weaker than the outer ones, as expected from three folds of training instead of four. On the same strips the camera's total height RMSE is 0.541 m against 0.532 m, and the forecast + wind's is 0.270 m against 0.267 m.

**Inventory with the inner models.** For outer fold k, the fold-j calibration predictions now come from the pair {j, k} model. Of fold k's groups, that model trained on none and validated on none. It only predicted them, and those predictions are not used for calibration. Counts are train/val/test, checked against each run's `predictions.csv`.

| Outer fold k | Groups (buoys, strips) | Calibration fold j | Historical: cached fold-j models | Nested: pair {j, k} model |
|---|---|---|---|---|
| 0 | 1 (18, 9,440) | 1, 2, 3, 4 | 1/0/0 each | 0/0/1 each |
| 1 | 9 (17, 7,823) | 0, 2, 3, 4 | 7/2/0, 4/5/0, 5/4/0, 6/3/0 | 0/0/9 each |
| 2 | 10 (15, 8,039) | 0, 1, 3, 4 | 9/1/0, 6/4/0, 6/4/0, 4/6/0 | 0/0/10 each |
| 3 | 12 (15, 8,063) | 0, 1, 2, 4 | 11/1/0, 6/6/0, 7/5/0, 8/4/0 | 0/0/12 each |
| 4 | 11 (14, 7,726) | 0, 1, 2, 3 | 9/2/0, 9/2/0, 9/2/0, 8/3/0 | 0/0/11 each |

**RMSE in metres, forecast with measured wind.**

| Arm | Significant height | Wind-wave height |
|---|---:|---:|
| Forecast + measured wind (the bar's comparator) | 0.2670 | 0.2258 |
| Camera alone | 0.5193 | 0.2544 |
| **Gated swap, nested (primary)** | **0.2623** | **0.2079** |
| Gated swap, camera degraded | 0.2813 | 0.4005 |
| Variance-weighted wind sea, nested (secondary) | 0.2578 | 0.2020 |
| Gated swap, historical calibration (superseded) | 0.2635 | 0.2095 |

**Differences in metres.** Below zero favours the first arm.
- The bar uses the 95% interval from 2,000 resamples of whole station groups, RNG seed 0.
- Whole buoys and equal-group weights are the registered sensitivities.
- The equal-group column gives its own estimate, followed by its interval.

| Difference | Target | Estimate | Station groups (bar) | Whole buoys | Equal-group weights |
|---|---|---:|---:|---:|---:|
| Gated swap − forecast + wind | Significant height | −0.0047 | −0.0083 to −0.0007 | −0.0085 to −0.0010 | −0.0031 (−0.0085 to +0.0027) |
| Gated swap − forecast + wind | Wind-wave height | −0.0179 | −0.0253 to −0.0088 | −0.0266 to −0.0095 | −0.0151 (−0.0254 to −0.0049) |
| Variance-weighted − forecast + wind | Significant height | −0.0092 | −0.0142 to −0.0040 | −0.0150 to −0.0044 | −0.0064 (−0.0118 to −0.0015) |
| Variance-weighted − forecast + wind | Wind-wave height | −0.0238 | −0.0293 to −0.0176 | −0.0301 to −0.0176 | −0.0229 (−0.0304 to −0.0152) |
| Intact − degraded camera | Significant height | −0.0190 | −0.0303 to −0.0104 | −0.0250 to −0.0138 | −0.0196 (−0.0286 to −0.0128) |
| Intact − degraded camera | Wind-wave height | −0.1926 | −0.2119 to −0.1669 | −0.2108 to −0.1743 | −0.1942 (−0.2194 to −0.1679) |

**What the numbers mean.**
- **Wind-wave height.** The camera removes 1.8 cm of the 22.6 cm error left by forecast + measured wind, or 8%. It does so in every outer fold; the smallest gain is fold 3's, 0.6 cm.
- **Total height.** The gain is 4.7 mm, or 1.8%. The bar's interval clears zero by only 0.7 mm. With each group weighted equally, the interval reaches +2.7 mm. The total-height gain is real on the registered unit, but thin.
- **Degraded camera.** The camera vector was permuted within buoy and forecast-height quartile, which moves 99% of strips. The fused heights then become worse than the forecast alone: 0.2813 against 0.2670 m, and 0.4005 against 0.2258 m. The intact-minus-degraded intervals exclude zero in the expected direction. The gain therefore comes from what each image shows at that hour, not from a buoy's typical camera output. The bar requires this before crediting individual images.
- **Camera alone.** The camera is worse than forecast + measured wind on both endpoints: 0.5193 against 0.2670 m, and 0.2544 against 0.2258 m. Its value is as an added input.
- **Variance-weighted arm.** This secondary arm beats the gated swap on both endpoints: 0.2578 m total and 0.2020 m wind-wave. It had no pass mark of its own, so it does not replace the primary arm without a confirmation.
- **Nested against historical.** Calibrating on clean inner predictions did not shrink the gain; the nested swap scores slightly better than the historical one, at 0.2623 against 0.2635 m and 0.2079 against 0.2095 m. Its intervals are wider: −0.0083 to −0.0007 m against the historical −0.0054 to −0.0019 m for total height. The refitted gates pass more of the camera's wind-sea correction into the total: the fitted transfer k is 0.20–0.70 by fold, against 0.12–0.39 historically.

**Without measured wind (anemometer withheld).** The forecast alone scores 0.2670 m total and 0.2740 m wind-wave.
- The gated swap gives 0.2631 m total and 0.2262 m wind-wave. Its total-height difference, −0.0038 m (−0.0082 to +0.0042), does not clear zero. Its wind-wave difference, −0.0477 m (−0.0627 to −0.0247), does.
- The variance-weighted arm gives 0.2587 m (−0.0083, −0.0131 to −0.0013) and 0.2160 m (−0.0580, −0.0673 to −0.0429).

**Check beside bar S2, not a new bar.** The stage-2 combiners of `scripts/buoycam_partition_fusion.py` were calibrated the same way, on inner camera predictions only. Both endpoints still clear zero by whole buoys, S2's unit:
- total height: the gated swap on ERA5 + wind scores 0.2667 m against 0.2883 m (−0.0215; whole buoys −0.0306 to −0.0128; station groups −0.0345 to −0.0089);
- wind-wave height: 0.2407 against 0.2928 m (−0.0521; whole buoys −0.0657 to −0.0387; groups −0.0679 to −0.0349).

The historical S2 arrangement, re-run by the same code, gives 0.2668 and 0.2398 m. S2's scored result, 0.267 against 0.288 m, does not depend on the leak. Bar S2 and its fresh-strip confirmation on 2026-11-27 are unchanged.

**Cost.** The 30 inner runs took 1,265 GPU-minutes (21.1 GPU-hours) on the shared server. That is above the 18–20-hour estimate, because the runs shared busy GPUs, and below the amendment's 22-hour stop. Two runs ran at commit `0a9668670` and 28 at `9f7426637`; their training code is identical.

## Reproduce

Run `scripts/sensing_forecast_corrector.py` with the v2cv package labels, `runs/gfs-wave-strip-match-v1.parquet`, the hourly `era5_at_buoys.parquet` and the measurement branch's `runs/prep-v1/station-depths.csv`. The script uses scikit-learn and the repository dependencies. Run `scripts/sensing_fusion_diagnostic.py` with those labels, the saved forecast predictions and the existing `oof_dinov3-gentle.csv`. The row-level outputs are `runs/forecast-baseline-oof-v1.parquet` and `runs/forecast-fusion-historical-v1.parquet`.

For the nested result:
1. Build the pair-split package with `scripts/sensing_nested_splits.py`.
2. Train the 30 inner runs: split `pair<ab>`, seeds 0–2, with the outer folds' DINOv3-gentle options. Use `deploy/dgx/run.sh` with `PKG=ndbc_buoycam_heights_v2cvn TRAIN_SCRIPT=scripts/train_wave_heights.py TAG=nested-gentle`.
3. Fit the inner correctors: `scripts/sensing_forecast_corrector.py` with `--splits pair01 … pair34`, written to `runs/forecast-baseline-inner-v1.parquet`.
4. Score with `scripts/sensing_nested_fusion.py`. Its inputs are the v2cvn labels, the existing `oof_dinov3-gentle.csv`, both forecast tables and the inner runs. It writes `runs/forecast-fusion-nested-v1/` (`rows.parquet`, `summary.json`, `report.md`).
5. `scripts/sensing_nested_inventory.py` regenerates the inventory. [Registered protocol](sensing_model_bar.md)
