# Test 1 · Forecast baseline and camera diagnostic, November 19

The measured-wind and forecast combination narrows the remaining opportunity for cameras. At the 71 eligible stations (41 geographic groups), choose ridge or histogram boosting independently on each outer fold's validation groups. Heights are in metres. Predictions use the fold's training groups, and all arms in each row share the same evaluation strips.

| Instrument target | Strips | Existing ERA5 + measured wind | ERA5 + GFS + location and depth | Same inputs plus measured wind |
|---|---:|---:|---:|---:|
| Significant height | 36,819 | 0.2927 | 0.2665 | 0.2667 |
| Wind-wave height | 34,723 | 0.2972 | 0.2740 | 0.2258 |

The distributed GFS station bulletins supplied usable total heights on 38,093 of the 41,091 package rows. Additional required source fields and target availability determine the sizes above. The GFS systems and ERA5 directions use circular features. Fit the historical ERA5 + wind log-height ridge on the same rows with its historical penalty. The new ridge and boosted models fit heights directly. Deterministic fits use the same predictions for each seed. These are retrospective nominal issue-time comparisons.

## Camera diagnostic using the historical stage-2 procedure

Combine the forecast with the three-seed mean of the held-fold DINOv3 camera outputs using the existing learned wind-sea gate. The table uses the 34,723 strips with both reference targets and inputs. Below-zero differences favor fusion. The intervals resample 41 whole geographic groups, 2,000 times.

| Comparator | Target | Forecast alone | Camera alone | Combined | 95% interval, combined minus forecast |
|---|---|---:|---:|---:|---:|
| Forecast + measured wind | Significant height | 0.2670 | 0.5193 | 0.2635 | −0.0054 to −0.0019 |
| Forecast + measured wind | Wind-wave height | 0.2258 | 0.2544 | 0.2095 | −0.0239 to −0.0063 |
| Forecast without measured wind | Significant height | 0.2670 | 0.5193 | 0.2606 | −0.0091 to −0.0034 |
| Forecast without measured wind | Wind-wave height | 0.2740 | 0.2544 | 0.2266 | −0.0624 to −0.0241 |

Shuffling the camera outputs and image-quality inputs jointly within buoy and training-defined forecast-height bin raises combined RMSE to 0.2711 m for total and 0.3974 m for wind-wave height with measured wind, averaging the three shuffled predictions. Those contrasts describe disruption of the inputs.

**Interpretation.** The camera has its clearest incremental value for wind waves. The total-height difference with measured wind is about 3.5 millimetres of aggregate RMSE. The gate was trained on other folds' out-of-fold predictions using the historical stage-2 design. Some base predictors producing those calibration rows saw geographic groups held out from the gate. The registered primary analysis requires inner cross-fitting confined to each outer training fold. This diagnostic therefore does not declare a pass on test 1. The pre-existing fresh-strip confirmation keeps its separate date and hypotheses.

## Reproduce

Run `scripts/sensing_forecast_corrector.py` with the v2cv package labels, `runs/gfs-wave-strip-match-v1.parquet`, the hourly `era5_at_buoys.parquet` and the measurement branch's `runs/prep-v1/station-depths.csv`. The script uses scikit-learn and the repository dependencies. Run `scripts/sensing_fusion_diagnostic.py` with those labels, the saved forecast predictions and the existing `oof_dinov3-gentle.csv`. The row-level outputs are `runs/forecast-baseline-oof-v1.parquet` and `runs/forecast-fusion-historical-v1.parquet`. [Registered protocol](sensing_model_bar.md)
