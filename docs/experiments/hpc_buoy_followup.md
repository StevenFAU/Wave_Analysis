# Claude handoff for two buoy-camera experiments

## Fixed historical comparisons

The storm-video workflow has its own queue. These are separate experiments on the September–November historical buoy archive. Both use existing geographic folds. The November confirmation and the subsequent 14-day wind-wave replication have their own frozen models and scoring windows.

### A. Train before October 29 and score later at unseen buoys

Use the original 79-buoy height package and five outer geographic folds. For each fold use only its assigned training and validation strips dated before October 29, 2026 at 00:00 UTC. This cutoff applies to training labels, target normalization, checkpoint selection and forecast-corrector selection. Test on its held-out buoys from October 31 at 00:00 UTC. Use the exact 7,855 matched strips (67 buoys, 39 groups) of the historical [date-cutoff comparison](wind_wave_reliability.md#earlier-calibration-later-scoring).

Train one seed-0 camera per fold from public DINOv3 ConvNeXt-Tiny weights using the established gentle height recipe and total-height validation criterion. Fit the original ERA5 + GFS-Wave + measured-wind corrector on the same calendar and geographic roles with its original eligibility checks. Report camera alone, corrected forecast alone and the arithmetic mean of their wind-wave height predictions in metres. The sole primary contrast is the 50/50 blend minus forecast in pooled RMSE. Require its paired 95% upper endpoint below zero over 2,000 whole geographic-group resamples with seed 0. Report equal-group and per-fold results. Use all five folds at seed 0 regardless of interim outcomes. This is a retrospective test with ERA5 and an already inspected historical scoring window.

### B. Estimate swell from the part of the image below the horizon

Use the same five original seed-0 geographic height checkpoints for whole views and water crops. Use the existing below-horizon transform and its minimum of three usable water views. Extract the water embeddings and reuse the matching cached whole-view embeddings. Predict the lowest fixed band (0.030–0.095 Hz) as equivalent height in metres, four times the square root of its integrated variance. A zero-total spectrum is missing. A zero swell band in an otherwise valid spectrum is allowed.

Use the identical eligible training and test rows for each arm within each fold. Fit the established training-standardized ridge (penalty 10) on training buoys. Evaluate whole features alone, water features alone, measured wind alone, and each image input with wind. Wind features are speed, gust, both squares and their product. Clip heights at zero. The primary contrast is water + wind minus wind in pooled RMSE, with the same paired geographic bootstrap as A. Compare water + wind against whole + wind to estimate the cost of the crop. Report pooled R², equal-group contrasts, per-fold outcomes and coverage before and after filtering. Resizing and discarding part of the scene can also affect skill. This control tests information retained under that specific crop.
