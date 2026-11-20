# Test 6 · Label-free site adaptation and a few-shot calibration curve

Fixed on 2026-11-20, before any adapted head was fitted or any few-shot correction scored.

**Hypothesis.** Part of the camera's error at an unseen buoy is a per-camera shift in its features. The shift can come from mounting, lens, exposure or hull. It can be removed using only that camera's own unlabelled strips, if the camera is compared with the training cameras at the same hindcast sea state. A camera's mean feature also encodes its sea climate, so a plain per-camera standardisation would remove real signal. Source: rank 6 of `reports/Wave sensing landscape and multimodal options.md`. The bar there is "a paired buoy-level gain larger than the seed spread".

**Two weak spots this targets.**
- **The US West Coast fold.** Fold 0 is one group of 18 buoys, with H_s RMSE 0.70 m out of fold against 0.532 m overall.
- **Under-covered buoys.** Eleven buoys have buoy-level conformal 90 % coverage below 0.80 for H_s: 46081, 46042, 46047, 44013, 42002, 46013, 46069, 46022, 46014, 44008 and 45002.

## Features

The features come from the five grouped folds of `ndbc_buoycam_heights_v2cv` and the 15 DINOv3-gentle fold models behind the out-of-fold 0.532 m (folds 0–4 × seeds 0–2).
- **What they are.** Each strip's pooled 256-dimensional vector, the input of the model's own heads (LayerNorm, then a linear layer onto standardised log heights).
- **Where they come from.** The measurement session's probe saved them for every strip of each fold (`scripts/measurement_buoy_probe.py --save-features`; full views, not water-only). The probe's run records name the checkpoints, and they are the 15 runs behind `oof_dinov3-gentle.csv`.
- **Check.** On the CPU, the checkpoint's own head applied to these features reproduces the run's predictions. The median strip differs by 0.35 %, from bf16 against float32 inference and float16 storage. The fold RMSE matches within 0.001 m.
- **Cost.** No new extraction is needed and nothing is trained on a GPU.

## Adaptation (per fold k, per seed)

- **Camera and archive.** A camera is a buoy (one camera code per buoy in the package). Its archive is every strip of that buoy in the package, images only. No label of an adapted camera is used. For a fold's test buoys, the archive is their own test strips, used unlabelled.
- **Hindcast bins.** The raw interpolated ERA5 total height at the buoy (`era5_hs_raw_m`), cut into five bins at the quintiles of fold k's training strips. Strips without ERA5 (2 %) are adapted but do not enter any statistic.
- **Reference.** For each bin b, the mean μ_b and per-dimension variance s²_b of the raw features over fold k's training strips, all training cameras pooled.
- **Camera statistics.** For camera c, use the bins with at least 5 of its strips, with weights w_cb = n_cb / Σ n_cb.
  - Camera mean: a_c = Σ w_cb m_cb.
  - Reference mean at the camera's own sea-state mix: r_c = Σ w_cb μ_b.
  - Within-bin variances: v_c = Σ w_cb var_cb and u_c = Σ w_cb s²_b, per dimension, floored at 1e-6.
- **Adapted features.** x′ = r_c + (x − a_c) · √(u_c / v_c). The camera's offset from what the training cameras show at the same ERA5 states is removed, and its within-bin spread is matched. The camera's own variation across sea states is kept.
- **Applied to every camera.** Training, validation and test cameras are all adapted, each with its own archive.

## Arms

All arms use the same strips, every arm is fitted per fold and seed, and the three seeds' log heights are averaged per strip before scoring (the stage-1 rule).

| Arm | Features | Head |
|---|---|---|
| A0, the comparator | Raw | The model's own head: the out-of-fold `oof_dinov3-gentle.csv` (0.532 m total, 0.260 m wind-wave) |
| A1, head-matched control | Raw | Refitted: LayerNorm without affine, then ridge onto each log height |
| **A2, primary** | Adapted (binned mean and scale) | Refitted as A1 |
| A3, sensitivity | Adapted, mean only (scale factor 1) | Refitted as A1 |
| A4, sensitivity | Plain per-camera standardisation, no bins (r_c = global training mean, u_c = global training variance, a_c and v_c over all the camera's strips) | Refitted as A1 |
| A5, sensitivity | Only non-training cameras adapted to the reference | The model's own head, not refitted |

- **Fitting the refitted heads.** Ridge regression on fold k's training strips. Log heights are floored at 0.05 m, and each target is fitted where it is present. The penalty is chosen from {0.1, 1, 10, 100, 1000} by the lowest log-RMSE on fold k's validation strips, per arm, seed and target.

## Few-shot calibration curve (descriptive, with intervals; no pass mark)

- **Labelled window.** Each buoy's first 14 days in the package, counted from its first strip.
- **Evaluation strips.** The buoy's strips from day 17 on, after a 72-hour gap, so that no labelled hour shares a swell with an evaluation hour.
- **Draws.** k ∈ {1, 3, 10, 30, 100} labelled strips, drawn uniformly without replacement from the window strips where the target is present. Use RNG seeds 0–9, per buoy and k.
- **Inclusion.** A buoy enters the curve at k if it has at least k labelled window strips and at least 30 evaluation strips.
- **Correction.** A shrunk log bias: b = Σ (log y − log ŷ) / (k + λ), and ŷ′ = ŷ · exp(b).
  - λ is chosen from {0, 1, 3, 10, 30} on fold k's validation buoys, by the same window, draws and evaluation strips, averaged over the five k values, per fold and target.
  - The validation buoys' predictions come from the fold models: the out-of-fold seeds merged for A0, the refitted head for A2.
- **Curves.** The curve is computed on A0 and on A2. Each point is reported against k = 0 on the same evaluation strips, as the RMSE averaged over the ten draws (the mean squared error over draws, then the root).
- **Interval.** A paired 95 % interval from 2,000 resamples of whole station groups, RNG seed 0.
- **Summary.** Report the smallest k whose gain exceeds the endpoint's seed spread with an interval below zero.

## Bar

Two co-primary endpoints: total significant height and wind-wave height, out of fold at all 79 buoys.
- **Seed spread.** An endpoint's seed spread is the range of the three seeds' own out-of-fold RMSE. It is 0.016 m for H_s (0.539–0.555 m), the figure in the spec. For wind-wave height it is 0.0075 m (0.268–0.276 m). The spec quotes 0.016 m for both, so whether the wind-wave gain also exceeds 0.016 m is reported as well.
- **Intervals.** Paired 95 % percentile intervals from 2,000 resamples of whole station groups, RNG seed 0. Sensitivities: whole buoys, and equal-group weights.

**Pass, per endpoint.** All three must hold:
1. RMSE(A2) − RMSE(A0) ≤ −(seed spread).
2. The upper end of its interval is below zero.
3. The upper end of the interval for RMSE(A2) − RMSE(A1) is below zero, so the gain belongs to the adaptation, not to the refitted head.

Test 6 is met if both endpoints pass. Each is reported separately.

**Reported beside the bar.**
- **West Coast fold.** Fold 0 alone: A0, A1 and A2, with whole-buoy intervals, since the fold is a single group.
- **Coverage.** Buoy-level conformal 90 % coverage of H_s at the eleven buoys and overall, before (A0) and after (A2). The method is stage 1's: score |log y − μ| / σ, and fold k's threshold from the other folds' scores with each buoy weighted equally. After adaptation, μ is shifted by the change in the point prediction, log ŷ_A2 − log ŷ_A0, and σ is kept. The same applies after the few-shot correction at each k, on the evaluation strips, with thresholds recomputed from the other folds' corrected scores.
- **The other arms** A1, A3, A4 and A5.

No setting is changed after any A2 result is seen. A miss is reported as plainly as a pass.
