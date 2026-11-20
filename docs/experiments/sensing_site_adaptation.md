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

## Result, 2026-11-20: not met

**Test 6 is not met.** Each camera's features were standardised against the training cameras at the same ERA5 total height.
- **Total height.** RMSE fell from 0.532 to 0.500 m, a gain larger than the seed spread. But the interval over station groups crosses zero, and the gain does not beat the head-matched control.
- **Wind-wave height.** It got much worse: 0.260 to 0.377 m.

Nearly all of the total-height gain comes from the US West Coast fold, 0.699 to 0.586 m.

Everything was computed on the CPU from the cached features of the 15 fold models; nothing was trained on a GPU. The output is `runs/site-adaptation-v1/` from `scripts/sensing_site_adaptation.py`.

**RMSE in metres, out of fold.** H_s covers 41,091 strips at 79 buoys; wind-wave covers 37,652 strips at 77 buoys. Three seeds are averaged per strip.

| Arm | Significant height | Wind-wave height |
|---|---:|---:|
| A0, the model as trained (comparator) | 0.5315 | 0.2599 |
| A1, refitted head on raw features | 0.5246 | 0.2571 |
| **A2, adapted features, refitted head (primary)** | **0.5002** | **0.3765** |
| A3, adapted mean only | 0.5025 | 0.3980 |
| A4, plain per-camera standardisation, no bins | 0.9583 | 0.4879 |
| A5, own head, only held-out cameras adapted | 0.5340 | 0.4161 |

The model's own head on the cached features reproduces A0 within 0.0001 m.

**Differences in metres** (95 % intervals; the bar uses station groups):

| Difference | Target | Estimate | Station groups (bar) | Whole buoys | Equal-group weights |
|---|---|---:|---:|---:|---:|
| A2 − A0 | Significant height | −0.0313 | −0.0689 to +0.0193 | −0.0628 to +0.0024 | +0.0018 (−0.0263 to +0.0355) |
| A2 − A1 | Significant height | −0.0244 | −0.0568 to +0.0214 | −0.0555 to +0.0088 | +0.0047 (−0.0243 to +0.0401) |
| A1 − A0 | Significant height | −0.0069 | −0.0135 to +0.0021 | | |
| A2 − A0 | Wind-wave height | +0.1166 | +0.0881 to +0.1412 | +0.0905 to +0.1439 | +0.1177 (+0.0808 to +0.1554) |
| A2 − A1 | Wind-wave height | +0.1194 | +0.0930 to +0.1429 | +0.0949 to +0.1459 | +0.1229 (+0.0870 to +0.1588) |
| A1 − A0 | Wind-wave height | −0.0028 | −0.0074 to +0.0001 | | |

**Against the pass criteria.**
- **H_s:** the gain clears the 0.016 m seed spread, but the interval is not below zero and A2 does not beat A1. The endpoint is not met.
- **Wind-wave height:** none of the three criteria holds. The endpoint is not met.

**What the numbers mean.**
- **Total height.** The 3.1 cm average gain sits in one group.
  - The West Coast fold (one group, 18 buoys) goes from 0.699 to 0.586 m: −0.113 m, whole-buoy interval −0.191 to −0.022. Against the head-matched control it is −0.093 m (−0.170 to −0.002).
  - Folds 1 and 2 get worse (0.528 → 0.540 and 0.462 → 0.471 m).
  - Folds 3 and 4 get slightly better (0.423 → 0.411 and 0.463 → 0.457 m).
  - With each group weighted equally there is no gain (+0.002 m).
- **Wind-wave height.** It is worse in every fold, for example 0.310 → 0.450 m in fold 1. Bins of ERA5 total height do not hold the wind-sea share fixed, so a camera's offset at a given total height includes its local wind-sea climate, and removing the offset removes that signal. This is the report's sea-climate caveat, realised for the wind sea. The validation-chosen ridge penalty for A2's wind-wave head went to the top of the grid (1,000) in every fold-0 seed, which is consistent with adapted features that no longer predict it. The registered grid was not widened.
- **Without bins.** Plain per-camera standardisation (A4) nearly doubles total-height error, to 0.958 m. A camera's mean feature does carry its sea climate. Height bins are necessary, but for the wind sea they are not enough.
- **The refitted head alone** (A1) is level with the model's own head.

**Coverage of the H_s 90 % intervals** (buoy-level conformal, stage 1's method).

| Buoy | 46081 | 46042 | 46047 | 44013 | 42002 | 46013 | 46069 | 46022 | 46014 | 44008 | 45002 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Before (A0) | 0.545 | 0.589 | 0.600 | 0.694 | 0.726 | 0.731 | 0.741 | 0.749 | 0.761 | 0.777 | 0.794 |
| After (A2) | 0.346 | 0.941 | 0.969 | 0.772 | 0.824 | 0.891 | 0.881 | 0.966 | 0.965 | 0.818 | 0.820 |

- **The eleven buoys.** Nine of them rise above 0.80. 46081 (Gulf of Alaska) falls to 0.35, and 44013 stays below.
- **Other buoys.** Five others fall below 0.80: 46087 0.49, 46077 0.54, 46088 0.63, 44020 0.76 and 44027 0.78. So the number of buoys below 0.80 goes from 11 to 7, not to 2.
- **Overall coverage** goes from 0.904 to 0.915.

## Few-shot curve (descriptive)

The curve uses the model as trained (A0). For each buoy, k labelled hours are drawn from its first 14 days, with ten draws. The buoy is scored from day 17 on, against k = 0 on the same strips. Differences are in metres, with intervals over station groups.

| k labelled hours | H_s buoys | H_s RMSE, k = 0 → k | Difference | Wind-wave buoys | Wind-wave RMSE, k = 0 → k | Difference |
|---:|---:|---|---:|---:|---|---:|
| 1 | 76 | 0.567 → 0.544 | −0.022 (−0.039 to −0.002) | 70 | 0.261 → 0.260 | −0.002 (−0.003 to −0.001) |
| 3 | 76 | 0.567 → 0.534 | −0.033 (−0.050 to −0.010) | 70 | 0.261 → 0.258 | −0.004 (−0.005 to −0.002) |
| 10 | 76 | 0.567 → 0.514 | −0.052 (−0.075 to −0.021) | 70 | 0.261 → 0.254 | −0.007 (−0.010 to −0.004) |
| 30 | 76 | 0.567 → 0.507 | −0.060 (−0.082 to −0.025) | 70 | 0.261 → 0.251 | −0.011 (−0.015 to −0.006) |
| 100 | 75 | 0.568 → 0.507 | −0.061 (−0.083 to −0.026) | 66 | 0.259 → 0.249 | −0.010 (−0.016 to −0.003) |

The shrinkage λ chosen on the validation buoys was 3–30 for H_s and 10–30 for wind-wave.

**What the curve means.**
- **Total height.** One labelled hour already removes more than the seed spread (−0.022 m, interval below zero). Ten hours remove 9 %. The curve flattens at about 0.507 m (11 % lower) by 30 hours. Most of the camera's remaining total-height error at a new buoy is a per-buoy offset, which a few labelled hours remove and the label-free route did not.
- **Wind-wave height.** It gains little. Thirty hours are needed to exceed its 0.0075 m seed spread (−0.011 m, 4 %).
- **Coverage.** At k = 10, 8 of the 11 buoys reach 0.80, with overall coverage 0.911. Still below 0.80: 44013, 44008 and 46081 (0.79). At k = 30 and 100 the three below are 44013, 44008 and 46047.
- **On the adapted arm (A2).** The few-shot correction brings total height to 0.500 m at k = 100, against 0.507 m for the model as trained on the same strips. Wind-wave reaches only 0.281 m, against 0.249 m. Once 30 or more labelled hours exist, adaptation adds nothing for total height, and it still hurts the wind sea.

**Where this leaves the two weak spots.**
- **West Coast.** Label-free adaptation helps the West Coast fold a great deal for total height, but it is one group, so this design cannot confirm it.
- **What works in general.** A few labelled hours at a new buoy are the lever that works across buoys.
- **A possible next bar.** Binning by ERA5 wind-sea and swell heights jointly, not by total height, might keep the wind-sea climate that this version removed. That would be a new bar; it was not tried here.
