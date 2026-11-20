# When should the forecast get more weight?

## Historical experiment protocol, November 20, 2026

Buoy 41040 motivated this experiment after its held-out errors were inspected. This is an exploratory comparison on the existing historical archive. The two tests below use wind-wave height as the sole endpoint. The November fresh-strip confirmation has its own fixed models and scoring protocol.

### 1. Transfer to an unseen geographic group

Fit a convex combination of the camera height C and corrected forecast height F, both in metres. The prediction is F + w(C − F), with 0 ≤ w ≤ 1. Use the measured-wind forecast and exactly the 34,723 paired strips, five outer folds and 41 geographic groups of the [nested evaluation](sensing_forecast_results.md). For outer fold k, calibrate on predictions for j from models that held out both j and k. Outer predictions come from the frozen fold-k models. The weight sees three quantities available at prediction time: forecast wind-wave height, measured wind speed, and signed disagreement C − F.

- Fit the constant weight by least squares, clipping the result to [0, 1]. Call it w₀. Write D = C − F and Y for the buoy reference.
- For the primary rule, bin each of the three inputs at its two calibration-set tertiles (at most 27 cells). Within a cell use w = clip((ΣD(Y − F) + λw₀) / (ΣD² + λ), 0, 1), where λ = 100 × mean(D²) across the calibration set. Empty cells use w₀. A zero-disagreement calibration set uses the forecast. Use all eligible training strips with equal weight.
- Controls are the forecast, camera, historical registered gate, historical inverse-variance energy blend, and the fitted constant blend. A two-input ablation uses the same formula and excludes signed disagreement from the binning.
- Compare pooled RMSE on identical outer rows. For evidence that the rule improves on available alternatives, require a paired 95% interval below zero for candidate minus **both** the registered gate and the existing variance-weighted blend. Report the constant-weight and two-input differences to distinguish conditional weighting from a change of mixing formula.
- Report equal-group sensitivity, outer folds, the number of geographic groups improved, and the primary comparison excluding g41040. Inspect 41040 separately as the motivating case. Record learned weights and calibration sizes. There is no search over cuts, features, estimators or penalties.

### 2. Does one week of local wave measurements help?

A distinct use case has one week of reference heights at a new buoy. Start the clock at its first eligible historical strip (UTC). Use [start, start + 7 days) for calibration, leave a 48-hour gap, then score rows at or after start + 9 days. Require at least 24 calibration strips and one later strip per buoy. Use each buoy's fold-specific frozen forecast, camera and primary rule from test 1.

Fit one additive offset to the primary camera weight per buoy, using its early strips only. With P the primary prediction, the offset is ΣD(Y − P) / (ΣD² + 50 × mean(D²) from that outer fold's inner calibration). Use zero for a zero denominator and clip the final weight to [0, 1]. Compare with the unchanged primary rule on the same later strips. Also report a simpler local control: apply the same offset procedure to the constant blend. Report calibration and evaluation coverage, 41040, and intervals against both the primary rule and the local constant control. Evidence for local calibration requires the primary-plus-offset rule to beat the primary alone. Added value over the simpler local approach requires it also to beat the local constant control.

### Scoring and interpretation

Use 2,000 paired whole-station-group bootstrap draws, NumPy RNG seed 0, to estimate percentile intervals for differences in metres. For equal-group sensitivity average each group's MSE equally before taking the square root. For each evaluated population, compute each arm on the same rows. The chronological comparison simulates the order of local label availability within this archive. Base models and reanalysis remain historical. Interpret both tests in light of prior inspection of this archive and reserve a future independent window for any new claim.

## Results

### Transfer to unseen groups

The conditional rule achieved **19.92 cm RMSE** on 34,723 historical strips from 71 buoys. The forecast alone scored 22.58 cm and the camera alone scored 25.44 cm. Both comparisons specified for success favored the rule. Negative differences below indicate lower RMSE for the conditional rule.

| Comparator | RMSE (cm) | Difference (cm, paired 95% interval) | Groups improved |
|---|---:|---:|---:|
| Forecast | 22.58 | −2.66 [−3.18, −2.03] | 37/41 |
| Registered gate | 20.79 | −0.87 [−1.21, −0.58] | 34/41 |
| Variance-weighted energy blend | 20.20 | −0.28 [−0.38, −0.14] | 33/41 |
| Constant camera weight | 20.10 | −0.18 [−0.26, −0.09] | 30/41 |
| Conditional rule without disagreement | 20.05 | −0.13 [−0.19, −0.05] | 23/41 |

The rule improved on both the registered gate and variance blend in each of the five outer folds. Giving each geographic group equal weight yielded differences of −0.98 cm [−1.37, −0.56] against the gate and −0.20 cm [−0.31, −0.07] against the variance blend. Removing g41040 yielded strip-weighted differences of −0.82 cm [−1.13, −0.52] and −0.27 cm [−0.37, −0.13] respectively. The fitted constant camera weights ranged from 0.367 to 0.423. Conditioning on the three inputs contributed a modest additional gain.

At buoy 41040, the conditional rule reduced RMSE from the gate's 16.18 cm to 10.85 cm across 631 strips. The forecast alone still did best at 7.49 cm.

### One week of local measurements

Of the 71 buoys, 68 met the calibration and evaluation requirements. They supplied 5,548 first-week strips, 1,544 strips in the two-day gap, and 27,046 later strips across 39 groups. The local offset scored **19.84 cm** on later strips, versus **19.74 cm** for the unchanged conditional rule. The paired difference was +0.11 cm [−0.20, +0.52], with 20 of 39 groups improving. Although the local rule beat the locally fitted constant control by 0.12 cm [0.04, 0.20], these results give no evidence that a week of local labels improves the transferred rule overall.

Buoy 41040 was an exception. On its 518 later strips the offset reduced RMSE from 10.70 to 8.46 cm. The forecast alone scored 7.63 cm. Its improvement illustrates why checking the pooled result matters after selecting a station for its large errors.

### Reproduction

The protocol was committed in `c82d9cdbd` before fitting. From the repository root, with the nested historical run archive available, use a new output directory:

```bash
sha256sum --quiet -c data/manifests/ml/forecast-fusion-nested-v1.sha256
uv run --locked --group forecast python scripts/wind_wave_reliability.py --out runs/wind-wave-reliability-reproduction
```

The output includes row-level predictions, per-fold weights and cuts, calibration counts, fold and geographic scores, and paired intervals. The small result files are tracked in `reports/wind-wave-reliability/`. The SHA-256 in `fits.json` identifies the archived outer predictions used for scoring.

## Earlier calibration, later scoring

### Protocol fixed after the two results above, before fitting this comparison

To test whether conditional weighting survives a shift in dates, use one fixed historical cutoff across all buoys. For each held-out geographic fold, fit every combiner on its usual nested calibration strips dated **before October 29, 2026 at 00:00 UTC**. Score every eligible strip in the held-out fold **from October 31 at 00:00 UTC** to the end of the same historical archive. This leaves a common 48-hour gap. Choose one cutoff to reserve roughly the last two weeks for scoring. Do not select among dates or change the fit after scoring.

Use the same wind-wave endpoint, three-input rule, penalty and paired group-bootstrap intervals as above. Refit the constant blend, the variance-weighted energy blend (including its forecast-total quartiles), and the two-input control on the same early rows. Report the raw forecast and camera as references. The primary comparisons are the three-input rule against the constant and variance-weighted blends. Require the strip-weighted 95% upper endpoints for both differences to fall below zero. Report group counts, fold scores, equal-group sensitivity, and calibration and test date ranges.

This cutoff applies to blend calibration. The archived base models were fitted geographically and can draw on later dates at other stations. ERA5 is reanalysis. This tests calendar transfer of the weights within an already inspected archive. An end-to-end prospective claim requires a separate window and base predictions available at issue time.

### Results on October 31 through November 13

The conditional rule scored **21.56 cm RMSE** on 7,855 later strips from 67 buoys in 39 geographic groups. Each fold used 19,692–20,826 calibration strips from September 23 through October 28. The camera alone scored 27.85 cm.

| Earlier-fit comparator | Later RMSE (cm) | Conditional minus comparator (cm, paired 95% interval) | Groups improved |
|---|---:|---:|---:|
| Forecast | 24.26 | −2.69 [−4.08, −1.84] | 34/39 |
| Variance-weighted energy blend | 21.83 | −0.27 [−0.43, −0.03] | 27/39 |
| Constant camera weight | 21.77 | −0.20 [−0.41, −0.04] | 25/39 |
| Conditional rule without disagreement | 21.65 | −0.08 [−0.28, +0.09] | 24/39 |

Both pooled comparisons met the specified criterion. Giving each group equal weight widened the uncertainty about a typical group's gain. The difference was −0.12 cm [−0.38, +0.18] against the variance blend and −0.22 cm [−0.45, +0.02] against the constant blend. The conditional rule beat the variance blend in four of five folds and the constant blend in three. The later data also leave open whether the disagreement input itself adds value. These results support testing the frozen three-input rule and the simple constant blend on a separate future window.

The cutoff and decision criterion are in commit `2fae15151`. To reproduce this comparison, use the same verified archive:

```bash
uv run --locked --group forecast python scripts/wind_wave_reliability.py --forward --out runs/wind-wave-reliability-forward-reproduction
```

The per-fold fits and dates, scores and contrasts are tracked in `reports/wind-wave-reliability/forward/`.
