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
