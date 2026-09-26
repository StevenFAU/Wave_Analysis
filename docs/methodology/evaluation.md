# Evaluation

Code: `evaluation/`. One global metric is not enough. This page defines the
**required reporting set**.

## Regression (scalar targets)

`regression_report` returns: n, bias (pred − true), MAE, RMSE, centred RMSE,
NRMSE by **mean**, **range**, and **std**, scatter index (centred RMSE / mean
observed), R², Pearson, Spearman, OLS slope and intercept of prediction on truth.

- **Name the NRMSE normaliser.** Kamagata et al.'s 28.81 % is RMSE / range; the
  same errors normalised by the mean give a different number.
- **Report the slope.** A slope far below 1 with moderate correlation is
  regression dilution toward the training mean (Kamagata et al.: 0.12 for H_s).
  RMSE hides it and R² only hints at it.
- **R² can be negative** on held-out data. Report it as is.

## Agreement

`bland_altman`: bias, SD, limits of agreement (bias ± 1.96 SD), and a
proportional-bias slope and p-value. By default the difference is regressed on
the **reference** value, because regressing on the pair mean (classic
Bland–Altman) creates a spurious slope when the estimator is much noisier than
the reference instrument (demonstrated in `tests/unit/test_evaluation.py`).

## Directions

`circular_report`: circular bias, MAE, and RMSE on wrapped differences in
(−180°, 180°], plus the Jammalamadaka–SenGupta circular correlation. Linear RMSE
on raw degrees is invalid near north: truth 355° and 5° against predictions
5° and 355° give a linear RMSE of 350° but a circular RMSE of 10°.

## Classification (binned sea states)

`classification_report`: accuracy, **balanced accuracy**, macro and weighted
P/R/F1, per-class table, confusion matrix, and **ordinal** metrics (class-index
MAE, within-one accuracy, quadratic-weighted κ). Kim et al. (2023) find that most
errors fall in neighbouring classes, which only ordinal metrics credit.
`bin_values` maps out-of-range values to −1 (excluded) rather than clipping,
matching Choi et al.'s exclusion of the intermediate band.

## Uncertainty of the metrics

`block_bootstrap_ci` gives percentile CIs under a **cluster bootstrap** over
days or events, or a **moving-block bootstrap** otherwise. An i.i.d. bootstrap
of autocorrelated errors gives intervals that are too narrow; the test suite
shows block CIs more than twice as wide for AR(1) errors with φ = 0.95.

## Probabilistic predictions

For models that output uncertainty: interval coverage (PICP) together with mean
width, Gaussian NLL, closed-form Gaussian CRPS, and a calibration curve
(empirical vs nominal coverage).

## Stratification

`metrics_by_stratum` over at least: station, illumination category, sea-state
bin (`sea_state_bins`), season, |Δt| bin, reference distance bin, and image
quality bins. Thin strata are reported with their n, never dropped.

## Baselines are part of every result

`models.baselines.statistical`: mean, climatology (station × month),
persistence (causal), ridge regression on tabular covariates. A vision model's
skill is stated relative to the best baseline on the same split.

## Operational metrics (deployment studies)

Latency per clip, throughput, peak memory, power (embedded), missing-prediction
rate, and out-of-distribution detection rate.
