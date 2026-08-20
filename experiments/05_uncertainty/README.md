# Experiment 05 — Predictive Uncertainty

This experiment evaluates whether point predictions are accompanied by
prediction intervals with reasonable empirical coverage.

## Method

Primary method: split conformal prediction using absolute calibration residuals.

Target coverage:

90% (`alpha = 0.10`)

## 05a — Temporal uncertainty

- Fit: 2020–2022
- Calibration: 2023
- Test: 2024

This prevents calibration observations from being used to fit the predictive
model.

## 05b — Spatial uncertainty

The outer evaluation uses state-grouped cross-validation.

Inside each outer training fold, states are again partitioned into:

- model-fit states
- calibration states

The final test states remain unseen by both model fitting and conformal
calibration.

## Reported quantities

- MAE
- RMSE
- R²
- empirical interval coverage
- mean interval width
- median interval width

Coverage must be interpreted jointly with interval width.

## Limitations

The dataset is small, so calibration sets are also small. Nominal 90%
coverage should therefore not be expected to equal exactly 90% in finite
samples.

The experiment evaluates predictive uncertainty under the observed dataset and
split protocols; it does not establish causal uncertainty.