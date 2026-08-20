# Baseline Evaluation Protocol

## Dataset

Nigeria Maize Weather-Yield Dataset v1.0.

SHA-256:

`09ab62faed51d7f391595068587d33595f49058ae1687621925cf12408c2f97c`

## Features

The eight frozen weather features are used without post-freeze feature
engineering:

- Seasonal_Rainfall_mm
- Seasonal_GDD_C
- Max_CDD_days
- Mean_Tmax_C
- Mean_Tmin_C
- Mean_RH_pct
- Mean_Solar_Radiation_MJ_m2_day
- Rainfall_Anomaly_Z_2000_2019

## Target

`Yield_MT_Ha`

## Experiment 01 — Random baseline

20% observation-level test split with `random_state=42`.

This is a diagnostic baseline only. State and year may appear in both
training and testing.

## Experiment 02 — Temporal holdout

Train: 2020–2023.

Test: 2024.

This evaluates performance on one held-out future year.

## Experiment 03 — Spatial generalization

5-fold `GroupKFold`, grouped by `State`.

A state cannot appear in both training and testing within a fold.

`shuffle=False`; the resulting fold assignment is saved as an experiment
artifact.

## Models

- Ridge
- Random Forest
- LightGBM

Hyperparameter optimization is deliberately deferred until the baseline
evaluation is frozen.

## Metrics

- MAE
- RMSE
- R²
- Mean prediction error

## Leakage policy

Any transformation that learns parameters from the data must be fit only on
the training partition. Ridge scaling is implemented inside an sklearn
Pipeline. Tree models do not require feature scaling.