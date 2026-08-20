# Experiment 07 — Held-Out Explainability

## Status

COMPLETE.

## Validation

SHAP additive reconstruction checks passed for both Random Forest and
LightGBM.

Explanations were computed only for held-out observations.

## Temporal explainability

For 2024 held-out predictions, both tree models ranked:

1. Seasonal GDD
2. Mean solar radiation
3. Mean maximum temperature

as their three largest mean absolute SHAP contributors.

Seasonal rainfall and rainfall-derived dry-spell variables generally received
lower attribution.

## Spatial explainability

Across state-held-out folds, Seasonal GDD remained the highest-ranked feature
for both nonlinear models.

Random Forest showed particularly stable feature rankings:

- GDD: rank 1 in every fold
- solar radiation: rank 2 in every fold
- Tmax: rank 3 in every fold
- CDD: rank 8 in every fold

LightGBM showed greater fold-to-fold variation but retained GDD as rank 1 or 2
in every fold.

## Agreement with Experiment 06

The SHAP findings are consistent with grouped ablation results.

Removing GDD substantially reduced Random Forest and LightGBM performance,
while temperature-only feature sets remained competitive, especially in the
2024 temporal holdout.

Together, ablation and SHAP indicate substantial model reliance on
temperature-related information.

## Correlated predictors

Mean Tmax and relative humidity are strongly correlated in the dataset.

Ridge assigned large coefficients to both variables, whereas tree-model SHAP
ranked RH substantially lower. Therefore, individual attribution among
correlated weather predictors should be interpreted cautiously.

## Interpretation

SHAP values explain the behavior of fitted prediction models. They do not
identify causal effects of weather variables on maize yield.