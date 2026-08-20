# Experiment 06 — Grouped Feature Ablation

## Status

COMPLETE.

## Temporal holdout

Temperature-only features produced the strongest 2024 performance for the
nonlinear models:

- LightGBM R²: 0.200 vs 0.142 for the full set
- Random Forest R²: 0.187 vs 0.158 for the full set

Removing GDD caused substantial performance degradation for both nonlinear
models.

## Spatial generalization

Removing RH produced small improvements for both nonlinear models:

- LightGBM R²: -0.141 vs -0.147
- Random Forest R²: -0.062 vs -0.071

Ridge behaved differently. Rainfall-only features improved mean state-held-out
R² from -0.318 to -0.171, although spatial performance remained weak.

## Interpretation

Temperature-derived information appears to carry much of the predictive signal
available to the nonlinear models.

GDD is particularly important for Random Forest and LightGBM.

The full linear Ridge model appears more sensitive to multicollinearity and
benefits spatially from aggressive feature reduction.

No ablated feature set consistently dominates the full feature set across
models and evaluation protocols.

## Research implication

Feature importance should not be inferred from a single model or a single
holdout. The ablation results indicate model-dependent reliance on correlated
weather variables.