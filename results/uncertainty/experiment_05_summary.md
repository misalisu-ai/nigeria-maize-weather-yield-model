# Experiment 05 — Predictive Uncertainty

## Temporal conformal

Fit: 2020–2022  
Calibration: 2023  
Test: 2024  
Target coverage: 90%

Observed coverage:

- Ridge: 91.9%
- Random Forest: 89.2%
- LightGBM: 81.1%

Mean interval widths were approximately 1.45–1.56 MT/ha.

## Spatial conformal

Nested state-grouped split conformal prediction produced pooled coverage of:

- Ridge: 90.3%
- Random Forest: 89.7%
- LightGBM: 90.8%

Mean interval widths ranged from approximately 1.55 to 1.67 MT/ha.

## Interpretation

Spatial point-prediction performance remained weak, but conformal intervals
achieved approximately nominal pooled coverage by producing relatively wide
prediction intervals.

Fold-level coverage varied substantially because each outer fold had only six
calibration states.

## Limitation

Intervals are symmetric and use a single calibration quantile per fold.
They do not adapt their width to local prediction difficulty.