# Experiment 04a Diagnostic Summary

## Status

PASS AS FEATURE-SPACE SENSITIVITY ANALYSIS WITH LIMITATIONS.

## Baseline identity

The T+0_R0 scenario reproduced baseline model predictions exactly.

## Drought monotonicity

State-level predicted yield was non-increasing under progressively lower
rainfall for only:

- LightGBM: 40.5–43.2% of states
- Random Forest: 40.5–45.9% of states
- Ridge: 54.1% of states

Therefore, drought sensitivity should not be interpreted as a stable
state-level dose-response relationship.

## Training support

For the +2 °C scenarios:

- 45.9% of states exceeded the training GDD range.
- 67.6% exceeded the training Tmin range.
- 78.4% exceeded at least one stressed-feature training range.

Severe heat scenarios therefore involve substantial extrapolation.

## Interpretation

Experiment 04a measures sensitivity of fitted weather-yield models to
controlled feature perturbations.

It is not a physical crop simulation and must not be interpreted as a
climate-change projection or causal yield estimate.