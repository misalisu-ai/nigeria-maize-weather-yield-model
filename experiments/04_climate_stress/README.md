# Experiment 04 — Climate-Stress Sensitivity

This experiment evaluates model sensitivity to controlled heat and rainfall
perturbations.

It is **not a climate-change projection**.

## 04a — Feature-space counterfactuals

Models are trained on 2020–2023 and evaluated on the 2024 temporal holdout.

The observed 2024 weather state is treated as the baseline.

Stress scenarios modify:

- mean Tmax
- mean Tmin
- seasonal GDD
- seasonal rainfall
- rainfall anomaly

`Max_CDD_days` remains fixed because dry-spell duration cannot be reconstructed
from seasonal rainfall totals alone.

### Temperature

Uniform warming scenarios:

- 0°C
- +1°C
- +2°C

### Rainfall

Seasonal rainfall scenarios:

- 0%
- -10%
- -20%
- -30%

The factorial design creates 12 scenarios.

## Interpretation

Outputs represent model sensitivity under controlled counterfactual feature
shifts.

They must not be described as future climate projections.

## 04b — Weather-recomputed stress

A later version will perturb raw daily weather and regenerate all seasonal
features, including GDD and CDD, before prediction.