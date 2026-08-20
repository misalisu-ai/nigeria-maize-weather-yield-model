# Experiment 08 — Robustness Synthesis

## Status

COMPLETE.

## Purpose

This experiment synthesizes the temporal, spatial, climate-stress, uncertainty, ablation, and explainability experiments without fitting additional predictive models.

## Paired model comparison

- Temporal_2024: RandomForest minus LightGBM MAE = 0.0023, 95% bootstrap CI [-0.0305, 0.0343].
- Temporal_2024: RandomForest minus Ridge MAE = -0.0185, 95% bootstrap CI [-0.1045, 0.0629].
- Temporal_2024: LightGBM minus Ridge MAE = -0.0208, 95% bootstrap CI [-0.1119, 0.0685].
- Spatial_State_Held_Out: RandomForest minus LightGBM MAE = -0.0153, 95% bootstrap CI [-0.0344, 0.0030].
- Spatial_State_Held_Out: RandomForest minus Ridge MAE = -0.0441, 95% bootstrap CI [-0.1018, 0.0130].
- Spatial_State_Held_Out: LightGBM minus Ridge MAE = -0.0288, 95% bootstrap CI [-0.0857, 0.0273].

## Temporal vs spatial generalization

- Ridge: temporal MAE 0.3818; spatial MAE 0.3975; difference +0.0157.
- RandomForest: temporal MAE 0.3633; spatial MAE 0.3534; difference -0.0099.
- LightGBM: temporal MAE 0.3610; spatial MAE 0.3687; difference +0.0077.

## Climate-stress method agreement

- LightGBM: 04a–04b Spearman agreement 0.997; directional agreement 99.3%; mean absolute difference 0.093 percentage points.
- RandomForest: 04a–04b Spearman agreement 0.999; directional agreement 99.5%; mean absolute difference 0.014 percentage points.
- Ridge: 04a–04b Spearman agreement 0.916; directional agreement 99.0%; mean absolute difference 0.280 percentage points.

## Predictive uncertainty

- Temporal LightGBM: coverage 81.1%; mean interval width 1.476 MT/ha (66.4% of mean observed yield).
- Temporal RandomForest: coverage 89.2%; mean interval width 1.452 MT/ha (65.3% of mean observed yield).
- Temporal Ridge: coverage 91.9%; mean interval width 1.560 MT/ha (70.2% of mean observed yield).
- Spatial LightGBM: coverage 90.8%; mean interval width 1.547 MT/ha (73.3% of mean observed yield).
- Spatial RandomForest: coverage 89.7%; mean interval width 1.577 MT/ha (74.6% of mean observed yield).
- Spatial Ridge: coverage 90.3%; mean interval width 1.667 MT/ha (78.9% of mean observed yield).

## Ablation consistency

- Worsened_Both: 11 model-feature-set comparisons.
- Mixed: 10 model-feature-set comparisons.

## Explainability stability

- LightGBM top spatial SHAP features: Seasonal_GDD_C, Mean_Solar_Radiation_MJ_m2_day, Mean_Tmax_C.
- RandomForest top spatial SHAP features: Seasonal_GDD_C, Mean_Solar_Radiation_MJ_m2_day, Mean_Tmax_C.

## Interpretation

The synthesis should be interpreted as evidence about predictive robustness and model behavior. It does not convert the observational weather-yield relationships into causal or climate-projection estimates.