# 3MTT Capstone Summary

## Project Title
Nigeria Maize Yield Intelligence

## Problem
Maize yield varies across Nigerian states and years, while weather conditions such as rainfall, temperature, dry spells, humidity and solar radiation also vary substantially. Decision makers need a practical way to explore these patterns and understand what weather-only ML models can—and cannot—predict.

## Solution
An interactive Streamlit dashboard combining official maize-yield observations with NASA POWER weather data. The system supports state-level exploration, model comparison, controlled climate-stress sensitivity analysis, predictive uncertainty and explainability.

## Dataset
- 37 states/FCT
- 2020–2024
- 185 state-year observations
- Official agricultural survey yield records
- NASA POWER daily weather aggregated into maize-season features

## Models
- Ridge Regression
- Random Forest
- LightGBM

## Key Findings
- Best 2024 MAE: about 0.361 MT/ha (LightGBM)
- Best state-held-out MAE: about 0.353 MT/ha (Random Forest)
- No model showed a statistically clear MAE advantage in paired bootstrap comparisons
- Seasonal GDD, solar radiation and Tmax were the most stable tree-model SHAP features
- Conformal intervals approached nominal coverage but remained wide
- Climate-stress outputs are model sensitivity diagnostics, not climate forecasts

## Dashboard Pages
1. Overview
2. Yield Explorer
3. Model Performance
4. Climate Stress
5. Explainability
