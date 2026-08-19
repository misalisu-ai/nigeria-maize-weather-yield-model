# Nigeria Maize Weather-Yield Modeling

Research repository for evaluating weather-driven maize yield prediction across Nigerian states under temporal and spatial distribution shift.

## Frozen dataset

**Nigeria Maize Weather-Yield Dataset v1.0**

- Data repository: https://github.com/misalisu-ai/nigeria-maize-weather-yield-data
- Rows: 185
- States: 37
- Years: 2020–2024
- Target: `Yield_MT_Ha`
- Dataset SHA-256: `09ab62faed51d7f391595068587d33595f49058ae1687621925cf12408c2f97c`

The loader verifies this hash before experiments run.

## Research question

> How well can weather-only ML models generalize across Nigerian states and future years, and how does predictive reliability change under controlled climate perturbations?

## Experimental hierarchy

1. Random baseline — diagnostic only
2. Temporal holdout — train 2020–2023, evaluate 2024
3. State-held-out evaluation — grouped by state
4. Climate stress testing — controlled temperature and rainfall perturbations
5. Uncertainty estimation
6. SHAP interpretation
7. Feature ablation

## Models

- Ridge regression
- Random Forest
- LightGBM
- PyTorch MLP benchmark

The MLP is treated as a deep-learning benchmark. The small panel means deep learning is not assumed to outperform simpler models.

## Getting started

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy the frozen dataset into:

`data/nigeria_maize_weather_yield_2020_2024_v1.0.csv`

Then verify the checksum before running experiments.
