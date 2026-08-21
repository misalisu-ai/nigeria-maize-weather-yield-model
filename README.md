# 🌽 Nigeria Maize Yield Intelligence

> **Weather-based machine learning for state-level maize yield analysis, robustness testing, uncertainty estimation, and explainable decision support across Nigeria.**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-Ridge%20%7C%20Random%20Forest%20%7C%20LightGBM-green.svg)](#)
[![Status](https://img.shields.io/badge/Status-3MTT%20Capstone%20Ready-success.svg)](#)

## 📌 Project Overview

**Nigeria Maize Yield Intelligence** is an end-to-end machine-learning project combining Nigerian state-level maize yield records with NASA POWER weather data.

It serves two complementary purposes:

1. **3MTT Capstone:** an interactive Streamlit application for exploring yield, weather patterns, model performance, controlled climate-stress scenarios, and model explanations.
2. **Research:** a reproducible study of temporal and geographic generalization, robustness, predictive uncertainty, feature dependence, and explainability in weather-only maize-yield prediction.

> **Research question:** How well can weather-only machine-learning models generalize across Nigerian states and future years, and how does predictive reliability change under controlled climate perturbations?

---

## 🚀 Live Application

**Streamlit App:** `https://nigeriamaizeyield.streamlit.app/`

Run locally:

```bash
streamlit run app/app.py
```

---

## 🎯 Problem

Maize yield varies across Nigerian states and years. Weather conditions such as rainfall, temperature, dry spells, humidity, and solar radiation may provide useful predictive information, but models can fail when asked to predict a future year, an unseen state, or weather conditions outside their training distribution.

This project therefore evaluates not only prediction accuracy, but also **generalization, robustness, uncertainty, and explainability**.

---

## 💡 Solution

The dashboard allows users to:

- explore state-level maize yield histories;
- inspect rainfall and temperature patterns;
- compare Ridge, Random Forest, and LightGBM models;
- examine temporal and state-held-out performance;
- explore controlled daily-weather climate-stress scenarios;
- inspect predictive uncertainty;
- view SHAP-based model explanations.

The dashboard is an **analytical decision-support prototype**, not a climate forecasting or causal inference system.

---

## 🗺️ Dataset

| Item | Value |
|---|---:|
| Nigerian states/FCT | 37 |
| Years | 2020–2024 |
| State-year observations | 185 |
| Weather features | 8 |
| Target | Maize yield (MT/ha) |

### Target
`Yield_MT_Ha`

Yield values come from Nigerian Agricultural Performance / Productivity Survey reports, with source provenance preserved.

### Weather source
Daily meteorological data come from **NASA POWER** and are aggregated over reproducible maize-season windows.

### Weather features
- `Seasonal_Rainfall_mm`
- `Seasonal_GDD_C`
- `Max_CDD_days`
- `Mean_Tmax_C`
- `Mean_Tmin_C`
- `Mean_RH_pct`
- `Mean_Solar_Radiation_MJ_m2_day`
- `Rainfall_Anomaly_Z_2000_2019`

Rainfall anomaly uses a state-specific **2000–2019 historical climatology**, avoiding target-period leakage.

---

## 🧠 Models

- **Ridge Regression** — regularized linear baseline
- **Random Forest** — nonlinear tree ensemble
- **LightGBM** — gradient-boosted tree model

No model is presented as universally superior; performance is evaluated under multiple generalization protocols.

---

## 📊 Dashboard Pages

### 1. Overview
Dataset coverage, annual yield patterns, top-yielding states, and feature distributions.

### 2. Yield Explorer
Interactive state selection with 2020–2024 yield, rainfall, temperature, and dry-spell trends.

### 3. Model Performance
Comparison across random diagnostic, 2024 temporal holdout, state-held-out evaluation, and conformal uncertainty.

### 4. Climate Stress
Interactive controlled scenarios with:
- state
- model
- temperature shift: `0`, `+1`, `+2 °C`
- rainfall shift: `0`, `-10`, `-20`, `-30%`

Outputs include baseline yield, stressed yield, absolute change, percentage change, and state-level response patterns.

### 5. Explainability
Held-out SHAP attribution for Random Forest and LightGBM.

---

## 🔬 Research Experiments

| Experiment | Purpose |
|---|---|
| **01** | Random-split diagnostic baseline |
| **02** | Future-year temporal generalization |
| **03** | State-held-out spatial generalization |
| **04a** | Aggregate feature-space climate stress |
| **04b** | Daily-weather-recomputed climate stress |
| **05** | Temporal and spatial conformal uncertainty |
| **06** | Grouped feature ablation |
| **07** | Held-out SHAP explainability |
| **08** | Statistical robustness synthesis |

---

## 📈 Selected Findings

### 2024 temporal holdout

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge | 0.382 | 0.481 | 0.090 |
| Random Forest | 0.363 | 0.463 | 0.158 |
| LightGBM | 0.361 | 0.467 | 0.142 |

### State-held-out generalization

| Model | MAE | RMSE | Mean R² |
|---|---:|---:|---:|
| Random Forest | 0.354 | 0.437 | -0.071 |
| LightGBM | 0.370 | 0.451 | -0.147 |
| Ridge | 0.396 | 0.486 | -0.318 |

Geographic transfer to unseen states remains difficult.

### Statistical comparison
Experiment 08 paired-bootstrap confidence intervals for model MAE differences all included zero, so the project does **not** claim a statistically clear model winner.

### Predictive uncertainty
Pooled state-held-out 90% conformal coverage:
- LightGBM: **90.8%**
- Random Forest: **89.7%**
- Ridge: **90.3%**

The intervals were relatively wide, indicating limited sharpness.

### Explainability
Both nonlinear models consistently ranked:
1. **Seasonal Growing Degree Days**
2. **Mean Solar Radiation**
3. **Mean Maximum Temperature**

as their top three mean-absolute-SHAP contributors.

### Feature ablation
Removing GDD consistently degraded Random Forest and LightGBM. No reduced feature set consistently improved both temporal and spatial performance.

---

## 🌦️ Climate-Stress Analysis

Two controlled stress methods were evaluated:

- **04a:** direct perturbation of aggregate features
- **04b:** perturb daily 2024 weather and recompute the original seasonal features

Experiment 04b is the preferred implementation.

Stress grid:
- Temperature: `0`, `+1`, `+2 °C`
- Rainfall: `0`, `-10`, `-20`, `-30%`

> These scenarios are **controlled sensitivity tests**, not future climate projections or causal estimates.

---

## 🧪 Reproducibility

Frozen Dataset v1.0 SHA256:

```text
09ab62faed51d7f391595068587d33595f49058ae1687621925cf12408c2f97c
```

Safeguards include:
- checksum verification;
- explicit temporal and state-grouped splits;
- saved split assignments;
- experiment metadata;
- fixed random seeds;
- climate baseline identity tests;
- repaired rainfall climatology validation;
- held-out-only SHAP explanations.

---

## 📁 Repository Structure

```text
.
├── app/
│   ├── app.py
│   ├── components/
│   └── pages/
├── configs/
├── data/
├── docs/
├── experiments/
│   ├── 01_random_baseline/
│   ├── 02_temporal_holdout/
│   ├── 03_state_held_out/
│   ├── 04_climate_stress/
│   ├── 05_uncertainty/
│   ├── 06_ablation/
│   ├── 07_explainability/
│   └── 08_synthesis/
├── results/
├── src/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/misalisu-ai/nigeria-maize-weather-yield-model.git
cd nigeria-maize-weather-yield-model
python -m venv .venv
pip install -r requirements.txt
streamlit run app/app.py
```

---

## ☁️ Streamlit Deployment

Deploy from Streamlit Community Cloud using:

```text
Repository: misalisu-ai/nigeria-maize-weather-yield-model
Branch: main
Entrypoint: app/app.py
```

After deployment, replace the placeholder live-app URL above with the public `streamlit.app` URL.

---

## ⚠️ Limitations

- only five target years;
- one representative NASA POWER point per state;
- weather-only modeling omits management, soil, fertilizer, pests, irrigation, seed variety, economics, and policy;
- some nearby locations share gridded weather profiles;
- state-held-out generalization is weak;
- uncertainty intervals are relatively wide;
- climate-stress scenarios are controlled model-sensitivity tests;
- SHAP explains fitted-model behavior, not causality.

---

## 🔭 Future Work

- longer yield histories;
- LGA-level modeling;
- soil and management variables;
- satellite vegetation indices;
- adaptive uncertainty estimation;
- independent external validation;
- downscaled climate-model scenarios.

---

## 🧰 Technology Stack

Python · pandas · NumPy · scikit-learn · LightGBM · SHAP · Streamlit · Plotly · Matplotlib · SciPy

---

## 📦 Related Data Repository

Dataset construction, provenance, quality control, coordinates, and feature engineering:

**https://github.com/misalisu-ai/nigeria-maize-weather-yield-data**

---

## 📜 Responsible Use

This system is an analytical research prototype. Predictions should not be used as standalone agronomic recommendations or official crop-production forecasts.

---

## 👤 Author

**Muhammad Ibrahim Salisu**  
3MTT AI/ML Capstone Project

---

## ⭐ Project Status

- **3MTT capstone:** dashboard complete; public deployment and demo submission pending.
- **Research:** Experiments 01–08 complete; manuscript preparation will continue after capstone submission.
