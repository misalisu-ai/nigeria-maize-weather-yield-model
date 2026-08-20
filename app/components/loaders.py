from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


DATASET = (
    ROOT
    / "data"
    / "nigeria_maize_weather_yield_2020_2024_v1.0.csv"
)


BASELINE_COMPARISON = (
    ROOT
    / "results"
    / "tables"
    / "baseline_comparison.csv"
)


CLIMATE_STRESS = (
    ROOT
    / "results"
    / "climate_stress"
    / "weather_recomputed_detailed.csv"
)


TEMPORAL_SHAP = (
    ROOT
    / "results"
    / "explainability"
    / "tables"
    / "temporal_shap_importance.csv"
)


SPATIAL_SHAP = (
    ROOT
    / "results"
    / "explainability"
    / "tables"
    / "spatial_shap_importance.csv"
)


UNCERTAINTY_TEMPORAL = (
    ROOT
    / "results"
    / "tables"
    / "temporal_conformal_summary.csv"
)


UNCERTAINTY_SPATIAL = (
    ROOT
    / "results"
    / "tables"
    / "spatial_conformal_overall.csv"
)


@st.cache_data
def load_main_dataset():
    return pd.read_csv(DATASET)


@st.cache_data
def load_baseline_results():
    return pd.read_csv(BASELINE_COMPARISON)


@st.cache_data
def load_climate_stress():
    return pd.read_csv(CLIMATE_STRESS)


@st.cache_data
def load_temporal_shap():
    return pd.read_csv(TEMPORAL_SHAP)


@st.cache_data
def load_spatial_shap():
    return pd.read_csv(SPATIAL_SHAP)


@st.cache_data
def load_temporal_uncertainty():
    return pd.read_csv(UNCERTAINTY_TEMPORAL)


@st.cache_data
def load_spatial_uncertainty():
    return pd.read_csv(UNCERTAINTY_SPATIAL)