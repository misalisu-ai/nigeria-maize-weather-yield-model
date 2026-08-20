from __future__ import annotations

import streamlit as st
import plotly.express as px

from components.loaders import (
    load_temporal_shap,
    load_spatial_shap,
)


st.set_page_config(
    page_title="Explainability",
    page_icon="🔍",
    layout="wide",
)


st.title(
    "🔍 Model Explainability"
)


st.markdown(
    """
SHAP values are used to examine how strongly individual weather variables
contribute to predictions made by the fitted tree models.

The explanations shown here are calculated only on observations that were
held out from model training.
"""
)


protocol = st.radio(
    "Explanation protocol",
    [
        "2024 Temporal Holdout",
        "State-Held-Out",
    ],
    horizontal=True,
)


model = st.selectbox(
    "Model",
    [
        "RandomForest",
        "LightGBM",
    ],
)


if protocol == "2024 Temporal Holdout":

    df = load_temporal_shap()

else:

    df = load_spatial_shap()


model_df = (
    df[
        df["Model"] == model
    ]
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    )
)


# ---------------------------------------------------------
# Top feature cards
# ---------------------------------------------------------

top = model_df.head(3)


cols = st.columns(3)


for col, (_, row) in zip(
    cols,
    top.iterrows(),
):

    col.metric(
        f"Rank {int(row['Rank'])}",
        row["Feature"],
        f"|SHAP| = {row['Mean_Absolute_SHAP']:.3f}",
    )


st.divider()


# ---------------------------------------------------------
# SHAP ranking chart
# ---------------------------------------------------------

plot_df = (
    model_df
    .sort_values(
        "Mean_Absolute_SHAP"
    )
)


fig = px.bar(
    plot_df,
    x="Mean_Absolute_SHAP",
    y="Feature",
    orientation="h",
    title=(
        f"{model} — Held-Out Feature Attribution"
    ),
    labels={
        "Mean_Absolute_SHAP":
            "Mean Absolute SHAP Value",
    },
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


st.markdown(
    """
### Main pattern

Across both Random Forest and LightGBM, the most consistently influential
features are:

1. **Seasonal Growing Degree Days (GDD)**
2. **Mean Solar Radiation**
3. **Mean Maximum Temperature**

The ranking is especially stable for Random Forest under state-held-out
evaluation.
"""
)


st.info(
    """
SHAP explains the behavior of the fitted prediction model.

A high SHAP ranking does not imply that the variable is independently causal.
This is particularly important because some weather variables are correlated.
"""
)


with st.expander(
    "View feature-attribution table"
):

    st.dataframe(
        model_df,
        hide_index=True,
        use_container_width=True,
    )
