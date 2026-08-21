from __future__ import annotations
from components.style import (
    apply_dashboard_style,
    sidebar_branding,
)

apply_dashboard_style()
sidebar_branding()

import streamlit as st
import plotly.express as px
import pandas as pd

from components.loaders import (
    load_climate_stress,
)


st.set_page_config(
    page_title="Climate Stress",
    page_icon="🌦️",
    layout="wide",
)


st.title("🌦️ Controlled Climate-Stress Explorer")

st.markdown(
    """
Explore how fitted machine-learning models respond when 2024 daily weather
is subjected to controlled temperature and rainfall perturbations.

The daily weather is modified first, then seasonal weather features such as
rainfall, growing degree days and consecutive dry days are recomputed before
prediction.
"""
)


df = load_climate_stress()


# ---------------------------------------------------------
# Controls
# ---------------------------------------------------------

st.sidebar.markdown(
    "### Scenario Controls"
)


state = st.sidebar.selectbox(
    "State",
    sorted(
        df["State"].unique()
    ),
)


model = st.sidebar.selectbox(
    "Model",
    [
        "RandomForest",
        "LightGBM",
        "Ridge",
    ],
)


temperature = st.sidebar.select_slider(
    "Temperature Shift",
    options=[
        0.0,
        1.0,
        2.0,
    ],
    value=0.0,
    format_func=lambda x:
        f"+{x:.0f} °C",
)


rainfall = st.sidebar.select_slider(
    "Rainfall Change",
    options=[
        0.0,
        -10.0,
        -20.0,
        -30.0,
    ],
    value=0.0,
    format_func=lambda x:
        f"{x:.0f}%",
)


selected = df[
    (df["State"] == state)
    & (df["Model"] == model)
    & (
        df["Temperature_Shift_C"]
        == temperature
    )
    & (
        df["Rainfall_Shift_pct"]
        == rainfall
    )
]


if len(selected) != 1:
    st.error(
        "Scenario lookup failed."
    )
    st.stop()


row = selected.iloc[0]


# ---------------------------------------------------------
# Scenario label
# ---------------------------------------------------------

st.subheader(
    f"{state} — {model}"
)

st.caption(
    f"Scenario: +{temperature:.0f} °C temperature, "
    f"{rainfall:.0f}% rainfall"
)


# ---------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------

baseline_yield = (
    row[
        "Predicted_Yield_Baseline"
    ]
)

stressed_yield = (
    row[
        "Predicted_Yield_Stress"
    ]
)

absolute_change = (
    row[
        "Yield_Change_MT_Ha"
    ]
)

percentage_change = (
    row[
        "Yield_Change_pct"
    ]
)


col1, col2, col3, col4 = (
    st.columns(4)
)


col1.metric(
    "Baseline Prediction",
    f"{baseline_yield:.2f} MT/ha",
)


col2.metric(
    "Stressed Prediction",
    f"{stressed_yield:.2f} MT/ha",
)


col3.metric(
    "Absolute Change",
    f"{absolute_change:+.2f} MT/ha",
)


col4.metric(
    "Relative Change",
    f"{percentage_change:+.1f}%",
)


# ---------------------------------------------------------
# Baseline vs stress visualization
# ---------------------------------------------------------

comparison = pd.DataFrame(
    {
        "Condition": [
            "Baseline",
            "Stress Scenario",
        ],
        "Predicted Yield": [
            baseline_yield,
            stressed_yield,
        ],
    }
)


fig = px.bar(
    comparison,
    x="Condition",
    y="Predicted Yield",
    text_auto=".2f",
    title=(
        f"Predicted Yield Response — {state}"
    ),
    labels={
        "Predicted Yield":
            "Yield (MT/ha)",
    },
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


# ---------------------------------------------------------
# Scenario response surface
# ---------------------------------------------------------

st.divider()

st.subheader(
    "Scenario response across all stress levels"
)


state_model = df[
    (df["State"] == state)
    & (df["Model"] == model)
].copy()


pivot = state_model.pivot_table(
    index="Temperature_Shift_C",
    columns="Rainfall_Shift_pct",
    values="Yield_Change_pct",
)


fig = px.imshow(
    pivot,
    text_auto=".1f",
    aspect="auto",
    title=(
        "Predicted Yield Change (%)"
    ),
    labels={
        "x":
            "Rainfall Change (%)",

        "y":
            "Temperature Shift (°C)",

        "color":
            "Yield Change (%)",
    },
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


# ---------------------------------------------------------
# National/model-level distribution
# ---------------------------------------------------------

st.subheader(
    "State-level responses under the selected scenario"
)


scenario_all_states = df[
    (df["Model"] == model)
    & (
        df["Temperature_Shift_C"]
        == temperature
    )
    & (
        df["Rainfall_Shift_pct"]
        == rainfall
    )
].copy()


scenario_all_states = (
    scenario_all_states
    .sort_values(
        "Yield_Change_pct"
    )
)


fig = px.bar(
    scenario_all_states,
    x="State",
    y="Yield_Change_pct",
    title=(
        f"{model}: State-Level Predicted Response"
    ),
    labels={
        "Yield_Change_pct":
            "Predicted Yield Change (%)",
    },
)


fig.add_hline(
    y=0,
    line_dash="dash",
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


st.warning(
    """
These scenarios are controlled counterfactual sensitivity tests.

They are **not climate forecasts**, **not causal estimates**, and should not
be interpreted as predictions of future maize yield under climate change.
"""
)