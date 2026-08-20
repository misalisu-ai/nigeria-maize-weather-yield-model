from __future__ import annotations

import streamlit as st
import plotly.express as px

from components.loaders import (
    load_baseline_results,
    load_temporal_uncertainty,
    load_spatial_uncertainty,
)


st.set_page_config(
    page_title="Model Performance",
    page_icon="🤖",
    layout="wide",
)


st.title("🤖 Model Performance")

st.markdown(
    """
This page compares model performance under different evaluation settings.

The random split is included only as a diagnostic reference.
The main evaluation protocols are:

- **Temporal holdout:** train on 2020–2023, test on 2024.
- **State-held-out:** evaluate generalization to unseen Nigerian states.
"""
)


baseline = load_baseline_results()


# ---------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------

temporal = (
    baseline[
        baseline["Experiment"] == "Temporal"
    ]
    .sort_values("MAE")
)

spatial = (
    baseline[
        baseline["Experiment"] == "Spatial"
    ]
    .sort_values("MAE")
)


best_temporal = temporal.iloc[0]
best_spatial = spatial.iloc[0]


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Best 2024 MAE",
    f"{best_temporal['MAE']:.3f}",
    help=f"{best_temporal['Model']} on the temporal holdout",
)

col2.metric(
    "Best State-Held-Out MAE",
    f"{best_spatial['MAE']:.3f}",
    help=f"{best_spatial['Model']} across state-held-out folds",
)

col3.metric(
    "Best 2024 R²",
    f"{temporal['R2'].max():.3f}",
)

col4.metric(
    "Best Spatial R²",
    f"{spatial['R2'].max():.3f}",
)


st.divider()


# ---------------------------------------------------------
# MAE comparison
# ---------------------------------------------------------

st.subheader("Error across evaluation protocols")

fig = px.bar(
    baseline,
    x="Model",
    y="MAE",
    color="Experiment",
    barmode="group",
    title="Mean Absolute Error",
    labels={
        "MAE": "MAE (MT/ha)",
    },
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ---------------------------------------------------------
# R2 comparison
# ---------------------------------------------------------

fig = px.bar(
    baseline,
    x="Model",
    y="R2",
    color="Experiment",
    barmode="group",
    title="R² Across Evaluation Protocols",
)

fig.add_hline(
    y=0,
    line_dash="dash",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


st.caption(
    """
Negative R² indicates that the model performs worse than predicting the
test-set mean under that evaluation protocol.
"""
)


# ---------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------

st.divider()

st.subheader("Prediction uncertainty")

temporal_uncertainty = (
    load_temporal_uncertainty()
)

spatial_uncertainty = (
    load_spatial_uncertainty()
)


left, right = st.columns(2)


with left:

    st.markdown("#### Temporal conformal intervals")

    temp_display = (
        temporal_uncertainty[
            [
                "Model",
                "Coverage",
                "Mean_Interval_Width",
            ]
        ]
        .copy()
    )

    temp_display["Coverage"] = (
        temp_display["Coverage"] * 100
    )

    temp_display = temp_display.rename(
        columns={
            "Coverage":
                "Coverage (%)",

            "Mean_Interval_Width":
                "Mean Width (MT/ha)",
        }
    )

    st.dataframe(
        temp_display,
        hide_index=True,
        use_container_width=True,
    )


with right:

    st.markdown("#### State-held-out conformal intervals")

    spatial_display = (
        spatial_uncertainty[
            [
                "Model",
                "Coverage",
                "Mean_Interval_Width",
            ]
        ]
        .copy()
    )

    spatial_display["Coverage"] = (
        spatial_display["Coverage"] * 100
    )

    spatial_display = spatial_display.rename(
        columns={
            "Coverage":
                "Coverage (%)",

            "Mean_Interval_Width":
                "Mean Width (MT/ha)",
        }
    )

    st.dataframe(
        spatial_display,
        hide_index=True,
        use_container_width=True,
    )


st.info(
    """
Prediction intervals achieve approximately 90% empirical coverage for
most model/protocol combinations, but the intervals remain relatively wide.
This reflects substantial predictive uncertainty in a small weather-only dataset.
"""
)


# ---------------------------------------------------------
# Detailed results
# ---------------------------------------------------------

with st.expander(
    "View complete baseline comparison"
):

    st.dataframe(
        baseline,
        hide_index=True,
        use_container_width=True,
    )