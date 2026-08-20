from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def yield_trend_chart(
    df: pd.DataFrame,
    state: str,
):
    state_df = (
        df[
            df["State"] == state
        ]
        .sort_values("Year")
    )

    fig = px.line(
        state_df,
        x="Year",
        y="Yield_MT_Ha",
        markers=True,
        title=f"Maize Yield Trend — {state}",
        labels={
            "Yield_MT_Ha":
                "Yield (MT/ha)",
        },
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear"
        ),
        hovermode="x unified",
    )

    return fig


def rainfall_trend_chart(
    df: pd.DataFrame,
    state: str,
):
    state_df = (
        df[
            df["State"] == state
        ]
        .sort_values("Year")
    )

    fig = px.bar(
        state_df,
        x="Year",
        y="Seasonal_Rainfall_mm",
        title=f"Seasonal Rainfall — {state}",
        labels={
            "Seasonal_Rainfall_mm":
                "Rainfall (mm)",
        },
    )

    return fig


def temperature_chart(
    df: pd.DataFrame,
    state: str,
):
    state_df = (
        df[
            df["State"] == state
        ]
        .sort_values("Year")
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=state_df["Year"],
            y=state_df["Mean_Tmax_C"],
            mode="lines+markers",
            name="Mean Tmax",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=state_df["Year"],
            y=state_df["Mean_Tmin_C"],
            mode="lines+markers",
            name="Mean Tmin",
        )
    )

    fig.update_layout(
        title=f"Seasonal Temperature — {state}",
        xaxis_title="Year",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
    )

    return fig


def model_performance_chart(
    results: pd.DataFrame,
):
    fig = px.bar(
        results,
        x="Model",
        y="MAE",
        color="Experiment",
        barmode="group",
        title="Model MAE Across Evaluation Protocols",
        labels={
            "MAE":
                "Mean Absolute Error",
        },
    )

    return fig


def shap_importance_chart(
    df: pd.DataFrame,
    model: str,
):
    model_df = (
        df[
            df["Model"] == model
        ]
        .sort_values(
            "Mean_Absolute_SHAP"
        )
    )

    fig = px.bar(
        model_df,
        x="Mean_Absolute_SHAP",
        y="Feature",
        orientation="h",
        title=f"{model} — Feature Attribution",
        labels={
            "Mean_Absolute_SHAP":
                "Mean |SHAP value|",
        },
    )

    return fig