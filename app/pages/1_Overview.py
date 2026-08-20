from __future__ import annotations

import streamlit as st
import plotly.express as px

from components.loaders import (
    load_main_dataset,
)


st.set_page_config(
    page_title="Overview",
    page_icon="📊",
    layout="wide",
)


df = load_main_dataset()


st.title(
    "📊 Dataset Overview"
)


col1, col2, col3, col4 = (
    st.columns(4)
)

col1.metric(
    "States",
    df["State"].nunique(),
)

col2.metric(
    "Years",
    df["Year"].nunique(),
)

col3.metric(
    "Observations",
    len(df),
)

col4.metric(
    "Mean Yield",
    f"{df['Yield_MT_Ha'].mean():.2f} MT/ha",
)


st.divider()


left, right = st.columns(2)


with left:

    yearly = (
        df.groupby(
            "Year",
            as_index=False,
        )["Yield_MT_Ha"]
        .mean()
    )

    fig = px.line(
        yearly,
        x="Year",
        y="Yield_MT_Ha",
        markers=True,
        title="Average Maize Yield by Year",
        labels={
            "Yield_MT_Ha":
                "Mean Yield (MT/ha)",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


with right:

    state_mean = (
        df.groupby(
            "State",
            as_index=False,
        )["Yield_MT_Ha"]
        .mean()
        .sort_values(
            "Yield_MT_Ha",
            ascending=False,
        )
        .head(10)
    )

    fig = px.bar(
        state_mean,
        x="Yield_MT_Ha",
        y="State",
        orientation="h",
        title="Top 10 States by Mean Yield",
        labels={
            "Yield_MT_Ha":
                "Mean Yield (MT/ha)",
        },
    )

    fig.update_layout(
        yaxis={
            "categoryorder":
                "total ascending"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


st.subheader(
    "Weather feature distributions"
)


feature = st.selectbox(
    "Select feature",
    [
        "Seasonal_Rainfall_mm",
        "Seasonal_GDD_C",
        "Max_CDD_days",
        "Mean_Tmax_C",
        "Mean_Tmin_C",
        "Mean_RH_pct",
        "Mean_Solar_Radiation_MJ_m2_day",
        "Rainfall_Anomaly_Z_2000_2019",
    ],
)


fig = px.histogram(
    df,
    x=feature,
    nbins=25,
    title=f"Distribution of {feature}",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)