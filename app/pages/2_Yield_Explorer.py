from __future__ import annotations

import streamlit as st

from components.loaders import (
    load_main_dataset,
)

from components.charts import (
    yield_trend_chart,
    rainfall_trend_chart,
    temperature_chart,
)


st.set_page_config(
    page_title="Yield Explorer",
    page_icon="🌾",
    layout="wide",
)


df = load_main_dataset()


st.title(
    "🌾 State Yield Explorer"
)


state = st.selectbox(
    "Select a state",
    sorted(
        df["State"].unique()
    ),
)


state_df = (
    df[
        df["State"] == state
    ]
    .sort_values("Year")
)


latest = state_df.iloc[-1]


col1, col2, col3, col4 = (
    st.columns(4)
)


col1.metric(
    "2024 Yield",
    f"{latest['Yield_MT_Ha']:.2f} MT/ha",
)

col2.metric(
    "Seasonal Rainfall",
    f"{latest['Seasonal_Rainfall_mm']:.0f} mm",
)

col3.metric(
    "Mean Tmax",
    f"{latest['Mean_Tmax_C']:.1f} °C",
)

col4.metric(
    "Maximum Dry Spell",
    f"{int(latest['Max_CDD_days'])} days",
)


st.plotly_chart(
    yield_trend_chart(
        df,
        state,
    ),
    use_container_width=True,
)


left, right = st.columns(2)


with left:

    st.plotly_chart(
        rainfall_trend_chart(
            df,
            state,
        ),
        use_container_width=True,
    )


with right:

    st.plotly_chart(
        temperature_chart(
            df,
            state,
        ),
        use_container_width=True,
    )


with st.expander(
    "View state-level data"
):

    st.dataframe(
        state_df,
        use_container_width=True,
        hide_index=True,
    )