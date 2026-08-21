from __future__ import annotations

import streamlit as st

from components.loaders import (
    load_main_dataset,
)

from components.style import hero
hero(
    "Nigeria Maize Yield Intelligence",
    "Weather-based machine learning for state-level yield analysis, climate-stress sensitivity and explainable decision support.",
)

from components.style import apply_dashboard_style, sidebar_branding
apply_dashboard_style()
sidebar_branding()

st.set_page_config(
    page_title="Nigeria Maize Yield Intelligence",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)


df = load_main_dataset()


st.title(
    "🌽 Nigeria Maize Yield Intelligence"
)

st.subheader(
    "Weather-Based Machine Learning for "
    "Maize Yield Analysis Across Nigerian States"
)


st.markdown(
    """
This dashboard explores how seasonal weather patterns relate to
state-level maize yield across Nigeria.

It combines official maize-yield observations with NASA POWER
weather data and machine-learning models to examine:

- historical yield patterns,
- model generalization,
- controlled climate-stress sensitivity,
- and model explainability.
"""
)


st.divider()


col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:
    st.metric(
        "States / FCT",
        df["State"].nunique(),
    )

with col2:
    st.metric(
        "Observations",
        len(df),
    )

with col3:
    st.metric(
        "Years",
        (
            f"{df['Year'].min()}–"
            f"{df['Year'].max()}"
        ),
    )

with col4:
    st.metric(
        "Weather Features",
        8,
    )


st.divider()


left, right = st.columns(
    [1.5, 1]
)


with left:

    st.markdown(
        "### Why this project matters"
    )

    st.markdown(
        """
Maize production is sensitive to rainfall, temperature,
dry spells, humidity and solar conditions.

The goal of this project is not to claim that weather alone
determines agricultural output. Instead, it investigates how
much predictive information can be extracted from weather
variables and how reliably machine-learning models generalize
across time and geography.
"""
    )


with right:

    st.markdown(
        "### Data sources"
    )

    st.markdown(
        """
**Yield**

Nigerian Agricultural Performance / Productivity Survey reports.

**Weather**

NASA POWER daily meteorological observations.

**Coverage**

37 states/FCT × 5 years = 185 state-year observations.
"""
    )


st.divider()


st.info(
    """
Climate-stress scenarios shown in this dashboard are controlled
counterfactual sensitivity tests. They are not future climate
projections and should not be interpreted as causal estimates.
"""
)


st.markdown(
    "### Navigate using the sidebar"
)

st.markdown(
    """
1. **Overview**
2. **Yield Explorer**
3. **Model Performance**
4. **Climate Stress**
5. **Explainability**
"""
)