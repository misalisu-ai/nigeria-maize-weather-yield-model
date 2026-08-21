import streamlit as st


def apply_dashboard_style():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1250px;
        }
        h1, h2, h3 { letter-spacing: -0.02em; }
        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #e2e8e4;
            padding: 0.9rem 1rem;
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(31, 109, 66, 0.05);
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid #e6ebe7;
        }
        .capstone-hero {
            background: linear-gradient(135deg, #1f6d42 0%, #2e8b57 100%);
            border-radius: 20px;
            padding: 2rem 2.2rem;
            margin-bottom: 1.25rem;
            color: white;
        }
        .capstone-hero h1 {
            color: white;
            margin: 0 0 0.55rem 0;
            font-size: 2.35rem;
        }
        .capstone-hero p {
            color: #edf6f0;
            margin: 0;
            font-size: 1.02rem;
            max-width: 850px;
        }
        .small-note {
            color: #6b7280;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="capstone-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_branding():
    with st.sidebar:
        # Custom CSS applied strictly within the sidebar scope
        st.markdown(
            """
            <style>
            /* Beautiful styling for the branding card */
            .custom-sidebar-card {
                background: linear-gradient(135deg, #f0fdf4 0%, #e6f4ea 100%);
                border: 1px solid #d1ebd9;
                padding: 1.2rem;
                border-radius: 12px;
                margin-bottom: 1.5rem;
            }
            .custom-sidebar-title {
                color: #164e2b;
                font-size: 1.2rem;
                font-weight: 700;
                margin: 0 0 0.4rem 0 !important;
            }
            .custom-sidebar-sub {
                color: #476e54;
                font-size: 0.85rem;
                line-height: 1.4;
                margin: 0 !important;
            }
            /* Beautiful style for the bottom notice */
            .custom-sidebar-notice {
                background-color: #fff9f0;
                border-left: 4px solid #f59e0b;
                padding: 0.75rem 0.9rem;
                border-radius: 0 8px 8px 0;
                margin-top: 2rem;
                color: #78350f;
                font-size: 0.8rem;
                line-height: 1.4;
            }
            </style>
            
            <div class="custom-sidebar-card">
                <h3 class="custom-sidebar-title">🌽 Nigeria Maize Intelligence</h3>
                <p class="custom-sidebar-sub">3MTT Capstone • Weather-based ML decision support</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # A clean, native Streamlit line split
        st.divider()
        
        # The beautiful notice box at the bottom
        st.markdown(
            """
            <div class="custom-sidebar-notice">
                <strong>Notice:</strong> Climate-stress outputs are controlled sensitivity tests, not forecasts.
            </div>
            """,
            unsafe_allow_html=True
        )

