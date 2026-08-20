from __future__ import annotations

import pandas as pd


def apply_feature_space_stress(
    X: pd.DataFrame,
    *,
    temperature_shift_c: float,
    rainfall_shift_pct: float,
    rainfall_baselines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply a controlled feature-space climate stress.

    Important:
    - This is NOT a physical climate simulation.
    - Temperature changes update Tmax, Tmin and approximate GDD.
    - Rainfall changes update seasonal rainfall and its climatological Z-score.
    - Max_CDD_days is held fixed because it cannot be reconstructed from
      aggregate seasonal rainfall alone.
    """

    stressed = X.copy()

    # ---------------------------------------------------------
    # Temperature
    # ---------------------------------------------------------

    stressed["Mean_Tmax_C"] = (
        stressed["Mean_Tmax_C"]
        + temperature_shift_c
    )

    stressed["Mean_Tmin_C"] = (
        stressed["Mean_Tmin_C"]
        + temperature_shift_c
    )

    # Under a uniform warming assumption, approximate the
    # change in accumulated GDD across valid seasonal days.
    stressed["Seasonal_GDD_C"] = (
        stressed["Seasonal_GDD_C"]
        + (
            temperature_shift_c
            * stressed["Weather_Valid_Days"]
        )
    )

    # ---------------------------------------------------------
    # Rainfall
    # ---------------------------------------------------------

    rainfall_factor = (
        1.0
        + rainfall_shift_pct / 100.0
    )

    if rainfall_factor < 0:
        raise ValueError(
            "Rainfall stress cannot produce negative rainfall."
        )

    stressed["Seasonal_Rainfall_mm"] = (
        stressed["Seasonal_Rainfall_mm"]
        * rainfall_factor
    )

    # ---------------------------------------------------------
    # Recalculate rainfall anomaly consistently
    # ---------------------------------------------------------

    required_baseline_cols = {
        "State",
        "Baseline_Mean_mm",
        "Baseline_SD_mm",
    }

    missing = (
        required_baseline_cols
        - set(rainfall_baselines.columns)
    )

    if missing:
        raise ValueError(
            "Rainfall baseline file missing columns: "
            f"{sorted(missing)}"
        )

    lookup = (
        rainfall_baselines[
            [
                "State",
                "Baseline_Mean_mm",
                "Baseline_SD_mm",
            ]
        ]
        .drop_duplicates("State")
        .set_index("State")
    )

    baseline_mean = (
        stressed["State"]
        .map(
            lookup[
                "Baseline_Mean_mm"
            ]
        )
    )

    baseline_sd = (
        stressed["State"]
        .map(
            lookup[
                "Baseline_SD_mm"
            ]
        )
    )

    if (
        baseline_mean.isna().any()
        or baseline_sd.isna().any()
    ):
        missing_states = stressed.loc[
            baseline_mean.isna()
            | baseline_sd.isna(),
            "State",
        ].unique()

        raise ValueError(
            "Missing climatology for states: "
            f"{sorted(missing_states)}"
        )

    if (baseline_sd <= 0).any():
        raise ValueError(
            "Rainfall baseline SD must be positive."
        )

    stressed[
        "Rainfall_Anomaly_Z_2000_2019"
    ] = (
        (
            stressed["Seasonal_Rainfall_mm"]
            - baseline_mean
        )
        / baseline_sd
    )

    return stressed