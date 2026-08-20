from __future__ import annotations


FULL_FEATURES = [
    "Seasonal_Rainfall_mm",
    "Seasonal_GDD_C",
    "Max_CDD_days",
    "Mean_Tmax_C",
    "Mean_Tmin_C",
    "Mean_RH_pct",
    "Mean_Solar_Radiation_MJ_m2_day",
    "Rainfall_Anomaly_Z_2000_2019",
]


FEATURE_SETS = {

    "Full": FULL_FEATURES,

    "Rainfall_Only": [
        "Seasonal_Rainfall_mm",
        "Max_CDD_days",
        "Rainfall_Anomaly_Z_2000_2019",
    ],

    "Temperature_Only": [
        "Seasonal_GDD_C",
        "Mean_Tmax_C",
        "Mean_Tmin_C",
    ],

    "Atmospheric_Only": [
        "Mean_RH_pct",
        "Mean_Solar_Radiation_MJ_m2_day",
    ],

    "No_Rainfall_Anomaly": [
        "Seasonal_Rainfall_mm",
        "Seasonal_GDD_C",
        "Max_CDD_days",
        "Mean_Tmax_C",
        "Mean_Tmin_C",
        "Mean_RH_pct",
        "Mean_Solar_Radiation_MJ_m2_day",
    ],

    "No_RH": [
        "Seasonal_Rainfall_mm",
        "Seasonal_GDD_C",
        "Max_CDD_days",
        "Mean_Tmax_C",
        "Mean_Tmin_C",
        "Mean_Solar_Radiation_MJ_m2_day",
        "Rainfall_Anomaly_Z_2000_2019",
    ],

    "No_GDD": [
        "Seasonal_Rainfall_mm",
        "Max_CDD_days",
        "Mean_Tmax_C",
        "Mean_Tmin_C",
        "Mean_RH_pct",
        "Mean_Solar_Radiation_MJ_m2_day",
        "Rainfall_Anomaly_Z_2000_2019",
    ],

    "No_Temperature_Pair": [
        "Seasonal_Rainfall_mm",
        "Seasonal_GDD_C",
        "Max_CDD_days",
        "Mean_RH_pct",
        "Mean_Solar_Radiation_MJ_m2_day",
        "Rainfall_Anomaly_Z_2000_2019",
    ],
}


def validate_feature_sets():
    full = set(FULL_FEATURES)

    for name, features in FEATURE_SETS.items():

        unknown = set(features) - full

        if unknown:
            raise ValueError(
                f"{name} contains unknown features: "
                f"{sorted(unknown)}"
            )

        if len(features) == 0:
            raise ValueError(
                f"{name} has no features."
            )