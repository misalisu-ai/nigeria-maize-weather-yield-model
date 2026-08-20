from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

A = (
    ROOT
    / "results"
    / "tables"
    / "climate_stress_feature_space_summary.csv"
)

B = (
    ROOT
    / "results"
    / "tables"
    / "climate_stress_weather_recomputed_summary.csv"
)

FEATURE_A = (
    ROOT
    / "results"
    / "climate_stress"
    / "feature_space_detailed.csv"
)

FEATURE_B = (
    ROOT
    / "results"
    / "climate_stress"
    / "weather_recomputed_features.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "climate_stress"
    / "method_comparison"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. SUMMARY-LEVEL METHOD COMPARISON
# ============================================================

a = pd.read_csv(A)
b = pd.read_csv(B)

keys = [
    "Model",
    "Scenario",
    "Scenario_Type",
    "Temperature_Shift_C",
    "Rainfall_Shift_pct",
]

compare = a.merge(
    b,
    on=keys,
    suffixes=("_04a", "_04b"),
    validate="one_to_one",
)

compare["Difference_Mean_MT_Ha_04b_minus_04a"] = (
    compare["Mean_Yield_Change_MT_Ha_04b"]
    - compare["Mean_Yield_Change_MT_Ha_04a"]
)

compare["Difference_Mean_pct_points_04b_minus_04a"] = (
    compare["Mean_Yield_Change_pct_04b"]
    - compare["Mean_Yield_Change_pct_04a"]
)

compare["Absolute_Method_Difference_pct_points"] = (
    compare[
        "Difference_Mean_pct_points_04b_minus_04a"
    ].abs()
)

compare.to_csv(
    OUT_DIR / "04a_vs_04b_summary.csv",
    index=False,
)


# ============================================================
# 2. FEATURE-LEVEL DIFFERENCES
# ============================================================

b_features = pd.read_csv(FEATURE_B)

baseline = (
    b_features[
        b_features["Scenario"] == "T+0_R0"
    ][
        [
            "State",
            "Max_CDD_days",
            "Seasonal_GDD_C",
            "Seasonal_Rainfall_mm",
            "Rainfall_Anomaly_Z_2000_2019",
        ]
    ]
    .copy()
)

baseline = baseline.rename(
    columns={
        "Max_CDD_days":
            "Baseline_Max_CDD_days",

        "Seasonal_GDD_C":
            "Baseline_Seasonal_GDD_C",

        "Seasonal_Rainfall_mm":
            "Baseline_Seasonal_Rainfall_mm",

        "Rainfall_Anomaly_Z_2000_2019":
            "Baseline_Rainfall_Anomaly_Z",
    }
)

feature_compare = b_features.merge(
    baseline,
    on="State",
    validate="many_to_one",
)

feature_compare["CDD_Change_days"] = (
    feature_compare["Max_CDD_days"]
    - feature_compare["Baseline_Max_CDD_days"]
)

feature_compare["GDD_Change_C"] = (
    feature_compare["Seasonal_GDD_C"]
    - feature_compare["Baseline_Seasonal_GDD_C"]
)

feature_compare["Rainfall_Change_mm"] = (
    feature_compare["Seasonal_Rainfall_mm"]
    - feature_compare[
        "Baseline_Seasonal_Rainfall_mm"
    ]
)

feature_compare["Rainfall_Anomaly_Change"] = (
    feature_compare[
        "Rainfall_Anomaly_Z_2000_2019"
    ]
    - feature_compare[
        "Baseline_Rainfall_Anomaly_Z"
    ]
)

feature_compare.to_csv(
    OUT_DIR / "04b_feature_changes.csv",
    index=False,
)


# ============================================================
# 3. CDD SUMMARY
# ============================================================

cdd_summary = (
    feature_compare
    .groupby(
        [
            "Scenario",
            "Scenario_Type",
            "Temperature_Shift_C",
            "Rainfall_Shift_pct",
        ],
        as_index=False,
    )
    .agg(
        States_with_CDD_Increase=(
            "CDD_Change_days",
            lambda x: int((x > 0).sum()),
        ),
        States_with_CDD_Decrease=(
            "CDD_Change_days",
            lambda x: int((x < 0).sum()),
        ),
        States_with_CDD_Unchanged=(
            "CDD_Change_days",
            lambda x: int((x == 0).sum()),
        ),
        Mean_CDD_Change_days=(
            "CDD_Change_days",
            "mean",
        ),
        Median_CDD_Change_days=(
            "CDD_Change_days",
            "median",
        ),
        Max_CDD_Increase_days=(
            "CDD_Change_days",
            "max",
        ),
    )
)

cdd_summary.to_csv(
    OUT_DIR / "04b_cdd_change_summary.csv",
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 70)
print("EXPERIMENT 04 — 04a vs 04b METHOD COMPARISON")
print("=" * 70)

focus = compare[
    compare["Scenario"].isin(
        [
            "T+0_R-10",
            "T+0_R-20",
            "T+0_R-30",
            "T+1_R0",
            "T+2_R0",
            "T+2_R-30",
        ]
    )
]

print("\n[1] MODEL RESPONSE COMPARISON")

print(
    focus[
        [
            "Model",
            "Scenario",
            "Mean_Yield_Change_pct_04a",
            "Mean_Yield_Change_pct_04b",
            "Difference_Mean_pct_points_04b_minus_04a",
        ]
    ].to_string(index=False)
)

print("\n[2] CDD RECOMPUTATION")

print(
    cdd_summary.to_string(index=False)
)

print("\nLargest method disagreements:")

print(
    compare
    .sort_values(
        "Absolute_Method_Difference_pct_points",
        ascending=False,
    )
    [
        [
            "Model",
            "Scenario",
            "Mean_Yield_Change_pct_04a",
            "Mean_Yield_Change_pct_04b",
            "Difference_Mean_pct_points_04b_minus_04a",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print(
    "\nSaved to:",
    OUT_DIR.resolve(),
)