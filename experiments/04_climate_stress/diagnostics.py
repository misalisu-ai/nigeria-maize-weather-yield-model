from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import (
    load_dataset,
    FEATURES,
    TARGET,
)
from src.data.splits import temporal_holdout
from src.robustness.climate_shift import (
    apply_feature_space_stress,
)
from src.robustness.scenarios import (
    build_scenarios,
)


DATASET = (
    ROOT
    / "data"
    / "nigeria_maize_weather_yield_2020_2024_v1.0.csv"
)

RAINFALL_BASELINES = (
    ROOT
    / "data"
    / "reference"
    / "rainfall_climatology_2000_2019_v1.0.csv"
)

DETAILED_RESULTS = (
    ROOT
    / "results"
    / "climate_stress"
    / "feature_space_detailed.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "climate_stress"
    / "diagnostics"
)


STRESS_FEATURES = [
    "Seasonal_Rainfall_mm",
    "Seasonal_GDD_C",
    "Mean_Tmax_C",
    "Mean_Tmin_C",
    "Rainfall_Anomaly_Z_2000_2019",
]


def save_state_extremes(
    detailed: pd.DataFrame,
):
    """
    Save five most negative and five most positive
    state responses for every model/scenario.
    """

    rows = []

    for (
        model,
        scenario,
    ), group in detailed.groupby(
        ["Model", "Scenario"]
    ):

        group = group.sort_values(
            "Yield_Change_MT_Ha"
        )

        negative = group.head(5).copy()
        positive = group.tail(5).copy()

        negative["Extreme_Type"] = (
            "Most_Negative"
        )

        positive["Extreme_Type"] = (
            "Most_Positive"
        )

        selected = pd.concat(
            [negative, positive],
            ignore_index=True,
        )

        rows.append(selected)

    result = pd.concat(
        rows,
        ignore_index=True,
    )

    cols = [
        "Model",
        "Scenario",
        "Scenario_Type",
        "Extreme_Type",
        "State",
        "Year",
        "Temperature_Shift_C",
        "Rainfall_Shift_pct",
        "Predicted_Yield_Baseline",
        "Predicted_Yield_Stress",
        "Yield_Change_MT_Ha",
        "Yield_Change_pct",
    ]

    result = result[cols]

    result.to_csv(
        OUT_DIR
        / "climate_stress_state_extremes.csv",
        index=False,
    )

    return result


def drought_monotonicity(
    detailed: pd.DataFrame,
):
    """
    Test whether predicted yield is non-increasing as
    rainfall is progressively reduced.

    This is a diagnostic, not a physical constraint.
    """

    rows = []

    temperature_levels = sorted(
        detailed[
            "Temperature_Shift_C"
        ].unique()
    )

    models = sorted(
        detailed["Model"].unique()
    )

    states = sorted(
        detailed["State"].unique()
    )

    rainfall_order = [
        0.0,
        -10.0,
        -20.0,
        -30.0,
    ]

    for model in models:

        model_df = detailed[
            detailed["Model"] == model
        ]

        for temp in temperature_levels:

            temp_df = model_df[
                model_df[
                    "Temperature_Shift_C"
                ] == temp
            ]

            monotonic_states = 0

            state_rows = []

            for state in states:

                state_df = temp_df[
                    temp_df["State"] == state
                ].copy()

                state_df = (
                    state_df
                    .set_index(
                        "Rainfall_Shift_pct"
                    )
                    .reindex(
                        rainfall_order
                    )
                )

                if (
                    state_df[
                        "Predicted_Yield_Stress"
                    ]
                    .isna()
                    .any()
                ):
                    raise RuntimeError(
                        f"Missing scenario for "
                        f"{model}, {state}, T+{temp:g}"
                    )

                values = (
                    state_df[
                        "Predicted_Yield_Stress"
                    ]
                    .to_numpy()
                )

                diffs = np.diff(values)

                is_monotonic = bool(
                    np.all(
                        diffs <= 1e-10
                    )
                )

                if is_monotonic:
                    monotonic_states += 1

                state_rows.append(
                    {
                        "Model": model,
                        "Temperature_Shift_C":
                            temp,
                        "State": state,
                        "Monotonic_Drought_Response":
                            is_monotonic,
                        "Yield_R0":
                            values[0],
                        "Yield_Rminus10":
                            values[1],
                        "Yield_Rminus20":
                            values[2],
                        "Yield_Rminus30":
                            values[3],
                    }
                )

            state_result = pd.DataFrame(
                state_rows
            )

            state_result.to_csv(
                OUT_DIR
                / (
                    f"drought_monotonicity_"
                    f"{model}_T{temp:g}.csv"
                ),
                index=False,
            )

            rows.append(
                {
                    "Model": model,
                    "Temperature_Shift_C":
                        temp,
                    "Number_of_states":
                        len(states),
                    "Monotonic_drought_states":
                        monotonic_states,
                    "Non_monotonic_drought_states":
                        len(states)
                        - monotonic_states,
                    "Monotonicity_rate":
                        monotonic_states
                        / len(states),
                }
            )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUT_DIR
        / "drought_monotonicity_summary.csv",
        index=False,
    )

    return summary


def extrapolation_diagnostics():
    """
    Measure whether stressed 2024 feature values fall
    outside the 2020-2023 training support.
    """

    df = load_dataset(
        DATASET
    )

    rainfall_baselines = pd.read_csv(
        RAINFALL_BASELINES
    )

    train, test = temporal_holdout(
        df,
        train_end=2023,
    )

    scenarios = build_scenarios()

    train_bounds = {}

    for feature in STRESS_FEATURES:

        train_bounds[feature] = {
            "min": train[feature].min(),
            "max": train[feature].max(),
        }

    stress_columns = [
        "State",
        "Year",
        "Weather_Valid_Days",
        *FEATURES,
    ]

    X_test_full = test[
        stress_columns
    ].copy()

    detail_rows = []
    summary_rows = []

    for _, scenario in scenarios.iterrows():

        stressed = apply_feature_space_stress(
            X_test_full,
            temperature_shift_c=
                scenario[
                    "Temperature_Shift_C"
                ],
            rainfall_shift_pct=
                scenario[
                    "Rainfall_Shift_pct"
                ],
            rainfall_baselines=
                rainfall_baselines,
        )

        scenario_total_outside = (
            np.zeros(
                len(stressed),
                dtype=bool,
            )
        )

        for feature in STRESS_FEATURES:

            lower = (
                train_bounds[
                    feature
                ]["min"]
            )

            upper = (
                train_bounds[
                    feature
                ]["max"]
            )

            values = stressed[
                feature
            ].to_numpy()

            below = values < lower
            above = values > upper
            outside = below | above

            scenario_total_outside |= (
                outside
            )

            summary_rows.append(
                {
                    "Scenario":
                        scenario[
                            "Scenario"
                        ],
                    "Scenario_Type":
                        scenario[
                            "Scenario_Type"
                        ],
                    "Temperature_Shift_C":
                        scenario[
                            "Temperature_Shift_C"
                        ],
                    "Rainfall_Shift_pct":
                        scenario[
                            "Rainfall_Shift_pct"
                        ],
                    "Feature":
                        feature,
                    "Training_Min":
                        lower,
                    "Training_Max":
                        upper,
                    "Below_Training_Min":
                        int(
                            below.sum()
                        ),
                    "Above_Training_Max":
                        int(
                            above.sum()
                        ),
                    "Outside_Training_Range":
                        int(
                            outside.sum()
                        ),
                    "Outside_Training_Range_pct":
                        float(
                            outside.mean()
                            * 100
                        ),
                }
            )

            for i, row in (
                stressed
                .reset_index(drop=True)
                .iterrows()
            ):

                detail_rows.append(
                    {
                        "Scenario":
                            scenario[
                                "Scenario"
                            ],
                        "State":
                            row["State"],
                        "Year":
                            row["Year"],
                        "Feature":
                            feature,
                        "Value":
                            row[feature],
                        "Training_Min":
                            lower,
                        "Training_Max":
                            upper,
                        "Below_Training_Min":
                            bool(
                                below[i]
                            ),
                        "Above_Training_Max":
                            bool(
                                above[i]
                            ),
                        "Outside_Training_Range":
                            bool(
                                outside[i]
                            ),
                    }
                )

        summary_rows.append(
            {
                "Scenario":
                    scenario[
                        "Scenario"
                    ],
                "Scenario_Type":
                    scenario[
                        "Scenario_Type"
                    ],
                "Temperature_Shift_C":
                    scenario[
                        "Temperature_Shift_C"
                    ],
                "Rainfall_Shift_pct":
                    scenario[
                        "Rainfall_Shift_pct"
                    ],
                "Feature":
                    "__ANY_STRESS_FEATURE__",
                "Training_Min":
                    np.nan,
                "Training_Max":
                    np.nan,
                "Below_Training_Min":
                    np.nan,
                "Above_Training_Max":
                    np.nan,
                "Outside_Training_Range":
                    int(
                        scenario_total_outside.sum()
                    ),
                "Outside_Training_Range_pct":
                    float(
                        scenario_total_outside.mean()
                        * 100
                    ),
            }
        )

    detail = pd.DataFrame(
        detail_rows
    )

    summary = pd.DataFrame(
        summary_rows
    )

    detail.to_csv(
        OUT_DIR
        / "training_support_detailed.csv",
        index=False,
    )

    summary.to_csv(
        OUT_DIR
        / "training_support_summary.csv",
        index=False,
    )

    return summary


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DETAILED_RESULTS.exists():
        raise FileNotFoundError(
            "Run feature_space.py first."
        )

    detailed = pd.read_csv(
        DETAILED_RESULTS
    )

    print("=" * 70)
    print(
        "EXPERIMENT 04 — DIAGNOSTIC AUDIT"
    )
    print("=" * 70)

    # ---------------------------------------------------------
    # State extremes
    # ---------------------------------------------------------

    extremes = save_state_extremes(
        detailed
    )

    print(
        "\n[1] STATE RESPONSE EXTREMES"
    )

    focus = extremes[
        extremes[
            "Scenario"
        ].isin(
            [
                "T+0_R-30",
                "T+2_R0",
                "T+2_R-30",
            ]
        )
    ]

    print(
        focus[
            [
                "Model",
                "Scenario",
                "Extreme_Type",
                "State",
                "Yield_Change_MT_Ha",
                "Yield_Change_pct",
            ]
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Monotonicity
    # ---------------------------------------------------------

    monotonic = drought_monotonicity(
        detailed
    )

    print(
        "\n[2] DROUGHT MONOTONICITY"
    )

    print(
        monotonic.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Training support
    # ---------------------------------------------------------

    support = (
        extrapolation_diagnostics()
    )

    print(
        "\n[3] TRAINING-SUPPORT / EXTRAPOLATION"
    )

    focus_support = support[
        support["Scenario"].isin(
            [
                "T+0_R0",
                "T+0_R-30",
                "T+2_R0",
                "T+2_R-30",
            ]
        )
    ]

    print(
        focus_support.to_string(
            index=False
        )
    )

    print(
        "\nSaved diagnostics to:"
    )

    print(
        OUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()