from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT),
)


from src.data.loader import (
    load_dataset,
    FEATURES,
    TARGET,
    sha256_file,
)

from src.data.splits import (
    temporal_holdout,
)

from src.models.baselines import (
    ridge_model,
    random_forest_model,
)

from src.models.lightgbm_model import (
    lightgbm_model,
)

from src.robustness.scenarios import (
    build_scenarios,
)

from src.robustness.weather_recompute import (
    build_raw_file_map,
    load_state_daily,
    perturb_daily_weather,
    recompute_state_year_features,
)

from src.utils.experiment_metadata import (
    collect_metadata,
    save_metadata,
)


# =============================================================
# CONFIG
# =============================================================

SEED = 42
YEAR = 2024
T_BASE = 10.0
DRY_THRESHOLD_MM = 1.0


DATASET = (
    ROOT
    / "data"
    / "nigeria_maize_weather_yield_2020_2024_v1.0.csv"
)


RAW_WEATHER_DIR = (
    ROOT.parent
    / "nigeria-maize-weather-yield-data"
    / "data"
    / "raw"
    / "nasa_power_daily"
)


RAINFALL_CLIMATOLOGY = (
    ROOT.parent
    / "nigeria-maize-weather-yield-data"
    / "data"
    / "metadata"
    / "rainfall_climatology_2000_2019_v1.0.csv"
)


RESULT_DIR = (
    ROOT
    / "results"
    / "climate_stress"
)


TABLE_DIR = (
    ROOT
    / "results"
    / "tables"
)


META_DIR = (
    ROOT
    / "results"
    / "metadata"
)


# =============================================================
# IDENTITY CHECK
# =============================================================

def verify_recomputed_baseline(
    observed: pd.DataFrame,
    recomputed: pd.DataFrame,
    tolerance: float = 1e-6,
):
    """
    T+0_R0 must reconstruct the frozen Dataset v1.0
    weather features from the repaired daily weather.
    """

    check_features = [
        *FEATURES,
        "Weather_Valid_Days",
        "Weather_Total_Days",
    ]

    merged = observed[
        [
            "State",
            "Year",
            *check_features,
        ]
    ].merge(
        recomputed[
            [
                "State",
                "Year",
                *check_features,
            ]
        ],
        on=[
            "State",
            "Year",
        ],
        suffixes=(
            "_frozen",
            "_recomputed",
        ),
        validate="one_to_one",
    )

    mismatch_rows = []

    for feature in check_features:

        frozen = merged[
            f"{feature}_frozen"
        ]

        recalculated = merged[
            f"{feature}_recomputed"
        ]

        difference = (
            recalculated
            - frozen
        ).abs()

        max_difference = float(
            difference.max()
        )

        if max_difference > tolerance:

            bad = merged.loc[
                difference
                > tolerance,
                [
                    "State",
                    "Year",
                    f"{feature}_frozen",
                    f"{feature}_recomputed",
                ],
            ].copy()

            bad["Feature"] = feature

            bad["Difference"] = (
                difference[
                    difference > tolerance
                ].values
            )

            mismatch_rows.append(
                bad
            )

    if mismatch_rows:

        mismatches = pd.concat(
            mismatch_rows,
            ignore_index=True,
        )

        output = (
            RESULT_DIR
            / "weather_recomputed_identity_mismatches.csv"
        )

        mismatches.to_csv(
            output,
            index=False,
        )

        raise RuntimeError(
            "T+0_R0 daily-weather reconstruction does not "
            "match frozen Dataset v1.0. "
            f"See {output}"
        )

    print(
        "✓ T+0_R0 reproduces frozen 2024 weather features."
    )


# =============================================================
# MAIN
# =============================================================

def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    df = load_dataset(
        DATASET
    )

    dataset_hash = sha256_file(
        DATASET
    )

    train, test = temporal_holdout(
        df,
        train_end=2023,
    )

    test = test[
        test["Year"] == YEAR
    ].copy()

    if len(test) != 37:
        raise RuntimeError(
            f"Expected 37 observations for {YEAR}, "
            f"found {len(test)}"
        )

    # ---------------------------------------------------------
    # External data references
    # ---------------------------------------------------------

    if not RAW_WEATHER_DIR.exists():

        raise FileNotFoundError(
            f"Raw weather directory not found:\n"
            f"{RAW_WEATHER_DIR}"
        )

    if not RAINFALL_CLIMATOLOGY.exists():

        raise FileNotFoundError(
            f"Rainfall climatology not found:\n"
            f"{RAINFALL_CLIMATOLOGY}"
        )

    rainfall_climatology = (
        pd.read_csv(
            RAINFALL_CLIMATOLOGY
        )
    )

    raw_file_map = (
        build_raw_file_map(
            RAW_WEATHER_DIR
        )
    )

    missing_states = (
        set(
            test["State"]
        )
        - set(
            raw_file_map
        )
    )

    if missing_states:

        raise RuntimeError(
            "Missing daily weather files for: "
            f"{sorted(missing_states)}"
        )

    # ---------------------------------------------------------
    # Load each state's daily file once
    # ---------------------------------------------------------

    daily_by_state = {}

    for state in sorted(
        test["State"].unique()
    ):

        daily_by_state[
            state
        ] = load_state_daily(
            state,
            raw_file_map,
        )

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------

    models = {
        "Ridge":
            ridge_model(),

        "RandomForest":
            random_forest_model(
                random_state=SEED
            ),

        "LightGBM":
            lightgbm_model(
                random_state=SEED
            ),
    }

    for model in models.values():

        model.fit(
            train[FEATURES],
            train[TARGET],
        )

    # ---------------------------------------------------------
    # Scenario grid
    # ---------------------------------------------------------

    scenarios = (
        build_scenarios()
    )

    scenario_features = {}

    # ---------------------------------------------------------
    # Recompute daily-weather scenarios
    # ---------------------------------------------------------

    for _, scenario in scenarios.iterrows():

        scenario_rows = []

        for state in sorted(
            test["State"].unique()
        ):

            daily = daily_by_state[
                state
            ]

            stressed_daily = (
                perturb_daily_weather(
                    daily,
                    temperature_shift_c=
                        float(
                            scenario[
                                "Temperature_Shift_C"
                            ]
                        ),
                    rainfall_shift_pct=
                        float(
                            scenario[
                                "Rainfall_Shift_pct"
                            ]
                        ),
                )
            )

            features = (
                recompute_state_year_features(
                    state=state,
                    year=YEAR,
                    daily=stressed_daily,
                    rainfall_climatology=
                        rainfall_climatology,
                    t_base=T_BASE,
                    dry_threshold_mm=
                        DRY_THRESHOLD_MM,
                )
            )

            scenario_rows.append(
                features
            )

        scenario_df = pd.DataFrame(
            scenario_rows
        )

        scenario_features[
            scenario["Scenario"]
        ] = scenario_df

    # ---------------------------------------------------------
    # CRITICAL: reconstruct frozen baseline
    # ---------------------------------------------------------

    verify_recomputed_baseline(
        observed=test,
        recomputed=scenario_features[
            "T+0_R0"
        ],
    )

    # ---------------------------------------------------------
    # Baseline prediction uses the RECOMPUTED baseline.
    #
    # Because identity has passed, this must match the frozen
    # 2024 feature representation.
    # ---------------------------------------------------------

    baseline_features = (
        scenario_features[
            "T+0_R0"
        ]
        .set_index("State")
        .loc[
            test["State"]
        ]
        .reset_index()
    )

    baseline_predictions = {}

    for model_name, model in (
        models.items()
    ):

        baseline_predictions[
            model_name
        ] = model.predict(
            baseline_features[
                FEATURES
            ]
        )

    # ---------------------------------------------------------
    # Counterfactual predictions
    # ---------------------------------------------------------

    rows = []

    for _, scenario in scenarios.iterrows():

        stressed = (
            scenario_features[
                scenario[
                    "Scenario"
                ]
            ]
            .set_index("State")
            .loc[
                test["State"]
            ]
            .reset_index()
        )

        for model_name, model in (
            models.items()
        ):

            stressed_predictions = (
                model.predict(
                    stressed[
                        FEATURES
                    ]
                )
            )

            baseline_prediction = (
                baseline_predictions[
                    model_name
                ]
            )

            for (
                state,
                year,
                baseline_yield,
                stressed_yield,
            ) in zip(
                test["State"],
                test["Year"],
                baseline_prediction,
                stressed_predictions,
            ):

                change = (
                    stressed_yield
                    - baseline_yield
                )

                pct_change = (
                    (
                        change
                        / baseline_yield
                    )
                    * 100.0
                    if baseline_yield != 0
                    else np.nan
                )

                rows.append(
                    {
                        "State":
                            state,

                        "Year":
                            year,

                        "Model":
                            model_name,

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

                        "Predicted_Yield_Baseline":
                            baseline_yield,

                        "Predicted_Yield_Stress":
                            stressed_yield,

                        "Yield_Change_MT_Ha":
                            change,

                        "Yield_Change_pct":
                            pct_change,
                    }
                )

    detailed = pd.DataFrame(
        rows
    )

    # ---------------------------------------------------------
    # Verify prediction identity too
    # ---------------------------------------------------------

    baseline_result = detailed[
        detailed[
            "Scenario"
        ] == "T+0_R0"
    ]

    max_baseline_change = (
        baseline_result[
            "Yield_Change_MT_Ha"
        ]
        .abs()
        .max()
    )

    if max_baseline_change > 1e-10:

        raise RuntimeError(
            "Baseline prediction identity failed. "
            f"Maximum change = {max_baseline_change}"
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = (
        detailed
        .groupby(
            [
                "Model",
                "Scenario",
                "Scenario_Type",
                "Temperature_Shift_C",
                "Rainfall_Shift_pct",
            ],
            as_index=False,
        )
        .agg(
            Mean_Yield_Change_MT_Ha=(
                "Yield_Change_MT_Ha",
                "mean",
            ),

            Median_Yield_Change_MT_Ha=(
                "Yield_Change_MT_Ha",
                "median",
            ),

            Q25_Yield_Change_MT_Ha=(
                "Yield_Change_MT_Ha",
                lambda x:
                    x.quantile(
                        0.25
                    ),
            ),

            Q75_Yield_Change_MT_Ha=(
                "Yield_Change_MT_Ha",
                lambda x:
                    x.quantile(
                        0.75
                    ),
            ),

            Min_Yield_Change_MT_Ha=(
                "Yield_Change_MT_Ha",
                "min",
            ),

            Max_Yield_Change_MT_Ha=(
                "Yield_Change_MT_Ha",
                "max",
            ),

            Mean_Yield_Change_pct=(
                "Yield_Change_pct",
                "mean",
            ),

            Median_Yield_Change_pct=(
                "Yield_Change_pct",
                "median",
            ),
        )
    )

    # ---------------------------------------------------------
    # Save scenario features themselves.
    # Useful for auditing CDD/GDD differences later.
    # ---------------------------------------------------------

    feature_frames = []

    for _, scenario in scenarios.iterrows():

        frame = scenario_features[
            scenario["Scenario"]
        ].copy()

        frame.insert(
            0,
            "Scenario",
            scenario["Scenario"],
        )

        frame.insert(
            1,
            "Scenario_Type",
            scenario["Scenario_Type"],
        )

        frame.insert(
            2,
            "Temperature_Shift_C",
            scenario[
                "Temperature_Shift_C"
            ],
        )

        frame.insert(
            3,
            "Rainfall_Shift_pct",
            scenario[
                "Rainfall_Shift_pct"
            ],
        )

        feature_frames.append(
            frame
        )

    all_features = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Save artifacts
    # ---------------------------------------------------------

    detailed.to_csv(
        RESULT_DIR
        / "weather_recomputed_detailed.csv",
        index=False,
    )

    all_features.to_csv(
        RESULT_DIR
        / "weather_recomputed_features.csv",
        index=False,
    )

    summary.to_csv(
        TABLE_DIR
        / "climate_stress_weather_recomputed_summary.csv",
        index=False,
    )

    scenarios.to_csv(
        RESULT_DIR
        / "weather_recomputed_scenario_grid.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = collect_metadata(
        experiment_name=(
            "04b_climate_stress_weather_recomputed"
        ),

        dataset_path=str(
            DATASET
        ),

        dataset_sha256=
            dataset_hash,

        features=
            FEATURES,

        target=
            TARGET,

        seed=
            SEED,

        split_definition={
            "training":
                "2020-2023",

            "counterfactual_year":
                2024,

            "counterfactual_level":
                "daily weather",

            "temperature_perturbation":
                "uniform additive shift to daily Tmax and Tmin",

            "rainfall_perturbation":
                "uniform multiplicative shift to daily precipitation",

            "GDD":
                (
                    "sum(max(((Tmax + Tmin)/2) - 10, 0))"
                ),

            "CDD":
                (
                    "maximum consecutive days with rainfall < 1 mm"
                ),

            "rainfall_climatology":
                (
                    "fixed repaired 2000-2019 Dataset v1.0 climatology"
                ),

            "RH":
                "unperturbed",

            "solar_radiation":
                "unperturbed",

            "physical_climate_projection":
                False,
        },

        model_parameters={
            name:
                model.get_params()

            for name, model
            in models.items()
        },
    )

    save_metadata(
        metadata,
        META_DIR
        / "climate_stress_weather_recomputed.json",
    )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPERIMENT 04b — WEATHER-RECOMPUTED CLIMATE STRESS"
    )

    print(
        "=" * 70
    )

    print(
        "\n✓ Baseline feature reconstruction: PASS"
    )

    print(
        "✓ Baseline prediction identity: PASS"
    )

    print(
        "\nScenario summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()