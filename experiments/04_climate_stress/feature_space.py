from __future__ import annotations

from pathlib import Path
import sys

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
from src.data.splits import temporal_holdout
from src.models.baselines import (
    ridge_model,
    random_forest_model,
)
from src.models.lightgbm_model import (
    lightgbm_model,
)
from src.robustness.climate_shift import (
    apply_feature_space_stress,
)
from src.robustness.scenarios import (
    build_scenarios,
)
from src.utils.experiment_metadata import (
    collect_metadata,
    save_metadata,
)


SEED = 42

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


def main():

    # =========================================================
    # Load frozen data
    # =========================================================

    df = load_dataset(
        DATASET
    )

    dataset_hash = sha256_file(
        DATASET
    )

    baseline_stats = pd.read_csv(
        RAINFALL_BASELINES
    )

    # =========================================================
    # Temporal training protocol
    # =========================================================

    train, test = temporal_holdout(
        df,
        train_end=2023,
    )

    # Keep metadata required by stress generator.
    stress_columns = [
        "State",
        "Year",
        "Weather_Valid_Days",
        *FEATURES,
    ]

    X_test_full = (
        test[
            stress_columns
        ]
        .copy()
    )

    X_train = train[
        FEATURES
    ]

    y_train = train[
        TARGET
    ]

    # =========================================================
    # Models
    # =========================================================

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
            X_train,
            y_train,
        )

    # =========================================================
    # Baseline predictions
    # =========================================================

    baseline_predictions = {}

    for name, model in models.items():

        baseline_predictions[
            name
        ] = model.predict(
            test[FEATURES]
        )

    # =========================================================
    # Scenario grid
    # =========================================================

    scenarios = build_scenarios()

    rows = []

    for _, scenario in (
        scenarios.iterrows()
    ):

        stressed = (
            apply_feature_space_stress(
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
                    baseline_stats,
            )
        )

        X_stressed = stressed[
            FEATURES
        ]

        for model_name, model in (
            models.items()
        ):

            stressed_pred = (
                model.predict(
                    X_stressed
                )
            )

            baseline_pred = (
                baseline_predictions[
                    model_name
                ]
            )

            for i, (
                state,
                year,
                base,
                stress,
            ) in enumerate(
                zip(
                    test["State"],
                    test["Year"],
                    baseline_pred,
                    stressed_pred,
                )
            ):

                absolute_change = (
                    stress
                    - base
                )

                pct_change = (
                    (
                        absolute_change
                        / base
                    )
                    * 100
                    if base != 0
                    else float("nan")
                )

                rows.append(
                    {
                        "State": state,
                        "Year": year,
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
                            base,
                        "Predicted_Yield_Stress":
                            stress,
                        "Yield_Change_MT_Ha":
                            absolute_change,
                        "Yield_Change_pct":
                            pct_change,
                    }
                )

    detailed = pd.DataFrame(
        rows
    )

    if (
    scenario["Temperature_Shift_C"] == 0
    and scenario["Rainfall_Shift_pct"] == 0
    ):

        original = (
            X_test_full[
                FEATURES
            ]
            .reset_index(
                drop=True
            )
        )

        reconstructed = (
            stressed[
                FEATURES
            ]
            .reset_index(
                drop=True
            )
        )

        difference = (
            reconstructed
            - original
        ).abs()

        max_difference = (
            difference
            .to_numpy()
            .max()
        )

        if max_difference > 1e-6:

            raise RuntimeError(
                "Baseline scenario is not identity-preserving. "
                f"Maximum feature difference = {max_difference}"
            )

    # =========================================================
    # Summary
    # =========================================================

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
                    x.quantile(0.25),
            ),
            Q75_Yield_Change_MT_Ha=(
                "Yield_Change_MT_Ha",
                lambda x:
                    x.quantile(0.75),
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

    # =========================================================
    # Save
    # =========================================================

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed.to_csv(
        RESULT_DIR
        / "feature_space_detailed.csv",
        index=False,
    )

    summary.to_csv(
        TABLE_DIR
        / "climate_stress_feature_space_summary.csv",
        index=False,
    )

    scenarios.to_csv(
        RESULT_DIR
        / "scenario_grid.csv",
        index=False,
    )

    # =========================================================
    # Metadata
    # =========================================================

    metadata = collect_metadata(
        experiment_name=(
            "04a_climate_stress_feature_space"
        ),
        dataset_path=str(
            DATASET
        ),
        dataset_sha256=
            dataset_hash,
        features=FEATURES,
        target=TARGET,
        seed=SEED,
        split_definition={
            "training":
                "2020-2023",
            "stress_evaluation":
                "2024 temporal holdout",
            "scenario_type":
                "controlled feature-space counterfactual",
            "physical_climate_projection":
                False,
            "CDD_treatment":
                "held fixed",
            "GDD_treatment":
                (
                    "approximated as "
                    "GDD + deltaT * Weather_Valid_Days"
                ),
            "rainfall_anomaly":
                (
                    "recomputed from state-specific "
                    "2000-2019 climatology"
                ),
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
        / "climate_stress_feature_space.json",
    )

    # =========================================================
    # Display
    # =========================================================

    print(
        "\nClimate stress scenarios:"
    )

    print(
        scenarios.to_string(
            index=False
        )
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