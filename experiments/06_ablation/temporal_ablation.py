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
    TARGET,
    sha256_file,
)

from src.models.baselines import (
    ridge_model,
    random_forest_model,
)

from src.models.lightgbm_model import (
    lightgbm_model,
)

from src.evaluation.metrics import (
    regression_metrics,
)

from src.evaluation.feature_sets import (
    FEATURE_SETS,
    validate_feature_sets,
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


TABLE_DIR = (
    ROOT
    / "results"
    / "tables"
)


PRED_DIR = (
    ROOT
    / "results"
    / "predictions"
)


META_DIR = (
    ROOT
    / "results"
    / "metadata"
)


def main():

    for directory in [
        TABLE_DIR,
        PRED_DIR,
        META_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    validate_feature_sets()

    df = load_dataset(
        DATASET
    )

    dataset_hash = sha256_file(
        DATASET
    )

    train = df[
        df["Year"] <= 2023
    ].copy()

    test = df[
        df["Year"] == 2024
    ].copy()

    if len(train) != 148:
        raise RuntimeError(
            f"Expected 148 training rows, got {len(train)}"
        )

    if len(test) != 37:
        raise RuntimeError(
            f"Expected 37 test rows, got {len(test)}"
        )

    model_factories = {

        "Ridge":
            ridge_model,

        "RandomForest":
            lambda:
                random_forest_model(
                    random_state=SEED
                ),

        "LightGBM":
            lambda:
                lightgbm_model(
                    random_state=SEED
                ),
    }

    result_rows = []
    prediction_rows = []

    for feature_set_name, features in (
        FEATURE_SETS.items()
    ):

        for model_name, factory in (
            model_factories.items()
        ):

            model = factory()

            model.fit(
                train[features],
                train[TARGET],
            )

            pred = model.predict(
                test[features]
            )

            metrics = regression_metrics(
                test[TARGET],
                pred,
            )

            result_rows.append(
                {
                    "Feature_Set":
                        feature_set_name,

                    "Feature_Count":
                        len(features),

                    "Model":
                        model_name,

                    **metrics,
                }
            )

            for (
                state,
                year,
                observed,
                predicted,
            ) in zip(
                test["State"],
                test["Year"],
                test[TARGET],
                pred,
            ):

                prediction_rows.append(
                    {
                        "Feature_Set":
                            feature_set_name,

                        "Model":
                            model_name,

                        "State":
                            state,

                        "Year":
                            year,

                        "Observed_Yield":
                            observed,

                        "Predicted_Yield":
                            predicted,

                        "Residual":
                            observed
                            - predicted,
                    }
                )

    results = pd.DataFrame(
        result_rows
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    # ---------------------------------------------------------
    # Compare each ablation against Full for same model
    # ---------------------------------------------------------

    full = (
        results[
            results["Feature_Set"]
            == "Full"
        ]
        .set_index("Model")
    )

    comparisons = []

    for _, row in results.iterrows():

        baseline = full.loc[
            row["Model"]
        ]

        comparisons.append(
            {
                **row.to_dict(),

                "Delta_MAE_vs_Full":
                    row["MAE"]
                    - baseline["MAE"],

                "Delta_RMSE_vs_Full":
                    row["RMSE"]
                    - baseline["RMSE"],

                "Delta_R2_vs_Full":
                    row["R2"]
                    - baseline["R2"],
            }
        )

    comparison = pd.DataFrame(
        comparisons
    )

    results.to_csv(
        TABLE_DIR
        / "temporal_ablation_metrics.csv",
        index=False,
    )

    comparison.to_csv(
        TABLE_DIR
        / "temporal_ablation_vs_full.csv",
        index=False,
    )

    predictions.to_csv(
        PRED_DIR
        / "temporal_ablation_predictions.csv",
        index=False,
    )

    metadata = collect_metadata(
        experiment_name=(
            "06a_temporal_feature_ablation"
        ),

        dataset_path=str(
            DATASET
        ),

        dataset_sha256=
            dataset_hash,

        features=[
            f"{name}: {features}"
            for name, features
            in FEATURE_SETS.items()
        ],

        target=
            TARGET,

        seed=
            SEED,

        split_definition={
            "train":
                "2020-2023",

            "test":
                "2024",

            "ablation_type":
                "predefined grouped feature sets",

            "selection_based_on_test_performance":
                False,
        },

        model_parameters={
            name:
                factory()
                .get_params()

            for name, factory
            in model_factories.items()
        },
    )

    save_metadata(
        metadata,
        META_DIR
        / "temporal_ablation.json",
    )

    print(
        "=" * 70
    )

    print(
        "EXPERIMENT 06a — TEMPORAL FEATURE ABLATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nResults:"
    )

    print(
        comparison[
            [
                "Feature_Set",
                "Feature_Count",
                "Model",
                "MAE",
                "RMSE",
                "R2",
                "Delta_MAE_vs_Full",
                "Delta_R2_vs_Full",
            ]
        ]
        .sort_values(
            [
                "Model",
                "R2",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()