from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from sklearn.model_selection import (
    GroupKFold,
)


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
N_SPLITS = 5


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


SPLIT_DIR = (
    ROOT
    / "results"
    / "splits"
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
        SPLIT_DIR,
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

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    folds = list(
        group_kfold.split(
            df,
            groups=df["State"],
        )
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

    fold_rows = []
    prediction_rows = []
    split_rows = []

    # ---------------------------------------------------------
    # Save one canonical outer split assignment.
    # It is shared by all models and feature sets.
    # ---------------------------------------------------------

    for fold_number, (
        train_idx,
        test_idx,
    ) in enumerate(
        folds,
        start=1,
    ):

        train_states = sorted(
            df.iloc[
                train_idx
            ]["State"].unique()
        )

        test_states = sorted(
            df.iloc[
                test_idx
            ]["State"].unique()
        )

        if (
            set(train_states)
            & set(test_states)
        ):
            raise RuntimeError(
                f"State leakage in fold {fold_number}"
            )

        for state in train_states:

            split_rows.append(
                {
                    "Fold":
                        fold_number,

                    "State":
                        state,

                    "Role":
                        "train",
                }
            )

        for state in test_states:

            split_rows.append(
                {
                    "Fold":
                        fold_number,

                    "State":
                        state,

                    "Role":
                        "test",
                }
            )

    # ---------------------------------------------------------
    # Ablations
    # ---------------------------------------------------------

    for feature_set_name, features in (
        FEATURE_SETS.items()
    ):

        for fold_number, (
            train_idx,
            test_idx,
        ) in enumerate(
            folds,
            start=1,
        ):

            train = (
                df.iloc[
                    train_idx
                ]
                .copy()
            )

            test = (
                df.iloc[
                    test_idx
                ]
                .copy()
            )

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

                metrics = (
                    regression_metrics(
                        test[TARGET],
                        pred,
                    )
                )

                fold_rows.append(
                    {
                        "Feature_Set":
                            feature_set_name,

                        "Feature_Count":
                            len(features),

                        "Fold":
                            fold_number,

                        "Model":
                            model_name,

                        "Test_State_Count":
                            test[
                                "State"
                            ]
                            .nunique(),

                        "Test_Size":
                            len(test),

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

                            "Fold":
                                fold_number,

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

    fold_results = pd.DataFrame(
        fold_rows
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    split_assignments = pd.DataFrame(
        split_rows
    )

    # ---------------------------------------------------------
    # Summarize across folds
    # ---------------------------------------------------------

    summary = (
        fold_results
        .groupby(
            [
                "Feature_Set",
                "Feature_Count",
                "Model",
            ],
            as_index=False,
        )
        .agg(
            MAE_Mean=(
                "MAE",
                "mean",
            ),

            MAE_SD=(
                "MAE",
                "std",
            ),

            RMSE_Mean=(
                "RMSE",
                "mean",
            ),

            RMSE_SD=(
                "RMSE",
                "std",
            ),

            R2_Mean=(
                "R2",
                "mean",
            ),

            R2_SD=(
                "R2",
                "std",
            ),

            Mean_Error_Mean=(
                "Mean_Error",
                "mean",
            ),
        )
    )

    # ---------------------------------------------------------
    # Compare each feature set with Full
    # ---------------------------------------------------------

    full = (
        summary[
            summary[
                "Feature_Set"
            ] == "Full"
        ]
        .set_index(
            "Model"
        )
    )

    comparison_rows = []

    for _, row in (
        summary.iterrows()
    ):

        baseline = full.loc[
            row["Model"]
        ]

        comparison_rows.append(
            {
                **row.to_dict(),

                "Delta_MAE_vs_Full":
                    row["MAE_Mean"]
                    - baseline[
                        "MAE_Mean"
                    ],

                "Delta_RMSE_vs_Full":
                    row["RMSE_Mean"]
                    - baseline[
                        "RMSE_Mean"
                    ],

                "Delta_R2_vs_Full":
                    row["R2_Mean"]
                    - baseline[
                        "R2_Mean"
                    ],
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    fold_results.to_csv(
        TABLE_DIR
        / "spatial_ablation_fold_metrics.csv",
        index=False,
    )

    summary.to_csv(
        TABLE_DIR
        / "spatial_ablation_summary.csv",
        index=False,
    )

    comparison.to_csv(
        TABLE_DIR
        / "spatial_ablation_vs_full.csv",
        index=False,
    )

    predictions.to_csv(
        PRED_DIR
        / "spatial_ablation_predictions.csv",
        index=False,
    )

    split_assignments.to_csv(
        SPLIT_DIR
        / "spatial_ablation_assignments.csv",
        index=False,
    )

    metadata = collect_metadata(
        experiment_name=(
            "06b_spatial_feature_ablation"
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
            "method":
                "5-fold GroupKFold",

            "group":
                "State",

            "n_splits":
                N_SPLITS,

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
        / "spatial_ablation.json",
    )

    print(
        "=" * 70
    )

    print(
        "EXPERIMENT 06b — SPATIAL FEATURE ABLATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nSummary:"
    )

    print(
        comparison[
            [
                "Feature_Set",
                "Feature_Count",
                "Model",
                "MAE_Mean",
                "RMSE_Mean",
                "R2_Mean",
                "R2_SD",
                "Delta_MAE_vs_Full",
                "Delta_R2_vs_Full",
            ]
        ]
        .sort_values(
            [
                "Model",
                "R2_Mean",
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