from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

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
    FEATURES,
    TARGET,
    sha256_file,
)

from src.models.baselines import (
    random_forest_model,
)

from src.models.lightgbm_model import (
    lightgbm_model,
)

from src.explainability.shap_analysis import (
    tree_shap_values,
    mean_absolute_shap,
    long_format_shap,
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


OUT_DIR = (
    ROOT
    / "results"
    / "explainability"
    / "spatial"
)


TABLE_DIR = (
    ROOT
    / "results"
    / "explainability"
    / "tables"
)


FIG_DIR = (
    ROOT
    / "results"
    / "figures"
    / "explainability"
    / "spatial"
)


META_DIR = (
    ROOT
    / "results"
    / "metadata"
)


def main():

    for directory in [
        OUT_DIR,
        TABLE_DIR,
        FIG_DIR,
        META_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    df = load_dataset(
        DATASET
    )

    dataset_hash = sha256_file(
        DATASET
    )

    gkf = GroupKFold(
        n_splits=N_SPLITS
    )

    folds = list(
        gkf.split(
            df,
            groups=df["State"],
        )
    )

    model_factories = {
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

    long_frames = []

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

        train_states = set(
            train["State"]
        )

        test_states = set(
            test["State"]
        )

        if train_states & test_states:
            raise RuntimeError(
                f"State leakage in fold {fold_number}"
            )

        for model_name, factory in (
            model_factories.items()
        ):

            model = factory()

            model.fit(
                train[FEATURES],
                train[TARGET],
            )

            X_test = (
                test[
                    FEATURES
                ]
                .copy()
            )

            values, expected_value = (
                tree_shap_values(
                    model,
                    X_test,
                )
            )

            predictions = (
                model.predict(
                    X_test
                )
            )

            reconstructed = (
                expected_value
                + values.sum(
                    axis=1
                )
            )

            max_error = float(
                np.max(
                    np.abs(
                        predictions
                        - reconstructed
                    )
                )
            )

            if max_error > 1e-4:
                raise RuntimeError(
                    f"Fold {fold_number} "
                    f"{model_name} SHAP reconstruction "
                    f"failed: {max_error}"
                )

            long_df = (
                long_format_shap(
                    shap_values=
                        values,

                    X=
                        X_test,

                    states=
                        test[
                            "State"
                        ],

                    years=
                        test[
                            "Year"
                        ],

                    model_name=
                        model_name,

                    fold=
                        fold_number,
                )
            )

            long_frames.append(
                long_df
            )

    long_all = pd.concat(
        long_frames,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Overall state-held-out importance
    # ---------------------------------------------------------

    overall_rows = []

    for (
        model_name,
        feature,
    ), group in (
        long_all.groupby(
            [
                "Model",
                "Feature",
            ]
        )
    ):

        overall_rows.append(
            {
                "Model":
                    model_name,

                "Feature":
                    feature,

                "Mean_Absolute_SHAP":
                    group[
                        "Absolute_SHAP"
                    ].mean(),

                "Median_Absolute_SHAP":
                    group[
                        "Absolute_SHAP"
                    ].median(),

                "Mean_SHAP":
                    group[
                        "SHAP_Value"
                    ].mean(),
            }
        )

    overall = pd.DataFrame(
        overall_rows
    )

    overall["Rank"] = (
        overall
        .groupby(
            "Model"
        )[
            "Mean_Absolute_SHAP"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    overall = (
        overall
        .sort_values(
            [
                "Model",
                "Rank",
            ]
        )
    )

    # ---------------------------------------------------------
    # Fold-specific importance
    # ---------------------------------------------------------

    fold_importance = (
        long_all
        .groupby(
            [
                "Model",
                "Fold",
                "Feature",
            ],
            as_index=False,
        )
        .agg(
            Mean_Absolute_SHAP=(
                "Absolute_SHAP",
                "mean",
            )
        )
    )

    fold_importance[
        "Rank"
    ] = (
        fold_importance
        .groupby(
            [
                "Model",
                "Fold",
            ]
        )[
            "Mean_Absolute_SHAP"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    # ---------------------------------------------------------
    # Rank stability
    # ---------------------------------------------------------

    rank_stability = (
        fold_importance
        .groupby(
            [
                "Model",
                "Feature",
            ],
            as_index=False,
        )
        .agg(
            Mean_Rank=(
                "Rank",
                "mean",
            ),

            Rank_SD=(
                "Rank",
                "std",
            ),

            Best_Rank=(
                "Rank",
                "min",
            ),

            Worst_Rank=(
                "Rank",
                "max",
            ),
        )
    )

    # ---------------------------------------------------------
    # Save tables
    # ---------------------------------------------------------

    long_all.to_csv(
        OUT_DIR
        / "spatial_shap_values_long.csv",
        index=False,
    )

    overall.to_csv(
        TABLE_DIR
        / "spatial_shap_importance.csv",
        index=False,
    )

    fold_importance.to_csv(
        TABLE_DIR
        / "spatial_shap_fold_importance.csv",
        index=False,
    )

    rank_stability.to_csv(
        TABLE_DIR
        / "spatial_shap_rank_stability.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Plot pooled held-out SHAP values
    # ---------------------------------------------------------

    for model_name in (
        model_factories
    ):

        model_long = (
            long_all[
                long_all[
                    "Model"
                ] == model_name
            ]
        )

        pivot_shap = (
            model_long
            .pivot_table(
                index=[
                    "State",
                    "Year",
                ],
                columns=
                    "Feature",
                values=
                    "SHAP_Value",
                aggfunc="first",
            )
            .reindex(
                columns=
                    FEATURES
            )
        )

        pivot_values = (
            model_long
            .pivot_table(
                index=[
                    "State",
                    "Year",
                ],
                columns=
                    "Feature",
                values=
                    "Feature_Value",
                aggfunc="first",
            )
            .reindex(
                columns=
                    FEATURES
            )
        )

        shap.summary_plot(
            pivot_shap.to_numpy(),
            pivot_values,
            feature_names=
                FEATURES,
            show=False,
        )

        plt.tight_layout()

        plt.savefig(
            FIG_DIR
            / (
                f"{model_name}_"
                f"spatial_shap_beeswarm.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        shap.summary_plot(
            pivot_shap.to_numpy(),
            pivot_values,
            feature_names=
                FEATURES,
            plot_type="bar",
            show=False,
        )

        plt.tight_layout()

        plt.savefig(
            FIG_DIR
            / (
                f"{model_name}_"
                f"spatial_shap_bar.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    metadata = collect_metadata(
        experiment_name=(
            "07b_spatial_held_out_explainability"
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
            "method":
                "5-fold GroupKFold",

            "group":
                "State",

            "n_splits":
                N_SPLITS,

            "explanation_target":
                "held-out states only",

            "global_importance":
                "mean absolute SHAP pooled across held-out folds",

            "fold_rank_stability":
                True,

            "training_rows_explained":
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
        / "spatial_explainability.json",
    )

    print(
        "=" * 70
    )

    print(
        "EXPERIMENT 07b — SPATIAL HELD-OUT EXPLAINABILITY"
    )

    print(
        "=" * 70
    )

    print(
        "\nOverall SHAP importance:"
    )

    print(
        overall[
            [
                "Model",
                "Feature",
                "Mean_Absolute_SHAP",
                "Mean_SHAP",
                "Rank",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRank stability across folds:"
    )

    print(
        rank_stability[
            [
                "Model",
                "Feature",
                "Mean_Rank",
                "Rank_SD",
                "Best_Rank",
                "Worst_Rank",
            ]
        ]
        .sort_values(
            [
                "Model",
                "Mean_Rank",
            ]
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()