from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


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
    ridge_model,
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


DATASET = (
    ROOT
    / "data"
    / "nigeria_maize_weather_yield_2020_2024_v1.0.csv"
)


OUT_DIR = (
    ROOT
    / "results"
    / "explainability"
    / "temporal"
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
    / "temporal"
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

    train = df[
        df["Year"] <= 2023
    ].copy()

    test = df[
        df["Year"] == 2024
    ].copy()

    if len(test) != 37:
        raise RuntimeError(
            f"Expected 37 test rows, found {len(test)}"
        )

    models = {
        "RandomForest":
            random_forest_model(
                random_state=SEED
            ),

        "LightGBM":
            lightgbm_model(
                random_state=SEED
            ),
    }

    importance_frames = []
    long_frames = []

    for model_name, model in (
        models.items()
    ):

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

        if values.shape != (
            len(test),
            len(FEATURES),
        ):
            raise RuntimeError(
                f"{model_name} SHAP shape mismatch: "
                f"{values.shape}"
            )

        # -----------------------------------------------------
        # Global importance
        # -----------------------------------------------------

        importance = (
            mean_absolute_shap(
                values,
                FEATURES,
            )
        )

        importance.insert(
            0,
            "Model",
            model_name,
        )

        importance_frames.append(
            importance
        )

        # -----------------------------------------------------
        # Observation-level values
        # -----------------------------------------------------

        long_df = long_format_shap(
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
        )

        long_frames.append(
            long_df
        )

        # -----------------------------------------------------
        # SHAP reconstruction check
        #
        # expected + sum(SHAP) should equal model prediction
        # -----------------------------------------------------

        predictions = model.predict(
            X_test
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

        print(
            f"{model_name} reconstruction max error: "
            f"{max_error:.10f}"
        )

        if max_error > 1e-4:
            raise RuntimeError(
                f"SHAP reconstruction failed for {model_name}."
            )

        # -----------------------------------------------------
        # Summary plot
        # -----------------------------------------------------

        shap.summary_plot(
            values,
            X_test,
            feature_names=
                FEATURES,
            show=False,
        )

        plt.tight_layout()

        plt.savefig(
            FIG_DIR
            / (
                f"{model_name}_"
                f"temporal_shap_beeswarm.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        # -----------------------------------------------------
        # Mean absolute SHAP bar plot
        # -----------------------------------------------------

        shap.summary_plot(
            values,
            X_test,
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
                f"temporal_shap_bar.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    importance_all = pd.concat(
        importance_frames,
        ignore_index=True,
    )

    long_all = pd.concat(
        long_frames,
        ignore_index=True,
    )

    importance_all.to_csv(
        TABLE_DIR
        / "temporal_shap_importance.csv",
        index=False,
    )

    long_all.to_csv(
        OUT_DIR
        / "temporal_shap_values_long.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Ridge standardized coefficients
    # ---------------------------------------------------------

    ridge = ridge_model()

    ridge.fit(
        train[FEATURES],
        train[TARGET],
    )

    coefficients = (
        ridge
        .named_steps["model"]
        .coef_
    )

    ridge_table = pd.DataFrame(
        {
            "Feature":
                FEATURES,

            "Standardized_Coefficient":
                coefficients,

            "Absolute_Coefficient":
                np.abs(
                    coefficients
                ),
        }
    )

    ridge_table["Rank"] = (
        ridge_table[
            "Absolute_Coefficient"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    ridge_table = (
        ridge_table
        .sort_values(
            "Absolute_Coefficient",
            ascending=False,
        )
    )

    ridge_table.to_csv(
        TABLE_DIR
        / "temporal_ridge_coefficients.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = collect_metadata(
        experiment_name=(
            "07a_temporal_explainability"
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
            "train":
                "2020-2023",

            "explained_test":
                "2024",

            "tree_explainer":
                "SHAP TreeExplainer",

            "global_importance":
                "mean absolute held-out SHAP",

            "ridge_explanation":
                "standardized coefficients",

            "training_rows_explained":
                False,
        },

        model_parameters={
            "RandomForest":
                models[
                    "RandomForest"
                ].get_params(),

            "LightGBM":
                models[
                    "LightGBM"
                ].get_params(),

            "Ridge":
                ridge.get_params(),
        },
    )

    save_metadata(
        metadata,
        META_DIR
        / "temporal_explainability.json",
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPERIMENT 07a — TEMPORAL EXPLAINABILITY"
    )

    print(
        "=" * 70
    )

    print(
        "\nTree-model global importance:"
    )

    print(
        importance_all[
            [
                "Model",
                "Feature",
                "Mean_Absolute_SHAP",
                "Rank",
            ]
        ]
        .sort_values(
            [
                "Model",
                "Rank",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nRidge standardized coefficients:"
    )

    print(
        ridge_table[
            [
                "Feature",
                "Standardized_Coefficient",
                "Absolute_Coefficient",
                "Rank",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()