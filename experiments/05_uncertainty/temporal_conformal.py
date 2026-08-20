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

from src.uncertainty.conformal import (
    conformal_quantile,
    symmetric_interval,
    interval_metrics,
)

from src.utils.experiment_metadata import (
    collect_metadata,
    save_metadata,
)


SEED = 42
ALPHA = 0.10

DATASET = (
    ROOT
    / "data"
    / "nigeria_maize_weather_yield_2020_2024_v1.0.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "uncertainty"
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

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
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
        df["Year"].between(
            2020,
            2022,
        )
    ].copy()

    calibration = df[
        df["Year"] == 2023
    ].copy()

    test = df[
        df["Year"] == 2024
    ].copy()

    if len(calibration) != 37:
        raise RuntimeError(
            f"Expected 37 calibration rows, got {len(calibration)}"
        )

    if len(test) != 37:
        raise RuntimeError(
            f"Expected 37 test rows, got {len(test)}"
        )

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

    detailed_rows = []
    summary_rows = []

    for model_name, model in (
        models.items()
    ):

        # -----------------------------------------------------
        # Fit only on 2020-2022
        # -----------------------------------------------------

        model.fit(
            train[FEATURES],
            train[TARGET],
        )

        # -----------------------------------------------------
        # Calibration residuals
        # -----------------------------------------------------

        calibration_pred = (
            model.predict(
                calibration[
                    FEATURES
                ]
            )
        )

        calibration_residuals = (
            calibration[
                TARGET
            ].to_numpy()
            - calibration_pred
        )

        absolute_residuals = abs(
            calibration_residuals
        )

        qhat = conformal_quantile(
            absolute_residuals,
            alpha=ALPHA,
        )

        # -----------------------------------------------------
        # Test prediction
        # -----------------------------------------------------

        test_pred = model.predict(
            test[FEATURES]
        )

        lower, upper = (
            symmetric_interval(
                test_pred,
                qhat,
            )
        )

        # Yield is physically non-negative.
        lower = lower.clip(
            min=0
        )

        covered = (
            (
                test[
                    TARGET
                ].to_numpy()
                >= lower
            )
            &
            (
                test[
                    TARGET
                ].to_numpy()
                <= upper
            )
        )

        widths = (
            upper
            - lower
        )

        # -----------------------------------------------------
        # Detailed output
        # -----------------------------------------------------

        for (
            state,
            year,
            observed,
            predicted,
            lo,
            hi,
            width,
            is_covered,
        ) in zip(
            test["State"],
            test["Year"],
            test[TARGET],
            test_pred,
            lower,
            upper,
            widths,
            covered,
        ):

            detailed_rows.append(
                {
                    "State":
                        state,

                    "Year":
                        year,

                    "Model":
                        model_name,

                    "Observed_Yield":
                        observed,

                    "Predicted_Yield":
                        predicted,

                    "Lower_90":
                        lo,

                    "Upper_90":
                        hi,

                    "Interval_Width":
                        width,

                    "Covered":
                        bool(
                            is_covered
                        ),

                    "Calibration_qhat":
                        qhat,
                }
            )

        point_metrics = (
            regression_metrics(
                test[TARGET],
                test_pred,
            )
        )

        uncertainty_metrics = (
            interval_metrics(
                test[TARGET],
                lower,
                upper,
            )
        )

        summary_rows.append(
            {
                "Model":
                    model_name,

                "Target_Coverage":
                    1 - ALPHA,

                "Calibration_Size":
                    len(
                        calibration
                    ),

                "Calibration_qhat":
                    qhat,

                **point_metrics,

                **uncertainty_metrics,
            }
        )

    detailed = pd.DataFrame(
        detailed_rows
    )

    summary = pd.DataFrame(
        summary_rows
    )

    detailed.to_csv(
        OUT_DIR
        / "temporal_conformal_detailed.csv",
        index=False,
    )

    summary.to_csv(
        TABLE_DIR
        / "temporal_conformal_summary.csv",
        index=False,
    )

    metadata = collect_metadata(
        experiment_name=(
            "05a_temporal_split_conformal"
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
            "model_training":
                "2020-2022",

            "conformal_calibration":
                "2023",

            "test":
                "2024",

            "alpha":
                ALPHA,

            "target_coverage":
                1 - ALPHA,

            "interval_type":
                "symmetric split conformal",

            "nonnegative_lower_bound":
                True,
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
        / "temporal_conformal.json",
    )

    print(
        "=" * 70
    )

    print(
        "EXPERIMENT 05a — TEMPORAL SPLIT CONFORMAL"
    )

    print(
        "=" * 70
    )

    print(
        "\nSummary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()