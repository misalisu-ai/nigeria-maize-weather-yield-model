from pathlib import Path
import sys

import numpy as np
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

OUTER_SPLITS = 5
INNER_SPLITS = 5

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

    for path in [
        OUT_DIR,
        TABLE_DIR,
        SPLIT_DIR,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    df = load_dataset(
        DATASET
    )

    dataset_hash = sha256_file(
        DATASET
    )

    outer = GroupKFold(
        n_splits=OUTER_SPLITS
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

    detailed_rows = []
    fold_summary_rows = []
    split_rows = []

    for outer_fold, (
        outer_train_idx,
        test_idx,
    ) in enumerate(
        outer.split(
            df,
            groups=df[
                "State"
            ],
        ),
        start=1,
    ):

        outer_train = (
            df.iloc[
                outer_train_idx
            ].copy()
        )

        test = (
            df.iloc[
                test_idx
            ].copy()
        )

        # -----------------------------------------------------
        # Inner grouped split for calibration.
        #
        # We use the first deterministic GroupKFold split.
        # This creates one fixed calibration-state subset
        # inside each outer fold.
        # -----------------------------------------------------

        inner = GroupKFold(
            n_splits=INNER_SPLITS
        )

        inner_fit_idx, calibration_idx = (
            next(
                inner.split(
                    outer_train,
                    groups=outer_train[
                        "State"
                    ],
                )
            )
        )

        fit = (
            outer_train.iloc[
                inner_fit_idx
            ].copy()
        )

        calibration = (
            outer_train.iloc[
                calibration_idx
            ].copy()
        )

        fit_states = set(
            fit["State"]
        )

        calibration_states = set(
            calibration[
                "State"
            ]
        )

        test_states = set(
            test["State"]
        )

        # -----------------------------------------------------
        # Leakage assertions
        # -----------------------------------------------------

        if (
            fit_states
            & calibration_states
        ):
            raise RuntimeError(
                "Fit/calibration state leakage."
            )

        if (
            fit_states
            & test_states
        ):
            raise RuntimeError(
                "Fit/test state leakage."
            )

        if (
            calibration_states
            & test_states
        ):
            raise RuntimeError(
                "Calibration/test state leakage."
            )

        for state in sorted(
            fit_states
        ):
            split_rows.append(
                {
                    "Outer_Fold":
                        outer_fold,
                    "State":
                        state,
                    "Role":
                        "fit",
                }
            )

        for state in sorted(
            calibration_states
        ):
            split_rows.append(
                {
                    "Outer_Fold":
                        outer_fold,
                    "State":
                        state,
                    "Role":
                        "calibration",
                }
            )

        for state in sorted(
            test_states
        ):
            split_rows.append(
                {
                    "Outer_Fold":
                        outer_fold,
                    "State":
                        state,
                    "Role":
                        "test",
                }
            )

        # -----------------------------------------------------
        # Models
        # -----------------------------------------------------

        for model_name, factory in (
            model_factories.items()
        ):

            model = factory()

            model.fit(
                fit[FEATURES],
                fit[TARGET],
            )

            calibration_pred = (
                model.predict(
                    calibration[
                        FEATURES
                    ]
                )
            )

            residuals = abs(
                calibration[
                    TARGET
                ].to_numpy()
                - calibration_pred
            )

            qhat = conformal_quantile(
                residuals,
                alpha=ALPHA,
            )

            test_pred = model.predict(
                test[FEATURES]
            )

            lower, upper = (
                symmetric_interval(
                    test_pred,
                    qhat,
                )
            )

            lower = lower.clip(
                min=0
            )

            observed = (
                test[TARGET]
                .to_numpy()
            )

            covered = (
                (observed >= lower)
                &
                (observed <= upper)
            )

            widths = (
                upper
                - lower
            )

            for (
                state,
                year,
                actual,
                predicted,
                lo,
                hi,
                width,
                is_covered,
            ) in zip(
                test["State"],
                test["Year"],
                observed,
                test_pred,
                lower,
                upper,
                widths,
                covered,
            ):

                detailed_rows.append(
                    {
                        "Outer_Fold":
                            outer_fold,

                        "State":
                            state,

                        "Year":
                            year,

                        "Model":
                            model_name,

                        "Observed_Yield":
                            actual,

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
                    observed,
                    test_pred,
                )
            )

            uncertainty_metrics = (
                interval_metrics(
                    observed,
                    lower,
                    upper,
                )
            )

            fold_summary_rows.append(
                {
                    "Outer_Fold":
                        outer_fold,

                    "Model":
                        model_name,

                    "Fit_State_Count":
                        len(
                            fit_states
                        ),

                    "Calibration_State_Count":
                        len(
                            calibration_states
                        ),

                    "Test_State_Count":
                        len(
                            test_states
                        ),

                    "Calibration_Size":
                        len(
                            calibration
                        ),

                    "Test_Size":
                        len(
                            test
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

    fold_summary = pd.DataFrame(
        fold_summary_rows
    )

    split_assignments = pd.DataFrame(
        split_rows
    )

    overall_rows = []

    for model_name, group in (
        detailed.groupby(
            "Model"
        )
    ):

        observed = (
            group[
                "Observed_Yield"
            ].to_numpy()
        )

        predicted = (
            group[
                "Predicted_Yield"
            ].to_numpy()
        )

        lower = (
            group[
                "Lower_90"
            ].to_numpy()
        )

        upper = (
            group[
                "Upper_90"
            ].to_numpy()
        )

        overall_rows.append(
            {
                "Model":
                    model_name,

                "Target_Coverage":
                    1 - ALPHA,

                **regression_metrics(
                    observed,
                    predicted,
                ),

                **interval_metrics(
                    observed,
                    lower,
                    upper,
                ),
            }
        )

    overall = pd.DataFrame(
        overall_rows
    )

    detailed.to_csv(
        OUT_DIR
        / "spatial_conformal_detailed.csv",
        index=False,
    )

    fold_summary.to_csv(
        TABLE_DIR
        / "spatial_conformal_fold_summary.csv",
        index=False,
    )

    overall.to_csv(
        TABLE_DIR
        / "spatial_conformal_overall.csv",
        index=False,
    )

    split_assignments.to_csv(
        SPLIT_DIR
        / "spatial_conformal_assignments.csv",
        index=False,
    )

    metadata = collect_metadata(
        experiment_name=(
            "05b_spatial_grouped_split_conformal"
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
            "outer_method":
                "GroupKFold",

            "outer_group":
                "State",

            "outer_splits":
                OUTER_SPLITS,

            "inner_calibration_method":
                "first deterministic GroupKFold split",

            "inner_group":
                "State",

            "inner_splits":
                INNER_SPLITS,

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
                factory()
                .get_params()

            for name, factory
            in model_factories.items()
        },
    )

    save_metadata(
        metadata,
        META_DIR
        / "spatial_conformal.json",
    )

    print(
        "=" * 70
    )

    print(
        "EXPERIMENT 05b — SPATIAL GROUPED SPLIT CONFORMAL"
    )

    print(
        "=" * 70
    )

    print(
        "\nOverall:"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    print(
        "\nFold summary:"
    )

    print(
        fold_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()