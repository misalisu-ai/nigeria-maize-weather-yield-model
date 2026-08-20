from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import (
    load_dataset,
    FEATURES,
    TARGET,
    sha256_file,
)
from src.data.splits import state_group_folds
from src.evaluation.metrics import regression_metrics
from src.models.baselines import (
    ridge_model,
    random_forest_model,
)
from src.models.lightgbm_model import lightgbm_model
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

RESULTS = ROOT / "results"

TABLE_DIR = RESULTS / "tables"
SPLIT_DIR = RESULTS / "splits"
META_DIR = RESULTS / "metadata"


def main():

    df = load_dataset(DATASET)

    dataset_sha256 = sha256_file(
        DATASET
    )

    model_factories = {
        "Ridge": ridge_model,
        "RandomForest": lambda:
            random_forest_model(
                random_state=SEED
            ),
        "LightGBM": lambda:
            lightgbm_model(
                random_state=SEED
            ),
    }

    rows = []

    fold_assignments = []

    # ---------------------------------------------------------
    # State GroupKFold
    # ---------------------------------------------------------

    splits = state_group_folds(
        df,
        n_splits=N_SPLITS,
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splits,
        start=1,
    ):

        train = df.iloc[
            train_idx
        ]

        test = df.iloc[
            test_idx
        ]

        test_states = sorted(
            test["State"]
            .unique()
        )

        train_states = sorted(
            train["State"]
            .unique()
        )

        # Save exact fold assignment.
        for state in train_states:

            fold_assignments.append(
                {
                    "Fold": fold,
                    "State": state,
                    "Role": "train",
                }
            )

        for state in test_states:

            fold_assignments.append(
                {
                    "Fold": fold,
                    "State": state,
                    "Role": "test",
                }
            )

        # -----------------------------------------------------
        # Models
        # -----------------------------------------------------

        for name, factory in (
            model_factories.items()
        ):

            model = factory()

            model.fit(
                train[FEATURES],
                train[TARGET],
            )

            predictions = model.predict(
                test[FEATURES]
            )

            rows.append(
                {
                    "Fold": fold,
                    "Model": name,
                    "Train_State_Count":
                        len(train_states),
                    "Test_State_Count":
                        len(test_states),
                    "Test_States":
                        ";".join(
                            test_states
                        ),
                    **regression_metrics(
                        test[TARGET],
                        predictions,
                    ),
                }
            )

    result = pd.DataFrame(rows)

    fold_df = pd.DataFrame(
        fold_assignments
    )

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_df.to_csv(
        SPLIT_DIR
        / "state_fold_assignments.csv",
        index=False,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        TABLE_DIR
        / "state_held_out.csv",
        index=False,
    )

    summary = (
        result
        .groupby("Model")[
            ["MAE", "RMSE", "R2"]
        ]
        .agg(
            ["mean", "std"]
        )
        .reset_index()
    )

    summary.to_csv(
        TABLE_DIR
        / "state_held_out_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = collect_metadata(
        experiment_name="03_state_held_out",
        dataset_path=str(DATASET),
        dataset_sha256=dataset_sha256,
        features=FEATURES,
        target=TARGET,
        seed=SEED,
        split_definition={
            "method": "GroupKFold",
            "group": "State",
            "n_splits": N_SPLITS,
            "shuffle": False,
        },
        model_parameters={
            name:
            (
                factory()
                .get_params()
            )
            for name, factory
            in model_factories.items()
        },
    )

    save_metadata(
        metadata,
        META_DIR
        / "state_held_out.json",
    )

    print(
        "\nState-held-out results:"
    )

    print(
        result.to_string(
            index=False
        )
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