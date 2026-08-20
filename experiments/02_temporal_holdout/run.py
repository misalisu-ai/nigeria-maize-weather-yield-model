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
from src.data.splits import temporal_holdout
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
TRAIN_END = 2023

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

    train, test = temporal_holdout(
        df,
        train_end=TRAIN_END,
    )

    # ---------------------------------------------------------
    # Save exact temporal split
    # ---------------------------------------------------------

    assignment = df[
        ["State", "Year"]
    ].copy()

    assignment["Split"] = assignment[
        "Year"
    ].apply(
        lambda year:
        "train"
        if year <= TRAIN_END
        else "test"
    )

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignment.to_csv(
        SPLIT_DIR
        / "temporal_holdout_assignments.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    models = {
        "Ridge": ridge_model(),
        "RandomForest": random_forest_model(
            random_state=SEED
        ),
        "LightGBM": lightgbm_model(
            random_state=SEED
        ),
    }

    rows = []

    for name, model in models.items():

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        rows.append(
            {
                "Model": name,
                "Train_Years": "2020-2023",
                "Test_Years": "2024",
                **regression_metrics(
                    y_test,
                    predictions,
                ),
            }
        )

    result = pd.DataFrame(rows)

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        TABLE_DIR
        / "temporal_holdout.csv",
        index=False,
    )

    metadata = collect_metadata(
        experiment_name="02_temporal_holdout",
        dataset_path=str(DATASET),
        dataset_sha256=dataset_sha256,
        features=FEATURES,
        target=TARGET,
        seed=SEED,
        split_definition={
            "method": "chronological_holdout",
            "train_years": "2020-2023",
            "test_years": "2024",
        },
        model_parameters={
            name: model.get_params()
            for name, model in models.items()
        },
    )

    save_metadata(
        metadata,
        META_DIR
        / "temporal_holdout.json",
    )

    print(
        "\nTemporal holdout:"
    )

    print(
        result.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()