from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset, FEATURES, TARGET
from src.data.splits import state_group_folds
from src.evaluation.metrics import regression_metrics
from src.models.baselines import ridge_model, random_forest_model
from src.models.lightgbm_model import lightgbm_model

DATASET = ROOT / "data/nigeria_maize_weather_yield_2020_2024_v1.0.csv"
OUT = ROOT / "results/tables/state_held_out.csv"

def main():
    df = load_dataset(DATASET)
    factories = {
        "Ridge": ridge_model,
        "RandomForest": random_forest_model,
        "LightGBM": lightgbm_model,
    }

    rows = []
    for fold, (train_idx, test_idx) in enumerate(
        state_group_folds(df, n_splits=5), start=1
    ):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        for name, factory in factories.items():
            model = factory()
            model.fit(train[FEATURES], train[TARGET])
            pred = model.predict(test[FEATURES])
            rows.append({
                "Fold": fold,
                "Model": name,
                "Test_States": ";".join(sorted(test["State"].unique())),
                **regression_metrics(test[TARGET], pred),
            })

    result = pd.DataFrame(rows)
    summary = result.groupby("Model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).reset_index()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    summary.to_csv(ROOT / "results/tables/state_held_out_summary.csv", index=False)

    print("Fold results:")
    print(result.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
