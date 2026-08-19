from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset, FEATURES, TARGET
from src.data.splits import random_holdout
from src.evaluation.metrics import regression_metrics
from src.models.baselines import ridge_model, random_forest_model
from src.models.lightgbm_model import lightgbm_model

DATASET = ROOT / "data/nigeria_maize_weather_yield_2020_2024_v1.0.csv"
OUT = ROOT / "results/tables/random_baseline.csv"

def main():
    df = load_dataset(DATASET)
    train, test = random_holdout(df, random_state=42)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    models = {
        "Ridge": ridge_model(),
        "RandomForest": random_forest_model(),
        "LightGBM": lightgbm_model(),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({"Model": name, **regression_metrics(y_test, pred)})

    result = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
