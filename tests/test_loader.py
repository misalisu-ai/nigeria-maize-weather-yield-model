from pathlib import Path
import pytest
from src.data.loader import load_dataset

def test_frozen_dataset_exists_and_loads():
    path = Path("data/nigeria_maize_weather_yield_2020_2024_v1.0.csv")
    if not path.exists():
        pytest.skip("Copy frozen Dataset v1.0 into data/ first.")
    df = load_dataset(path)
    assert len(df) == 185
    assert df["State"].nunique() == 37
