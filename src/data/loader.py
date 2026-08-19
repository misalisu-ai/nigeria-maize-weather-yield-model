from __future__ import annotations
from pathlib import Path
import hashlib
import pandas as pd

EXPECTED_SHA256 = "09ab62faed51d7f391595068587d33595f49058ae1687621925cf12408c2f97c"

FEATURES = [
    "Seasonal_Rainfall_mm",
    "Seasonal_GDD_C",
    "Max_CDD_days",
    "Mean_Tmax_C",
    "Mean_Tmin_C",
    "Mean_RH_pct",
    "Mean_Solar_Radiation_MJ_m2_day",
    "Rainfall_Anomaly_Z_2000_2019",
]
TARGET = "Yield_MT_Ha"

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_dataset(path: str | Path, verify_hash: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if verify_hash:
        actual = sha256_file(path)
        if actual.lower() != EXPECTED_SHA256.lower():
            raise ValueError(
                f"Frozen dataset SHA-256 mismatch. Expected {EXPECTED_SHA256}, got {actual}."
            )
    df = pd.read_csv(path)
    required = ["State", "Year", TARGET, *FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[["State", "Year"]].duplicated().any():
        raise ValueError("Duplicate State-Year observations detected.")
    return df
