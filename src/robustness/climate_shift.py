import pandas as pd

def perturb_temperature(X: pd.DataFrame, shift_c: float) -> pd.DataFrame:
    out = X.copy()
    out["Mean_Tmax_C"] += shift_c
    out["Mean_Tmin_C"] += shift_c
    return out

def perturb_rainfall(X: pd.DataFrame, shift_pct: float) -> pd.DataFrame:
    out = X.copy()
    factor = 1.0 + shift_pct / 100.0
    if factor < 0:
        raise ValueError("Rainfall multiplier cannot be negative.")
    out["Seasonal_Rainfall_mm"] *= factor
    return out
