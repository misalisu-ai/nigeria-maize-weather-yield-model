from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def ridge_model(alpha: float = 1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])

def random_forest_model(random_state: int = 42):
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
