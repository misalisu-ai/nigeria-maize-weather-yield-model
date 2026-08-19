from lightgbm import LGBMRegressor

def quantile_model(alpha: float, random_state: int = 42):
    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=random_state,
        verbosity=-1,
    )
