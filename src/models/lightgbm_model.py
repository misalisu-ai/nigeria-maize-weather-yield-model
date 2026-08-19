from lightgbm import LGBMRegressor

def lightgbm_model(random_state: int = 42):
    return LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=random_state,
        verbosity=-1,
    )
