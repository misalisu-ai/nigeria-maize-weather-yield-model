from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split


def random_holdout(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    return train.copy(), test.copy()


def temporal_holdout(
    df: pd.DataFrame,
    train_end: int = 2023,
):
    train = df[df["Year"] <= train_end].copy()
    test = df[df["Year"] > train_end].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Temporal split produced an empty partition."
        )

    return train.copy(), test.copy()


def state_group_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
):
    if df["State"].nunique() < n_splits:
        raise ValueError(
            "n_splits exceeds the number of unique states."
        )

    splitter = GroupKFold(
        n_splits=n_splits
    )

    return splitter.split(
        df,
        groups=df["State"],
    )