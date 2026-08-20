from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def tree_shap_values(
    model,
    X: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """
    Compute SHAP values for a fitted tree-based regression model.

    Returns
    -------
    shap_values:
        Array with shape (n_samples, n_features).

    expected_value:
        Scalar model baseline prediction.
    """

    explainer = shap.TreeExplainer(
        model
    )

    explanation = explainer(
        X
    )

    values = np.asarray(
        explanation.values
    )

    if values.ndim != 2:
        raise RuntimeError(
            f"Unexpected SHAP shape: {values.shape}"
        )

    base_values = np.asarray(
        explanation.base_values
    )

    if base_values.ndim == 0:
        expected_value = float(
            base_values
        )
    else:
        expected_value = float(
            np.mean(base_values)
        )

    return (
        values,
        expected_value,
    )


def mean_absolute_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Global importance from mean absolute SHAP magnitude.
    """

    importance = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    result = pd.DataFrame(
        {
            "Feature":
                feature_names,

            "Mean_Absolute_SHAP":
                importance,
        }
    )

    result["Rank"] = (
        result[
            "Mean_Absolute_SHAP"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return (
        result
        .sort_values(
            "Mean_Absolute_SHAP",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def long_format_shap(
    *,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    states,
    years,
    model_name: str,
    fold: int | None = None,
) -> pd.DataFrame:
    """
    Save observation-level SHAP values in long format.
    """

    rows = []

    states = list(states)
    years = list(years)

    for i in range(
        len(X)
    ):

        for j, feature in enumerate(
            X.columns
        ):

            row = {
                "Model":
                    model_name,

                "State":
                    states[i],

                "Year":
                    years[i],

                "Feature":
                    feature,

                "Feature_Value":
                    float(
                        X.iloc[i, j]
                    ),

                "SHAP_Value":
                    float(
                        shap_values[i, j]
                    ),

                "Absolute_SHAP":
                    float(
                        abs(
                            shap_values[i, j]
                        )
                    ),
            }

            if fold is not None:
                row["Fold"] = fold

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )