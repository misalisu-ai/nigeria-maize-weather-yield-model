from __future__ import annotations

import numpy as np


def conformal_quantile(
    residuals,
    alpha: float = 0.10,
) -> float:
    """
    Split-conformal absolute-residual quantile.

    Uses the finite-sample corrected rank:
        ceil((n + 1) * (1 - alpha)) / n
    """

    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    if residuals.ndim != 1:
        raise ValueError(
            "residuals must be one-dimensional"
        )

    if len(residuals) == 0:
        raise ValueError(
            "Calibration residuals are empty."
        )

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    n = len(residuals)

    rank = int(
        np.ceil(
            (n + 1)
            * (1 - alpha)
        )
    )

    rank = min(
        rank,
        n,
    )

    sorted_residuals = np.sort(
        residuals
    )

    return float(
        sorted_residuals[
            rank - 1
        ]
    )


def symmetric_interval(
    predictions,
    qhat: float,
):
    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    lower = predictions - qhat
    upper = predictions + qhat

    return lower, upper


def interval_metrics(
    y_true,
    lower,
    upper,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    lower = np.asarray(
        lower,
        dtype=float,
    )

    upper = np.asarray(
        upper,
        dtype=float,
    )

    covered = (
        (y_true >= lower)
        & (y_true <= upper)
    )

    widths = upper - lower

    return {
        "Coverage": float(
            covered.mean()
        ),
        "Mean_Interval_Width": float(
            widths.mean()
        ),
        "Median_Interval_Width": float(
            np.median(widths)
        ),
        "Min_Interval_Width": float(
            widths.min()
        ),
        "Max_Interval_Width": float(
            widths.max()
        ),
    }