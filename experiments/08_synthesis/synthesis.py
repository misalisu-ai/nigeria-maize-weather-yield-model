from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


SEED = 42
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95


OUT_DIR = (
    ROOT
    / "results"
    / "synthesis"
)


# ============================================================
# INPUTS
# ============================================================

TEMPORAL_PREDICTIONS = (
    ROOT
    / "results"
    / "predictions"
    / "temporal_ablation_predictions.csv"
)

SPATIAL_PREDICTIONS = (
    ROOT
    / "results"
    / "predictions"
    / "spatial_ablation_predictions.csv"
)

CLIMATE_04A = (
    ROOT
    / "results"
    / "climate_stress"
    / "feature_space_detailed.csv"
)

CLIMATE_04B = (
    ROOT
    / "results"
    / "climate_stress"
    / "weather_recomputed_detailed.csv"
)

TEMPORAL_UNCERTAINTY = (
    ROOT
    / "results"
    / "uncertainty"
    / "temporal_conformal_detailed.csv"
)

SPATIAL_UNCERTAINTY = (
    ROOT
    / "results"
    / "uncertainty"
    / "spatial_conformal_detailed.csv"
)

TEMPORAL_ABLATION = (
    ROOT
    / "results"
    / "tables"
    / "temporal_ablation_vs_full.csv"
)

SPATIAL_ABLATION = (
    ROOT
    / "results"
    / "tables"
    / "spatial_ablation_vs_full.csv"
)

TEMPORAL_SHAP = (
    ROOT
    / "results"
    / "explainability"
    / "tables"
    / "temporal_shap_importance.csv"
)

SPATIAL_SHAP = (
    ROOT
    / "results"
    / "explainability"
    / "tables"
    / "spatial_shap_importance.csv"
)

SPATIAL_SHAP_STABILITY = (
    ROOT
    / "results"
    / "explainability"
    / "tables"
    / "spatial_shap_rank_stability.csv"
)


MODELS = [
    "Ridge",
    "RandomForest",
    "LightGBM",
]


# ============================================================
# HELPERS
# ============================================================

def require_files(paths):
    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Experiment 08 input files:\n"
            + "\n".join(missing)
        )


def percentile_ci(values, level=0.95):
    values = np.asarray(values, dtype=float)

    alpha = 1.0 - level

    return (
        float(
            np.quantile(
                values,
                alpha / 2.0,
            )
        ),
        float(
            np.quantile(
                values,
                1.0 - alpha / 2.0,
            )
        ),
    )


def paired_state_bootstrap(
    wide: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    cluster_by_state: bool,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
):
    """
    Bootstrap difference in mean absolute error:

        MAE(model_a) - MAE(model_b)

    Negative:
        model_a has lower MAE.

    Positive:
        model_b has lower MAE.

    For spatial evaluation, whole states are resampled to
    preserve within-state dependence across years.
    """

    rng = np.random.default_rng(seed)

    required = {
        "State",
        "Observed_Yield",
        model_a,
        model_b,
    }

    missing = required - set(wide.columns)

    if missing:
        raise RuntimeError(
            f"Missing bootstrap columns: {sorted(missing)}"
        )

    observed = wide["Observed_Yield"].to_numpy()

    error_a = np.abs(
        observed
        - wide[model_a].to_numpy()
    )

    error_b = np.abs(
        observed
        - wide[model_b].to_numpy()
    )

    point_difference = float(
        np.mean(error_a)
        - np.mean(error_b)
    )

    bootstrap_differences = []

    if cluster_by_state:

        states = np.asarray(
            sorted(
                wide["State"].unique()
            )
        )

        indices_by_state = {
            state:
                wide.index[
                    wide["State"] == state
                ].to_numpy()
            for state in states
        }

        for _ in range(n_bootstrap):

            sampled_states = rng.choice(
                states,
                size=len(states),
                replace=True,
            )

            sampled_indices = np.concatenate(
                [
                    indices_by_state[state]
                    for state in sampled_states
                ]
            )

            boot_error_a = (
                np.abs(
                    wide.loc[
                        sampled_indices,
                        "Observed_Yield",
                    ].to_numpy()
                    -
                    wide.loc[
                        sampled_indices,
                        model_a,
                    ].to_numpy()
                )
            )

            boot_error_b = (
                np.abs(
                    wide.loc[
                        sampled_indices,
                        "Observed_Yield",
                    ].to_numpy()
                    -
                    wide.loc[
                        sampled_indices,
                        model_b,
                    ].to_numpy()
                )
            )

            bootstrap_differences.append(
                np.mean(boot_error_a)
                - np.mean(boot_error_b)
            )

    else:

        n = len(wide)

        for _ in range(n_bootstrap):

            idx = rng.integers(
                0,
                n,
                size=n,
            )

            bootstrap_differences.append(
                np.mean(error_a[idx])
                - np.mean(error_b[idx])
            )

    lower, upper = percentile_ci(
        bootstrap_differences,
        CI_LEVEL,
    )

    return {
        "Model_A":
            model_a,

        "Model_B":
            model_b,

        "MAE_Difference_A_minus_B":
            point_difference,

        "CI_Lower":
            lower,

        "CI_Upper":
            upper,

        "CI_Excludes_Zero":
            bool(
                lower > 0
                or upper < 0
            ),
    }


def prepare_prediction_wide(
    path: Path,
    *,
    spatial: bool,
):
    df = pd.read_csv(path)

    df = df[
        df["Feature_Set"] == "Full"
    ].copy()

    keys = [
        "State",
        "Year",
    ]

    if spatial and "Fold" in df.columns:
        keys.append("Fold")

    observed_check = (
        df.groupby(
            ["State", "Year"]
        )["Observed_Yield"]
        .nunique()
        .max()
    )

    if observed_check != 1:
        raise RuntimeError(
            "Observed yield differs between models."
        )

    observed = (
        df[
            [
                "State",
                "Year",
                "Observed_Yield",
            ]
        ]
        .drop_duplicates()
    )

    predictions = (
        df
        .pivot_table(
            index=[
                "State",
                "Year",
            ],
            columns="Model",
            values="Predicted_Yield",
            aggfunc="first",
        )
        .reset_index()
    )

    wide = observed.merge(
        predictions,
        on=[
            "State",
            "Year",
        ],
        validate="one_to_one",
    )

    return wide


# ============================================================
# 1. PAIRED MODEL COMPARISONS
# ============================================================

def paired_model_comparisons():

    temporal = prepare_prediction_wide(
        TEMPORAL_PREDICTIONS,
        spatial=False,
    )

    spatial = prepare_prediction_wide(
        SPATIAL_PREDICTIONS,
        spatial=True,
    )

    pairs = [
        (
            "RandomForest",
            "LightGBM",
        ),
        (
            "RandomForest",
            "Ridge",
        ),
        (
            "LightGBM",
            "Ridge",
        ),
    ]

    rows = []

    for protocol, data, clustered in [
        (
            "Temporal_2024",
            temporal,
            False,
        ),
        (
            "Spatial_State_Held_Out",
            spatial,
            True,
        ),
    ]:

        for model_a, model_b in pairs:

            result = paired_state_bootstrap(
                data,
                model_a,
                model_b,
                cluster_by_state=clustered,
            )

            result["Protocol"] = protocol
            result["Bootstrap_Replicates"] = (
                N_BOOTSTRAP
            )

            rows.append(result)

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT_DIR
        / "paired_model_comparisons.csv",
        index=False,
    )

    return result


# ============================================================
# 2. GENERALIZATION DEGRADATION
# ============================================================

def generalization_degradation():

    temporal = prepare_prediction_wide(
        TEMPORAL_PREDICTIONS,
        spatial=False,
    )

    spatial = prepare_prediction_wide(
        SPATIAL_PREDICTIONS,
        spatial=True,
    )

    rows = []

    for model in MODELS:

        temporal_error = np.abs(
            temporal["Observed_Yield"]
            - temporal[model]
        )

        spatial_error = np.abs(
            spatial["Observed_Yield"]
            - spatial[model]
        )

        temporal_mae = float(
            temporal_error.mean()
        )

        spatial_mae = float(
            spatial_error.mean()
        )

        rows.append(
            {
                "Model":
                    model,

                "Temporal_MAE":
                    temporal_mae,

                "Spatial_MAE":
                    spatial_mae,

                "Spatial_minus_Temporal_MAE":
                    spatial_mae
                    - temporal_mae,

                "Relative_MAE_Change_pct":
                    (
                        (
                            spatial_mae
                            - temporal_mae
                        )
                        / temporal_mae
                        * 100.0
                    ),
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT_DIR
        / "generalization_degradation.csv",
        index=False,
    )

    return result


# ============================================================
# 3. 04a vs 04b CLIMATE METHOD AGREEMENT
# ============================================================

def climate_method_agreement():

    a = pd.read_csv(
        CLIMATE_04A
    )

    b = pd.read_csv(
        CLIMATE_04B
    )

    keys = [
        "State",
        "Year",
        "Model",
        "Scenario",
        "Scenario_Type",
        "Temperature_Shift_C",
        "Rainfall_Shift_pct",
    ]

    merged = a.merge(
        b,
        on=keys,
        suffixes=(
            "_04a",
            "_04b",
        ),
        validate="one_to_one",
    )

    merged = merged[
        merged["Scenario"]
        != "T+0_R0"
    ].copy()

    merged["Absolute_Method_Difference_pct_points"] = (
        merged[
            "Yield_Change_pct_04b"
        ]
        - merged[
            "Yield_Change_pct_04a"
        ]
    ).abs()

    rows = []

    for model, group in (
        merged.groupby(
            "Model"
        )
    ):

        rho, _ = spearmanr(
            group[
                "Yield_Change_pct_04a"
            ],
            group[
                "Yield_Change_pct_04b"
            ],
        )

        sign_a = np.sign(
            group[
                "Yield_Change_MT_Ha_04a"
            ]
        )

        sign_b = np.sign(
            group[
                "Yield_Change_MT_Ha_04b"
            ]
        )

        sign_agreement = float(
            np.mean(
                sign_a == sign_b
            )
        )

        rows.append(
            {
                "Model":
                    model,

                "Scenario_State_Count":
                    len(group),

                "Spearman_04a_vs_04b":
                    float(rho),

                "Mean_Absolute_Difference_pct_points":
                    float(
                        group[
                            "Absolute_Method_Difference_pct_points"
                        ].mean()
                    ),

                "Median_Absolute_Difference_pct_points":
                    float(
                        group[
                            "Absolute_Method_Difference_pct_points"
                        ].median()
                    ),

                "Max_Absolute_Difference_pct_points":
                    float(
                        group[
                            "Absolute_Method_Difference_pct_points"
                        ].max()
                    ),

                "Directional_Agreement":
                    sign_agreement,
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT_DIR
        / "climate_method_agreement.csv",
        index=False,
    )

    return result


# ============================================================
# 4. UNCERTAINTY TRADE-OFF
# ============================================================

def uncertainty_tradeoff():

    rows = []

    protocols = {
        "Temporal":
            pd.read_csv(
                TEMPORAL_UNCERTAINTY
            ),

        "Spatial":
            pd.read_csv(
                SPATIAL_UNCERTAINTY
            ),
    }

    for protocol, df in protocols.items():

        for model, group in (
            df.groupby(
                "Model"
            )
        ):

            coverage = float(
                group[
                    "Covered"
                ].mean()
            )

            mean_width = float(
                group[
                    "Interval_Width"
                ].mean()
            )

            mean_yield = float(
                group[
                    "Observed_Yield"
                ].mean()
            )

            rows.append(
                {
                    "Protocol":
                        protocol,

                    "Model":
                        model,

                    "Target_Coverage":
                        0.90,

                    "Empirical_Coverage":
                        coverage,

                    "Coverage_Gap":
                        coverage
                        - 0.90,

                    "Mean_Interval_Width":
                        mean_width,

                    "Mean_Observed_Yield":
                        mean_yield,

                    "Width_to_Mean_Yield_Ratio":
                        mean_width
                        / mean_yield,

                    "Width_as_pct_of_Mean_Yield":
                        (
                            mean_width
                            / mean_yield
                            * 100.0
                        ),
                }
            )

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT_DIR
        / "uncertainty_tradeoff.csv",
        index=False,
    )

    return result


# ============================================================
# 5. ABLATION CONSISTENCY
# ============================================================

def ablation_consistency():

    temporal = pd.read_csv(
        TEMPORAL_ABLATION
    )

    spatial = pd.read_csv(
        SPATIAL_ABLATION
    )

    temporal = temporal[
        [
            "Feature_Set",
            "Model",
            "Delta_MAE_vs_Full",
            "Delta_R2_vs_Full",
        ]
    ].rename(
        columns={
            "Delta_MAE_vs_Full":
                "Temporal_Delta_MAE",

            "Delta_R2_vs_Full":
                "Temporal_Delta_R2",
        }
    )

    spatial = spatial[
        [
            "Feature_Set",
            "Model",
            "Delta_MAE_vs_Full",
            "Delta_R2_vs_Full",
        ]
    ].rename(
        columns={
            "Delta_MAE_vs_Full":
                "Spatial_Delta_MAE",

            "Delta_R2_vs_Full":
                "Spatial_Delta_R2",
        }
    )

    merged = temporal.merge(
        spatial,
        on=[
            "Feature_Set",
            "Model",
        ],
        validate="one_to_one",
    )

    def classify(row):

        if row["Feature_Set"] == "Full":
            return "Reference"

        temporal_better = (
            row["Temporal_Delta_MAE"] < 0
            and row["Temporal_Delta_R2"] > 0
        )

        spatial_better = (
            row["Spatial_Delta_MAE"] < 0
            and row["Spatial_Delta_R2"] > 0
        )

        temporal_worse = (
            row["Temporal_Delta_MAE"] > 0
            and row["Temporal_Delta_R2"] < 0
        )

        spatial_worse = (
            row["Spatial_Delta_MAE"] > 0
            and row["Spatial_Delta_R2"] < 0
        )

        if (
            temporal_better
            and spatial_better
        ):
            return "Improved_Both"

        if (
            temporal_worse
            and spatial_worse
        ):
            return "Worsened_Both"

        return "Mixed"

    merged[
        "Consistency_Class"
    ] = merged.apply(
        classify,
        axis=1,
    )

    merged.to_csv(
        OUT_DIR
        / "ablation_consistency.csv",
        index=False,
    )

    return merged


# ============================================================
# 6. SHAP CONSISTENCY
# ============================================================

def shap_consistency():

    temporal = pd.read_csv(
        TEMPORAL_SHAP
    )

    spatial = pd.read_csv(
        SPATIAL_SHAP
    )

    stability = pd.read_csv(
        SPATIAL_SHAP_STABILITY
    )

    temporal = temporal[
        [
            "Model",
            "Feature",
            "Mean_Absolute_SHAP",
            "Rank",
        ]
    ].rename(
        columns={
            "Mean_Absolute_SHAP":
                "Temporal_Mean_Abs_SHAP",

            "Rank":
                "Temporal_Rank",
        }
    )

    spatial = spatial[
        [
            "Model",
            "Feature",
            "Mean_Absolute_SHAP",
            "Rank",
        ]
    ].rename(
        columns={
            "Mean_Absolute_SHAP":
                "Spatial_Mean_Abs_SHAP",

            "Rank":
                "Spatial_Rank",
        }
    )

    stability = stability[
        [
            "Model",
            "Feature",
            "Mean_Rank",
            "Rank_SD",
            "Best_Rank",
            "Worst_Rank",
        ]
    ]

    merged = (
        temporal
        .merge(
            spatial,
            on=[
                "Model",
                "Feature",
            ],
            validate="one_to_one",
        )
        .merge(
            stability,
            on=[
                "Model",
                "Feature",
            ],
            validate="one_to_one",
        )
    )

    merged[
        "Temporal_Spatial_Rank_Difference"
    ] = (
        merged[
            "Temporal_Rank"
        ]
        - merged[
            "Spatial_Rank"
        ]
    ).abs()

    merged.to_csv(
        OUT_DIR
        / "shap_consistency.csv",
        index=False,
    )

    return merged


# ============================================================
# SUMMARY MARKDOWN
# ============================================================

def write_summary(
    paired,
    degradation,
    climate,
    uncertainty,
    ablation,
    shap_table,
):

    path = (
        OUT_DIR
        / "experiment_08_summary.md"
    )

    lines = [
        "# Experiment 08 — Robustness Synthesis",
        "",
        "## Status",
        "",
        "COMPLETE.",
        "",
        "## Purpose",
        "",
        (
            "This experiment synthesizes the temporal, spatial, "
            "climate-stress, uncertainty, ablation, and explainability "
            "experiments without fitting additional predictive models."
        ),
        "",
        "## Paired model comparison",
        "",
    ]

    for _, row in paired.iterrows():

        lines.append(
            (
                f"- {row['Protocol']}: "
                f"{row['Model_A']} minus {row['Model_B']} MAE = "
                f"{row['MAE_Difference_A_minus_B']:.4f}, "
                f"95% bootstrap CI "
                f"[{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]."
            )
        )

    lines.extend(
        [
            "",
            "## Temporal vs spatial generalization",
            "",
        ]
    )

    for _, row in degradation.iterrows():

        lines.append(
            (
                f"- {row['Model']}: temporal MAE "
                f"{row['Temporal_MAE']:.4f}; spatial MAE "
                f"{row['Spatial_MAE']:.4f}; difference "
                f"{row['Spatial_minus_Temporal_MAE']:+.4f}."
            )
        )

    lines.extend(
        [
            "",
            "## Climate-stress method agreement",
            "",
        ]
    )

    for _, row in climate.iterrows():

        lines.append(
            (
                f"- {row['Model']}: 04a–04b Spearman agreement "
                f"{row['Spearman_04a_vs_04b']:.3f}; directional "
                f"agreement {row['Directional_Agreement']:.1%}; "
                f"mean absolute difference "
                f"{row['Mean_Absolute_Difference_pct_points']:.3f} "
                "percentage points."
            )
        )

    lines.extend(
        [
            "",
            "## Predictive uncertainty",
            "",
        ]
    )

    for _, row in uncertainty.iterrows():

        lines.append(
            (
                f"- {row['Protocol']} {row['Model']}: coverage "
                f"{row['Empirical_Coverage']:.1%}; mean interval width "
                f"{row['Mean_Interval_Width']:.3f} MT/ha "
                f"({row['Width_as_pct_of_Mean_Yield']:.1f}% of mean "
                "observed yield)."
            )
        )

    lines.extend(
        [
            "",
            "## Ablation consistency",
            "",
        ]
    )

    counts = (
        ablation[
            ablation[
                "Feature_Set"
            ] != "Full"
        ][
            "Consistency_Class"
        ]
        .value_counts()
    )

    for category, count in counts.items():

        lines.append(
            f"- {category}: {count} model-feature-set comparisons."
        )

    lines.extend(
        [
            "",
            "## Explainability stability",
            "",
        ]
    )

    for model in sorted(
        shap_table[
            "Model"
        ].unique()
    ):

        model_df = (
            shap_table[
                shap_table[
                    "Model"
                ] == model
            ]
            .sort_values(
                "Spatial_Rank"
            )
            .head(3)
        )

        features = ", ".join(
            model_df[
                "Feature"
            ].tolist()
        )

        lines.append(
            f"- {model} top spatial SHAP features: {features}."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The synthesis should be interpreted as evidence about "
                "predictive robustness and model behavior. It does not "
                "convert the observational weather-yield relationships "
                "into causal or climate-projection estimates."
            ),
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    require_files(
        [
            TEMPORAL_PREDICTIONS,
            SPATIAL_PREDICTIONS,
            CLIMATE_04A,
            CLIMATE_04B,
            TEMPORAL_UNCERTAINTY,
            SPATIAL_UNCERTAINTY,
            TEMPORAL_ABLATION,
            SPATIAL_ABLATION,
            TEMPORAL_SHAP,
            SPATIAL_SHAP,
            SPATIAL_SHAP_STABILITY,
        ]
    )

    print(
        "=" * 72
    )

    print(
        "EXPERIMENT 08 — ROBUSTNESS SYNTHESIS"
    )

    print(
        "=" * 72
    )

    paired = (
        paired_model_comparisons()
    )

    degradation = (
        generalization_degradation()
    )

    climate = (
        climate_method_agreement()
    )

    uncertainty = (
        uncertainty_tradeoff()
    )

    ablation = (
        ablation_consistency()
    )

    shap_table = (
        shap_consistency()
    )

    write_summary(
        paired,
        degradation,
        climate,
        uncertainty,
        ablation,
        shap_table,
    )

    print(
        "\n[1] PAIRED MODEL COMPARISONS"
    )

    print(
        paired.to_string(
            index=False
        )
    )

    print(
        "\n[2] TEMPORAL VS SPATIAL GENERALIZATION"
    )

    print(
        degradation.to_string(
            index=False
        )
    )

    print(
        "\n[3] 04a vs 04b CLIMATE METHOD AGREEMENT"
    )

    print(
        climate.to_string(
            index=False
        )
    )

    print(
        "\n[4] UNCERTAINTY TRADE-OFF"
    )

    print(
        uncertainty.to_string(
            index=False
        )
    )

    print(
        "\n[5] ABLATION CONSISTENCY"
    )

    print(
        ablation[
            [
                "Feature_Set",
                "Model",
                "Temporal_Delta_MAE",
                "Temporal_Delta_R2",
                "Spatial_Delta_MAE",
                "Spatial_Delta_R2",
                "Consistency_Class",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n[6] SHAP CONSISTENCY"
    )

    print(
        shap_table[
            [
                "Model",
                "Feature",
                "Temporal_Rank",
                "Spatial_Rank",
                "Mean_Rank",
                "Rank_SD",
                "Temporal_Spatial_Rank_Difference",
            ]
        ]
        .sort_values(
            [
                "Model",
                "Spatial_Rank",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nSaved synthesis artifacts to:"
    )

    print(
        OUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()