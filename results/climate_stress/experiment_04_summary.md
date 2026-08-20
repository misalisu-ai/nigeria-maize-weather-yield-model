# Experiment 04 — Climate-Stress Sensitivity

## Status

COMPLETE.

## 04a: Feature-space counterfactuals

Seasonal aggregate weather features were directly perturbed.

The experiment passed baseline identity checks but showed substantial
state-level non-monotonicity and extrapolation under severe warming.

## 04b: Weather-recomputed counterfactuals

Daily 2024 NASA POWER weather was perturbed and the original Dataset v1.0
feature-engineering definitions were reapplied.

The unperturbed T+0_R0 scenario reproduced the frozen 2024 weather features
and model predictions exactly.

## Main findings

### Drought

Weather-recomputed -30% rainfall produced mean predicted-yield changes of:

- LightGBM: approximately -2.74%
- Random Forest: approximately -1.25%
- Ridge: approximately +0.17%

The nonlinear models therefore showed modest average yield declines under
severe rainfall reduction, while Ridge did not provide a directionally
consistent drought response.

### Heat

Under +2°C with unchanged rainfall:

- Ridge: approximately +3.33%
- LightGBM: approximately +1.14%
- Random Forest: approximately -0.30%

Model disagreement indicates substantial model-form uncertainty.

### Compound stress

For +2°C and -30% rainfall:

- LightGBM: approximately -2.01%
- Random Forest: approximately -1.65%
- Ridge: approximately +3.49%

No single robust model-independent response was observed.

## 04a vs 04b

Temperature-only scenarios were effectively identical between the two methods.

Daily-weather recomputation changed rainfall scenarios primarily through
recalculation of maximum consecutive dry days.

At -30% rainfall:

- 11 of 37 states experienced increased CDD.
- Mean CDD increase was approximately 1.41 days.
- Maximum increase was 12 days.

Ridge was substantially more sensitive to this feature recomputation than
Random Forest or LightGBM.

## Interpretation

Experiment 04 measures sensitivity of fitted weather-yield models under
controlled counterfactual weather shifts.

The results do not constitute causal estimates or climate-change projections.