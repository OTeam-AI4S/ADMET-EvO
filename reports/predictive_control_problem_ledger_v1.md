# Predictive-control autonomous problem ledger

- Material losses: 8；material gains: 3
- Policy mutated after evaluation: false
- Evaluation seed reused for confirmation: false
- Test loaded: false

| Task | Category | Delta | Agent selection | AutoML selection |
|---|---|---:|---|---|
| Clearance_Microsome_AZ | excretion | -0.1388 | control | lightgbm |
| CYP2C9_Veith | metabolism | -0.0533 | treatment | lightgbm |
| DILI | toxicity | -0.0465 | control | lightgbm |
| Lipophilicity_AstraZeneca | absorption | -0.0366 | treatment | lightgbm |
| PPBR_AZ | distribution | -0.0340 | control | lightgbm |
| CYP2D6_Veith | metabolism | -0.0265 | control | lightgbm |
| Clearance_Hepatocyte_AZ | excretion | -0.0260 | control | lightgbm |
| CYP3A4_Veith | metabolism | -0.0232 | treatment | lightgbm |

## 可证伪的下一版本假设

### 1. `separate-research-pair-from-predictive-anchor`

- Observation: All material losses selected LightGBM in the AutoML arm while the minimal Agent candidate budget was consumed by its scientific pair.
- Change: Give both adaptive arms three equal screen slots: Agent matched pair plus frozen LightGBM anchor; AutoML three scalar candidates.
- Protocol: Freeze before use; select on seed 13 and evaluate once on seed 14.
- Gate: Agent-vs-AutoML task-bootstrap lower bound >= -0.01 with no extra logical fit budget for Agent.

### 2. `large-metabolism-needs-model-anchor-not-only-imbalance-axis`

- Observation: All three large Veith endpoints favored the AutoML LightGBM selection on fresh evaluation.
- Change: Retain imbalance matched evidence as research output but prevent it from displacing the frozen model anchor in finalist selection.
- Protocol: Evaluate only in a newly frozen large-metabolism cohort; no seed-12 reuse.
- Gate: No material regression >0.02 on any Veith endpoint and mean oriented delta >= -0.005.

### 3. `rank-regression-feature-axis-can-miss-estimator-effect`

- Observation: Clearance_Microsome showed the largest loss when the feature-axis control was selected instead of LightGBM.
- Change: Trigger one model-anchor comparison when rank-metric feature evidence is unresolved or selects the basic feature control.
- Protocol: Matched future-seed comparison on excretion rank endpoints only.
- Gate: Correction improves mean normalized utility >=0.02 without more than one added configuration.

Post-evaluation diagnosis only. These hypotheses are not validated fixes and may only be tested on a newly frozen future split.
