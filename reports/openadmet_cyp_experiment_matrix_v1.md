# OpenADMET CYP one-week experiment matrix v1

This matrix is frozen before model selection. Test labels are unavailable. The
750 blind structures are limited to contamination filtering and final frozen
inference; their feature distribution cannot choose an architecture, external
source, hyperparameter or ensemble weight.

## Shared validation protocol

- Group unit: standardized InChIKey across every assay table.
- Chemistry groups: deterministic fingerprint/scaffold clusters derived from
  training structures only; all modalities for one identity share one group.
- Development: nested chemistry-group CV (five outer folds; inner tuning folds).
- Certification: one sealed 10–15% chemistry-cluster holdout, opened once after
  the recipe and ensemble rule are frozen.
- Regression primary metric: official macro ST-RAE, with MAE, Spearman and
  interval coverage as secondary outcomes.
- Classification primary metric: macro MCC, with PR-AUC, balanced accuracy,
  calibration error and Brier score as secondary outcomes.
- Uncertainty: task/cluster bootstrap, conformal coverage, risk–coverage and
  abstention-utility curves.

## Model families

| ID | Model | Purpose |
|---|---|---|
| B0 | Official-style RDKit descriptors + LightGBM, random row split | Tutorial comparability only; never a main scientific estimate |
| B1 | Rich 2D fingerprints/descriptors + endpoint-specific LightGBM | Strong chemistry-group-CV baseline |
| B2 | Frozen molecular foundation embedding + LightGBM | Pretrained-representation baseline |
| M0 | Shared molecule trunk + isoform/assay tokens + masked endpoint heads | Tests assay-conditioned multitask transfer |
| M1 | M0 + credible-interval-aware regression loss | Aligns training with ST-RAE and assay uncertainty |
| M2 | M1 + TDI-condition pIC50 and Emax auxiliary heads | Tests paired-condition multi-fidelity transfer |
| M3 | M2 + single-concentration auxiliary head | Tests cheap-assay multi-fidelity transfer |
| M4 | M3 + direct/TDI pIC50-shift consistency with TDI class head | Tests mechanistic cross-head consistency |
| E0 | Frozen OOF stack of B1, B2 and best M-model | Final variance-controlled ensemble |

The M-family input is a molecule representation plus explicit isoform,
preincubation condition, assay type, source and fidelity tokens. It predicts
sparse long-form observations with masks; it does not duplicate a molecule into
independent random-split samples.

## Required ablations

1. `M0 versus B1/B2`: shared assay-conditioned architecture.
2. `M1 versus M0`: interval-aware objective.
3. `M2 versus M1`: paired TDI/Emax auxiliary information.
4. `M3 versus M2`: single-concentration fidelity.
5. `M4 versus M3`: biochemical consistency constraint.
6. `M4 without isoform token`: isoform conditioning.
7. `M4 with shuffled assay tokens`: negative control for token use.
8. `E0 versus best component`: ensemble value rather than component cherry-pick.
9. Exact-source external data, bridge-only auxiliary data and pretraining-only
   data, each versus the same no-external-data recipe.
10. Identity-group CV versus raw row CV, reported strictly as leakage-bias
    quantification and never used to choose the final model.

E0 is evaluated by nested OOF stacking rather than by fitting and scoring
weights on the same OOF rows. For each outer fold, weights are learned on the
other four folds; final all-OOF weights are used only for the later sealed
certification/refit package.

## Compute-aware funnel

Stage A runs B1, B2, M0, M1 and M2 with three outer folds and one seed. Any
candidate with a task-macro metric worse than B1 and no calibrated/UQ advantage
is stopped. Stage B runs surviving models plus M3/M4 with five outer folds and
three seeds. Stage C freezes one model/stack and evaluates the sealed internal
holdout once. Stage D retrains the frozen recipe on all admitted training data
and creates one content-addressed blind submission.

The executable Stage-A gate is frozen in
`scripts/promote_openadmet_cyp_stage_a.py`. Relative to B1, a candidate must not
worsen mean macro ST-RAE by more than 0.05 or mean macro MCC by more than 0.03.
It must additionally improve at least one primary metric, or improve macro
classification Brier by at least 0.01 or ECE by at least 0.02. UQ cannot rescue
a material primary-metric regression, and blind leaderboard feedback cannot
override a failed internal gate.

Stage-B component admission is frozen in
`scripts/select_openadmet_cyp_stage_b.py`. It averages the three seeds within
each of the five chemistry-group outer folds before comparison with B1. A
candidate must satisfy mean ST-RAE noninferiority within 0.05, mean MCC
noninferiority within 0.03, improve at least one primary metric, and avoid a
worst-fold regression larger than 0.10 on either primary metric. If neither M3
nor M4 passes, the recipe falls back to single-component B1; a failed model is
not retained merely to manufacture an ensemble.

All stopped models, timeouts and failed hypotheses remain in the ledger. A
multi-fidelity or consistency claim requires its paired ablation to improve the
task-level bootstrap estimate or a preregistered calibration/UQ endpoint; the
blind leaderboard cannot rescue a failed internal claim.

## Frozen Stage-A/B outcome (2026-08-21 UTC)

The funnel was executed without loading the sealed certification labels or the
blind-test labels.  Stage A rejected B2, M0, M1 and M2.  Stage B then compared
the preregistered B1 baseline with M3 and M4 using all five outer chemistry
folds and seeds 1, 2 and 3:

| ID | Mean macro ST-RAE (lower is better) | Mean macro MCC (higher is better) | Stage-B decision |
|---|---:|---:|---|
| B1 | 0.7259417 | 0.2463464 | selected baseline |
| M3 | 0.7743852 | 0.1154448 | rejected |
| M4 | 0.9025530 | 0.1073952 | rejected |

M3 failed classification noninferiority and the worst-fold guard. M4 failed
regression and classification noninferiority and the worst-fold guard. The
frozen recipe mode is therefore `single_component_B1`; no post-hoc stack is
fit. This is a prospective falsification result for the proposed neural
multi-fidelity mechanisms, not evidence that they improved the endpoint.

Authoritative artifacts:

- `reports/openadmet_cyp_stage_b_component_gate_v1.json`
- `reports/openadmet_cyp_stage_b_cv_analysis_v1.json`
- `runs/openadmet-cyp-stage-b/b1_cv_result.json`
- `runs/openadmet-cyp-stage-b/m3_cv_result.json`
- `runs/openadmet-cyp-stage-b/m4_cv_result.json`

The component gate records `training_only=true`,
`sealed_certification_opened=false`, `blind_test_opened=false` and
`leaderboard_feedback_used=false`.
