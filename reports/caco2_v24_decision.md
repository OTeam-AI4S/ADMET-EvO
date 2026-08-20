# Caco2 v2.4 optimization decision

Date: 13 August 2026. Outcome: freeze the existing binary-rich LightGBM recipe as the validation control; reject the tested CatBoost, full-count, Avalon-addition, and RDKit-only nested-selection branches. No official-test or leaderboard claim is made.

## Evidence matrix

| Class | Source | Reported data | Transfer decision |
|---|---|---|---|
| `published_external` | [TDC Caco2 leaderboard](https://tdcommons.ai/benchmark/admet_group/01caco2/) | 906 compounds, scaffold split, MAE; CaliciBoost 0.256 ± 0.006, XG Boost 0.274 ± 0.004, MapLight 0.276 ± 0.005 on the access date | Treat rank as hypothesis only; do not tune with test feedback |
| `published_external` | [CaliciBoost, J. Cheminformatics 2025](https://doi.org/10.1186/s13321-025-01137-7) | selected PaDEL 2D+3D/XGBoost; official five-seed 0.256 ± 0.006 MAE; three 3D descriptors among 73 selected features associated with about 15.5% MAE reduction | Motivates richer descriptors and nested selection, not test-set selection |
| `published_external` | [MapLight preprint](https://arxiv.org/abs/2310.00174) | CatBoost over ECFP-count, Avalon-count, ErG and properties; Top-3 on 16/22 TDC tasks without GIN | Motivated matched CatBoost/feature ablation |
| `published_external` | [2025 feature benchmark](https://doi.org/10.1186/s13321-025-01041-0) | RDKit → ErG → ECFP4 → Avalon forward path; CatBoost average rank 2.07 for two-feature models | Motivated paired forward addition and learner controls |
| `repository_internal` | Caco2 v2.4 discovery and confirmation | Values below; validation only, official test sealed | Freeze current LightGBM control; reject tested challengers |
| `proposed` | PaDEL2D or maintained Mordred-compatible source | Not installed or measured | Add only as a pinned optional dependency with nested-fold controls |

Published leaderboard numbers and repository validation numbers use different held-out rows and are not directly comparable.

## Internal paired results

Protocol: TDC default scaffold validation; 637 train and 91 validation molecules per seed; discovery seeds 1–5; preregistered confirmation seeds 6–10; minimum useful reduction 0.01 MAE. Positive paired improvement means lower MAE.

| Candidate | Confirmation MAE | Improvement vs binary-rich LightGBM | 95% paired bootstrap CI | W/T/L | Decision |
|---|---:|---:|---:|---:|---|
| Binary-rich LightGBM | 0.36789 ± 0.03621 | control | — | — | `promote`: retain/freeze, no new gain claimed |
| Full-count LightGBM | 0.36644 ± 0.03377 | +0.00145 | [-0.00792, 0.00955] | 3/0/2 | `reject` |
| Full-count Extra Trees | 0.36968 ± 0.03216 | -0.00179 | [-0.01020, 0.00623] | 3/0/2 | `reject` |
| CatBoost RDKit+ErG+Morgan-count | 0.38762 ± 0.03290 | -0.01973 | [-0.02860, -0.01086] | 0/0/5 | `reject` |

The discovery binary-rich LightGBM mean was 0.35893 ± 0.05275. CatBoost's three-view addition initially improved its RDKit-only control by 0.00990 MAE, just below the gate, but fresh confirmation reduced that gain to 0.00037 with a crossing-zero interval. The apparent discovery signal did not replicate.

Nested RDKit2D selection was also rejected: all descriptors scored 0.36034 ± 0.04301, while nested top-N scored 0.36435 ± 0.04671. Its paired improvement was -0.00402, 95% CI [-0.00836, 0], with 0 wins, 3 ties, 2 losses and 114.07 additional seconds.

## Audit and freeze gate

- All ten outer splits: zero exact-SMILES overlap, zero Bemis–Murcko scaffold overlap, and zero invalid train/validation molecules.
- Target-dependent permutation ranking and top-N choice stayed inside outer training, with separate scaffold holdouts for ranking and inner validation.
- Raw predictions, targets, molecule identifiers, split audits, package versions, data SHA-256, code revision, and runtime are preserved.
- The official TDC test was never loaded; `test_labels_used` is false throughout.
- Frozen validation recipe: `2d_lightgbm` = LightGBM over `morgan + avalon + erg + rdkit2d`.
- The freeze manifest does not authorize official-test execution. Its digest and source hashes are in `reports/caco2_v24_freeze_manifest.json`.

## Next experiment

The next bounded branch is a richer descriptor source, not more tuning of the rejected recipes on seeds 1–10. Add PaDEL2D or a maintained Mordred-compatible package as an optional, pinned dependency; compare all descriptors versus leak-safe nested selection against the frozen binary-rich LightGBM control. A separate 2D-versus-3D experiment is justified only after the 2D descriptor lane clears the 0.01-MAE validation gate.
