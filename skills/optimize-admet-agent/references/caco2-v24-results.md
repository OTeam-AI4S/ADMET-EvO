# ADMET-EvO v2.4 Caco2 internal evidence

This is repository-internal validation evidence produced on 13 August 2026. It is not an official TDC test result and must not be compared numerically with leaderboard MAE as if the evaluation sets were identical.

## Protocol

- Endpoint: `Caco2_Wang`; metric: MAE, lower is better.
- TDC default scaffold train/validation split: 637/91 rows per seed.
- Discovery seeds: 1–5; fresh confirmation seeds: 6–10.
- Preregistered minimum useful reduction: 0.01 MAE.
- Every seed had zero exact-SMILES overlap, zero Bemis–Murcko scaffold overlap, and zero invalid molecules between train and validation.
- The official test was not loaded and test labels were not used.
- Paired intervals use 20,000 deterministic bootstrap samples; sign-flip tests and Holm adjustment are recorded in the JSON summaries.

## Decisions

| Candidate | Phase and MAE | Paired improvement versus control | Decision |
|---|---:|---:|---|
| Current binary-rich LightGBM (`morgan + avalon + erg + rdkit2d`) | discovery 0.35893 ± 0.05275; confirmation 0.36789 ± 0.03621 | control | `promote`: retain and freeze as validation control; no new gain claimed |
| Full-count LightGBM | confirmation 0.36644 ± 0.03377 | +0.00145, 95% CI [-0.00792, 0.00955], 3/0/2 | `reject`: misses 0.01 gate |
| Full-count Extra Trees | confirmation 0.36968 ± 0.03216 | -0.00179, 95% CI [-0.01020, 0.00623], 3/0/2 | `reject`: no gain and higher cost |
| CatBoost RDKit+ErG+Morgan-count | confirmation 0.38762 ± 0.03290 | -0.01973, 95% CI [-0.02860, -0.01086], 0/0/5 | `reject`: negative fresh-seed interval |
| CatBoost + Avalon | discovery 0.37088 ± 0.04290 | +0.00258 versus CatBoost RDKit, 95% CI [-0.00718, 0.01400], 2/0/3 | `reject`: misses gate |
| Nested RDKit2D permutation top-N | discovery 0.36435 ± 0.04671 | -0.00402 versus all RDKit2D, 95% CI [-0.00836, 0], 0/3/2 | `reject`: regression and +114.07 s |

Positive improvement means lower MAE. Five repeated scaffold splits remain a small, dependent sample; the fresh confirmation phase controls selection pressure but does not create independent datasets.

## Artifacts

- `runs/caco2-feature-ablation-v24`: discovery results and raw predictions.
- `runs/caco2-feature-confirmation-v24`: preregistered confirmation results and raw predictions.
- `runs/caco2-nested-selection-v24`: inner-fold selector audits, selected feature names, and predictions.
- `reports/caco2_v24_freeze_manifest.json`: hashes for comparison summaries, implementation files, data, and all ten frozen-baseline prediction artifacts.

## Next bounded experiment

Do not retune the rejected recipes on seeds 1–10. The next descriptor experiment should add a pinned, optional PaDEL2D or maintained Mordred-compatible source, compare all descriptors with selection inside nested scaffold folds, and retain the frozen binary-rich LightGBM as control. Neither PaDEL nor Mordred is installed in the inspected environment, so this remains `proposed`, not measured.
