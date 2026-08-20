# ADMET-EvO integration map

This map reflects the repository inspected on 13 August 2026. Reinspect before editing because the worktree may contain active user changes.

## Table of contents

1. [Existing capabilities](#existing-capabilities)
2. [Safe extension pattern](#safe-extension-pattern)
3. [Recommended implementation sequence](#recommended-implementation-sequence)
4. [Tests required](#tests-required-for-any-promoted-method)
5. [Worktree rule](#worktree-rule)

## Existing capabilities

| Area | Path | Existing role |
|---|---|---|
| Typed experiment | `admet_agent/schemas.py` | serializable task, representations, model, imbalance, hyperparameters, budget |
| Registry | `admet_agent/registry.py` | registered representations and models |
| Features/cache | `admet_agent/features.py` | RDKit fingerprints/descriptors and optional Uni-Mol2 cache |
| Nested selection | `admet_agent/feature_selection.py` | outer-train-only scaffold CV and permutation top-N selection |
| Guard | `admet_agent/guard.py` | rejects unknown tools, hyperparameters, leakage fields, and invalid budgets |
| Planner | `admet_agent/brain.py`, `openai_brain.py` | bounded rule/LLM proposal generation |
| Training/scoring | `admet_agent/selection.py`, `runner.py` | fit, predict, metric scoring, validation search, finalization |
| Tuning | `admet_agent/tuning.py` | bounded tuning support |
| Evidence ledger | `admet_agent/experiment_db.py`, `memory.py` | structured experiment history |
| Tests | `tests/test_contracts.py`, `test_agent_safety.py`, `test_v2.py`, `test_v21.py` | protocol, schema, selection, freeze, and Uni-Mol2 behavior |

Current representations include Morgan variants, Morgan count, Avalon, ErG, MACCS, RDKit path, atom pair, topological torsion, RDKit 2D, and optional Uni-Mol2. Current learners include Extra Trees, Random Forest, RBF SVM, XGBoost, LightGBM, CatBoost, and a balanced Extra Trees variant, subject to optional dependencies.

The v2.4 Caco2 research harnesses are `scripts/run_caco2_feature_ablation.py` and `scripts/run_caco2_nested_selection.py`. The former separates discovery seeds 1–5 from confirmation seeds 6–10 and saves safe per-molecule NPZ artifacts; the latter fits all target-dependent ranking and cutoff choices inside outer-train scaffold folds. `scripts/build_caco2_v24_freeze_manifest.py` validates the gate and hashes the retained recipe's code and prediction artifacts.

Relevant existing evidence:

- `runs/caco2-official-v1/final/results.json`: older Caco2 frozen Random Forest rich-2D run, reported 0.293 ± 0.002 MAE.
- `reports/goal_scorecard.md`: dated leaderboard comparison; on 13 August 2026 it recorded Caco2 0.283 and rank 4. This is a planning snapshot, not validation evidence.
- `runs/unimol2-pilot-v22/summary.md`: six-task 84M study.
- `runs/v23-ablation-study/summary.md`: routing, transform/fusion, and 84M/164M scale conclusions.
- `docs/v24_research_plan.md`: current sequential evidence-acquisition proposal.
- `reports/caco2_v24_decision.md`: completed validation-only rich-2D, learner, forward-addition, and nested-selector decisions.
- `reports/caco2_v24_freeze_manifest.json`: hashed Caco2 validation-candidate freeze record; it explicitly does not authorize official-test access.

## Safe extension pattern

When adding a representation:

1. add a stable registry name and dependency availability check;
2. separate target-independent extraction from fitted preprocessing/selection;
3. include implementation/package versions and parameters in the feature cache key;
4. define invalid-SMILES and descriptor-failure behavior;
5. update the guard and OpenAI schema allowlist;
6. add shape, finite-value, deterministic-cache, and missing-dependency tests;
7. update CLI tool discovery and installation extras.

When adding a model:

1. register the model and optional dependency;
2. add only bounded hyperparameters to the guard and planner schema;
3. implement classification/regression and metric-appropriate prediction consistently;
4. make device and worker behavior explicit;
5. test parameter forwarding, fixed-seed reproducibility, and unsupported combinations.

When adding target-dependent feature selection:

- model it as a fitted pipeline component, not a representation cache;
- fit inside every inner fold and outer training split;
- serialize selector type, feature ordering, cutoff rule, and fitted feature names;
- ensure finalization reconstructs and refits the selector without reading test labels;
- record selection frequency across outer seeds.

## Recommended implementation sequence

### P0: strengthen the cheap Caco2 lane

Completed in v2.4. Do not repeat unchanged; consult `references/caco2-v24-results.md`.

1. Reconstruct the strongest current validation-only Caco2 2D candidate.
2. Ensure CatBoost receives the exact same rich feature matrix as other learners.
3. Add a forward-addition experiment over RDKit → ErG → Morgan count/ECFP → Avalon.
4. Run paired outer seeds and retain raw validation predictions.

This tests MapLight and the 2025 feature benchmark using already available components.

### P1: add nested descriptor selection

The leak-safe selector infrastructure is implemented and RDKit-only selection was rejected on current evidence. The next use of this path should involve a richer descriptor source or a materially different selector, not retuning top-N values against the same outer validation seeds.

Implement a training-fold selector over RDKit 2D first. Compare no selection with preregistered top-N or threshold rules. Only after the selector path is leak-safe, add PaDEL or Mordred descriptors as an optional dependency.

Prefer a maintained Python-compatible descriptor package and pin its version. PaDEL invokes a Java-backed tool and has different operational failure modes; Mordred community packages may be easier to integrate but must be checked against the repository's Python/RDKit range. Do not silently substitute descriptors when extraction fails.

### P2: test learned representations

Add Chemprop-RDKit as two explicit candidates, `chemprop` and `chemprop_rdkit`, so descriptor contribution is measurable. Add multitask transfer only after dataset overlap and unit audits. Reuse the existing cost-aware evidence router to decide whether Uni-Mol2 or Chemprop training is justified.

### P3: AutoML only after pipeline safety

If integrating AutoGluon or another AutoML system:

- expose a bounded preset, time limit, allowed learner list, and ensemble cap;
- run it only inside outer training data;
- prevent it from loading official test or arbitrary paths;
- serialize every ensemble member and weight;
- compare it to the best constituent model at equal and unconstrained cost.

## Tests required for any promoted method

- schema round-trip and freeze-hash stability;
- guard rejects test-related and unknown selector fields;
- no train/validation SMILES or Bemis–Murcko scaffold overlap;
- selector is refit per outer seed and never sees outer-validation labels;
- predictions align with saved molecule identifiers;
- repeated fixed-seed run matches within documented numerical tolerance;
- optional dependency failure is clear and leaves the base install functional;
- finalization requires at least five unique seeds and a fresh freeze directory.

## Worktree rule

The inspected repository contains substantial modified and untracked work. Preserve user changes, avoid formatting unrelated files, and place new experiments in new run directories. Before editing a file, inspect its diff and current tests; do not assume `HEAD` is the active design.
