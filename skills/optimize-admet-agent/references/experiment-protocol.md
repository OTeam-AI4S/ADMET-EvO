# Protocol-safe ADMET ablations

## Table of contents

1. [Data firewall](#data-firewall)
2. [Paired experimental design](#paired-experimental-design)
3. [Caco2 ablation matrix](#caco2-ablation-matrix)
4. [Statistics and decisions](#statistics-and-decisions)
5. [Machine-readable schema](#machine-readable-schema)
6. [Artifact checklist](#artifact-checklist)

## Data firewall

Use four conceptual zones:

| Zone | Permitted use |
|---|---|
| Outer-train | fit representations, selectors, models, and inner folds |
| Outer-validation | compare preregistered candidates and estimate paired gain |
| Confirmation validation | confirm the selected mechanism on fresh, preregistered seeds/splits |
| Official test | one-shot evaluation of the frozen finalist only |

During search, do not expose official test labels, per-molecule test errors, test predictions, test metrics, or current rank gaps to the planner. A dated leaderboard snapshot may set the project objective, but must not select a recipe.

For external data or pretrained weights, record:

- source URL/DOI and immutable version or hash;
- endpoint definition, unit, assay direction, pH/temperature when relevant;
- canonical-SMILES and InChIKey overlap with every TDC split;
- whether TDC or equivalent records could have entered pretraining;
- deduplication and conflicting-label policy.

## Paired experimental design

Use identical outer scaffold splits and seeds for baseline and candidate. For each outer seed:

1. derive all target-dependent choices from outer-train only;
2. use inner scaffold OOF folds for feature selection, hyperparameters, calibration, stacking, and early stopping;
3. refit the chosen pipeline on the full outer-train;
4. evaluate once on outer-validation;
5. persist predictions in stable row order plus molecule identifiers.

Use at least five discovery outer seeds. If several representations, top-N cutoffs, or learners were screened, preregister the chosen mechanism and confirm it on fresh seeds such as 6–10 before freezing. Repeated seeds are repeated resplits, not independent new datasets; state this limitation.

Prevent feature-selection leakage:

- compute target-independent descriptors before splitting only if the cache contains no fitted statistics;
- fit imputation, variance filters, scaling, permutation importance, SHAP aggregation, and top-N choice inside inner training folds;
- never use outer-validation or official test values to rank descriptors;
- store selected descriptor names per outer seed and selection stability across seeds.

## Caco2 ablation matrix

Run stages sequentially and stop a branch that fails its acceptance threshold.

| ID | Scientific factor | Candidate | Control | Primary readout |
|---|---|---|---|---|
| A0 | Current baseline | current frozen rich-2D recipe | none | MAE, runtime |
| A1 | Feature family | RDKit; +ErG; +Morgan/ECFP; +Avalon | A0 or RDKit-only | paired MAE gain |
| A2 | Learner | CatBoost | ExtraTrees, LightGBM on same matrix | paired MAE gain/cost |
| A3 | Selection | nested permutation top-N | same descriptors, no selector | MAE and selection stability |
| A4 | Descriptor source | PaDEL2D or Mordred2D | RDKit descriptors | gain per feature-second |
| A5 | Geometry | 3D-only; 2D+3D | matched 2D-only | paired gain/conformer failures |
| A6 | Automation | validation-only AutoML ensemble | best single model | gain, wall time, model count |
| A7 | Learned graph | Chemprop; Chemprop-RDKit | rich-2D CatBoost | gain and variance |
| A8 | Transfer | multitask Chemprop-RDKit | single-task version | gain and overlap audit |
| A9 | 3D pretraining | Uni-Mol2 84M or hybrid | cheap 2D champion | LCB utility and GPU cost |

Recommended minimum useful effects must be chosen before results. For Caco2, consider a validation MAE reduction of 0.01 a meaningful initial gate, but adjust this using existing paired variability and project cost. Do not set the threshold from observed candidate scores.

## Statistics and decisions

For each candidate calculate improvement with a common sign:

```text
improvement_i = candidate_i - baseline_i     if higher is better
improvement_i = baseline_i - candidate_i     if lower is better
```

Report:

- mean and sample standard deviation of each method;
- mean paired improvement and its bootstrap 95% interval;
- median improvement;
- win/tie/loss counts;
- exact paired sign-flip p-value;
- runtime/GPU-hour and peak-memory delta;
- invalid/failed molecule counts.

With only five pairs, the smallest possible two-sided exact sign-flip p-value is 0.0625. Do not demand `p < 0.05` as the only gate or claim significance from five favorable seeds. Prefer practical thresholds, interval estimates, confirmation seeds, and consistency. When comparing many methods across many endpoints, use a Friedman test followed by a multiplicity-aware post-hoc procedure such as Nemenyi, or a preregistered paired hierarchical analysis.

Decision meanings:

- `promote`: confirmation evidence clears the practical threshold and all audits pass;
- `confirm`: discovery evidence is positive but uncertainty or selection pressure remains;
- `reject`: candidate misses the threshold, regresses, or costs more without justified gain;
- `blocked`: required provenance, dependency, compute, or valid artifact is unavailable.

## Machine-readable schema

Use this input with `scripts/compare_ablation.py`:

```json
{
  "metric": "mae",
  "higher_is_better": false,
  "seeds": [1, 2, 3, 4, 5],
  "baseline": {
    "name": "rich_2d_catboost",
    "scores": [0.31, 0.30, 0.29, 0.32, 0.30],
    "cost": {"gpu_hours": 0.0, "wall_seconds": 120.0}
  },
  "candidates": [
    {
      "name": "nested_mordred_selector",
      "scores": [0.29, 0.28, 0.28, 0.30, 0.29],
      "cost": {"gpu_hours": 0.0, "wall_seconds": 420.0}
    }
  ],
  "protocol": {
    "split": "TDC default scaffold validation",
    "test_labels_used": false,
    "selection": "inner scaffold OOF only"
  }
}
```

The script rejects non-finite scores, unequal paired lengths, duplicate names, fewer than two pairs, and inconsistent seed counts. Its bootstrap is deterministic for a fixed `--random-seed`.

## Artifact checklist

- task name, endpoint definition, metric, direction, and TDC/PyTDC version;
- raw data hash and split molecule hashes;
- train/validation molecule and scaffold overlap audit;
- candidate specification and hypothesis;
- code revision and dirty-worktree note;
- dependency lock, device, worker count, and random seeds;
- feature schema, cache key, descriptor names, conformer settings, and failure counts;
- per-seed predictions, identifiers, scores, runtime, peak memory, and logs;
- inner-fold selection records and OOF predictions;
- paired comparison JSON;
- frozen finalist serialization and SHA-256 digest;
- explicit `test_labels_used: false` for all search artifacts.
