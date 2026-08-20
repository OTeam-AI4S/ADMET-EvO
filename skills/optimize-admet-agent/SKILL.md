---
name: optimize-admet-agent
description: Improve ADMET-EvO or another TDC ADMET modeling agent by converting current leaderboard methods and peer-reviewed molecular-property literature into auditable implementations, validation-only ablations, paired comparisons, and freeze-gated benchmark candidates. Use for ADMET agent optimization, TDC Caco2 or other ADMET leaderboard research, molecular representation or model selection, feature-selection and AutoML comparisons, Uni-Mol2 or Chemprop integration, ablation design, evidence-backed research plans, and claims that require literature data without test-set leakage.
---

# Optimize ADMET Agent

Turn published ideas into protocol-safe changes and evidence. Treat leaderboard results as hypotheses to test, not feedback for selecting a finalist.

## Load the right context

1. Inspect the repository before proposing changes. Read the active experiment schema, representation/model registry, guard, selection logic, tests, frozen results, and relevant run summaries.
2. Read [references/repo-integration.md](references/repo-integration.md) when working in ADMET-EvO or adapting a method to its typed pipeline.
3. Read [references/evidence-base.md](references/evidence-base.md) when comparing models, quoting published numbers, prioritizing Caco2 work, or writing scientific rationale.
4. Read [references/experiment-protocol.md](references/experiment-protocol.md) before designing, running, or interpreting an ablation.
5. Read [references/caco2-v24-results.md](references/caco2-v24-results.md) before repeating Caco2 rich-2D, CatBoost, or RDKit-selector experiments in ADMET-EvO.
6. Search the web when the request depends on a current leaderboard, recent paper, repository state, or package API. Prefer TDC, papers, official documentation, and authors' repositories. Record the access date.

## Execute the workflow

### 1. Define one falsifiable optimization

State:

- endpoint and official metric;
- current validation baseline and its artifact path;
- one mechanism to add or remove;
- expected direction and minimum useful effect;
- compute budget and failure condition.

Do not start from “use a newer/larger model.” Start from a mechanism supported by evidence, such as nested descriptor selection, complementary fingerprints, multitask transfer, or cost-aware 3D acquisition.

### 2. Build an evidence table

For every candidate method, capture:

- bibliographic source, year, peer-review status, code URL, and license if known;
- dataset, size, split, metric, mean, dispersion, and number of runs;
- exact mechanism that could transfer to this repository;
- protocol caveats and implementation cost.

Separate three evidence classes in the output:

1. `published_external`: values reported by a paper or official leaderboard;
2. `repository_internal`: values produced by checked-in artifacts;
3. `proposed`: expected outcomes that have not been measured.

Never merge these classes into one results table without a source column.

### 3. Translate the method into a bounded experiment

Map every method to registered, serializable components. Add new representations or models through the repository registry and guard; do not let the planner execute arbitrary code.

Keep target-dependent operations inside the training fold:

- feature ranking and top-N selection;
- normalization, imputation, calibration, and target transforms;
- hyperparameter or ensemble-weight selection;
- OOF stacker fitting.

Cache target-independent molecular features only. Store the fitted selector and ordered feature names in the finalist recipe or artifact manifest.

### 4. Run staged ablations

Use the same scaffold splits and seeds for every paired comparison. Run cheap, diagnostic stages before expensive stages:

1. current 2D champion;
2. representation ablation;
3. learner ablation;
4. nested feature-selection ablation;
5. 2D versus 3D descriptor ablation;
6. single model versus validation-only ensemble;
7. deep or pretrained expert only after a positive acquisition gate.

Change one scientific factor per row. Include runtime, peak memory, invalid molecule count, and cache provenance alongside the predictive metric.

### 5. Compare paired evidence

Prepare a JSON score file in the schema documented in [references/experiment-protocol.md](references/experiment-protocol.md), then run:

```bash
python skills/optimize-admet-agent/scripts/compare_ablation.py ablation.json \
  --output ablation-summary.json \
  --bootstrap-samples 20000
```

Interpret positive `mean_improvement` as better regardless of metric direction. Report paired deltas, bootstrap confidence intervals, win/tie/loss counts, exact sign-flip p-values, and compute cost. Treat five outer seeds as preliminary evidence; use a preregistered confirmation split when selection pressure is substantial.

### 6. Apply the freeze gate

Promote a candidate only when:

- the implementation and artifacts are reconstructable;
- all selection used train/validation data only;
- paired evidence meets the preregistered practical threshold;
- the gain is not driven by one split or unexplained data loss;
- the cost is acceptable for the stated objective;
- the exact recipe, dependencies, seeds, data hash, and code revision are frozen.

Evaluate the official test only after freezing. Do not change representations, hyperparameters, weights, routing, or transforms in response to test or leaderboard results.

## Caco2 priority ladder

For `Caco2_Wang`, prefer this order unless repository evidence contradicts it:

1. Reproduce the current rich 2D baseline on paired outer scaffold seeds.
2. Compare RDKit descriptors, ErG, ECFP/Morgan, and Avalon by forward feature addition.
3. Add CatBoost as the MapLight-style learner comparison if unavailable.
4. Add nested permutation-based descriptor selection; never rank features on validation or test rows.
5. Compare RDKit with PaDEL or Mordred 2D descriptors.
6. Compare 2D-only with 3D-only and 2D+3D descriptors using deterministic conformer provenance.
7. Compare the best single tabular model with a validation-only weighted ensemble.
8. Test Chemprop-RDKit or another pretrained expert only as a separate evidence source.

CaliciBoost's leaderboard result motivates steps 4–6, but does not waive nested selection. MapLight and the 2025 feature benchmark motivate steps 2–3. The repository's completed Uni-Mol2 studies motivate selective, not universal, 3D acquisition.

In ADMET-EvO v2.4, steps 1–4 now have repository-internal evidence. The fresh-seed confirmation retained `morgan + avalon + erg + rdkit2d` LightGBM as the validation control; MapLight-style CatBoost, full-count features, Avalon forward addition, and nested RDKit-only selection did not clear the preregistered 0.01-MAE gate. Do not rerun them unchanged. Continue with a genuinely richer descriptor source or a new selector mechanism, and keep the current LightGBM recipe as the paired control.

## Required deliverable

End optimization work with:

- a concise evidence matrix with linked sources and reported data;
- the code/config changes made;
- an ablation matrix with baseline, controls, and acceptance thresholds;
- validation results clearly separated from official test and external reports;
- protocol and reproducibility audit results;
- a decision: `promote`, `confirm`, `reject`, or `blocked`, with the next bounded experiment.

Do not claim improvement from architecture plausibility, a public rank, an in-sample blend, or a single favorable seed.
