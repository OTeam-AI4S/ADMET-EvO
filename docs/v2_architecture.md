# ADMET-EvO architecture and evaluation protocol

## Objective

ADMET-EvO searches for endpoint-specific molecular property models across the 22 datasets in the TDC ADMET benchmark group while keeping model selection isolated from the official test split.

## Components

1. **Task profiler** computes training/validation-only dataset statistics, label balance, invalid-SMILES counts, scaffold diversity, and validation scaffold shift.
2. **Experiment database** records candidate specifications, validation scores, timing, and reusable cross-task recipes.
3. **Planner** proposes bounded candidates from the registered models, representations, hyperparameters, and compute budgets. It can be rule based or use an optional structured-output language-model planner.
4. **Protocol guard** validates every proposal and blocks arbitrary code, unregistered tools, unsupported hyperparameters, and test access during search.
5. **Executor** featurizes molecules, trains candidates, and produces validation predictions. Optional Uni-Mol2 support runs the frozen official 84M encoder over SMILES-derived conformers, caches molecular CLS embeddings, and can concatenate them with registered 2D features without changing split ownership.
6. **Selection layer** promotes model-diverse candidates to multiple validation seeds and optionally builds prediction-level ensembles from complementary finalists. Ranking metrics can use fixed equal-weight percentile-rank aggregation so that differently calibrated model families remain comparable.
7. **Freeze lock** serializes the selected finalist and computes its SHA-256 digest before test evaluation.
8. **Finalizer** trains and evaluates the frozen recipe for at least five independent TDC seeds and records the same digest in every audit record.

## Data flow

```text
TDC train_val
    -> task profile
    -> bounded candidate proposals
    -> validation evaluation
    -> robust multi-seed selection
    -> optional validation-selected ensemble
    -> serialized finalist + SHA-256 freeze lock
    -> TDC test evaluation across seeds 1..5
    -> mean and standard deviation
```

The test split is never used to rank, refine, blend, or replace candidates. Test predictions are exposed only to the TDC evaluator during finalization.

## Experimental v2.3 evidence-routing extension

The v2.3 study replaces the assumption that every endpoint should use the same
multimodal stack with sequential, task-conditional evidence acquisition:

1. The task profiler diagnoses metric, endpoint type, sample count, imbalance,
   target skew, scaffold shift, and molecular geometry statistics.
2. A low-cost stage establishes 2D topology and descriptor experts plus
   metric-aligned target/loss variants.
3. The planner requests expensive fine-tuned 3D evidence only when comparable
   cross-scaffold experiments predict positive value relative to compute cost.
4. The experiment database acts as an evidence ledger, retaining OOF
   predictions, outer-seed gains, variance, complementarity, and failed
   acquisitions instead of storing only winners.
5. A reliability controller scores recipes with lower-confidence-bound utility
   and can select a single expert, a probability/rank fusion, or a conservative
   backoff. Fusion is an action to justify, not a mandatory architecture.
6. Self-evolution updates task-to-recipe priors from train/validation evidence
   only. Test scores are excluded from memory updates and planning context.

In a 22-task held-out-seed replay, the cost-aware controller used a global
cost penalty of 0.005. Relative to LCB-only selection it retained 19.1% of the
aggregate selected-recipe time proxy with no detected score difference
(mean oriented gain +0.0003; Wilcoxon p=0.953). Leave-one-task-out penalty
calibration gave a similar 18.5% cost ratio and p=0.721. These are validation
selection results, not official-test performance claims.

The initial six-task ablation supports this design: fine-tuned Uni-Mol2 helped
Lipophilicity strongly and hERG slightly but was not the best single expert on
the other four representative endpoints; unconditional fusion also hurt some
tasks. A paired five-seed Lipophilicity scale study retained 84M over 164M
(0.4036 versus 0.4129 MAE) while using about half the GPU time. On the highly
skewed Half-Life task, transformed-target experts were
weak alone but their complementary ranks improved leave-one-scaffold-seed-out
selection over the previous fixed rank ensemble. Full results and the freeze
gate are recorded in `runs/v23-ablation-study/summary.md`.

## Initial public finalists

The AMES finalist blends Extra Trees and XGBoost over Morgan, Avalon, ErG, and RDKit 2D features. PPBR_AZ blends two LightGBM models using complementary count/topological and binary-fingerprint feature views. Solubility_AqSolDB blends two LightGBM models using Morgan radius-2 and radius-3 views with shared descriptors.

Exact serialized recipes and freeze hashes are provided in `results/top1_finalists.json` and `results/tdc_top1_results.json`.
