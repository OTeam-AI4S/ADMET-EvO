# ADMET-EvO architecture and evaluation protocol

## Objective

ADMET-EvO searches for endpoint-specific molecular property models across the 22 datasets in the TDC ADMET benchmark group while keeping model selection isolated from the official test split.

## Components

1. **Task profiler** computes training/validation-only dataset statistics, label balance, invalid-SMILES counts, scaffold diversity, and validation scaffold shift.
2. **Experiment database** records candidate specifications, validation scores, timing, and reusable cross-task recipes.
3. **Planner** proposes bounded candidates from the registered models, representations, hyperparameters, and compute budgets. It can be rule based or use an optional structured-output language-model planner.
4. **Protocol guard** validates every proposal and blocks arbitrary code, unregistered tools, unsupported hyperparameters, and test access during search.
5. **Executor** featurizes molecules, trains candidates, and produces validation predictions.
6. **Selection layer** promotes the strongest candidates to multiple validation seeds and optionally builds prediction-level ensembles from complementary finalists.
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

## Initial public finalists

The AMES finalist blends Extra Trees and XGBoost over Morgan, Avalon, ErG, and RDKit 2D features. PPBR_AZ blends two LightGBM models using complementary count/topological and binary-fingerprint feature views. Solubility_AqSolDB blends two LightGBM models using Morgan radius-2 and radius-3 views with shared descriptors.

Exact serialized recipes and freeze hashes are provided in `results/top1_finalists.json` and `results/tdc_top1_results.json`.
