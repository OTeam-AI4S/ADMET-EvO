# TDC submission: ADMET-EvO v0.2.1

## Submitter

- Model: ADMET-EvO
- Contact: Yilin Wang
- Affiliation: Valhalla Technology
- Code: https://github.com/OTeam-AI4S/ADMET-EvO
- Technical report: https://github.com/OTeam-AI4S/ADMET-EvO/blob/main/MANUSCRIPT.md

## Short method description

ADMET-EvO is an evolutionary research agent for protocol-safe ADMET model discovery. For each endpoint, it profiles the TDC training/validation data, proposes bounded combinations of molecular fingerprints, RDKit descriptors, and classical or gradient-boosted learners, evaluates candidates only on validation splits, promotes finalists across multiple validation seeds, and optionally constructs validation-selected prediction ensembles. The selected recipe is serialized and SHA-256 locked before any official test evaluation. The reported values are the mean and standard deviation from five independent TDC runs using seeds 1, 2, 3, 4, and 5.

## Results submitted in the first batch

| TDC benchmark | Task | Metric | Mean | Standard deviation |
|---|---|---|---:|---:|
| AMES | Binary classification | AUROC | 0.874 | 0.001 |
| PPBR_AZ | Regression | MAE | 7.302 | 0.042 |
| Solubility_AqSolDB | Regression | MAE | 0.739 | 0.002 |

## Endpoint models

- **AMES:** a validation-selected 0.75/0.25 blend of Extra Trees and XGBoost using Morgan, Avalon, ErG, and RDKit 2D features.
- **PPBR_AZ:** a 0.50/0.50 blend of two LightGBM models using complementary Morgan-count/atom-pair/MACCS/RDKit 2D and Morgan/Avalon/ErG/RDKit 2D feature views.
- **Solubility_AqSolDB:** a 0.75/0.25 blend of two LightGBM models using complementary Morgan radius-2 and radius-3 views with Avalon, ErG, and RDKit 2D features.

## Parameters and hardware

The submission consists of tree ensembles rather than a single neural network, so a dense neural-network parameter count is not applicable. AMES contains 1,000 fitted trees, PPBR_AZ contains 2,000, and Solubility_AqSolDB contains 1,400; the median submitted model size is 1,400 trees. Individual estimator counts and hyperparameters are recorded in `results/top1_finalists.json`.

Experiments ran as Slurm jobs on Linux with Python 3.11, eight CPU workers, 64 GB host memory, and one allocated NVIDIA A100-SXM4 80 GB GPU. AMES used CPU Extra Trees and CUDA XGBoost. PPBR_AZ and Solubility_AqSolDB used CPU LightGBM; the allocated GPU was not used by those LightGBM components.

## Evaluation and reproducibility

- TDC ADMET BenchmarkGroup data and official endpoint metrics.
- Scaffold-based TDC train/validation splits.
- Five independent final runs: seeds 1, 2, 3, 4, and 5.
- Validation-only candidate selection and ensemble weighting.
- Frozen finalist configuration before test evaluation.
- Per-endpoint SHA-256 freeze hashes included in the machine-readable results.
- Exact package versions in `requirements-repro.txt`.
- One-command reproduction and verification in `scripts/reproduce_top1.sh`.

## API requirements

No external API is required to reproduce the submitted results. The optional language-model planner is limited to proposing experiments during validation-only search and never receives test labels or test metrics. Frozen finalist training, prediction, and TDC evaluation run locally without API calls. Users who wish to repeat optional LLM-guided search must provide their own credentials; API keys are never included in this repository.
