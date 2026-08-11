# ADMET-EvO: An Evolutionary Research Agent for Protocol-Safe ADMET Model Discovery

**Yilin Wang**

Valhalla Technology

Non-peer-reviewed technical report, version 0.2.1, 11 August 2026.

## Abstract

Reliable absorption, distribution, metabolism, excretion, and toxicity (ADMET) prediction requires models that adapt to heterogeneous endpoints without using the official test set for model selection. We present ADMET-EvO, an evolutionary research agent that profiles each endpoint, proposes bounded combinations of molecular representations and supervised learners, evaluates candidates on training/validation data only, promotes robust finalists across validation seeds, and optionally constructs validation-selected prediction ensembles. A protocol firewall restricts the search space and isolates the TDC test split. Before final evaluation, the complete finalist recipe is serialized and protected by a SHA-256 freeze lock. On an initial three-endpoint submission to the Therapeutics Data Commons (TDC) ADMET benchmark group, ADMET-EvO obtains AUROC 0.874 +/- 0.001 on AMES, MAE 7.302 +/- 0.042 on PPBR_AZ, and MAE 0.739 +/- 0.002 on Solubility_AqSolDB across five independent runs. Code, exact finalist recipes, software versions, and a one-command reproduction workflow are public.

## 1. Motivation

The 22 datasets in the TDC ADMET benchmark group span classification and regression endpoints with different sample sizes, label balances, metrics, and degrees of scaffold shift. A single fixed estimator is unlikely to be optimal across this range. ADMET-EvO treats model construction as a constrained research process: diagnose the endpoint, propose auditable experiments, retain validation evidence, and freeze the selected recipe before official test evaluation.

The primary design objective is protocol safety rather than unrestricted AutoML. Candidate proposals cannot modify the TDC split, metric, evaluator, test labels, or compute limits. They can only select from registered molecular representations, estimators, imbalance strategies, and validated hyperparameter ranges.

## 2. TDC benchmark protocol

We use the official `ADMET_Group` interface in PyTDC 1.1.15. For each endpoint and each seed in `{1, 2, 3, 4, 5}`, TDC supplies a default scaffold-based train/validation split from `train_val` and a fixed test set. Search and ensemble weighting use training/validation data only. After finalist freezing, the selected model is retrained on the concatenation of that seed's training and validation partitions and evaluated on the official test set. TDC's `BenchmarkGroup.evaluate_many` function reports the mean and standard deviation across the five runs.

The archived TDC ADMET group data used for the runs had SHA-256 checksum:

```text
bd0005246cb6f1672333c28ad202112a8bf071ab65d847848715030308faf1f1  admet_group.zip
```

The dataset itself is not redistributed by this repository and is obtained through PyTDC.

## 3. Agent architecture

ADMET-EvO contains eight auditable components:

1. A task profiler measures dataset size, label balance, invalid structures, scaffold diversity, and validation scaffold shift.
2. A bounded planner proposes registered model/representation configurations.
3. A protocol guard rejects arbitrary code, unknown tools, invalid hyperparameters, test-related fields, and budget violations.
4. A deterministic executor computes molecular features, fits the requested estimator, and produces validation predictions.
5. An experiment database stores candidate specifications, validation scores, runtime, and cross-task evidence.
6. A robust selector promotes top candidates across multiple validation seeds.
7. An ensemble selector tests simple convex blends using validation predictions only.
8. A finalizer serializes the complete recipe, computes a freeze hash, and only then accesses the official test evaluator.

Further component details are provided in [`docs/v2_architecture.md`](docs/v2_architecture.md).

## 4. Molecular representations and estimators

The registry includes Morgan radius-2 and radius-3 fingerprints, Morgan count fingerprints, Avalon, ErG, MACCS, RDKit path fingerprints, atom-pair fingerprints, topological torsions, and RDKit 2D descriptors. Candidate estimators include Extra Trees, Random Forest, radial-basis SVM, XGBoost, LightGBM, and CatBoost. The initial submitted finalists use Extra Trees, XGBoost, and LightGBM.

### 4.1 AMES

AMES uses a 0.75/0.25 prediction blend of 400 Extra Trees and 600 XGBoost trees. Both members use Morgan, Avalon, ErG, and RDKit 2D features. The Extra Trees component uses `min_samples_leaf=2`; XGBoost uses depth 6, learning rate 0.05, subsample 0.85, column subsample 0.7, minimum child weight 2.0, L2 regularization 2.0, histogram tree construction, and CUDA execution.

### 4.2 PPBR_AZ

PPBR_AZ uses an equal blend of two 1,000-tree LightGBM regressors. The first combines Morgan-count, atom-pair, MACCS, and RDKit 2D features with 15 leaves and depth 5. The second combines Morgan, Avalon, ErG, and RDKit 2D features with 49 leaves and depth 7. Complete numerical hyperparameters are recorded in [`results/top1_finalists.json`](results/top1_finalists.json).

### 4.3 Solubility_AqSolDB

Solubility_AqSolDB uses a 0.75/0.25 blend of a 600-tree LightGBM baseline and an 800-tree LightGBM model. The first uses Morgan radius-2, Avalon, ErG, and RDKit 2D features. The second substitutes Morgan radius-3 and uses 31 leaves, depth 7, learning rate 0.035, subsample 0.9, column subsample 0.7, and L2 regularization 3.0.

## 5. Test isolation and finalist provenance

During search, the agent receives neither test labels nor per-example test performance. The planner cannot execute shell commands, introduce unregistered code, replace the evaluator, or change a finalist after the freeze lock is written. Each initial endpoint was finalized in a separate run directory. The task-specific finalist hashes are:

| Endpoint | SHA-256 freeze hash |
|---|---|
| AMES | `15882461be0977edbdc4a32c36b3bcbbee6e659e6b5ab25bd1f044ea13e4e5b0` |
| PPBR_AZ | `0f9e2cfac44f5321b80e5cd3c010d23dfcf74ba05cf24c1b9c5d8c9c454029e0` |
| Solubility_AqSolDB | `a8c5944296c3616d0a1ed9d341ffab715ced71f4cc0c7574e65310d943113a86` |

The reproduction script finalizes each endpoint separately so that these hashes can be checked exactly.

## 6. Results

| TDC endpoint | Task | Metric | Mean | Standard deviation | Runs |
|---|---|---|---:|---:|---:|
| AMES | Binary classification | AUROC | 0.874 | 0.001 | 5 |
| PPBR_AZ | Regression | MAE | 7.302 | 0.042 | 5 |
| Solubility_AqSolDB | Regression | MAE | 0.739 | 0.002 | 5 |

These values are the direct output of `BenchmarkGroup.evaluate_many` and are rounded to three decimal places for TDC submission. Estimated leaderboard ranks are not treated as evaluation results and may change as the public leaderboard evolves.

## 7. Compute environment and model size

Each endpoint ran as a Slurm job with eight CPU workers, 64 GB host memory, and one allocated NVIDIA A100-SXM4 GPU with 80 GB memory. AMES used CPU Extra Trees and CUDA XGBoost. PPBR_AZ and Solubility_AqSolDB used CPU LightGBM; the allocated GPU was not used by those LightGBM components. The runtime environment used Linux, Python 3.11.15, PyTDC 1.1.15, RDKit 2026.3.5, scikit-learn 1.9.0, XGBoost 3.2.0, and LightGBM 4.7.0. Exact Python dependencies are pinned in [`requirements-repro.txt`](requirements-repro.txt).

Tree ensembles do not have a fixed dense parameter count comparable to a neural network. We therefore report fitted estimator counts as the reproducible model-size measure: 1,000 trees for AMES, 2,000 for PPBR_AZ, and 1,400 for Solubility_AqSolDB, with a median of 1,400 trees across the submitted endpoints.

## 8. API requirements

**No external API is required to reproduce the reported leaderboard results.** The optional language-model planner is used only to propose experiments during search and never receives test labels or test metrics. Final model construction, feature computation, training, prediction, and TDC evaluation are local deterministic components. The submitted scores can be reproduced directly from the published frozen finalist configurations.

Users who wish to repeat the optional LLM-guided discovery process may configure their own OpenAI-compatible credentials. API keys are not distributed, embedded in the repository, or required by `scripts/reproduce_top1.sh`. A rule-based planner is also available for API-free searches.

## 9. Reproduction

On a Linux machine with a CUDA-compatible NVIDIA GPU for exact AMES reproduction:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-repro.txt
python -m pip install -e . --no-deps
DEVICE=cuda WORKERS=8 ./scripts/reproduce_top1.sh
```

The script downloads the TDC benchmark through PyTDC, finalizes each endpoint for seeds 1 through 5, saves per-seed predictions, and verifies aggregate scores and task-specific freeze hashes against the published manifest. `DEVICE=cpu` is supported for inspection and portability, but the submitted AMES result used CUDA XGBoost.

## 10. Limitations

The initial report covers three of the 22 ADMET endpoints and does not claim universal superiority. The TDC test set is accessible through the public benchmark interface, so protocol compliance depends on auditable test isolation rather than a hidden evaluation server. Results may depend on the specified software versions, accelerator implementation, and dataset release. Tree counts are used as a model-size proxy and are not directly comparable to dense neural-network parameter counts. The optional LLM planner can introduce proposal variability, but it is outside the frozen finalization path used to reproduce the submitted scores.

## 11. Availability

Source code, documentation, tests, exact finalists, and result manifests are available at <https://github.com/YilinWang33/ADMET-EvO> under the MIT License. The initial result release is archived as `v0.2.1`.

## References

1. Huang, K. et al. Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development. *NeurIPS Datasets and Benchmarks* (2021).
2. Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research* 12, 2825-2830 (2011).
3. Chen, T. and Guestrin, C. XGBoost: A Scalable Tree Boosting System. *KDD* (2016).
4. Ke, G. et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS* (2017).
5. RDKit: Open-source cheminformatics. <https://www.rdkit.org/>.
