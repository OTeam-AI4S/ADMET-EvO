# ADMET optimization evidence base

Use this file to generate hypotheses and cite reported values. Recheck current web content before publication or leaderboard decisions. All leaderboard values below were accessed 13 August 2026.

## Table of contents

1. [Source-quality rules](#source-quality-rules)
2. [TDC and Caco2 benchmark facts](#tdc-and-caco2-benchmark-facts)
3. [CaliciBoost](#caliciboost-strongest-direct-caco2-lead)
4. [MapLight](#maplight-rich-classical-representations-remain-hard-to-beat)
5. [2025 systematic feature benchmark](#2025-systematic-feature-benchmark)
6. [ADMET-AI](#admet-ai-graph-plus-descriptors-and-multitask-transfer)
7. [Uni-Mol2](#uni-mol2-a-selectively-acquired-3d-expert)
8. [Protocol warning](#protocol-warning-and-large-model-ceiling)
9. [Evidence-to-action summary](#evidence-to-action-summary)

## Source-quality rules

- Prefer an official benchmark plus a peer-reviewed paper and author repository.
- Label arXiv-only work as preprint and very recent work as provisional.
- Quote a number only with its dataset, split, metric, run count, and source.
- Do not infer reproducibility from leaderboard rank alone.
- Treat papers that tune features or hyperparameters without a clearly isolated validation layer as inspiration, not as a protocol template.

## TDC and Caco2 benchmark facts

The official [TDC Caco2_Wang leaderboard](https://tdcommons.ai/benchmark/admet_group/01caco2/) lists 906 compounds, regression, MAE, and scaffold split. Lower is better. Its leading entries on the access date were:

| Rank | Method | MAE | Transferable mechanism |
|---:|---|---:|---|
| 1 | CaliciBoost | 0.256 ± 0.006 | descriptor screening, feature selection, AutoML ensemble |
| 2 | XG Boost (Rob Learsch) | 0.274 ± 0.004 | compact boosted-tree baseline |
| 3 | MapLight | 0.276 ± 0.005 | CatBoost with complementary fingerprints and descriptors |
| 4 | BaseBoosting | 0.285 ± 0.005 | learned/engineered base representation plus boosting |
| 5 | MolMapNet-D | 0.287 ± 0.005 | descriptor-map neural representation |

TDC introduced standardized therapeutic datasets, splits, metrics, and benchmark groups in Huang et al., *NeurIPS Datasets and Benchmarks* (2021), [paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/4c56ff4ce4aaf9573aa5dff913df997a-Abstract-round1.html), [arXiv](https://arxiv.org/abs/2102.09548).

## CaliciBoost: strongest direct Caco2 lead

Le et al., “CaliciBoost: Performance-Driven Evaluation of Molecular Representations for Caco-2 Permeability Prediction,” *Journal of Cheminformatics* 17, 184 (2025), [DOI 10.1186/s13321-025-01137-7](https://doi.org/10.1186/s13321-025-01137-7), [arXiv 2506.08059](https://arxiv.org/abs/2506.08059), [code](https://github.com/Calici/CaliciBoost).

Reported design:

- TDC Caco2: 906 compounds; paper describes 728 training and 182 test compounds.
- Eight representations: Morgan, Avalon, ErG, MACCS, RDKit, PaDEL, Mordred, and CDDD.
- AutoGluon `best_quality`, permutation/SHAP feature ranking, top-N screening, and Bayesian optimization.
- PaDEL has 1,875 2D/3D descriptors; Mordred has more than 1,800; CDDD has 512 dimensions.

Selected reported TDC results:

| Representation/configuration | MAE | RMSE | R² | Pearson r |
|---|---:|---:|---:|---:|
| Morgan, all | 0.3007 | 0.3729 | 0.7046 | 0.8410 |
| Morgan, selected | 0.2880 | 0.3620 | 0.7220 | 0.8510 |
| Avalon, all | 0.2941 | 0.3684 | 0.7118 | 0.8437 |
| Avalon, selected | 0.2750 | 0.3450 | 0.7480 | 0.8740 |
| RDKit, all | 0.2741 | not extracted | 0.7432 | 0.8640 |
| RDKit, selected/optimized | 0.2670 | 0.3390 | 0.7560 | 0.8700 |
| Mordred, selected/optimized | 0.2613 | 0.3413 | 0.7533 | 0.8687 |
| PaDEL, selected/optimized (arXiv table) | 0.2560 | 0.3224 | 0.7788 | 0.8839 |

The final journal article reports a development result of MAE 0.2525, RMSE 0.3216, R² 0.7805, and Pearson r 0.8835 for XGBoost over selected PaDEL 2D+3D descriptors; its official five-seed TDC result is 0.2560 ± 0.006. It also reports that three selected 3D descriptors among 73 PaDEL features produced an approximately 15.5% MAE reduction. Its most actionable result is that descriptor-family and nested feature-subset selection deserve explicit ablation. Do not reproduce selection against the official test; fit and choose selectors inside inner scaffold folds.

## MapLight: rich classical representations remain hard to beat

Notwell and Wood, “ADMET property prediction through combinations of molecular fingerprints” (2023 preprint), [arXiv 2310.00174](https://arxiv.org/abs/2310.00174), [code](https://github.com/maplightrx/MapLight-TDC).

Reported design and data:

- CatBoost with ECFP count 1,024 + Avalon count 1,024 + ErG 315 + about 200 molecular properties: 2,563 features.
- Optional supervised-masking GIN embedding adds 300 dimensions: 2,863 total.
- Only CatBoost `random_strength` was materially adjusted; extensive hyperparameter search overfit validation.
- Without GIN, the fixed recipe reported Top-1 on 6/22 and Top-3 on 16/22 TDC endpoints.
- With GIN, it reported Top-1 on 11/22 and Top-3 on 19/22.

Use this as evidence for a strong, cheap 2D control and for representation complementarity. It is not evidence that the GIN feature helps every endpoint.

## 2025 systematic feature benchmark

Kamuntavičius et al., “Benchmarking ML in ADMET predictions: the practical impact of feature representations in ligand-based models,” *Journal of Cheminformatics* 17, 108 (2025), [DOI 10.1186/s13321-025-01041-0](https://doi.org/10.1186/s13321-025-01041-0), [open PDF](https://jcheminf.biomedcentral.com/counter/pdf/10.1186/s13321-025-01041-0.pdf).

Scale and statistical design:

- 1,325 single-feature models and 1,125 two-feature models in the reported screening tables.
- Friedman rank tests followed by Nemenyi post-hoc comparisons.
- CatBoost average ranks: 2.36 for single features and 2.07 for two features.
- SVM: 2.52 and 3.01; LightGBM: 2.79 and 2.52; random forest: 3.29 and 3.13; MPNN: 4.03 and 4.28.
- With two representations, CatBoost was significantly preferred at p < 0.05 under Nemenyi.
- On the paper's Caco2 validation setting (`n=753`), RDKit-based CatBoost achieved R² 0.61 versus MPNN 0.52.

The regression forward-addition path was RDKit descriptors → +ErG → +ECFP4 → +Avalon. The authors report that feature optimization contributed more than subsequent hyperparameter tuning, and that extra features eventually reduced performance. Use forward addition with paired folds; do not concatenate every available representation by default.

## ADMET-AI: graph plus descriptors and multitask transfer

Swanson et al., “ADMET-AI: a machine learning ADMET platform for evaluation of large-scale chemical libraries,” *Bioinformatics* 40(7), btae416 (2024), [DOI 10.1093/bioinformatics/btae416](https://doi.org/10.1093/bioinformatics/btae416), [code](https://github.com/swansonk14/admet_ai), [archived data/models](https://doi.org/10.5281/zenodo.10372418).

Reported design and data:

- Chemprop directed message-passing network augmented with 200 RDKit physicochemical descriptors.
- 41 TDC datasets: 10 regression and 31 classification; 22 are leaderboard tasks.
- Single-task models and two multitask models, one regression and one classification.
- Deployed prediction uses five-model ensembles.
- The paper reported R² > 0.6 on 5/10 regression datasets and AUROC > 0.85 on 20/31 classification datasets.
- One million molecules took 3.1 h on 32 CPU cores plus GPU and 5 h on 8 CPU cores without GPU in the reported setup.

Use Chemprop-RDKit and multitask transfer as distinct ablations. Audit overlap, endpoint units, missing labels, and split ownership before transferring weights or data.

## Uni-Mol2: a selectively acquired 3D expert

Ji et al., “Uni-Mol2: Exploring Molecular Pretraining Model at Scale,” *NeurIPS* (2024), [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/53923bb44655a7defb31c7744c01b62b-Paper-Conference.pdf), [arXiv 2406.14969](https://arxiv.org/abs/2406.14969).

Uni-Mol2 scales a two-track atomic/graph/geometry transformer to 1.1B parameters using 800M conformations. The paper reports average improvements of 27% on QM9 and 14% on COMPAS-1D for the largest model. These are not TDC Caco2 results.

ADMET-EvO's internal six-task study is more relevant for routing: 84M fine-tuning improved Lipophilicity validation MAE from 0.5192 to 0.4066 and hERG AUROC from 0.8867 to 0.8916, but was not the best single expert on four other endpoints. Its matched scale ablation found 164M worse than 84M on DILI (0.8512 vs 0.8635 AUROC), hERG (0.8565 vs 0.8916), and a one-epoch Lipophilicity screen (1.0702 vs 0.8125 MAE) while costing 1.70–2.04× as much. Source: `runs/v23-ablation-study/summary.md`. Treat these as repository-internal validation results, not published claims.

## Protocol warning and large-model ceiling

Zeng et al., “ChemFM as a scaling law guided foundation model pre-trained on informative chemicals,” *Communications Chemistry* (2025), [article](https://www.nature.com/articles/s42004-025-01793-8). The authors report that their adapted 3B model surpassed reviewed leaderboard baselines on 20/22 ADMET datasets and then used 1B searches for Caco2 and HIA. More importantly for this skill, they explicitly excluded leaderboard entries whose public code optimized on test data or improperly combined train and validation data. Preserve this warning even if ChemFM itself is too expensive to integrate.

For a recent, non-peer-reviewed counterweight to blind scaling, Lim et al., “Do Larger Models Really Win in Drug Discovery?” (2026 preprint), [arXiv 2604.26498](https://arxiv.org/abs/2604.26498), reports 167,056 held-out task–molecule evaluations: classical models won 10 primary-metric tasks, GNNs 9, and pretrained molecular sequence models 3. Treat this as provisional evidence for cost-aware model routing.

## Evidence-to-action summary

| Evidence | Highest-value ADMET-EvO action | Required control |
|---|---|---|
| CaliciBoost | PaDEL/Mordred family + nested selector | same learner without selection; 2D-only |
| MapLight | CatBoost over ECFP/Avalon/ErG/RDKit | current LightGBM/ExtraTrees on identical features |
| 2025 feature benchmark | forward feature addition + rank tests | full concatenation and RDKit-only |
| ADMET-AI | Chemprop-RDKit and multitask transfer | Chemprop-only and 2D tree baseline |
| Uni-Mol2 | acquire 84M geometry only after gate | cheap 2D backoff; frozen embedding |
| ChemFM warning | strict split/freeze audit | no leaderboard-driven retuning |
