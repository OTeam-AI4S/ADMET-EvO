# Supplementary Information: ADMET-EvO v2.7

This document is generated from content-addressed evidence. It reports
negative results and missing measurements as well as positive results.

## S1. Complete conservative 22-task scorecard

| Task | Metric | Frozen finalist | Mean +/- SD | Estimated rank | Band |
|---|---|---|---:|---:|---|
| AMES | auroc | ensemble(extra_trees+xgboost) | 0.874 +/- 0.001 | 1 | Top-5 |
| BBB_Martins | auroc | ensemble(random_forest+random_forest) | 0.921 +/- 0.002 | 2 | Top-5 |
| Bioavailability_Ma | auroc | recipe(maplight_gnn) | 0.731 +/- 0.007 | 21 | below Top-10 |
| CYP2C9_Substrate_CarbonMangels | auprc | recipe(maplight_gnn) | 0.434 +/- 0.009 | 5 | Top-5 |
| CYP2C9_Veith | auprc | ensemble(lightgbm+xgboost) | 0.799 +/- 0.001 | 5 | Top-5 |
| CYP2D6_Substrate_CarbonMangels | auprc | ensemble(lightgbm+lightgbm) | 0.721 +/- 0.004 | 3 | Top-5 |
| CYP2D6_Veith | auprc | ensemble(lightgbm+lightgbm) | 0.721 +/- 0.001 | 4 | Top-5 |
| CYP3A4_Substrate_CarbonMangels | auroc | ensemble(svm_rbf+svm_rbf) | 0.633 +/- 0.000 | 9 | Top-10 |
| CYP3A4_Veith | auprc | ensemble(xgboost+xgboost) | 0.885 +/- 0.001 | 4 | Top-5 |
| Caco2_Wang | mae | lightgbm | 0.283 +/- 0.007 | 4 | Top-5 |
| Clearance_Hepatocyte_AZ | spearman | ensemble(xgboost+lightgbm) | 0.456 +/- 0.022 | 4 | Top-5 |
| Clearance_Microsome_AZ | spearman | ensemble(xgboost+xgboost) | 0.591 +/- 0.007 | 8 | Top-10 |
| DILI | auroc | recipe(raw(2d_lightgbm@0.33+2d_extra_trees@0.33+finetuned_unimol2@0.33)) | 0.920 +/- 0.004 | 3 | Top-5 |
| HIA_Hou | auroc | ensemble(extra_trees+extra_trees) | 0.990 +/- 0.001 | 2 | Top-5 |
| Half_Life_Obach | spearman | recipe(rank(finetuned_unimol2@0.33+fixed_rank_top3@0.33+rank_target_xgboost@0.33)) | 0.590 +/- 0.030 | 1 | Top-5 |
| LD50_Zhu | mae | ensemble(random_forest+random_forest) | 0.591 +/- 0.001 | 5 | Top-5 |
| Lipophilicity_AstraZeneca | mae | recipe(finetuned_unimol2) | 0.392 +/- 0.006 | 1 | Top-5 |
| PPBR_AZ | mae | ensemble(lightgbm+lightgbm) | 7.302 +/- 0.042 | 1 | Top-5 |
| Pgp_Broccatelli | auroc | random_forest | 0.912 +/- 0.003 | 8 | Top-10 |
| Solubility_AqSolDB | mae | ensemble(lightgbm+lightgbm) | 0.739 +/- 0.002 | 1 | Top-5 |
| VDss_Lombardo | spearman | extra_trees | 0.673 +/- 0.001 | 3 | Top-5 |
| hERG | auroc | recipe(raw(2d_lightgbm@0.50+finetuned_unimol2@0.50)) | 0.860 +/- 0.009 | 5 | Top-5 |

## S2. Preregistered router controls

| Contrast | Mean gain | 95% bootstrap CI | Improved/tied/worse | Wilcoxon p | Gate |
|---|---:|---:|---:|---:|---|
| Full-outcome minus winner-only memory | -0.00031 | [-0.00276, 0.00240] | 4/12/6 | 0.7213 | failed |
| Agent minus matched-cost random | 0.00083 | [-0.00380, 0.00542] | 7/5/10 | 0.8684 | failed |

Full-outcome/winner-only aggregate cost ratio: 0.1848. Maximum matched-random relative cost error: 3.958e-16.

## S3. Fresh-seed metric-risk guard

The guard covered 22 tasks on seeds [4, 5] with identity overlap 0. Its claim gate was False; guarded minus unguarded mean gain was 0.00000. The experiment did not change selection or the official freeze.

## S4. Primary paired model effects

| Task and contrast | n | Mean oriented gain | 95% paired-t CI | Corrected dz | Sign-flip p |
|---|---:|---:|---:|---:|---:|
| Bioavailability_Ma:maplight_gnn_minus_maplight | 5 | 0.02353 | [0.00014, 0.04692] | 0.999 | 0.1250 |
| CYP2C9_Substrate_CarbonMangels:maplight_gnn_minus_maplight | 5 | 0.03422 | [-0.04240, 0.11085] | 0.444 | 0.1875 |
| Lipophilicity_AstraZeneca:unimol2_164m_minus_84m | 5 | -0.00934 | [-0.02057, 0.00189] | -0.826 | 0.1250 |

## S5. Calibration and chemical-space diagnostics

| Task | Type | n train | n valid | Median nearest-train similarity | Low-minus-high error | Identity overlap | Scaffold overlap |
|---|---|---:|---:|---:|---:|---:|---:|
| AMES | classification_calibration | 5093 | 728 | 0.472 | 0.0742 | 0 | 0 |
| BBB_Martins | classification_calibration | 1421 | 203 | 0.480 | 0.0941 | 0 | 0 |
| Bioavailability_Ma | classification_calibration | 448 | 64 | 0.365 | -0.0630 | 0 | 0 |
| CYP2C9_Substrate_CarbonMangels | classification_calibration | 467 | 67 | 0.348 | 0.0167 | 0 | 0 |
| CYP2C9_Veith | classification_calibration | 8463 | 1210 | 0.500 | 0.0679 | 0 | 0 |
| CYP2D6_Substrate_CarbonMangels | classification_calibration | 465 | 67 | 0.325 | 0.0645 | 0 | 0 |
| CYP2D6_Veith | classification_calibration | 9191 | 1313 | 0.484 | 0.0383 | 0 | 0 |
| CYP3A4_Substrate_CarbonMangels | classification_calibration | 468 | 67 | 0.361 | 0.0779 | 0 | 0 |
| CYP3A4_Veith | classification_calibration | 8628 | 1233 | 0.422 | -0.0693 | 0 | 0 |
| Caco2_Wang | regression_residual | 637 | 91 | 0.364 | -0.0127 | 0 | 0 |
| Clearance_Hepatocyte_AZ | regression_residual | 848 | 122 | 0.392 | 0.0803 | 0 | 0 |
| Clearance_Microsome_AZ | regression_residual | 770 | 111 | 0.426 | 0.0962 | 0 | 0 |
| DILI | classification_calibration | 325 | 54 | 0.308 | 0.0363 | 0 | 0 |
| HIA_Hou | classification_calibration | 403 | 58 | 0.305 | 0.0481 | 0 | 0 |
| Half_Life_Obach | regression_residual | 465 | 67 | 0.346 | 0.0215 | 0 | 0 |
| LD50_Zhu | regression_residual | 5168 | 739 | 0.388 | 0.0837 | 0 | 0 |
| Lipophilicity_AstraZeneca | regression_residual | 2940 | 420 | 0.577 | 0.2723 | 0 | 0 |
| PPBR_AZ | regression_residual | 1952 | 279 | 0.473 | 9.4631 | 0 | 0 |
| Pgp_Broccatelli | classification_calibration | 851 | 122 | 0.531 | 0.0997 | 0 | 0 |
| Solubility_AqSolDB | regression_residual | 6986 | 999 | 0.432 | -0.0017 | 0 | 0 |
| VDss_Lombardo | regression_residual | 791 | 113 | 0.328 | 0.0073 | 0 | 0 |
| hERG | classification_calibration | 457 | 66 | 0.473 | 0.1207 | 0 | 0 |

## S6. Weak-task rescue ledger

| Task | Method | Baseline | Candidate | Gain | Decision |
|---|---|---:|---:|---:|---|
| Bioavailability_Ma | exact-overlap-excluded cross-endpoint teacher features | 0.6898 | 0.7175 | 0.0277 | retain_for_ablation_not_freeze |
| CYP2C9_Substrate_CarbonMangels | exact-overlap-excluded cross-endpoint teacher features | 0.4801 | 0.4738 | -0.0064 | reject |
| CYP2C9_Substrate_CarbonMangels | fresh-seed 6-10 fixed heterogeneous rank ensemble | 0.3695 | 0.3764 | 0.0069 | reject |
| CYP2C9_Substrate_CarbonMangels | balanced bagged Extra Trees | 0.4801 | 0.4801 | 0.0000 | reject |
| CYP2C9_Substrate_CarbonMangels | MiniMol 512-d fingerprint with fixed official MLP head | 0.4801 | 0.4484 | -0.0317 | reject |
| CYP2C9_Substrate_CarbonMangels | equal-rank MiniMol and LightGBM fusion | 0.4801 | 0.4676 | -0.0125 | reject |
| CYP2C9_Substrate_CarbonMangels | MiniMol inner-scaffold five-head ensemble | 0.4801 | 0.4628 | -0.0174 | reject |
| CYP2C9_Substrate_CarbonMangels | balanced eight-member MapLight+MiniMol CatBoost | 0.4801 | 0.4606 | -0.0195 | reject |

## S7. Compute-accounting boundary

Candidate-stage selected time was 0.256 h versus 2.488 h for score selection (ratio 0.1028). Historical peak-memory coverage was 0/31; no absolute memory claim is made.

## S8. Reproducibility firewall

The full finalist freeze SHA-256 is `693db6936485d59791eacfd9005d18efca00056274c11c1cc80efd962058c84b`. Candidate selection used train/validation evidence only. Fresh validation audits report zero canonical-molecule identity overlap. Official results cannot mutate a finalist within the same freeze lineage.
