# TDC-22 Discovery-only Hardness 基线

- 完整性：22/22
- 平均 Hardness：0.313
- 中位 Hardness：0.312
- Test loaded：false

## 动作路由分布

| Action | Tasks |
|---|---:|
| `data-leakage-safe` | 1 |
| `feature-domain-shift` | 17 |
| `model-imbalance-aware` | 4 |

## Endpoint 明细

| Task | Type | Metric | N train | Hardness | Domain shift | Confidence | Selected action | Conflicts | Train-valid overlap |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| CYP2D6_Substrate_CarbonMangels | classification | auprc | 465 | 0.383 | 0.801 | 0.335 | `model-imbalance-aware` | 0 | 0 |
| Bioavailability_Ma | classification | auroc | 448 | 0.377 | 0.852 | 0.320 | `model-imbalance-aware` | 0 | 0 |
| HIA_Hou | classification | auroc | 403 | 0.373 | 0.822 | 0.290 | `model-imbalance-aware` | 0 | 0 |
| CYP2C9_Substrate_CarbonMangels | classification | auprc | 467 | 0.360 | 0.743 | 0.335 | `model-imbalance-aware` | 0 | 0 |
| Clearance_Hepatocyte_AZ | regression | spearman | 848 | 0.340 | 0.736 | 0.610 | `feature-domain-shift` | 39 | 0 |
| Solubility_AqSolDB | regression | mae | 6986 | 0.338 | 0.778 | 1.000 | `feature-domain-shift` | 73 | 13 |
| LD50_Zhu | regression | mae | 5168 | 0.337 | 0.816 | 1.000 | `feature-domain-shift` | 3 | 0 |
| DILI | classification | auroc | 325 | 0.335 | 0.857 | 0.270 | `feature-domain-shift` | 0 | 0 |
| CYP3A4_Substrate_CarbonMangels | classification | auroc | 468 | 0.322 | 0.772 | 0.335 | `feature-domain-shift` | 0 | 0 |
| Caco2_Wang | regression | mae | 637 | 0.318 | 0.741 | 0.455 | `feature-domain-shift` | 1 | 0 |
| Clearance_Microsome_AZ | regression | spearman | 770 | 0.316 | 0.714 | 0.555 | `feature-domain-shift` | 0 | 0 |
| CYP2D6_Veith | classification | auprc | 9191 | 0.308 | 0.725 | 1.000 | `feature-domain-shift` | 3 | 1 |
| BBB_Martins | classification | auroc | 1421 | 0.302 | 0.681 | 1.000 | `feature-domain-shift` | 8 | 1 |
| PPBR_AZ | regression | mae | 1952 | 0.302 | 0.702 | 1.000 | `feature-domain-shift` | 47 | 0 |
| hERG | classification | auroc | 457 | 0.300 | 0.668 | 0.330 | `data-leakage-safe` | 0 | 1 |
| Half_Life_Obach | regression | spearman | 465 | 0.300 | 0.769 | 0.335 | `feature-domain-shift` | 0 | 0 |
| VDss_Lombardo | regression | spearman | 791 | 0.287 | 0.789 | 0.565 | `feature-domain-shift` | 0 | 0 |
| CYP3A4_Veith | classification | auprc | 8628 | 0.274 | 0.790 | 1.000 | `feature-domain-shift` | 1 | 0 |
| Lipophilicity_AstraZeneca | regression | mae | 2940 | 0.266 | 0.640 | 1.000 | `feature-domain-shift` | 0 | 0 |
| CYP2C9_Veith | classification | auprc | 8463 | 0.263 | 0.702 | 1.000 | `feature-domain-shift` | 5 | 3 |
| AMES | classification | auroc | 5093 | 0.253 | 0.713 | 1.000 | `feature-domain-shift` | 0 | 0 |
| Pgp_Broccatelli | classification | auroc | 851 | 0.237 | 0.650 | 0.610 | `feature-domain-shift` | 0 | 0 |

## 维度最高的 endpoints

- **data_quality**：mean=0.006, confidence=1.000; Clearance_Hepatocyte_AZ (0.040), PPBR_AZ (0.033), Solubility_AqSolDB (0.023), BBB_Martins (0.023), CYP2C9_Veith (0.002)
- **domain_shift**：mean=0.748, confidence=0.652; DILI (0.857), Bioavailability_Ma (0.852), HIA_Hou (0.822), LD50_Zhu (0.816), CYP2D6_Substrate_CarbonMangels (0.801)
- **imbalance**：mean=0.208, confidence=0.591; HIA_Hou (0.782), CYP2C9_Substrate_CarbonMangels (0.636), CYP2D6_Veith (0.606), Bioavailability_Ma (0.571), BBB_Martins (0.520)
- **local_roughness**：mean=0.216, confidence=0.652; CYP2D6_Substrate_CarbonMangels (0.344), CYP3A4_Substrate_CarbonMangels (0.337), Solubility_AqSolDB (0.313), Clearance_Hepatocyte_AZ (0.308), LD50_Zhu (0.271)
- **scarcity**：mean=0.196, confidence=1.000; DILI (0.321), HIA_Hou (0.295), Half_Life_Obach (0.289), hERG (0.284), CYP2D6_Substrate_CarbonMangels (0.284)
- **split_leakage_risk**：mean=0.002, confidence=1.000; hERG (0.015), Solubility_AqSolDB (0.013), BBB_Martins (0.005), CYP2C9_Veith (0.002), CYP2D6_Veith (0.001)
