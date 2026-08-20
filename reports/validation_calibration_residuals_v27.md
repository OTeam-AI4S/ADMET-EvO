# v2.7 validation calibration and chemical-space residual audit

one saved default scaffold validation split per task; seed 1 for v2.1/v2.4 lineages and preregistered fresh seed 6 for v2.7 MapLight+GNN. Official test data and official results were not loaded.

## Classification calibration

| Task | Seed | Brier | Brier skill | adaptive ECE | mean p - prevalence | low-high similarity Brier |
|---|---:|---:|---:|---:|---:|---:|
| AMES | 1 | 0.1233 | 0.478 | 0.0998 | 0.0201 | 0.0742 |
| BBB_Martins | 1 | 0.0847 | 0.593 | 0.0644 | 0.0200 | 0.0941 |
| Bioavailability_Ma | 6 | 0.1135 | 0.299 | 0.1110 | 0.0124 | -0.0630 |
| CYP2C9_Substrate_CarbonMangels | 6 | 0.1234 | 0.101 | 0.0567 | -0.0133 | 0.0167 |
| CYP2C9_Veith | 1 | 0.1240 | 0.444 | 0.0514 | -0.0129 | 0.0679 |
| CYP2D6_Substrate_CarbonMangels | 1 | 0.1564 | 0.139 | 0.1331 | 0.0461 | 0.0645 |
| CYP2D6_Veith | 1 | 0.1077 | 0.324 | 0.0485 | 0.0388 | 0.0383 |
| CYP3A4_Substrate_CarbonMangels | 1 | 0.2221 | 0.077 | 0.1909 | -0.0672 | 0.0779 |
| CYP3A4_Veith | 1 | 0.1010 | 0.536 | 0.0215 | 0.0005 | -0.0693 |
| DILI | 1 | 0.1485 | 0.084 | 0.1750 | 0.1518 | 0.0363 |
| HIA_Hou | 1 | 0.0305 | 0.712 | 0.0598 | -0.0297 | 0.0481 |
| Pgp_Broccatelli | 1 | 0.0721 | 0.711 | 0.0905 | 0.0407 | 0.0997 |
| hERG | 1 | 0.1180 | 0.357 | 0.1506 | -0.0881 | 0.1207 |

## Regression residuals

| Task | Metric | Error unit | Mean absolute error | Bias | p90 error | low-high similarity error | singleton-repeat scaffold error |
|---|---|---|---:|---:|---:|---:|---:|
| Caco2_Wang | mae | native_target | 0.3224 | -0.0135 | 0.6474 | -0.0127 | -0.0564 |
| Clearance_Hepatocyte_AZ | spearman | percentile_rank | 0.1928 | -0.0000 | 0.4254 | 0.0803 | -0.0040 |
| Clearance_Microsome_AZ | spearman | percentile_rank | 0.2087 | -0.0000 | 0.4595 | 0.0962 | 0.0039 |
| Half_Life_Obach | spearman | percentile_rank | 0.2292 | 0.0000 | 0.4970 | 0.0215 | -0.0307 |
| LD50_Zhu | mae | native_target | 0.4794 | 0.0077 | 1.0037 | 0.0837 | 0.0425 |
| Lipophilicity_AstraZeneca | mae | native_target | 0.3952 | -0.0472 | 0.8760 | 0.2723 | 0.0672 |
| PPBR_AZ | mae | native_target | 10.9078 | 3.3733 | 27.6417 | 9.4631 | 0.3713 |
| Solubility_AqSolDB | mae | native_target | 0.8004 | 0.1169 | 1.8752 | -0.0017 | 0.0344 |
| VDss_Lombardo | spearman | percentile_rank | 0.1444 | -0.0000 | 0.3398 | 0.0073 | 0.0495 |

## Aggregate observations

- Mean Brier across classification tasks: 0.1173.
- Mean adaptive ECE across classification tasks: 0.0964.
- Low-similarity molecules had worse Brier in 11 classification tasks and better Brier in 2.
- Low-similarity molecules had worse residual error in 7 regression tasks and better error in 2.

Finalists were selected using development validation evidence, so this is a post-selection diagnostic, not an unbiased performance estimate. Spearman residuals are computed in percentile-rank space.
