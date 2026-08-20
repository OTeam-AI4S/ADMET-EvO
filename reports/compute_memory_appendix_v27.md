# v2.7 compute and peak-memory appendix

## Router candidate-stage compute

Across 22 tasks and three saved validation seeds, the cost-aware policy used 0.256 recorded recipe-hours versus 2.488 for score selection (ratio 0.103).

| Task | Action | Cost-aware seconds | Score-policy seconds | Ratio | Peak memory |
|---|---|---:|---:|---:|---|
| AMES | fuse | 59.6 | 59.6 | 1.000 | not recorded |
| BBB_Martins | stop_2d | 3.9 | 83.9 | 0.046 | not recorded |
| Bioavailability_Ma | stop_2d | 3.0 | 12.7 | 0.234 | not recorded |
| CYP2C9_Substrate_CarbonMangels | stop_2d | 5.1 | 5.1 | 1.000 | not recorded |
| CYP2C9_Veith | stop_2d | 6.8 | 25.4 | 0.268 | not recorded |
| CYP2D6_Substrate_CarbonMangels | stop_2d | 1.6 | 3.5 | 0.453 | not recorded |
| CYP2D6_Veith | fuse | 49.6 | 49.6 | 1.000 | not recorded |
| CYP3A4_Substrate_CarbonMangels | fuse | 17.3 | 17.3 | 1.000 | not recorded |
| CYP3A4_Veith | stop_2d | 7.0 | 15.0 | 0.465 | not recorded |
| Caco2_Wang | stop_2d | 5.2 | 5.2 | 1.000 | not recorded |
| Clearance_Hepatocyte_AZ | fuse | 14.7 | 14.7 | 1.000 | not recorded |
| Clearance_Microsome_AZ | fuse | 16.2 | 16.2 | 1.000 | not recorded |
| DILI | stop_2d | 2.2 | 4.7 | 0.473 | not recorded |
| HIA_Hou | stop_2d | 10.4 | 17.2 | 0.607 | not recorded |
| Half_Life_Obach | fuse | 14.2 | 14.2 | 1.000 | not recorded |
| LD50_Zhu | stop_2d | 122.9 | 7957.3 | 0.015 | not recorded |
| Lipophilicity_AstraZeneca | stop_2d | 12.4 | 21.2 | 0.584 | not recorded |
| PPBR_AZ | fuse | 39.6 | 39.6 | 1.000 | not recorded |
| Pgp_Broccatelli | stop_2d | 3.2 | 3.2 | 1.000 | not recorded |
| Solubility_AqSolDB | stop_2d | 15.0 | 81.6 | 0.183 | not recorded |
| VDss_Lombardo | stop_2d | 509.1 | 509.1 | 1.000 | not recorded |
| hERG | stop_2d | 2.2 | 2.2 | 1.000 | not recorded |

## Recorded expert-family wall time

| Scope | Expert | Replicates | Mean seconds | Total hours | Peak memory |
|---|---|---:|---:|---:|---|
| DILI | 2d_lightgbm | 5 | 2.5 | 0.003 | not recorded |
| DILI | 2d_extra_trees | 5 | 5.1 | 0.007 | not recorded |
| DILI | finetuned_unimol2 | 5 | 360.8 | 0.501 | not recorded |
| hERG | 2d_lightgbm | 5 | 3.2 | 0.004 | not recorded |
| hERG | 2d_extra_trees | 5 | 5.2 | 0.007 | not recorded |
| hERG | finetuned_unimol2 | 5 | 384.8 | 0.534 | not recorded |
| Lipophilicity_AstraZeneca | finetuned_unimol2_84m | 5 | 3773.4 | 5.241 | not recorded |
| Lipophilicity_AstraZeneca | finetuned_unimol2_164m | 5 | 7662.8 | 10.643 | not recorded |
| Bioavailability_Ma + CYP2C9_Substrate_CarbonMangels | MapLight and MapLight+GNN confirmation campaign | 20 | 16.0 | 0.089 | not recorded |

## Measurement boundary

Wall times come from different experiment phases and environments and are not hardware-normalized. Only matched within-protocol ratios are interpreted.

Historical runs did not record process or CUDA peak memory. Missing values are reported explicitly; absolute peak-memory claims require a fresh instrumented rerun.

Peak-memory coverage: 0/31 records measured.
