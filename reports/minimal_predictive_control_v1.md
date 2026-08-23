# TDC ADMET-22 minimal three-arm predictive control

- Status: `complete`
- Logical charged fits: 154/154
- Physical unique fits: 113（reuse 26.6%）
- Discovery seed: 11；held-out evaluation seed: 12；test loaded: false

| Comparison | Mean normalized delta | 95% task-bootstrap CI | W/T/L |
|---|---:|---:|---:|
| streamlined_agent − expanded_automl | -0.0098 | [-0.0277, +0.0065] | 5/7/10 |
| streamlined_agent − strong_fixed | +0.0196 | [-0.0183, +0.0670] | 9/2/11 |
| expanded_automl − strong_fixed | +0.0294 | [-0.0004, +0.0730] | 5/14/3 |

Agent vs expanded AutoML noninferiority margin = -0.010; gate = FAIL.

## Per-task selections and evaluation

| Task | Metric | Agent selection | AutoML selection | Agent | AutoML | Strong fixed |
|---|---|---|---|---:|---:|---:|
| Caco2_Wang | mae | treatment | lightgbm | 0.34871 | 0.34525 | 0.34525 |
| HIA_Hou | auroc | treatment | extra_trees | 0.98747 | 0.98246 | 0.95990 |
| Pgp_Broccatelli | auroc | treatment | extra_trees | 0.92738 | 0.92738 | 0.92255 |
| Bioavailability_Ma | auroc | control | extra_trees | 0.73203 | 0.70065 | 0.70458 |
| Lipophilicity_AstraZeneca | mae | treatment | lightgbm | 0.55956 | 0.51499 | 0.51499 |
| Solubility_AqSolDB | mae | control | lightgbm | 0.77973 | 0.77973 | 0.77973 |
| BBB_Martins | auroc | treatment | extra_trees | 0.89908 | 0.90760 | 0.91151 |
| PPBR_AZ | mae | control | lightgbm | 12.55965 | 11.94091 | 11.94091 |
| VDss_Lombardo | spearman | treatment | extra_trees | 0.63179 | 0.63179 | 0.28621 |
| CYP2C9_Veith | auprc | treatment | lightgbm | 0.79462 | 0.84792 | 0.84792 |
| CYP2D6_Veith | auprc | control | lightgbm | 0.69358 | 0.72010 | 0.72010 |
| CYP3A4_Veith | auprc | treatment | lightgbm | 0.85947 | 0.88266 | 0.88266 |
| CYP2C9_Substrate_CarbonMangels | auprc | treatment | lightgbm | 0.36450 | 0.31435 | 0.31435 |
| CYP2D6_Substrate_CarbonMangels | auprc | control | lightgbm | 0.82416 | 0.74637 | 0.74637 |
| CYP3A4_Substrate_CarbonMangels | auroc | treatment | lightgbm | 0.72190 | 0.70476 | 0.70476 |
| Half_Life_Obach | spearman | treatment | extra_trees | 0.27364 | 0.27364 | -0.00588 |
| Clearance_Hepatocyte_AZ | spearman | control | lightgbm | 0.43833 | 0.46435 | 0.46435 |
| Clearance_Microsome_AZ | spearman | control | lightgbm | 0.39456 | 0.53336 | 0.53336 |
| LD50_Zhu | mae | control | lightgbm | 0.46802 | 0.46802 | 0.46802 |
| hERG | auroc | treatment | extra_trees | 0.88074 | 0.88074 | 0.86002 |
| AMES | auroc | treatment | extra_trees | 0.86591 | 0.86591 | 0.88380 |
| DILI | auroc | control | lightgbm | 0.79281 | 0.83932 | 0.83932 |

Fresh validation seeds 11->12 under a frozen protocol. This is predictive-control evidence, not frozen TDC test superiority or an application leaderboard update.
