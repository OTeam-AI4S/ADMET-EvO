# Search Redundancy Replay

- Completed tasks: 22
- Finalists captured within 8 actions: 17/22
- Practically equivalent within 8 actions: 18/22
- Finalists captured within 4 promoted configurations: 12/22
- Ensemble finalists: 17
- Material ensemble gains (> 0.005): 7
- Test loaded: false

| Task | Actions | Promotion | Finalist | Min action prefix | Min promotion width | Ensemble gain |
|---|---:|---:|---|---:|---:|---:|
| AMES | 12 | 6 | ensemble | 4 | 3 | 0.008370 |
| BBB_Martins | 12 | 6 | ensemble | 2 | 4 | 0.000834 |
| Bioavailability_Ma | 12 | 6 | ensemble | 11 | 3 | 0.004846 |
| CYP2C9_Substrate_CarbonMangels | 12 | 6 | lightgbm | 2 | 2 | 0.000000 |
| CYP2C9_Veith | 12 | 6 | lightgbm | 9 | 6 | 0.000000 |
| CYP2D6_Substrate_CarbonMangels | 12 | 6 | ensemble | 5 | 6 | 0.020464 |
| CYP2D6_Veith | 12 | 6 | ensemble | 5 | 6 | 0.003721 |
| CYP3A4_Substrate_CarbonMangels | 12 | 6 | ensemble | 6 | 5 | 0.003221 |
| CYP3A4_Veith | 12 | 6 | ensemble | 7 | 4 | 0.001346 |
| Caco2_Wang | 12 | 6 | ensemble | 4 | 6 | 0.003635 |
| Clearance_Hepatocyte_AZ | 12 | 6 | ensemble | 12 | 6 | 0.002055 |
| Clearance_Microsome_AZ | 12 | 6 | ensemble | 6 | 4 | 0.016533 |
| DILI | 12 | 6 | ensemble | 4 | 5 | 0.006655 |
| HIA_Hou | 12 | 6 | ensemble | 6 | 4 | 0.001201 |
| Half_Life_Obach | 12 | 6 | extra_trees | 1 | 1 | 0.000000 |
| LD50_Zhu | 9 | 6 | ensemble | 4 | 5 | 0.003501 |
| Lipophilicity_AstraZeneca | 12 | 6 | lightgbm | 3 | 3 | 0.000000 |
| PPBR_AZ | 12 | 6 | ensemble | 8 | 2 | 0.026500 |
| Pgp_Broccatelli | 12 | 6 | ensemble | 2 | 2 | 0.001827 |
| Solubility_AqSolDB | 12 | 6 | ensemble | 11 | 6 | 0.005102 |
| VDss_Lombardo | 12 | 6 | ensemble | 9 | 4 | 0.005468 |
| hERG | 12 | 6 | lightgbm | 3 | 5 | 0.000000 |

## Promotion-width replay

| Width | Tasks | Practical equivalents | Mean regret | Max regret |
|---:|---:|---:|---:|---:|
| 1 | 22 | 9 | 0.019462 | 0.164842 |
| 2 | 22 | 14 | 0.012426 | 0.127112 |
| 3 | 22 | 18 | 0.002779 | 0.023633 |
| 4 | 22 | 20 | 0.001572 | 0.021206 |
| 5 | 22 | 21 | 0.000515 | 0.008370 |
| 6 | 22 | 22 | 0.000000 | 0.000000 |

## Action-prefix replay

| Width | Tasks | Practical equivalents | Mean regret | Max regret |
|---:|---:|---:|---:|---:|
| 1 | 22 | 4 | 0.107381 | 1.559922 |
| 2 | 22 | 5 | 0.099438 | 1.559922 |
| 3 | 22 | 8 | 0.088728 | 1.559922 |
| 4 | 22 | 13 | 0.084510 | 1.559922 |
| 5 | 22 | 15 | 0.081974 | 1.559922 |
| 6 | 22 | 16 | 0.081163 | 1.559922 |
| 7 | 22 | 17 | 0.012050 | 0.127112 |
| 8 | 22 | 18 | 0.005974 | 0.068628 |
| 9 | 22 | 19 | 0.005566 | 0.068628 |
| 10 | 21 | 18 | 0.005831 | 0.068628 |
| 11 | 21 | 20 | 0.003268 | 0.068628 |
| 12 | 21 | 21 | 0.000000 | 0.000000 |

## Limitations

- Replay uses only candidates already promoted in the frozen run.
- Search summaries retain only the top three ensemble candidates.
- Validation replay cannot establish validation-to-test transfer.
