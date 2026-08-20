# ADMET-EvO leaderboard goal scorecard

Leaderboard snapshot: 2026-08-17T13:50:54.441113+00:00

Current: **18/22 Top-5**, **21/22 Top-10**. Target: at least 15 Top-5 and 22 Top-10.

Goal met: **no**

| Priority | Task | Metric | Score | Rank | Top-5 threshold | Gap to Top-5 | Top-10 threshold | Gap to Top-10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rescue_top10 | Bioavailability_Ma | auroc | 0.7310 | 21 | 0.9290 | 0.1980 | 0.9220 | 0.1910 |
| promote_top5 | Clearance_Microsome_AZ | spearman | 0.5910 | 8 | 0.6250 | 0.0340 | 0.5780 | 0.0000 |
| promote_top5 | Pgp_Broccatelli | auroc | 0.9120 | 8 | 0.9290 | 0.0170 | 0.8950 | 0.0000 |
| promote_top5 | CYP3A4_Substrate_CarbonMangels | auroc | 0.6330 | 9 | 0.6500 | 0.0170 | 0.6300 | 0.0000 |
| hold_top5 | AMES | auroc | 0.8740 | 1 | 0.8520 | 0.0000 | 0.8420 | 0.0000 |
| hold_top5 | Half_Life_Obach | spearman | 0.5900 | 1 | 0.5440 | 0.0000 | 0.3720 | 0.0000 |
| hold_top5 | Lipophilicity_AstraZeneca | mae | 0.3920 | 1 | 0.5120 | 0.0000 | 0.5390 | 0.0000 |
| hold_top5 | PPBR_AZ | mae | 7.3020 | 1 | 7.7880 | 0.0000 | 8.6800 | 0.0000 |
| hold_top5 | Solubility_AqSolDB | mae | 0.7390 | 1 | 0.7890 | 0.0000 | 0.8290 | 0.0000 |
| hold_top5 | BBB_Martins | auroc | 0.9210 | 2 | 0.9130 | 0.0000 | 0.9050 | 0.0000 |
| hold_top5 | HIA_Hou | auroc | 0.9900 | 2 | 0.9860 | 0.0000 | 0.9750 | 0.0000 |
| hold_top5 | CYP2D6_Substrate_CarbonMangels | auprc | 0.7210 | 3 | 0.7040 | 0.0000 | 0.6770 | 0.0000 |
| hold_top5 | DILI | auroc | 0.9200 | 3 | 0.9170 | 0.0000 | 0.8870 | 0.0000 |
| hold_top5 | VDss_Lombardo | spearman | 0.6730 | 3 | 0.6090 | 0.0000 | 0.4970 | 0.0000 |
| hold_top5 | CYP2D6_Veith | auprc | 0.7210 | 4 | 0.7190 | 0.0000 | 0.6490 | 0.0000 |
| hold_top5 | CYP3A4_Veith | auprc | 0.8850 | 4 | 0.8810 | 0.0000 | 0.8620 | 0.0000 |
| hold_top5 | Caco2_Wang | mae | 0.2830 | 4 | 0.2870 | 0.0000 | 0.3260 | 0.0000 |
| hold_top5 | Clearance_Hepatocyte_AZ | spearman | 0.4560 | 4 | 0.4460 | 0.0000 | 0.4300 | 0.0000 |
| hold_top5 | CYP2C9_Substrate_CarbonMangels | auprc | 0.4340 | 5 | 0.4330 | 0.0000 | 0.3920 | 0.0000 |
| hold_top5 | CYP2C9_Veith | auprc | 0.7990 | 5 | 0.7890 | 0.0000 | 0.7580 | 0.0000 |
| hold_top5 | LD50_Zhu | mae | 0.5910 | 5 | 0.6060 | 0.0000 | 0.6300 | 0.0000 |
| hold_top5 | hERG | auroc | 0.8600 | 5 | 0.8560 | 0.0000 | 0.8250 | 0.0000 |

A gap is expressed in the native metric and is always non-negative: higher-is-better metrics must gain that amount; MAE must decrease by it.

## v2.7 MapLight+GNN outcomes

- Bioavailability_Ma: 0.697 -> 0.731 +/- 0.007; estimated rank 21; remaining Top-10 gap 0.191.
- CYP2C9_Substrate_CarbonMangels: 0.340 -> 0.434 +/- 0.009; estimated rank 5; remaining Top-10 gap 0.000.

No post-test tuning or reweighting was performed.

Bioavailability has a separately verified material leaderboard discrepancy (`reports/bio_leaderboard_discrepancy_v27.json`). The displayed threshold is retained until an authoritative correction.
