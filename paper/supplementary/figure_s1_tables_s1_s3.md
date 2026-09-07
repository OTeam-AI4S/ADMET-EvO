# Supplementary Figure and Tables

## Figure S1 | Operational architecture of ADMET-EvO

![Figure S1. Operational architecture of ADMET-EvO](paper/methods/figures/figure1_nature_autonomous_research_architecture.png)

**Figure S1 | Operational architecture of ADMET-EvO.** **a,** The research loop updates an explicit state through contract, diagnosis, hypothesis, intervention, evaluation and revision. Evidence is retained as supported, refuted or inconclusive, and the frozen predictor is separated from blind certification. **b,** Data expansion is treated as evidence admission: candidate records pass endpoint definition, provenance preservation, assay harmonization, exact-analogue decontamination and role assignment before use. Endpoint-name agreement alone does not authorize merging.

Vector files: [PDF](paper/methods/figures/figure1_nature_autonomous_research_architecture.pdf) · [SVG](paper/methods/figures/figure1_nature_autonomous_research_architecture.svg)

## Table S1 | Raw values underlying the three-axis ablation heatmap

| Task | Metric | Feature Δ | Model Δ | Data Δ | Data outcome / reason |
|---|---:|---:|---:|---:|---|
| Caco2 Wang | MAE | 0.131852 | 0.008940 | 0.039558 | Selected Broad-DB candidate (ratio=4; seed=5) |
| HIA Hou | AUROC | 0.040329 | 0.034156 | 0.010000 | Selected Broad-DB candidate (ratio=0.5; seed=4) |
| Pgp Broccatelli | AUROC | 0.022511 | 0.010247 | 0.005593 | Selected Broad-DB candidate (ratio=2; seed=5) |
| Bioavailability Ma | AUROC | 0.071749 | -0.012970 | 0.039062 | Selected Broad-DB candidate (ratio=0.5; seed=4) |
| Lipophilicity AZ | MAE | 0.118820 | 0.000000 | 0.016310 | Selected Broad-DB candidate (ratio=1; seed=2) |
| Solubility AqSolDB | MAE | 0.464365 | -0.010787 | 0.012406 | Selected Broad-DB candidate (ratio=8; seed=5) |
| BBB Martins | AUROC | 0.034152 | 0.006782 | 0.013871 | Selected Broad-DB candidate (ratio=4; seed=2) |
| PPBR AZ | MAE | 1.032063 | -0.071893 | 0.570514 | Selected Broad-DB candidate (ratio=0.5; seed=2) |
| VDss Lombardo | Spearman | 0.069827 | 0.193243 | 0.031231 | Selected Broad-DB candidate (ratio=0.5; seed=3) |
| CYP2C9 Veith | AUPRC | 0.032602 | 0.005198 | 0.003649 | Selected Broad-DB candidate (ratio=2; seed=4) |
| CYP2D6 Veith | AUPRC | 0.060008 | 0.007398 | 0.003873 | Selected Broad-DB candidate (ratio=2; seed=3) |
| CYP3A4 Veith | AUPRC | 0.023548 | -0.000445 | 0.001838 | Selected Broad-DB candidate (ratio=2; seed=4) |
| CYP2C9 substrate | AUPRC | -0.002295 | -0.011720 | NA† | Rejected before training: 60 identities (<100), within-source label conflicts, shared/unresolved lineage and exact-overlap target conflicts; absence-as-negative risk |
| CYP2D6 substrate | AUPRC | 0.026019 | 0.070618 | NA† | Rejected before training: 88 identities (<100), within-source label conflicts, shared/unresolved lineage and exact-overlap target conflicts; absence-as-negative risk |
| CYP3A4 substrate | AUROC | -0.030289 | 0.018818 | 0.022894 | Selected Broad-DB candidate (ratio=0.5; seed=2) |
| Half-life Obach | Spearman | -0.065704 | 0.090341 | 0.000000 | Baseline retained; best explored candidate was non-improving (oriented delta=-0.004898) |
| Clearance hepatocyte | Spearman | 0.153525 | 0.002001 | NA† | Rejected before training: 17 identities (<30), no qualifying chemical-space coverage and two non-equivalent normalization units |
| Clearance microsome | Spearman | -0.008821 | 0.036476 | 0.103687 | Selected Broad-DB candidate (ratio=1; seed=5) |
| LD50 Zhu | MAE | 0.069553 | 0.023190 | 0.000000 | Baseline retained; best explored candidate was non-improving (oriented delta=-0.020667) |
| hERG | AUROC | 0.116053 | 0.069809 | 0.060470 | Selected Broad-DB candidate (ratio=1; seed=2) |
| AMES | AUROC | 0.032422 | 0.005246 | 0.000000 | Baseline retained; best explored candidate was non-improving (oriented delta=-0.003087) |
| DILI | AUROC | 0.060761 | 0.043913 | 0.054674 | Selected Broad-DB candidate (ratio=1; seed=3) |

Values are native-metric oriented differences: Δ = candidate − baseline for AUROC, AUPRC and Spearman, and Δ = baseline − candidate for MAE; positive values therefore indicate improvement. Feature and Model are four-seed certification means (seeds 41–44). Data is the Broad DB best-seed exploratory result and is not at the same evidence tier as Feature or Model. †For three tasks, the Broad-DB candidate source failed the admission gate before model fitting; the heatmap encoded the realised policy effect as zero, whereas Table S1 marks the unobserved augmentation effect as NA and reports the recorded rejection reason. This Broad-DB-specific label does not imply that no later evidence-packet experiment existed. Values are shown to six decimals; full stored precision and audit fields are retained in the CSV.

Complete baseline, candidate, SD and Data-selection fields: [table_s1_three_axis_ablation.csv](table_s1_three_axis_ablation.csv)

Source trace: [plotted three-axis values](figures/three_axis_ablation_v2/source_data.csv); [Broad-DB import audit](figures/five_axis_observed_gain_v2/data_axis_import_audit.csv); [upstream admission decisions](reports/all22_adaptive_agent_data_v1/development/).

## Table S2 | Definitions and classifications of the 43 ToxCast tasks (landscape layout)

| Liability class | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---|---|---|
| **Cytotoxicity (n=6)** | **T01 [H]** `APR_HepG2_CellLoss_72h_dn`<br>Hoechst nuclear-count decrease and cell loss in HepG2 cells<br>*Cellular; Cellular phenotype* | **T02 [H]** `ACEA_T47D_80hr_Negative`<br>RT-CES impedance loss-of-signal in human T47D cells<br>*Cellular; Cellular phenotype* | **T03 [H]** `BSK_4H_SRB_down`<br>SRB total-protein decrease in IL-4 plus histamine-stimulated HUVECs<br>*Cellular; Cellular phenotype* | **T04 [H]** `BSK_SAg_Proliferation_down`<br>SRB or total-protein decrease in TCR-activated HUVEC and PBMC coculture<br>*Cellular; Cellular phenotype* |
|  | **T05 [H]** `TOX21_MMP_viability`<br>CellTiter-Glo ATP viability decrease in HepG2 cells after 24 h<br>*Cellular; Cellular phenotype* | **T06 [H]** `BSK_KF3CT_SRB_down`<br>SRB total-protein decrease in keratinocyte and fibroblast coculture<br>*Cellular; Cellular phenotype* |  |  |
| **Cellular dysfunction (n=4)** | **T07 [M]** `TOX21_MMP_ratio_up`<br>Mito-MPS dye-ratio perturbation in HepG2 cells after 24 h<br>*Cellular; Cellular phenotype* | **T08 [H]** `APR_HepG2_MitoticArrest_72h_up`<br>High-content mitotic-arrest phenotype increase<br>*Cellular; Cellular phenotype* | **T09 [H]** `APR_HepG2_MicrotubuleCSK_72h_dn`<br>High-content microtubule cytoskeleton signal or conformation decrease<br>*Cellular; Cellular phenotype* | **T10 [H]** `APR_HepG2_CellCycleArrest_24h_dn`<br>High-content cell-cycle-arrest feature decrease<br>*Cellular; Cellular phenotype* |
| **Endocrine disruption (n=8)** | **T11 [H]** `OT_ER_ERaERb_0480`<br>ERalpha-ERbeta protein-fragment complementation or complex-stabilization signal<br>*Molecular; Mechanistic proxy* | **T12 [H]** `OT_ERa_EREGFP_0480`<br>Estrogen-response-element GFP reporter induction in HeLa cells at 8 h<br>*Molecular; Mechanistic proxy* | **T13 [H]** `TOX21_ERa_LUC_BG1_Antagonist`<br>Estrogen-responsive luciferase loss-of-signal in human BG1 ovarian cells<br>*Molecular; Mechanistic proxy* | **T14 [H]** `OT_ERa_EREGFP_0120`<br>Estrogen-response-element GFP reporter induction in HeLa cells at 2 h<br>*Molecular; Mechanistic proxy* |
|  | **T15 [H]** `ATG_ERE_CIS_up`<br>Estrogen-response-element cis-reporter transcription increase<br>*Molecular; Mechanistic proxy* | **T16 [H]** `OT_AR_ARSRC1_0960`<br>Androgen-receptor and SRC1 coactivator interaction signal<br>*Molecular; Mechanistic proxy* | **T17 [H]** `TOX21_PPARg_BLA_antagonist_ratio`<br>BLA FRET-ratio response for PPARgamma antagonism in HEK293 cells<br>*Molecular; Mechanistic proxy* | **T18 [L]** `ATG_DR5_CIS_dn`<br>Direct-repeat-5 retinoic-acid response element linked to RAR/RXR signaling<br>*Molecular; Mechanistic proxy* |
| **Stress response (n=6)** | **T19 [H]** `ATG_Ahr_CIS_up`<br>AhR/AhRE cis-reporter transcription increase in HepG2 cells<br>*Molecular; Mechanistic proxy* | **T20 [H]** `ATG_PBREM_CIS_up`<br>Phenobarbital-responsive enhancer module reporter increase linked mainly to CAR signaling<br>*Molecular; Mechanistic proxy* | **T21 [H]** `ATG_p53_CIS_dn`<br>TP53 response-element transcription decrease<br>*Molecular; Mechanistic proxy* | **T22 [H]** `ATG_CRE_CIS_up`<br>cAMP-response-element cis-reporter transcription increase<br>*Molecular; Mechanistic proxy* |
|  | **T23 [L]** `ATG_CRE_CIS_dn`<br>cAMP-response-element cis-reporter transcription decrease<br>*Molecular; Mechanistic proxy* | **T24 [M]** `ATG_GATA_CIS_dn`<br>GATA-family response-element transcription decrease<br>*Molecular; Mechanistic proxy* |  |  |
| **Immune dysregulation (n=4)** | **T25 [H]** `BSK_SAg_CD38_down`<br>CD38 protein decrease in TCR-activated HUVEC and PBMC coculture<br>*Cellular; Mechanistic proxy* | **T26 [H]** `BSK_3C_ICAM1_down`<br>ICAM-1 protein decrease in cytokine-stimulated HUVECs<br>*Cellular; Mechanistic proxy* | **T27 [H]** `ATG_STAT3_CIS_dn`<br>STAT3 response-element transcription decrease<br>*Molecular; Mechanistic proxy* | **T28 [H]** `BSK_LPS_TNFa_up`<br>TNF-alpha protein increase in LPS-stimulated HUVEC and PBMC coculture<br>*Cellular; Mechanistic proxy* |
| **Vascular dysfunction (n=3)** | **T29 [H]** `BSK_CASM3C_uPAR_down`<br>uPAR/PLAUR protein decrease in HUVEC and coronary-artery smooth-muscle coculture<br>*Cellular; Mechanistic proxy* | **T30 [H]** `BSK_CASM3C_LDLR_up`<br>LDL-receptor protein increase in vascular coculture<br>*Cellular; Mechanistic proxy* | **T31 [H]** `BSK_CASM3C_TissueFactor_down`<br>F3/tissue-factor protein decrease in vascular coculture<br>*Cellular; Mechanistic proxy* |  |
| **Tissue remodelling (n=2)** | **T32 [H]** `BSK_hDFCGF_MMP1_down`<br>MMP-1 protein decrease in stimulated human dermal fibroblasts<br>*Cellular; Mechanistic proxy* | **T33 [H]** `BSK_hDFCGF_EGFR_down`<br>EGFR protein decrease in stimulated human dermal fibroblasts<br>*Cellular; Mechanistic proxy* |  |  |
| **Developmental toxicity (n=2)** | **T34 [H]** `Tanguay_ZF_120hpf_AXIS_up`<br>Increased body-axis abnormality in zebrafish larvae at 120 hpf<br>*Organism; Organism phenotype* | **T35 [H]** `Tanguay_ZF_120hpf_TR_up`<br>Abnormal touch-response phenotype in zebrafish larvae at 120 hpf<br>*Organism; Organism phenotype* |  |  |
| **Assay interference (n=8)** | **T36 [H]** `TOX21_ERa_BLA_Agonist_ch2`<br>Cleaved-substrate fluorescence channel in an ERalpha agonist BLA assay<br>*Molecular; Technical control* | **T37 [H]** `TOX21_ARE_BLA_Agonist_ch1`<br>Uncleaved-substrate fluorescence channel in an ARE/NFE2L2 BLA assay<br>*Molecular; Technical control* | **T38 [H]** `TOX21_p53_BLA_p1_ch1`<br>Uncleaved-substrate fluorescence channel in a p53 BLA assay<br>*Molecular; Technical control* | **T39 [H]** `TOX21_p53_BLA_p4_ch1`<br>Uncleaved-substrate fluorescence channel in a p53 BLA assay<br>*Molecular; Technical control* |
|  | **T40 [H]** `ATG_GAL4_TRANS_dn`<br>Exogenous GAL4 trans-reporter internal marker decrease<br>*Molecular; Technical control* | **T41 [H]** `TOX21_GR_BLA_Agonist_ch1`<br>Uncleaved-substrate fluorescence channel in a glucocorticoid-receptor BLA assay<br>*Molecular; Technical control* | **T42 [H]** `TOX21_p53_BLA_p3_ch2`<br>Cleaved-substrate fluorescence channel in a p53 BLA assay<br>*Molecular; Technical control* | **T43 [H]** `TOX21_ERa_BLA_Antagonist_ch1`<br>Uncleaved-substrate fluorescence channel in an ERalpha antagonist BLA assay<br>*Molecular; Technical control* |

All 43 tasks are binary ToxCast classifications evaluated by MCC. The landscape matrix is read left to right within each liability class; all identifiers carry the `toxcast::` namespace. H, M and L denote high, medium and low curation confidence, respectively. Thirty-five rows are biologically interpretable readouts; the final eight assay-interference rows are technical control or interference-sensitive channels and should not be interpreted as standalone biological toxicity endpoints.

Complete classification notes and metadata fields: [table_s2_tox43_task_dictionary.csv](table_s2_tox43_task_dictionary.csv)

## Table S3 | ADMET-EvO versus the strongest fixed pipeline on the 43 tasks

| No. | Endpoint | ADMET-EvO MCC | Best fixed pipeline | Best fixed MCC | ΔMCC |
|---:|---|---:|---|---:|---:|
| T01 | HepG2 cell loss (72 h) | 0.464449 | RandomForest-Rich | 0.450237 | 0.014212 |
| T02 | T47D cell-index decrease (80 h) | 0.425626 | LightGBM-Rich | 0.407877 | 0.017748 |
| T03 | Endothelial cell-mass decrease | 0.401008 | RandomForest-Rich | 0.373654 | 0.027354 |
| T04 | Immune-coculture proliferation decrease | 0.378944 | RandomForest-Rich | 0.350283 | 0.028660 |
| T05 | MMP-assay viability decrease | 0.368909 | XGBoost-Rich | 0.340844 | 0.028066 |
| T06 | Skin-coculture cell-mass decrease | 0.267332 | XGBoost-Rich | 0.250455 | 0.016877 |
| T07 | Mitochondrial-potential ratio perturbation | 0.475257 | RandomForest-Rich | 0.402772 | 0.072486 |
| T08 | HepG2 mitotic arrest increase (72 h) | 0.349912 | XGBoost-Rich | 0.342026 | 0.007886 |
| T09 | HepG2 microtubule/cytoskeleton decrease (72 h) | 0.214916 | RandomForest-Rich | 0.152016 | 0.062899 |
| T10 | HepG2 cell-cycle feature decrease (24 h) | 0.146371 | RandomForest-Rich | 0.141821 | 0.004549 |
| T11 | ERalpha-ERbeta heterodimerization (8 h) | 0.566517 | RandomForest-Rich | 0.530157 | 0.036360 |
| T12 | ERalpha ERE-GFP response (8 h) | 0.469065 | ExtraTrees-Rich | 0.366018 | 0.103046 |
| T13 | ERalpha antagonism in BG1 cells | 0.392670 | ExtraTrees-Rich | 0.384052 | 0.008619 |
| T14 | ERalpha ERE-GFP response (2 h) | 0.388575 | ExtraTrees-Rich | 0.330271 | 0.058305 |
| T15 | Estrogen-response-element increase | 0.372142 | LightGBM-Rich | 0.369585 | 0.002557 |
| T16 | AR-SRC1 recruitment (16 h) | 0.344583 | CatBoost-Rich | 0.341774 | 0.002808 |
| T17 | PPARgamma antagonist response | 0.342532 | XGBoost-Rich | 0.335194 | 0.007338 |
| T18 | Retinoic-acid response-element decrease | 0.139679 | CatBoost-Rich | 0.132863 | 0.006815 |
| T19 | AhR response-element increase | 0.393021 | RandomForest-Rich | 0.386501 | 0.006520 |
| T20 | PBREM xenobiotic-response increase | 0.371961 | RandomForest-Rich | 0.363136 | 0.008825 |
| T21 | p53-responsive transcription decrease | 0.332619 | LightGBM-Rich | 0.324176 | 0.008442 |
| T22 | CRE reporter increase | 0.331205 | RandomForest-Rich | 0.296348 | 0.034857 |
| T23 | CRE reporter decrease | 0.168460 | RandomForest-Rich | 0.147277 | 0.021183 |
| T24 | GATA1-responsive transcription decrease | 0.104774 | LightGBM-Rich | 0.097077 | 0.007698 |
| T25 | CD38 immune-activation decrease | 0.406612 | XGBoost-Rich | 0.402108 | 0.004504 |
| T26 | Endothelial ICAM-1 decrease | 0.324497 | CatBoost-Rich | 0.305597 | 0.018900 |
| T27 | STAT3 reporter decrease | 0.257161 | ExtraTrees-Morgan | 0.243135 | 0.014026 |
| T28 | LPS-induced TNF-alpha increase | 0.115312 | RandomForest-Rich | 0.113954 | 0.001358 |
| T29 | Vascular uPAR decrease | 0.290789 | LightGBM-Rich | 0.274204 | 0.016585 |
| T30 | Vascular LDLR increase | 0.279988 | RandomForest-Rich | 0.263869 | 0.016119 |
| T31 | Vascular tissue-factor decrease | 0.233353 | ExtraTrees-Rich | 0.222782 | 0.010571 |
| T32 | Fibroblast MMP-1 decrease | 0.309318 | RandomForest-Rich | 0.248276 | 0.061042 |
| T33 | Fibroblast EGFR decrease | 0.186906 | CatBoost-Rich | 0.160867 | 0.026039 |
| T34 | Zebrafish axial malformation (120 hpf) | 0.195677 | CatBoost-Rich | 0.193295 | 0.002383 |
| T35 | Zebrafish touch-response abnormality (120 hpf) | 0.152095 | RandomForest-Rich | 0.147795 | 0.004300 |
| T36 | ERalpha agonist BLA channel 2 | 0.396660 | ExtraTrees-Rich | 0.393579 | 0.003082 |
| T37 | ARE BLA channel 1 | 0.368684 | RandomForest-Rich | 0.349350 | 0.019334 |
| T38 | p53 BLA p1 channel 1 | 0.308610 | ExtraTrees-Rich | 0.306616 | 0.001994 |
| T39 | p53 BLA p4 channel 1 | 0.307827 | RandomForest-Rich | 0.275031 | 0.032795 |
| T40 | GAL4 transactivation channel decrease | 0.292984 | RandomForest-Rich | 0.237094 | 0.055890 |
| T41 | GR agonist BLA channel 1 | 0.245760 | RandomForest-Rich | 0.232391 | 0.013369 |
| T42 | p53 BLA p3 channel 2 | 0.237625 | ExtraTrees-Rich | 0.226040 | 0.011585 |
| T43 | ERalpha antagonist BLA channel 1 | 0.221792 | ExtraTrees-Rich | 0.194327 | 0.027465 |

MCC is higher-is-better. The fixed comparison set comprised ExtraTrees-Morgan, ExtraTrees-Rich, LightGBM-Rich, RandomForest-Rich, XGBoost-Rich and CatBoost-Rich. Across these preselected rows, mean MCC was 0.310283 for the workbook-labelled ADMET-EvO models and 0.288528 for the per-task best fixed pipeline (mean paired ΔMCC = 0.021755).

ADMET-EvO MCC, best-fixed MCC and ΔMCC are independently reported six-decimal source fields; subtraction of the two displayed MCCs can therefore differ from the displayed ΔMCC by 0.000001.

These 43 rows are a post-hoc positive subset of 309 retained ToxCast endpoints, selected because the workbook-labelled ADMET-EvO point estimate exceeded all six fixed-pipeline point estimates on the same endpoint-wise local holdout. The source does not bind repeated-seed uncertainty, confidence intervals, retained predictions or a public leaderboard result; the table is therefore descriptive and does not establish prospective or statistical superiority.

Complete ADMET-EvO recipes and six-pipeline MCC matrix: [table_s3_tox43_model_comparison.csv](table_s3_tox43_model_comparison.csv)
