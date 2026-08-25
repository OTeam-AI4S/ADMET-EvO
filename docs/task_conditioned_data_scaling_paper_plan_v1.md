# Task-conditioned ADMET data scaling: minimal paper plan

## Core claim

The ADMET agent does not merely choose a model. For each endpoint it finds
database and primary-literature evidence, enforces an endpoint contract,
removes exact overlaps and all candidates with maximum Morgan Tanimoto
greater than or equal to 0.80, records source lineage, and measures whether
the admitted external data improves a fixed model recipe.

This is stronger and more defensible than the claim that adding more rows
always improves performance. Direct endpoint data, bridge/pretraining data,
and same-lineage data must be reported separately.

## Current evidence

| Task | External source | Raw rows | Post-exact identities | Baseline AUROC | Augmented AUROC | Delta | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| hERG | hERGCentral | 306,893 | 43,669 | 0.7302 | 0.8366 | +0.1064 | Supported on 3/3 seeds; bridge-grade source, discovery validation only |
| BBB_Martins | B3DB | 6,167 | 4,367 | 0.8993 | 0.9064 | +0.0071 | Supported on 2/3 seeds; source-lineage clearance still required |

Both pilots use a fixed LightGBM/Morgan recipe, never load TDC test labels,
and use the strict rule `max Tanimoto < 0.80` per validation fold.

## 22-task execution design

The machine-readable source and literature registry is
`reports/tdc22_data_source_registry_v1.json`. It covers all 22 endpoints and
specifies direct sources, bridge sources, acquisition method, endpoint guard,
primary benchmark reference, and claim status.

The time-efficient experiment matrix is:

1. Acquisition screen for all 22 tasks: source counts, endpoint-contract
   retention, exact-overlap retention, `<0.80` retention, conflict rate and
   license/provenance. No hyperparameter search.
2. One fixed-seed baseline versus augmented screen on every task with admitted
   direct data.
3. Three-seed confirmation only for tasks with positive screening delta.
4. Direct-versus-bridge ablation only where bridge data are actually used.
5. One evaluator-owned test evaluation after sources, thresholds and models
   are frozen. Do not repeatedly query test results.

Large model sweeps, many similarity thresholds, and all-pairs architecture
ablations are intentionally omitted. The fixed threshold is 0.80; it is not a
tuned hyperparameter.

## Acquisition order

1. hERG and BBB: pilots complete; next action is source-lineage audit and
   evaluator-owned confirmation.
2. AMES, LD50 and DILI: EPA ToxValDB/ToxRefDB and FDA DILIrank 2.0 offer the
   fastest high-provenance bulk path.
3. CYP inhibition/substrate tasks: one ChEMBL bulk snapshot can serve six
   tasks, but inhibitor and substrate records must never be conflated and the
   Veith/Carbon-Mangels lineages must be excluded from independent-gain claims.
4. Half-life, VDss, bioavailability, HIA and PPBR: PK-DB plus study-linked
   literature; species, route and units are mandatory.
5. Caco-2, solubility, lipophilicity and clearance: ChEMBL bulk DMPK records,
   with assay-system, pH, direction and unit harmonization.

## Paper figures/tables

- Figure 1: task-conditioned collection and admission workflow.
- Figure 2: per-task delta versus admitted independent external row count.
- Figure 3: retention waterfall (raw, valid, endpoint-matched, lineage-clean,
  exact-clean, `<0.80` clean).
- Table 1: 22 tasks, database sources and primary references.
- Table 2: baseline/augmented mean, standard deviation and positive-seed rate.
- One essential ablation: direct endpoint data versus bridge data versus both.

## Claim boundary

Until the 20 pending acquisition jobs and source-lineage audits are complete,
the defensible result is a two-task proof of concept plus a complete 22-task
collection protocol. It is not yet a demonstrated 22-task performance gain.
