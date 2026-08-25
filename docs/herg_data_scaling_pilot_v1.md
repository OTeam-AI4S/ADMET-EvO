# hERG External-Data Scaling Pilot v1

## Scope

This is a discovery-only matched pilot for the TDC hERG endpoint. It tests a
single frozen external source and a single inexpensive model family. It does
not load the TDC test table and does not establish leaderboard or certification
gain.

## Frozen source and curation

- Source: TDC-distributed hERGCentral, `hERG_inhib` bridge endpoint.
- Frozen raw SHA-256:
  `5775a6900c3be6a2db65d357411fd3f768189de611dc2da3456dab4acbcb3ee8`.
- Raw rows: 306,893.
- Deterministic compute cohort: every positive row plus 30,000 hash-ranked
  negative rows, 43,744 rows before standardization.
- Standardized unique identities: 43,696.
- Exact identities overlapping the complete TDC discovery table removed: 27.
- Post-exact-firewall identities: 43,669.
- Per-validation-fold Morgan radius-2 similarity exclusions at Tanimoto >=0.90:
  2, 2 and 4 rows for seeds 1, 2 and 3.

The source is bridge-grade rather than endpoint-equivalent direct evidence.

## Matched results

Model: LightGBM with 350 trees on Morgan radius-2 2048-bit fingerprints.

| Seed | TDC-only AUROC | Exact-only external | Firewalled external | Firewalled delta |
|---:|---:|---:|---:|---:|
| 1 | 0.673125 | 0.839375 | 0.840000 | +0.166875 |
| 2 | 0.706047 | 0.819149 | 0.828108 | +0.122060 |
| 3 | 0.811426 | 0.843276 | 0.848837 | +0.037412 |
| Mean | 0.730199 | 0.833933 | 0.838982 | **+0.108782** |

All three seeds improved. The preregistered pilot rule (mean delta >=0.005 and
at least two of three seeds positive) therefore returns `supported`.

## Interpretation and claim boundary

The experiment supports the narrow claim that a frozen, task-related external
source can substantially improve a matched low-cost TDC hERG discovery model
after identity and near-neighbour controls. It does not yet show improvement
over the repository's strongest multi-view hERG recipe, source-heldout transfer,
or frozen TDC test performance.

The minimum next experiment is one three-seed comparison using the frozen
strong multi-view hERG recipe with and without this same firewalled cohort.
Source-heldout, additional sources, five-seed expansion and cross-endpoint
experiments are optional until that gate is evaluated.
