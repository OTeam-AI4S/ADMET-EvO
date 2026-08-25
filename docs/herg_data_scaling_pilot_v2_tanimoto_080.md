# hERG External-Data Scaling Pilot v2: Tanimoto <0.80

## Frozen rule

External records are retained only when their maximum Morgan radius-2,
2048-bit Tanimoto similarity to every molecule in the current TDC validation
fold is strictly lower than 0.80. Thus all records with similarity greater
than or equal to 0.80 are excluded.

The source, deterministic 30,000-majority compute cap, structure
standardization, model and three TDC discovery seeds are identical to pilot
v1. The TDC test table is not opened.

## Curation

- Raw hERGCentral rows: 306,893.
- Rows before structure standardization: 43,744.
- Unique standardized identities: 43,696.
- Exact TDC discovery overlaps removed: 27.
- Post-exact-firewall identities: 43,669.
- Tanimoto >=0.80 exclusions by validation seed: 3, 3 and 5.

## Matched results

| Seed | TDC-only AUROC | Tanimoto <0.80 external AUROC | Delta |
|---:|---:|---:|---:|
| 1 | 0.673125 | 0.831250 | +0.158125 |
| 2 | 0.706047 | 0.834826 | +0.128779 |
| 3 | 0.811426 | 0.843782 | +0.032356 |
| Mean | 0.730199 | **0.836619** | **+0.106420** |

All three seeds improve and the pilot remains `supported`.

Relative to the Tanimoto <0.90 pilot, the stricter firewall changes mean AUROC
from 0.838982 to 0.836619 (difference -0.002362) while retaining a mean gain
above 0.10. The observed benefit therefore does not depend materially on
external molecules in the 0.80-0.90 similarity band.

## Claim boundary

This is discovery-validation evidence for a bridge-grade external source. It
does not establish gain over the strongest multi-view recipe, source-heldout
transfer, or frozen TDC test performance.
