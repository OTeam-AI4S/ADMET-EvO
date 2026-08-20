# External-data stewardship controls

Sealed local 11-dossier fixture; no TDC test data loaded.

| Arm | Decision accuracy | Issue F1 | Manifest accuracy | Unsafe merges | Safety gate |
|---|---:|---:|---:|---:|---|
| Agent contract steward | 1.000 | 1.000 | 1.000 | 0 | PASS |
| Complete rules steward | 1.000 | 1.000 | 1.000 | 0 | PASS |
| Expanded AutoML schema steward | 0.273 | 0.462 | 0.818 | 7 | FAIL |
| Conventional AutoML direct fit | 0.091 | 0.000 | 0.818 | 10 | FAIL |
| Seeded random | 0.364 | 0.000 | 0.364 | 4 | FAIL |

The Agent prevents unsafe merges that schema-only and direct-fit AutoML miss, but
it exactly ties the complete deterministic rules steward. This establishes a useful
data-governance safety capability, not learned autonomous reasoning. Independent
composite dossiers and evidence-dependent remediation are required for a stronger claim.
