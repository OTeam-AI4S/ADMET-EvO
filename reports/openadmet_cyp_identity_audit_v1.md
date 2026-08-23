# OpenADMET CYP canonical-identity audit v1

The identity gate passes: every unique raw SMILES in the official training and
blind tables was successfully standardized. However, raw SMILES are not a safe
split key. Standardization merges 1–2 apparent records per table into the same
chemical identity, including two collisions in the 750-molecule blinded set.

| Table | Raw unique SMILES | Valid standardized | Unique InChIKeys | Identity collisions |
|---|---:|---:|---:|---:|
| Direct inhibition | 4,905 | 4,905 | 4,904 | 1 |
| TDI | 6,145 | 6,145 | 6,144 | 1 |
| Emax | 6,146 | 6,146 | 6,145 | 1 |
| Single concentration | 4,376 | 4,376 | 4,374 | 2 |
| Blinded test | 750 | 750 | 748 | 2 |

The four training modalities are nearly nested views of the same compounds:
all 4,904 direct-inhibition identities occur in both TDI and Emax, while all
4,374 single-concentration identities occur in the other three modalities.
Consequently, rows from different assay tables are not independent examples.

The frozen modeling rule is therefore: group on standardized InChIKey before
any target-specific filtering, and assign every direct, TDI, Emax and
single-concentration observation for one identity to the same outer fold. A raw
row split or raw-SMILES split is prohibited. Test labels were not present or
loaded, and the blind-structure distribution was not used for model selection.
The remaining contamination gate applies when an external supervised source is
admitted.

Machine-readable evidence: `reports/openadmet_cyp_identity_audit_v1.json`.
