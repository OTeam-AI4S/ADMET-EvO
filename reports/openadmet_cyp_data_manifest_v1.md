# OpenADMET CYP challenge data intake v1

Status: **official raw data and canonical-identity rules frozen; external-source
contamination audit and modeling pending**.

## Frozen source

- Dataset: <https://huggingface.co/datasets/openadmet/cyp-challenge-train-test>
- Dataset commit: `85f8b358d0a2056a98b990dd75d3b3ec9247862b`
- Challenge Space commit: `13c5057b37d1e72b3f036dd0d59718b1823f8fdd`
- License declared by the official dataset card: Apache-2.0
- Local manifest: `reports/openadmet_cyp_data_manifest_v1.json`

All five CSVs and the dataset card are content-addressed in the JSON manifest.
The blinded test table has exactly two columns, `Molecule_Name` and `SMILES`, and
contains no label columns. No test label was loaded and no leaderboard feedback
was used.

## Dataset inventory

| File/track | Rows | Unique raw SMILES | Use |
|---|---:|---:|---|
| Direct-inhibition DRC | 4,905 | 4,905 | four pIC50 regression targets |
| TDI | 6,145 | 6,145 | CYP2D6/CYP3A4 classification and paired pIC50 auxiliary targets |
| Emax | 6,146 | 6,146 | effect/uncertainty auxiliary targets |
| Single concentration | 17,504 | 4,376 | 4 isoforms × 4,376 molecules, multi-fidelity auxiliary data |
| Blinded test | 750 | 750 | structure-only final inference |

The direct table is a sparse multitask matrix rather than 4,905 labels per
endpoint: CYP1A2/CYP2C9/CYP2D6/CYP3A4 have 1,412/1,285/1,493/2,335 labels. All
reported direct-inhibition credible intervals satisfy `low <= pIC50 <= high`.

TDI class balance is similar across the two scored endpoints: CYP2D6 has
324/1,497 positives (21.64%) and CYP3A4 has 764/3,584 (21.32%). This supports MCC
as the primary classification metric and requires group-stratified folds rather
than accuracy-driven selection.

## Admission boundary

Raw file integrity, schema and canonical-identity gates pass. The identity audit
found no invalid structures, but found 1–2 standardized identity collisions per
table and near-complete identity overlap among the four training modalities.
See `reports/openadmet_cyp_identity_audit_v1.json`. The following gates remain
closed:

1. molecule/scaffold/chemistry-cluster group assignment shared across all six
   targets and auxiliary tables;
2. sealed exact/near-neighbor exclusion for any external supervised data;
3. Endpoint Contracts for direct inhibition, TDI, Emax and single-concentration
   fidelity;
4. nested group-CV and one-time internal certification protocol;
5. source/assay tokens and external-data license manifest.

The blind test structures may be used only by the sealed contamination firewall
and the frozen inference pipeline. Their distribution must not be used to tune
the model or choose an external source.
