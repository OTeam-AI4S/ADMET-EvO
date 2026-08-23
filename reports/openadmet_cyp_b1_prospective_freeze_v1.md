# OpenADMET CYP B1 prospective freeze v1

Date: 2026-08-23
State: **frozen inference package ready; external upload/readout pending**

## Immutable selection lineage

- Stage-B selected the single B1 rich-2D LightGBM component; M3 and M4 were rejected before certification.
- Frozen recipe SHA-256: `2182b275b35c097f5739a3daa677964d2936576b6b893ceafc221016a5f03d1e`.
- Recipe file SHA-256: `6cb15a725f7e280a0f978a71ec0efc2759398afa2571efd42b8ca9cd974e2c10`.
- Official challenge Space commit: `13c5057b37d1e72b3f036dd0d59718b1823f8fdd`.
- Blind input SHA-256: `a342f8444a8dcb531ca12f3685293f0bd6c36ae9073f491e44a9bc1cc4b741f9`; it contains 750 rows and only `Molecule_Name`/`SMILES`.

## One-time internal certification

The certification protocol was hashed before the holdout was evaluated. All four frozen gates passed:

| Gate | Observed | Requirement |
|---|---:|---:|
| Macro ST-RAE | 0.684642 | <= 0.825942 |
| Macro MCC | 0.276628 | >= 0.146346 |
| Worst endpoint ST-RAE | 0.947342 | <= 1.25 |
| Worst endpoint MCC | 0.115551 | >= 0.00 |

Certification result SHA-256: `6385da4e34ba50efb0e107dd987495e96fd9a2a01ce730f1dc12917ee05b4c7b`.

## Blind inference artifacts

- Regression submission: four direct-inhibition pIC50 columns, 750 rows, SHA-256 `53d57a2829c43af383599da9ccf6052f1ba0b25368e5462631e1c80389f837ae`.
- Classification submission: two TDI binary columns, 750 rows, SHA-256 `614dc4450f84044b27d21734e2f0de27400e367259e41e1f19f723b11bf57231`.
- Both files pass the frozen official row/column, finite-regression and binary-classification schema checks.
- Blind labels loaded: false; leaderboard feedback used: false; selection reopened: false.

## Claim boundary

This package proves a prospective, content-addressed prediction freeze before leaderboard feedback. It does not yet prove blind predictive performance or rank. Upload requires the participant's external Hugging Face identity and is intentionally not performed by the local executor. Once uploaded, the first official readout is certification evidence only and cannot change this recipe.
