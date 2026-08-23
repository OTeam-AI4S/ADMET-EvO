# EC-VOI v5 public cohort gate v4

Date: 2026-08-23
Decision: **PASS — modeling-ready development cohort**

## Frozen evidence

- Protocol self-hash: `4cd0fd118065c1e6d2d1ee41646395b9361528737ff1867bd8e2c9025921660a`
- Protocol file SHA-256: `2eec8528cc6af3874002d0bde1352b6a34d33bfc707971aaf4a5635e573d51af`
- Cohort CSV SHA-256: `df83fbd5b8ebfc6c2199d2a752081abd8b15b5e3f38e1507dffd44dbf09dd3df`
- Manifest file SHA-256: `e5888d2640efdc2fe024e33dbb5cafac19edc4e86d50fe34566897b1af9c1301`
- Rows: 25,642 endpoint records; 17 candidate endpoints; 16 endpoints with at least 100 retained identities.
- Scientific sources: B3DB, MoleculeNet ClinTox, NCATS PAMPA and the Tox21 US Federal Challenge.
- Task types: classification and regression; all 16 scientific endpoints have five frozen outer folds.
- Identity overlap across folds: 0; chemistry-group overlap across folds: 0.
- Protected test labels loaded: false; blind/leaderboard feedback used: false.
- Targeted governance/policy/applicability regression: 16/16 tests passed.

## Firewall audit

The label-blind firewall considered 14,391 candidate identities. It removed 6,667 exact development overlaps, 701 exact protected-test overlaps and 784 protected-test near neighbours at frozen Morgan Tanimoto >= 0.90, leaving 6,239 allowed identities. Three protected structures that could not produce a fingerprint remained protected by exact identity and were separately counted. No acceptance decision depended on the observed removal counts.

## Compatibility history

v1, v2 and v3 each failed closed before writing a cohort or producing model/policy results. Their audits are retained. v4 changes only fold compatibility: the single identity mapping to multiple valid chemistry groups caused those groups to be joined into one label-blind connected component. Endpoints, targets, source permissions, firewall threshold, folds, seed and promotion gates were unchanged.

## Endpoint scope

The 16 scientific endpoints comprise B3DB classification/regression, ClinTox classification, NCATS PAMPA classification and 12 Tox21 classification endpoints. FreeSolv retained 68 identities after the frozen firewall, so it remains in the artifact for audit but is excluded from scientific endpoint counts by the preregistered >=100 rule.

## Claim boundary

This PASS proves cohort governance and readiness for development experiments only. It does not prove EC-VOI superiority, prospective blind performance, source-held-out generalization or external-data benefit. Those claims remain locked behind the matched-budget v5 experiment and its preregistered promotion gate.
