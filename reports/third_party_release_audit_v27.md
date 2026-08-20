# ADMET-EvO v2.7 third-party release audit

Observation date: `2026-08-17`.

| Dependency | Frozen tasks | Anonymous access | Checkpoint license | SHA-256 | Bundle policy |
|---|---|---:|---|---|---|
| unimol2_84m | DILI, Half_Life_Obach, Lipophilicity_AstraZeneca, hERG | yes | MIT | 5b9241630f1cf0b173fb06d1e76096e5daf5f91c3e51b066ba69524dafe60e35 | external_link_only_in_project_bundle |
| gin_supervised_masking | Bioavailability_Ma, CYP2C9_Substrate_CarbonMangels | yes | not explicit | 6d0f8febad73e437772ebffc2ac32253d79f86ee138cfc233590ae50fb1cfeb9 | external_link_only_verified_against_official_dgllife_parameters |

## Release gates

- Registration-free checkpoint access verified: **true**
- Source-code licenses verified: **true**
- Checkpoint-specific licenses verified: **false**
- Ready for external deposit: **true**

The Uni-Mol2 model repository explicitly applies MIT metadata to the public, ungated repository. The Molfeat GIN artifact is publicly downloadable and content-addressed. Its 57 parameter tensors are bitwise identical to DGL-LifeSci's official checkpoint, but its metadata has no license field. Apache-2.0 on source code is therefore not promoted into an artifact redistribution claim. The reviewer bundle links to both checkpoints and redistributes neither, so the bundle itself is ready for external deposit while checkpoint redistribution remains disallowed.

Remaining gate: deposit the prepared link-only bundle and mint a DOI; do not add third-party checkpoint files unless their artifact licenses are made explicit.

This audit is release-only: it did not load official test data and did not change model selection or the frozen scorecard.
