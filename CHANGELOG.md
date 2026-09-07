# Changelog

All notable repository releases are documented here. ADMET-EvO uses semantic
versioning for the installable Python package; experiment protocol identifiers
remain immutable and keep their own version suffixes.

## 0.3.0 - 2026-09-07

- Consolidate the newer `/data/projects/admet_evo` development tree with
  source-only additions preserved in `/home/wyilin/admet`.
- Add evidence-packet acquisition, splitting, multiseed evaluation, frozen
  recipe execution, and closed-loop orchestration.
- Add conservative external-data admission weights and exact held-out
  structure firewalls without exposing held-out labels.
- Add model/feature/data three-axis ablation and extended ADMET cohort tools.
- Move supplementary manuscript assets into `paper/supplementary` and document
  the repository layout and artifact policy.
- Exclude machine-local environments, caches, checkpoints, generated figures,
  and large intermediate datasets from source control.

## 0.2.1 - 2026-08-11

- Publish the reproducible TDC ADMET benchmark agent and audited research
  workflow.
