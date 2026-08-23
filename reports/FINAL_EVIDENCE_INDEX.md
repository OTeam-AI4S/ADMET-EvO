# Final evidence index

This index separates publishable evidence from development history. A file's
presence records provenance; it does not upgrade a failed gate.

## TDC ADMET-22 application qualification

- `goal_scorecard_v27.json`: frozen 22-task rank summary.
- `goal_status_v27_2026-08-17.md`: conservative 18/22 Top-5 and 21/22 Top-10
  interpretation and unresolved Bioavailability discrepancy.
- `leaderboard_snapshot_2026-08-17.json`: exact displayed-page snapshot used
  for rank insertion.
- `cross_endpoint_fixed_model_comparison_v1.{json,md}`: comparison against
  complete-coverage fixed models.
- `paper_evidence_manifest_v27.json` and
  `validation_calibration_residuals_v27.json`: paper evidence hashes and
  chemical-space risk analysis.

## Agent/process evidence

- `ai_scientist_goal_status_v1.md` and
  `ai_scientist_evidence_scorecard_v1.{json,md}`: consolidated claim gates.
- `minimal_predictive_control_v1.{json,md}`: formal predictive control; the
  Agent did not pass the frozen noninferiority gate against expanded AutoML.
- `composite_research_challenge_v1.{json,md}`: local sealed process challenge;
  useful process evidence, not independent blind validation.
- `data_stewardship_controls_v1.{json,md}` and
  `fault_policy_controls_v1.{json,md}`: data and endpoint safety controls.
- `conditional_committee_activation_v1.{json,md}` and
  `policy_self_correction_v1.{json,md}`: bounded multi-agent activation and
  validation-only correction behavior.

## EC-VOI public policy experiments

- `ecvoi_v5_public_cohort_protocol_v4.json` and
  `ecvoi_v5_public_cohort_manifest_v4.json`: final public cohort contract and
  source-level manifest; molecule-level cohort rows are omitted from Git.
- `ecvoi_v5_public_action_screen_protocol_v2.json` and
  `ecvoi_v5_public_action_screen_seed17_manifest_v2.json`: frozen action-screen
  lineage.
- `ecvoi_v5_public_policy_analysis_v1.json` and
  `ecvoi_v5_public_policy_gate_v1.md`: failed v5 promotion result retained as
  negative evidence.
- `ecvoi_v6_mechanism_protocol_v2.json` and
  `ecvoi_v6_v5_development_replay_v1.{json,csv}`: identifiability-aware v6
  mechanism and development replay. Fresh-cohort confirmation was not
  completed, so no promotion claim is allowed.

## OpenADMET CYP challenge

- `openadmet_cyp_data_manifest_v1.{json,md}` and
  `openadmet_cyp_identity_locked_splits_v1.manifest.json`: train-data and split
  provenance without publishing molecule-level training exports.
- `openadmet_cyp_stage_a_cv_analysis_v1.json`,
  `openadmet_cyp_stage_b_cv_analysis_v1.json`, and
  `openadmet_cyp_stage_b_component_gate_v1.json`: selection evidence.
- `openadmet_cyp_b1_certification_protocol_v1.json`,
  `openadmet_cyp_frozen_recipe_b1_v1.json`, and
  `openadmet_cyp_b1_prospective_freeze_v1.md`: frozen B1 certification and
  blind-inference lineage. Blind submission files and any external readout are
  excluded.
- `openadmet_cyp_applicability_analysis_v3.{json,md}`: applicability-domain
  analysis.

## Architecture and implementation

- `docs/admet_evo_research_agent_architecture.md`: detailed system design.
- `docs/ADMET_EvO_Agent_技术全景与顶刊升级蓝图.md`: Chinese technical
  overview and paper roadmap.
- `skills/dataskill/`: packaged external-data acquisition and governance skill.
- `admet_agent/`, `scripts/`, and `tests/`: executable implementation,
  reproduction entry points, and regression tests.

## Explicitly incomplete

- EC-VOI v6 fresh-cohort confirmation;
- independent expert blind task return;
- OpenADMET blind leaderboard performance/rank;
- authoritative resolution of the TDC Bioavailability leaderboard anomaly;
- any claim of predictive superiority over expanded AutoML.
