# ADMET-EvO v2.7 paper-readiness matrix

Snapshot date: 2026-08-17. This report distinguishes conservative displayed
leaderboard ranks, scientific reproduction evidence, and claims still requiring
external confirmation.

## Headline status

| Requirement | Current evidence | Status |
|---|---|---|
| At least 15/22 Top-5 | Verified v2.7 scorecard: 18/22. | achieved |
| All 22 Top-10 | Conservative displayed-page scorecard: 21/22. Bio is the only deficit. | not yet achieved |
| Reproducible upgrades | Full freeze `693db6936485d59791eacfd9005d18efca00056274c11c1cc80efd962058c84b`; five seeds; exact source/checkpoint/prediction hashes; test-label firewall. | supported |
| Novel publishable story | Protocol-safe evidence acquisition, an auditable full-outcome ledger, cost-aware backoff, immutable discovery-to-freeze boundary, and endpoint-specific expert acquisition. | supported with bounded claims |

## v2.7 weak-task resolution

| Task | Fresh validation | Official | Displayed rank | Verdict |
|---|---:|---:|---:|---|
| CYP2C9 substrate | 0.4367 AUPRC | 0.434 +/- 0.009 | 5 | resolved |
| Bioavailability | 0.7093 AUROC | 0.731 +/- 0.007 | 21 | method reproduced; reference page disputed |

The Bio result exactly matches the peer-reviewed independent MapLight+GNN
reproduction. The same implementation reproduces CYP2C9 within 0.003 of the
displayed MapLight+GNN score, serving as a positive control.

## Bio leaderboard integrity

The current Bio page contains every one of the 16 Pgp rows with identical
model, mean, and standard deviation. MiniMol's upstream June 2024 report lists
Bio `0.689 +/- 0.020` and Pgp `0.942 +/- 0.002`; the current Bio page displays
the latter Pgp value. A 2026 peer-reviewed independent audit and our pinned
reproduction both obtain MapLight+GNN Bio `0.731 +/- 0.007` rather than the
displayed `0.938 +/- 0.002`.

Verdict: `confirmed_material_leaderboard_discrepancy`. Root-cause mechanism and
corrected current ranks are not authoritative until TDC responds. The displayed
0.922 threshold remains in the conservative scorecard.

The same mapping problem has been open as TDC issue #394 since 2026-06-08,
without a maintainer comment at the 2026-08-17 audit. This corroborates that the
observation is independent, but does not constitute an authoritative fix.

## Claim boundary

- MapLight+GNN is a pinned external expert, not the paper's model novelty.
- The publishable contribution is the agent-level evidence-acquisition and
  protocol architecture plus its complete positive/negative evidence trail.
- The results support large aggregate compute savings, not universal accuracy
  gains or uniform per-task savings.
- Historical Bio ranks are context, not current-rank evidence.
- No official result is used to alter a finalist in the same freeze lineage.
- The full-outcome ledger reduced selected compute relative to winner-only
  memory, but did not improve mean held-out accuracy. Its demonstrated value is
  cost control and auditability, not predictive superiority.
- The agent did not beat a maximum-entropy random policy at exactly matched
  expected training cost under the preregistered 22-task gate. Selection
  intelligence is therefore not claimed.
- A preregistered metric-name safety guard was evaluated on fresh default
  scaffold seeds 4-5 with zero identity overlap. It retained 27.8% of measured
  score-policy fit time, but selected the same candidates as the unguarded
  router on every task and produced exactly zero guard-minus-unguarded gain.
  Metric-tail safety improvement is therefore not claimed.

## Router controls completed

The preregistered validation-only leave-one-task-out controls used candidate
generation seed 1 and held-out validation seeds 2 and 3; official test data,
official results, and leaderboard values were not loaded.

| Control | Mean oriented gain | 95% task-bootstrap CI | Improved/tied/worse | Wilcoxon p | Gate |
|---|---:|---:|---:|---:|---:|
| Full-outcome minus winner-only memory | -0.00031 | [-0.00276, 0.00240] | 4/12/6 | 0.7213 | failed |
| Agent minus exactly matched-cost random | 0.00083 | [-0.00380, 0.00542] | 7/5/10 | 0.8684 | failed |

Full-outcome memory selected 18.5% of winner-only memory's aggregate compute
proxy. The matched random policy reproduced each selected expected cost to a
maximum relative error of `3.96e-16`. Both failed gates are retained as
negative evidence and constrain the manuscript claims.

## Remaining paper work

1. Run an instrumented peak-memory benchmark only if the selected venue
   requires an absolute memory claim; recorded historical coverage is 0/31 and
   is explicitly disclosed in the completed compute appendix.
2. Supply author-only metadata, deposit the generated link-only reviewer bundle
   with a persistent DOI, and convert the Markdown package to the journal's
   final editable submission format. Do not add third-party checkpoint bytes
   unless their artifact licenses are made explicit.
3. Obtain a TDC correction or clearly present both conservative displayed and
   audited scientific scorecards at submission.

Normalized 15-entry BibTeX, full citation resolution, and venue selection are
complete. `paper/references.bib` and `reports/bibliography_audit_v27.json` are
the corresponding sources of truth; Journal of Cheminformatics is the primary
target and JCIM is a conditional stretch target.

Four main-text figures are complete in editable SVG and 300-dpi PNG: the
protocol architecture, conservative 22-task rank coverage, fresh-seed router
compute--accuracy trade-off, and chemical-space error inflation. Their inputs,
outputs, firewall status, and claim summaries are hashed in
`paper/figures/manifest.json`; synchronized captions are in
`paper/figures/captions.md`.

A journal-specific Research Article package is now generated under
`paper/submission/jcheminformatics/`. Its 255-word structured abstract includes
the required `Contribution` heading; numbered references resolve all 15 cited
works; SI covers all 22 tasks and the negative-result ledger; and a white
920x300 graphical abstract is 35 KB. The package is deliberately marked not
submission-ready until corresponding-author metadata, declarations, and an
archival DOI are supplied.

An upload-ready deterministic reviewer archive is generated at
`release/admet-evo-v2.7-reviewer-bundle.tar.gz`. It contains the source,
environments, 22-task frozen evidence, 110 official prediction files, figures,
submission package, and upstream reconstruction inputs; it excludes official
test labels, credentials, local data/model caches, and third-party checkpoints.
The expanded source tree passes the complete 125-test suite. A dated release
audit now verifies anonymous public access and immutable hashes for both
essential checkpoints;
Uni-Mol2 is explicitly MIT. The Molfeat GIN artifact has no explicit
artifact-level license, but all 57 parameter tensors are bitwise identical to
the DGL-LifeSci official checkpoint. Because the archive contains neither
checkpoint and provides only public links plus hashes, it is ready for external
deposit; DOI minting remains an author/external action. Redistribution of the
GIN bytes remains prohibited by the fail-closed policy.

## Primary artifacts

- `reports/goal_scorecard_v27.json`
- `reports/paper_evidence_manifest_v27.json`
- `reports/bio_leaderboard_discrepancy_v27.json`
- `reports/leaderboard_snapshot_2026-08-17.json`
- `runs/v27-maplight-gnn-freeze/full-22-task.json`
- `runs/v27-maplight-gnn-confirmation/result.json`
- `runs/v27-router-controls/preregistration.json`
- `runs/v27-router-controls/report.json`
- `runs/v27-router-risk-guard/preregistration.json`
- `runs/v27-router-risk-guard/report.json`
- `reports/primary_model_effects_v27.json`
- `reports/validation_calibration_residuals_v27.json`
- `reports/compute_memory_appendix_v27.json`
- `paper/references.bib`
- `reports/bibliography_audit_v27.json`
- `reports/third_party_release_audit_v27.json`
- `reports/third_party_release_audit_v27.md`
- `reports/gin_checkpoint_equivalence_v27.json`
- `reports/venue_strategy_v27.md`
- `paper/figures/manifest.json`
- `paper/figures/captions.md`
- `paper/submission/jcheminformatics/manifest.json`
- `paper/submission/jcheminformatics/manuscript.md`
- `paper/submission/jcheminformatics/supplementary_information.md`
- `paper/submission/jcheminformatics/submission_checklist.md`
- `release/admet-evo-v2.7-reviewer-bundle.archive.json`
- `release/admet-evo-v2.7-reviewer-bundle/MANIFEST.json`
- `runs/v27-official/Bioavailability_Ma/final/results.json`
- `runs/v27-official/CYP2C9_Substrate_CarbonMangels/final/results.json`
- `reports/tdc_bio_correction_packet_v27.md`
