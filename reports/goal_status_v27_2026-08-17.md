# ADMET Agent goal status — v2.7, 2026-08-17

## Outcome

The conservative displayed-page scorecard is **18/22 Top-5 and 21/22 Top-10**.
The Top-5 requirement is exceeded by three tasks. CYP2C9 substrate is now
resolved at `0.434 +/- 0.009` AUPRC, approximately rank 5. Bioavailability is
the only displayed Top-10 deficit at `0.731 +/- 0.007` AUROC versus the current
page threshold of 0.922.

## Why Bio is no longer a normal modeling gap

- All 16 Pgp rows occur verbatim on the Bio page.
- MiniMol's primary report assigns the current Bio MiniMol value to Pgp and
  reports a distinct Bio value.
- The 2026 peer-reviewed audit reproduces MapLight+GNN Bio at exactly
  `0.731 +/- 0.007`.
- Our pinned reproduction obtains the same value and reproduces CYP2C9 within
  0.003 of its displayed reference.
- Even the independent audit's deliberately test-overfit Bio model remained
  near 0.715 and rank 21, further weakening the plausibility of 0.922 as a
  current-data modeling target.

The evidence supports leaderboard-record or dataset/version corruption over a
local implementation failure. Nevertheless, no threshold is overridden until
TDC publishes an authoritative correction.

TDC issue #394 has reported the same Bio/Pgp mapping problem since 2026-06-08.
It remains open with no maintainer response, so the prepared correction packet
should be added there or sent to `contact@tdcommons.ai` only with user approval.

## Freeze and isolation

- Full v2.7 freeze: `693db6936485d59791eacfd9005d18efca00056274c11c1cc80efd962058c84b`.
- Fresh selection seeds: 6-10; four of five paired improvements for both tasks.
- Official seeds: 1-5, evaluated once after freeze.
- Official test loaded during selection: false.
- Test labels passed to executor: false for all ten official executions.
- Canonical train/validation identity overlap: zero on all fresh splits.

## Next decision

The highest-value path is an authoritative TDC correction using the prepared,
unsent evidence packet. Spending large compute to chase 0.922 would optimize
against a contradicted reference and weaken the paper's leakage-safe story.
Legitimate Bio research can continue only under fresh validation and a new
preregistration, but it should target robust current-data performance rather
than the duplicated page value.

The goal remains active because the conservative 22/22 Top-10 condition is not
yet externally verified.

## Paper-evidence progress

The preregistered 22-task validation-only router controls are complete. Full
outcome memory reduced aggregate selected compute to 18.5% of winner-only
memory but did not improve accuracy (mean oriented gain -0.00031, 95% CI
[-0.00276, 0.00240], 4/12/6 improved/tied/worse, p=0.7213). The agent likewise
did not pass the selection-intelligence gate against a maximum-entropy random
policy at exactly matched expected training cost (mean 0.00083, 95% CI
[-0.00380, 0.00542], 7/5/10, p=0.8684). These negative controls are retained:
the supported story is protocol safety, auditability, and aggregate cost
control, not universal routing accuracy.

The 22-task post-selection scaffold-validation audit now covers all 13
classification and nine regression endpoints. Canonical identity and scaffold
overlap are zero on every analyzed split. Low-similarity molecules have worse
Brier error on 11/13 classification tasks and worse residual error on 7/9
regression tasks, providing a concrete chemical-space risk result for the
paper. The compute appendix records matched wall-time evidence but also records
that peak-memory instrumentation was absent from all 31 historical entries;
no absolute memory claim is made.

A subsequent metric-tail guard was frozen before evaluating fresh default
scaffold seeds 4-5. It covered all 22 tasks, had zero canonical identity
overlap, and loaded neither official test data nor leaderboard evidence. The
guard used score selection for Spearman and cost-aware selection otherwise,
but the two policies selected the same candidate on every Spearman task after
development-seed aggregation. Guard-minus-unguarded gain was therefore exactly
zero on 22/22 tasks; the preregistered claim and significance gates failed.
Both policies retained 27.8% of newly measured score-policy fit time. This is
retained as a negative result and does not alter the v2.7 official freeze.

Paper metadata is now normalized in a 15-entry BibTeX file with a passing
citation-resolution audit. Journal of Cheminformatics is the primary target;
JCIM remains a conditional stretch target. Remaining packaging work is final
article/SI formatting, optional fresh peak-memory instrumentation if an
absolute claim is desired, and the unresolved authoritative Bio correction.

Four reproducible main-text figures are now complete in editable SVG and
300-dpi PNG and embedded in the working manuscript. They cover the protocol
architecture, conservative 22-task ranks, the negative fresh router safety
result with measured compute, and chemical-space error inflation. The figure
manifest hashes every source and output and records that figure generation did
not load official test data or change model selection/freeze state.

The Journal of Cheminformatics Research Article package is also generated and
hashed. The structured abstract is 255 words with the journal-specific
`Contribution` heading, all 15 citations are rendered as numbered references,
the SI covers the full 22-task scorecard and negative controls, and the
graphical abstract meets the 920x300 and 150-KB limits. Author identity,
declarations, and an archival DOI remain explicit pre-submission blockers.

A deterministic reviewer bundle is now built without copying the 167-GB run
tree. The compressed archive is approximately 4.2 MB and contains 110 official
prediction files plus the exact evidence required to rebuild the scorecard,
figures, SI, and paper manifest. It excludes official test labels, credentials,
data/model caches, and third-party checkpoints. After extraction into an
independent directory, the expanded source tree now passes all 125 tests.
Anonymous access and immutable hashes are verified for both essential
checkpoints. Uni-Mol2 is explicitly MIT. The Molfeat GIN artifact remains
link-only because its metadata has no explicit artifact-level license, but all
57 parameter tensors are bitwise identical to DGL-LifeSci's official
checkpoint. The checkpoint-free bundle is therefore ready for external
deposit; minting the archival DOI remains an external gate.
