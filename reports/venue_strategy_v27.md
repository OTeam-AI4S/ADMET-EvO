# ADMET-EvO v2.7 venue strategy

Decision date: 2026-08-17. This is a fit assessment, not an acceptance claim.

## Decision

Primary target: **Journal of Cheminformatics, Research Article**.

Conditional stretch target: **Journal of Chemical Information and Modeling
(JCIM), Article** only if the prospectively specified router safety guard gives
a positive held-out validation result and the final software/data archive is
reviewer-runnable.

Not selected as primary: **Bioinformatics**.

## Fit and risk matrix

| Venue | Fit to current evidence | Main editorial risk | Required action |
|---|---|---|---|
| Journal of Cheminformatics | Strong: cheminformatics method, 22-task benchmark, chemical-space reliability, open reproducibility, and a directly relevant 2026 ADMET reliability article | A leaderboard-centered framing could look incremental | Lead with freeze-safe evidence acquisition, negative-result completeness, and chemical-space risk; present ranks as one outcome rather than the method |
| JCIM | Plausible stretch: new cheminformatics methods, ML on chemical/biological data, algorithms/software | Current router controls do not demonstrate generic predictive superiority; QSAR/QSPR rigor and software/data availability expectations are high | Submit only with a positive preregistered safety result, complete data-availability statement, reviewer-runnable archive, 3--5 strong display items, and restrained claims |
| Bioinformatics | Weaker: ADMET-AI precedent exists and an official TDC split is independent, but the contribution is chemistry/protocol focused | Scope mismatch and desk-rejection risk for an established-method benchmark without a substantial biology-method advance | Keep as fallback only if the story is reframed around a broadly reusable ML-agent method and the official split is clearly independent |

## Official requirements used for the decision

- JCIM's author guidelines, updated 3 July 2026, explicitly include new
  chemical-informatics and molecular-modeling methodology, machine learning on
  chemical and biological data, and efficient algorithms/software. They require
  complete titled references at initial submission and a Data Availability
  Statement under ACS Research Data Policy Level 2:
  <https://researcher-resources.acs.org/publish/author_guidelines?coden=jcisd8>.
- Journal of Cheminformatics is the closest topical home, and the directly
  related 2026 ADMET reliability benchmark appeared there:
  <https://doi.org/10.1186/s13321-026-01217-2>.
- Bioinformatics author and scope guidance is maintained at:
  <https://academic.oup.com/bioinformatics/pages/author-guidelines> and
  <https://academic.oup.com/bioinformatics/pages/scope_guidelines>.

## Submission packaging gaps

1. Keep the conservative 21/22 Top-10 scorecard until TDC authoritatively
   resolves the Bioavailability page; do not present the independent audit as a
   corrected official rank.
2. Decide whether to run a fresh instrumented peak-memory benchmark. It is only
   necessary if the final text makes an absolute peak-memory claim; historical
   coverage is explicitly 0/31.
3. The Journal of Cheminformatics Research Article Markdown, publication
   figures, graphical abstract, and Supplementary Information are generated
   from the hashed evidence bundle. Remaining conversion is limited to the
   journal's final editable upload format after author metadata are supplied.
4. Deposit the code, frozen recipes, environment locks, prediction hashes, and
   non-restricted derived evidence in a versioned archive before submission.
5. Recheck publisher metadata in `paper/references.bib` during final typesetting.

The generated package and machine-verifiable checklist are in
`paper/submission/jcheminformatics/`. It retains explicit placeholders rather
than inventing ORCID, corresponding email, funding, competing-interest
confirmation, or an archive DOI.

The expanded source tree passes all 125 tests. Both essential checkpoints now
have verified anonymous public access and immutable hashes.
Uni-Mol2's model repository explicitly declares MIT. The Molfeat GIN artifact
metadata has no explicit license field, but all 57 parameter tensors are
bitwise identical to DGL-LifeSci's official checkpoint. The archive therefore
uses a link-only, content-addressed reproduction path and redistributes neither
checkpoint. It is ready for repository deposit; DOI minting remains an external
action, and adding GIN bytes remains prohibited without an explicit artifact
license.
