# Goal resume report — 2026-08-14

## Outcome

The dated scorecard remains **17/22 Top-5 and 20/22 Top-10**. The previously
reported Bioavailability score of 0.697 remains the scorecard value after a
new preregistered one-shot rescue failed to improve it.

Bioavailability now has a validation-supported promotion candidate. The locked
50/50 multiscale-Morgan-Tanimoto and MiniMol-cosine kernel (`C=10`) scored
0.7527 AUROC on discovery seeds 1–5. On fresh, previously unseen scaffold split
seeds 6–10 it reproduced at **0.7538 +/- 0.0630 AUROC** with scores
0.8115, 0.7122, 0.6496, 0.7930, and 0.8027. The fresh seeds were not used to
select the recipe. Its five-seed official AUROC was **0.675**, a drop of 0.022
from the existing 0.697 official result and 0.0788 from fresh scaffold
validation. The candidate is therefore rejected without post-test reweighting.
All five executor audits record that test labels were not passed to the recipe.
The ordinary fixed MiniMol head scored 0.7327 +/- 0.0734;
the 16-head balanced pairwise candidate scored 0.7414 +/- 0.0644 and was
selected unanimously by leave-one-outer-seed-out replay.

CYP2C9 substrate did not yield a promotable single model. Its best 16-head
candidate, focal loss, scored 0.4637 +/- 0.1622 AUPRC versus 0.4801 +/- 0.1719
for the existing LightGBM validation baseline. A leave-one-seed-out fixed 20%
focal-head / 80% LightGBM rank fusion reached 0.4895 +/- 0.1808 on discovery
seeds only. It is now preregistered for one confirmation on fresh scaffold
seeds 6-10; no official CYP2C9 test evaluation is authorized by that protocol.

## Leaderboard integrity warning

The frozen 2026-08-13 snapshot contains 16 exact `(model, mean, std)` matches
between the Bioavailability and Pgp pages, a set Jaccard overlap of 0.80. The
current Bioavailability Top-10 threshold of 0.922 therefore requires manual
verification. This warning does not identify which page is wrong and is not
used to silently alter ranks or declare the Goal achieved. See
`reports/leaderboard_integrity_audit_v24.json`.

## Reproduction branch

The MIT-licensed MapLight source was pinned at commit
`c249378c63232354d17083c83fe94fe728960a27`. Its original `int8` count-array
conversion overflows under the current NumPy/RDKit stack; the local evaluator
adds a provenance-recorded `int32` compatibility layer without editing the
vendored source. The corrected CYP2C9 substrate run completed on five fixed
scaffold seeds. The best fixed and leave-one-seed-out candidate was the same
balanced eight-member MapLight+MiniMol CatBoost, scoring **0.4606 +/- 0.1621
AUPRC** versus **0.4801 +/- 0.1719** for LightGBM. It improved only two of five
paired seeds and is rejected. Feature, split, embedding, code, and result hashes
are recorded in the machine-readable artifact.

## Gates

- Bioavailability MiniMol/Morgan kernel: **officially evaluated once and
  rejected**; retain the previous 0.697 score.
- CYP2C9 substrate focal head: **reject as a single model**.
- CYP2C9 20% rank fusion: **fresh-seed confirmation running**, not frozen for
  official evaluation.
- CYP2C9 MapLight/MapLight+MiniMol: **completed and rejected**.
- Bioavailability MapLight/MapLight+MiniMol: **not run; lower value than freezing
  the already fresh-seed-confirmed kernel candidate**.
- Official test access after the v2.5 freeze: **one Bioavailability five-seed
  finalization**; no test labels entered candidate execution or selection.

## Primary artifacts

- `runs/v24-minimol-kernel-confirmation/Bioavailability_Ma/locked_kernel_confirmation.json`
- `runs/v24-minimol-validation/Bioavailability_Ma/scaffold_kernels.json`
- `runs/v24-minimol-validation/Bioavailability_Ma/rank_heads.json`
- `runs/v24-minimol-validation/CYP2C9_Substrate_CarbonMangels/rank_heads.json`
- `runs/v24-minimol-validation/CYP2C9_Substrate_CarbonMangels/maplight_minimol.json`
- `reports/v25_bio_rescue_outcome.json`
- `runs/v25-cyp2c9-focal-confirmation/preregistration.json`
- `reports/leaderboard_integrity_audit_v24.json`
