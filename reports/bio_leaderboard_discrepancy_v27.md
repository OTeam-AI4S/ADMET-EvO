# Bioavailability leaderboard discrepancy audit (v2.7)

Status: **confirmed_material_leaderboard_discrepancy**

This is a confirmed material discrepancy, not an authoritative leaderboard correction. The displayed threshold remains binding for the goal scorecard.

## Independent evidence

| Evidence | Observation |
|---|---|
| Cross-page exact duplication | 16/16 Pgp rows are exact `(model, mean, std)` matches on Bio; containment 1.000, Jaccard 0.800. |
| Distinct data sets | Bio has 640 molecules; Pgp has 1212. |
| Published independent reproduction | MapLight+GNN Bio 0.731 +/- 0.007 (Table 2). |
| Historical primary report | MiniMol's upstream June 2024 table reports Bio 0.689 +/- 0.020 (rank 5) but Pgp 0.942 +/- 0.002; the current Bio page displays the latter Pgp value. |
| Our pinned reproduction | MapLight+GNN Bio 0.731 +/- 0.007; exact delta from published reproduction 0.000. |
| Positive control | CYP2C9 substrate reproduction 0.434 +/- 0.009 versus displayed 0.437; absolute delta 0.003. |
| Selection isolation | Fresh validation used no official test labels; canonical identity overlap across train/validation was zero. |
| Existing TDC issue | Issue #394 reported the same mapping problem on 2026-06-08; it remained open with 0 maintainer comments when audited. |

## Interpretation boundary

The displayed Bioavailability rows are inconsistent with current-data reproduction and are exactly copied from the distinct Pgp endpoint for all 16 Pgp entries. Dataset/version or leaderboard-record corruption is therefore substantially better supported than a local implementation failure.

Root cause proven: **False**.  
Authoritative correction obtained: **False**.  
Threshold override allowed: **False**.

## Goal impact

The displayed Bio Top-10 threshold remains 0.922. The conservative scorecard remains 18/22 Top-5 and 21/22 Top-10; the full goal is not yet met.

For context only, the local 0.731 exceeds MiniMol's historical June 2024 rank-5 Bio score of 0.689, implying a historical rank no worse than 5. This is not used as current rank evidence.

## Sources

- Current Bio page: https://tdcommons.ai/benchmark/admet_group/04bioav/
- Current Pgp page: https://tdcommons.ai/benchmark/admet_group/03pgp/
- Published audit: https://pubs.acs.org/doi/10.1021/acs.jcim.6c00819 (DOI `10.1021/acs.jcim.6c00819`)
- Audit repository: https://github.com/receptor-ai/tdc-admet-bench at `3e5472ce159a29b872219ffaaa4a11b4a2660f77`
- MiniMol primary report: https://github.com/graphcore-research/minimol at observed HEAD `41642328affbb35e4950b7b7a9020aa95919b503`
- Existing TDC bug report: https://github.com/mims-harvard/TDC/issues/394

Do not replace the displayed 0.922 Top-10 threshold and do not mark the 22-task goal complete until TDC issues an authoritative correction or a legitimate frozen model reaches the displayed threshold.
