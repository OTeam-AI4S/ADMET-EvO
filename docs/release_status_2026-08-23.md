# ADMET-EvO repository freeze — 2026-08-23

This repository is the compact, auditable research freeze. It intentionally
contains source code, tests, protocols, aggregate analyses, paper assets, and
small evidence tables. It intentionally does not contain local environments,
model/data caches, raw run trees, downloaded third-party repositories,
molecule-level training exports, or blind challenge submissions.

## Evidence that is retained

- The frozen TDC ADMET-22 application scorecard: estimated 18/22 Top-5 and
  21/22 Top-10 under the displayed-page snapshot, with a combined mean rank of
  4.545. These are snapshot-insertion estimates, not live leaderboard claims.
- The complete application and process evidence manifests, including negative
  router, memory, AutoML-control, and policy results.
- The EC-VOI v5 public cross-source action screen and final policy analysis.
  The v5 policy failed its preregistered promotion gate; this negative result
  is retained without reinterpretation.
- The frozen EC-VOI v6 mechanism protocol and its disclosed v5 development
  replay. The replay selected zero identifiable diagnostic states and exactly
  fell back to the no-diagnosis policy. It is not independent confirmation.
- The OpenADMET CYP B1 selection, certification, applicability, and freeze
  manifests. B1 passed the internal certification gates, but blind performance
  and rank remain unknown because no leaderboard readout is included.
- The external-data skill, test firewall, cohort governance, data stewardship,
  self-correction, and conditional committee implementations with their tests.

See `reports/FINAL_EVIDENCE_INDEX.md` for the compact paper-facing map.

## Artifacts that are not published in Git

The following are omitted because they are large, regenerable, private to the
execution environment, duplicated, or should remain sealed:

- `runs/`, `cache/`, `.venvs/`, `external/`, and `logs/`;
- generated archives and reviewer bundles under `release/`;
- nested third-party working copies under `vendor/`; the small, license-bearing
  MapLight/TDC audit snapshot required by reproduction tests is retained;
- molecule-level training/bridge/firewall CSV exports;
- blind OpenADMET submission CSV files;
- smoke runs, aborted payloads, and live-status files.

Their compact manifests, checksums, protocols, and final aggregate analyses
are retained where scientifically necessary. Omission from Git does not imply
that a result passed a promotion gate.

## Current claim boundary

The defensible application claim is that ADMET-EvO meets the project-level
TDC-22 qualification target on 18 Top-5 tasks and 21 Top-10 tasks under the
frozen snapshot. The defensible systems claim is that it implements a bounded,
auditable ADMET research workflow with explicit data, test-access, budget,
negative-result, and promotion controls.

The repository does not claim that the current research policy is predictively
superior to expanded AutoML. The formal three-arm predictive control did not
pass its preregistered noninferiority gate. It also does not claim EC-VOI v5 or
v6 promotion, independent expert validation, OpenADMET blind performance, or a
22/22 verified Top-10 result.

## Reproduction entry points

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
admet-agent tools
```

The large benchmark experiments are regenerated through the checked-in
scripts and require separately obtained public datasets/checkpoints according
to their upstream licenses. No held-out labels or credentials are included.
