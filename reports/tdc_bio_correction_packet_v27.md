# Draft correction request: TDC Bioavailability leaderboard

Status: **prepared, not sent**. External communication requires user approval.

Suggested recipients:

- Add evidence to existing issue #394:
  <https://github.com/mims-harvard/TDC/issues/394>
- Email: `contact@tdcommons.ai`

## Suggested subject

`Bioavailability_Ma leaderboard appears to contain Pgp_Broccatelli rows`

## Suggested message

Hello TDC maintainers,

We independently reproduced and extended the evidence in issue #394 concerning
the current `Bioavailability_Ma` leaderboard, and would appreciate an
authoritative check or correction.

1. The current Bio page contains all 16 Pgp leaderboard entries with exactly
   the same `(model, mean, std)` values. The two pages have set containment
   16/16 and Jaccard 0.80, although the data sets have 640 and 1,212 molecules,
   respectively.
2. MiniMol's upstream performance table (June 2024) reports Bioavailability
   `0.689 +/- 0.020` and Pgp `0.942 +/- 0.002`. The current Bio page displays
   the Pgp value `0.942 +/- 0.002` for MiniMol.
3. The peer-reviewed 2026 audit by Koleiev et al. (DOI
   `10.1021/acs.jcim.6c00819`, Table 2) independently reproduced MapLight+GNN
   on Bioavailability at `0.731 +/- 0.007`, not the displayed
   `0.938 +/- 0.002`.
4. Using the audit repository at commit
   `3e5472ce159a29b872219ffaaa4a11b4a2660f77`, its pinned upstream
   MapLight+GNN code and checkpoint, and the official five-seed TDC protocol,
   we independently obtained the same `0.731 +/- 0.007`. As a positive
   control, the same implementation obtained `0.434 +/- 0.009` on CYP2C9
   substrate versus the displayed MapLight+GNN value `0.437 +/- 0.008`.

Could you please confirm:

- whether the Bioavailability table accidentally imported Pgp records;
- the authoritative corrected Bioavailability values and ranks;
- the dataset/split version and checksums against which corrected scores apply;
- whether all affected rows will be recomputed or restored from a versioned
  source.

We have not replaced the displayed threshold in our own scorecard. The full
machine-readable evidence and hashes are in
`reports/bio_leaderboard_discrepancy_v27.json`; the readable summary is
`reports/bio_leaderboard_discrepancy_v27.md`.

Thank you.

## Attachment checklist

- `reports/bio_leaderboard_discrepancy_v27.json`
- `reports/bio_leaderboard_discrepancy_v27.md`
- `reports/leaderboard_snapshot_2026-08-17.json`
- `runs/v27-official/Bioavailability_Ma/final/results.json`
- `runs/v27-official/CYP2C9_Substrate_CarbonMangels/final/results.json`
- `runs/v27-maplight-gnn-confirmation/result.json`

No issue comment or email has been created by the agent.
