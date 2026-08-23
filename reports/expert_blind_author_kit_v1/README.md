# ADMET-EvO independent blind-task author kit v1

Status: **blank operational template, not experimental evidence**. No case in
this directory may be counted as independently authored or blind.

## Who should use this kit

- Two ADMET/cheminformatics task authors who did not develop ADMET-EvO.
- One custodian who keeps every completed hidden packet outside the Agent
  workspace until scoring and reveal.
- Two scorers who did not see system identities or outputs during case
  authorship.

Agent developers may explain the schema and run the supplied validation tools,
but may not write or edit the hidden fault, gold diagnosis, acceptable actions
or unsafe actions.

## Deadline for the one-week experiment

- Hidden cases authored and commitment hashes frozen: 2026-08-23 UTC.
- Anonymous system runs completed: 2026-08-25 UTC.
- Blinded scores and adjudication completed: 2026-08-27 UTC.

If those independence conditions are not met, report the cohort as pending.
Internally authored examples may be called engineering stress tests only.

## Author workflow

1. Copy `hidden_case_template.json` to a machine or account not accessible to
   ADMET-EvO. Never fill a hidden case inside this repository.
2. Create at least one case for each of the twelve categories in
   `../expert_blind_scientific_task_protocol_v1.md` across the two authors.
3. Make the discovery view sufficient to diagnose the issue, but do not expose
   a gold diagnosis, answer-like field name or unsafe-action list.
4. Freeze the gold packet before releasing the public packet.
5. The custodian runs:

   ```bash
   python scripts/seal_expert_blind_tasks.py seal \
     --hidden hidden-case.json \
     --public public-case.json \
     --commitment case-commitment.json
   ```

6. Release only `public-case.json` and `case-commitment.json` to the systems.
7. Randomize system identities before giving outputs to scorers.
8. Scorers fill rows following `blinded_score_template.json`. Complete scoring
   before the custodian reveals system identities.
9. After scoring, verify the reveal and run the frozen analyzer described in
   the main protocol.

## Minimum case quality checklist

- Endpoint species, biological system, assay type, units, relation operators
  and metric are explicit.
- Structures and rows have stable identifiers.
- At least one scientifically defensible matched treatment/control is in gold.
- At least one plausible but unsafe or invalid action is in gold.
- The correct outcome may be repair, replicate, abstain or retain/stop.
- Clean controls are genuinely clean; every case must not contain a fault.
- Permitted external resources and time allowance are explicit.
- Gold is based on scientific validity, not on whether a particular model wins.
- No test or leaderboard label is visible in the discovery packet.
- The author attestation is true, not merely checked for schema compliance.

## Files

- `hidden_case_template.json`: illustrative schema; fill only outside the
  workspace.
- `blinded_score_template.json`: frozen seven-component rubric row format.
- `system_output_contract.md`: the anonymous response fields every compared
  system must return.
