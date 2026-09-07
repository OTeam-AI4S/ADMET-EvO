# Repository layout

The Git repository contains source code, tests, protocols, compact evidence,
and manuscript material. Machine-local datasets, model weights, caches, and
generated outputs are deliberately excluded.

| Path | Purpose |
| --- | --- |
| `admet_agent/` | Installable Python package and reusable research logic. |
| `tests/` | Unit, safety, protocol, and evidence-integrity tests. |
| `scripts/` | Reproducible experiment, audit, summarization, and cluster entry points. |
| `scripts/slurm/` | Slurm launchers for extended cohort experiments. |
| `cluster/` | Legacy benchmark-scale launchers retained for compatibility. |
| `docs/` | Architecture, release, protocol, and maintenance documentation. |
| `paper/` | Manuscript sources, references, and supplementary material. |
| `reports/` | Curated compact evidence already reviewed for publication. New generated reports are ignored by default. |
| `skills/` | Packaged ADMET data-collection skill resources. |
| `environment-*.yml` | Reproducible CPU, GPU, docking, and external-data environments. |

## Core workflow boundaries

The package owns reusable policy and modeling behavior. Scripts compose those
APIs into immutable research protocols. Tests must target package behavior or a
specific protocol contract; experiment outputs do not belong in tests.

Held-out labels are available only to final evaluation paths. External-data
workflows may load the held-out structure column solely to construct identity
or similarity firewalls. Every such path records which columns were loaded and
asserts that labels were inaccessible.

## Local-only artifacts

The following stay in the working directory but are not committed:

- `data/`, `external/`, `runs/`, `outputs/`, and `checkpoints/`;
- `.cache/`, `.conda/`, `.venv/`, `.venvs/`, and Python caches;
- downloaded model assets and third-party working copies;
- generated figures, office documents, and large serialized arrays/models;
- the separate, artifact-heavy `discovery_research/` workspace, pending its own
  audited source release.

Use `git status --ignored --short` when auditing why an artifact is excluded.
Compact, reviewed evidence can be added explicitly when it is necessary to
support a published claim.
