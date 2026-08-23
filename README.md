# ADMET-EvO

ADMET-EvO is a reproducible workflow for model search and frozen test evaluation on the [Therapeutics Data Commons (TDC) ADMET benchmarks](https://tdcommons.ai/benchmark/admet_group/overview/).

It combines molecular representations, classical and gradient-boosted models, multi-seed validation, optional prediction ensembling, and a test-set firewall. Model selection happens on the training and validation splits. The selected configuration is then frozen and evaluated on the test split with at least five seeds.

The original three-task technical report remains in `MANUSCRIPT.md`. The
evidence-backed 22-task v2.4 working paper, including negative results and
explicit incomplete claims, is in `MANUSCRIPT_V24.md`.

The current repository freeze, included evidence, excluded local artifacts,
and exact claim boundaries are summarized in
[`docs/release_status_2026-08-23.md`](docs/release_status_2026-08-23.md). The
compact paper-facing result map is
[`reports/FINAL_EVIDENCE_INDEX.md`](reports/FINAL_EVIDENCE_INDEX.md).

## Workflow

The v3 research loop is restricted to the canonical 22 TDC ADMET endpoints. It
adds endpoint contracts, multi-dimensional hardness diagnosis, auditable data
curation actions, effect-first action routing, a hash-chained evidence ledger,
and a standardized autonomy scorecard. See
[`docs/tdc22_research_agent_v3.md`](docs/tdc22_research_agent_v3.md).

```mermaid
flowchart LR
    A[Load a TDC benchmark] --> B[Profile the training and validation data]
    B --> C[Generate bounded experiment candidates]
    C --> D[Train and score on validation data]
    D --> E[Compare candidates across validation seeds]
    E --> F[Select a model or prediction ensemble]
    F --> G[Freeze configuration and write SHA-256 lock]
    G --> H[Retrain and evaluate with five or more seeds]
    H --> I[Save scores, predictions, and audit records]
    D -->|experiment history| C
```

The workflow has two separate commands:

- `search` can access only training and validation data. It profiles the endpoint, evaluates candidate pipelines, repeats promising candidates across validation seeds, and writes a finalist configuration.
- `finalize` accepts a serialized finalist, writes a freeze lock, trains it independently for every requested seed, and calls TDC's official test evaluator. A different finalist cannot be reused in the same finalization directory.
- Frozen `recipe` finalists support preregistered heterogeneous raw/rank blends, including full fine-tuned Uni-Mol2, transformed-target experts, and OOF-selected sub-ensembles. The executor fits on `train+valid` and receives only test SMILES at prediction time.

## Installation

Python 3.10 through 3.12 is supported. Clone the repository and create the CPU environment:

```bash
git clone https://github.com/YilinWang33/ADMET-EvO.git
cd ADMET-EvO
conda env create -f environment-cpu.yml
conda activate admet-agent
```

For CUDA-enabled XGBoost, create the GPU environment from a GPU node whose driver is compatible with the installed CUDA runtime:

```bash
conda env create -f environment-gpu.yml
conda activate admet-agent-gpu
```

The TDC datasets are downloaded by PyTDC on first use and cached under the directory passed to `--data-dir`.

### Optional Uni-Mol2 representation

Install the optional Uni-Mol2 dependencies in the active CPU or GPU environment:

```bash
python -m pip install -e '.[unimol]'
```

The registered `unimol2` representation uses the frozen official 84M encoder and
its molecular CLS embedding. It can be combined with the existing 2D features in
an experiment specification, for example:

```json
{
  "representations": ["unimol2", "morgan", "rdkit2d"],
  "model": "extra_trees"
}
```

`admet-agent tools` reports whether the optional dependency is available. When it
is installed, the rule planner includes one Uni-Mol2 candidate in its initial
search. The official pretrained weights are downloaded on first use and the
resulting embeddings are stored in the run directory's feature cache. Use
`--device cuda` for practical benchmark-scale extraction; CPU inference is
supported but can be slow.

## Basic commands

List the available benchmark names:

```bash
admet-agent list
```

List the registered molecular representations, models, skills, and allowed hyperparameters:

```bash
admet-agent tools
```

### External-data skill

`admet-data-collector` is packaged with the Agent and appears in `admet-agent
tools`. It keeps endpoint semantics, source evidence, licensing, contamination
and modeling admission as separate gates. Discovery never downloads or merges
records automatically, and held-out labels are forbidden.

Inspect the immutable skill/resource manifest and discover registered source
candidates:

```bash
admet-agent data-skill info
admet-agent data-skill discover --task hERG
```

Initialize and audit a versioned evidence bundle:

```bash
admet-agent data-skill init \
  --task hERG \
  --out data/external-bundles/herg-v1 \
  --si-choice include

admet-agent data-skill audit \
  --bundle data/external-bundles/herg-v1 \
  --strict

admet-agent data-skill status \
  --bundle data/external-bundles/herg-v1
```

An evaluator-authorized, structure-only held-out file additionally requires
`--heldout-authorized`. A bundle cannot become direct training data until its
audit recommends `modeling_ready_discovery`, the versioned manifest is explicitly
promoted, scientific review is recorded, and held-out contamination auditing is
complete.
When the research planner selects an external-data action, `search` writes a
`data_skill_route_seed_<seed>.json` packet beside the research plan; it does not
silently admit the candidate records.

Validate an experiment specification without training it:

```bash
admet-agent validate-spec examples/experiment.json
```

All commands also expose their complete argument list through `--help`:

```bash
admet-agent search --help
admet-agent finalize --help
```

## Run a search

The default rule-based search does not require an API key:

```bash
admet-agent search \
  --task AMES \
  --brain rules \
  --seed 1 \
  --rounds 2 \
  --validation-seeds 1,2,3,4,5 \
  --robust-top-k 6 \
  --device cpu \
  --workers 8 \
  --data-dir data \
  --run-dir runs/ames-search
```

Important search arguments:

| Argument | Meaning |
|---|---|
| `--task NAME` | Select one benchmark. Repeat the argument to search several benchmarks. |
| `--all` | Search every registered benchmark instead of using `--task`. |
| `--seed N` | Seed used for the initial search round. |
| `--rounds N` | Number of propose, evaluate, and refine rounds. |
| `--validation-seeds LIST` | Comma-separated seeds used to check the strongest candidates for stability. |
| `--robust-top-k N` | Number of model-diverse candidates promoted to multi-seed validation. |
| `--no-ensemble` | Disable validation-selected prediction blending. |
| `--selection-policy score\|lcb\|cost-aware-lcb` | Select by mean score, reliability-adjusted LCB, or LCB minus normalized log compute cost. |
| `--uncertainty-penalty FLOAT` | Multiplier on cross-seed validation standard deviation for LCB selection. |
| `--cost-penalty FLOAT` | Native-metric penalty on normalized log compute for cost-aware LCB selection. |
| `--effect-priority FLOAT` | Weight on expected scientific effect when routing research actions; default `0.80`. |
| `--brain rules\|openai\|hybrid` | Choose the candidate planner. `rules` is local and is the default. |
| `--device auto\|cpu\|cuda` | Select the training device. |
| `--workers N` | Set the number of parallel CPU workers passed to supported models. |
| `--data-dir PATH` | Set the PyTDC data and feature-cache directory. |
| `--run-dir PATH` | Set the directory for experiment history and generated artifacts. |

To search several benchmarks in one invocation, repeat `--task`:

```bash
admet-agent search \
  --task AMES \
  --task PPBR_AZ \
  --validation-seeds 1,2,3 \
  --run-dir runs/multi-search
```

When search succeeds, the selected configuration for each benchmark is written to:

```text
<run-dir>/tasks/<task>/finalist_seed_<seed>.json
```

That JSON file is already in the task-to-configuration mapping format required by `finalize`.

## Finalize a selected configuration

Finalization requires at least five unique seeds. Use a new `--run-dir` whenever the finalist changes:

```bash
admet-agent finalize \
  --task AMES \
  --seeds 1,2,3,4,5 \
  --finalists runs/ames-search/tasks/AMES/finalist_seed_1.json \
  --device cpu \
  --workers 8 \
  --data-dir data \
  --run-dir runs/ames-final
```

For the v2.4 heterogeneous finalists, generate the validation-only freeze bundle first:

```bash
python scripts/build_v24_freeze_manifest.py
```

This writes `runs/v24-freeze/finalists.mapping.json`, which is directly accepted by
`--finalists`. The accompanying manifest hashes all validation evidence and runtime
sources and remains `ready_for_official_test: false` until a pinned GPU smoke test is
recorded.

For multiple benchmarks, pass a single JSON file containing one entry per benchmark and repeat `--task`:

```bash
admet-agent finalize \
  --task AMES \
  --task PPBR_AZ \
  --seeds 1,2,3,4,5 \
  --finalists results/top1_finalists.json \
  --run-dir runs/multi-final
```

The finalization output is:

```text
<run-dir>/
├── environment.json
├── memory.jsonl
├── experiments.sqlite
└── final/
    ├── freeze.json
    ├── results.json
    ├── predictions_seed_1.npz
    ├── predictions_seed_2.npz
    ├── predictions_seed_3.npz
    ├── predictions_seed_4.npz
    └── predictions_seed_5.npz
```

- `freeze.json` contains the exact finalist mapping and its SHA-256 hash.
- `results.json` contains the seeds, TDC metrics, finalist configuration, hash, and per-run audit records.
- `predictions_seed_*.npz` contains the test predictions produced by each independent seed.
- `environment.json` records Python, platform, device, worker, and GPU information.

## Optional model planner

The local `rules` planner is sufficient to run the complete workflow. An OpenAI-compatible planner can optionally propose candidates during `search`; it is never used by `finalize`.

Create the credential file outside the repository:

```bash
mkdir -p ~/.config/admet-agent
chmod 700 ~/.config/admet-agent
```

Save the following as `~/.config/admet-agent/openai.env` and restrict its permissions with `chmod 600 ~/.config/admet-agent/openai.env`:

```dotenv
OPENAI_API_KEY=your-key
# Optional for an OpenAI-compatible endpoint:
# OPENAI_BASE_URL=https://example.com/v1
```

Run either the API planner alone or the hybrid planner:

```bash
admet-agent search \
  --task AMES \
  --brain hybrid \
  --model gpt-5.6-sol \
  --api-config ~/.config/admet-agent/openai.env \
  --run-dir runs/ames-hybrid
```

Only task profiles, registered tool definitions, and validation history are sent to the planner. Credentials are not copied into the run directory. Request identifiers and token counts are recorded in `<run-dir>/api_usage.jsonl`.

## Reproduce the frozen examples

For the closest dependency match to the checked-in frozen configurations, use the pinned reproduction requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-repro.txt
python -m pip install -e . --no-deps
DEVICE=cuda WORKERS=8 ./scripts/reproduce_top1.sh
```

The script runs each frozen configuration with seeds 1 through 5, saves the predictions, evaluates them through TDC, and verifies both the recorded metrics and the freeze hashes. `DEVICE=cpu` can be used for a portable run; CUDA is needed to match configurations that use GPU XGBoost.

## Tests

```bash
python -m pytest -q
```

The tests cover experiment validation, task metadata, test-access guards, multi-seed selection, ensemble behavior, finalist freezing, and reproducibility artifacts.

For AUROC, AUPRC, and Spearman tasks, ensemble search evaluates both weighted
raw predictions and equal-weight percentile-rank blends. Rank blending removes
incompatible probability scales between model families. Multi-seed promotion
prefers model diversity before filling the remaining slots by validation score.

## More documentation

- [Architecture and protocol](docs/v2_architecture.md)
- [Technical report](MANUSCRIPT.md)
- [Optional planner contract](docs/gpt56_integration.md)
- [TDC submission notes](docs/tdc_submission.md)
- [Example finalist mapping](examples/finalists.example.json)

## Contact

Yilin Wang, Valhalla Technology

## License

ADMET-EvO is released under the [MIT License](LICENSE).
