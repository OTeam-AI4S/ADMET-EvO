# ADMET-EvO

ADMET-EvO is a reproducible workflow for model search and frozen test evaluation on the [Therapeutics Data Commons (TDC) ADMET benchmarks](https://tdcommons.ai/benchmark/admet_group/overview/).

It combines molecular representations, classical and gradient-boosted models, multi-seed validation, optional prediction ensembling, and a test-set firewall. Model selection happens on the training and validation splits. The selected configuration is then frozen and evaluated on the test split with at least five seeds.

## Workflow

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

## Basic commands

List the available benchmark names:

```bash
admet-agent list
```

List the registered molecular representations, models, and allowed hyperparameters:

```bash
admet-agent tools
```

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
  --robust-top-k 3 \
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
| `--robust-top-k N` | Number of candidates promoted to multi-seed validation. |
| `--no-ensemble` | Disable validation-selected prediction blending. |
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
