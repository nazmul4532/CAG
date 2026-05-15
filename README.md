# Mail-CAG

Mail-CAG is a cyclic adversarial game for robust email threat detection. The
current experiments train an ALBERT-based classifier, generate adversarial email
variants, fold successful attacks back into training, and repeat the loop across
rounds.

## Repository Layout

- `NLP_Adverserial_Game_v10.ipynb`: latest full cyclic adversarial game notebook.
- `NLP_Adverserial_Game_v10_dummy.ipynb`: latest small/debug cyclic run notebook.
- `modified_attacker.py`: patched TextAttack attacker used by the notebooks.
- `validation_set_0.csv`: small validation seed tracked in Git.
- `scripts/bootstrap.sh`: creates the Python environment and installs packages.
- `scripts/download_models.sh`: pulls local Ollama models and Hugging Face base models.
- `reports/notebook_exports/`: latest exported notebook HTML reports.
- `archive/past_attempts/`: older notebooks, older HTML exports, and old root links.
- `data/raw/`: local raw datasets, ignored by Git.
- `outputs/experiments/`: local experiment runs, checkpoints, and generated outputs,
  ignored by Git.
- `docs/reference/`: local reference documents, ignored unless explicitly promoted.

Large raw datasets, local caches, trained checkpoints, and generated experiment
folders are intentionally ignored by Git.

The root-level `CEAS_08.csv`, `albert_adversarial_game_model_v5`,
`albert_adversarial_game_model_v5dummy`, and `albert_adversarial_val_sets_v5dummy`
paths are compatibility symlinks into `data/raw/` and `outputs/experiments/`.
They keep the current notebooks runnable without rewriting old cells.

## Bootstrap

From the repository root:

```bash
scripts/bootstrap.sh
```

By default this creates or updates the original conda environment named
`nlp_game` with Python 3.12.
Override the defaults when needed:

```bash
ENV_NAME=mail-cag-dev PYTHON_VERSION=3.12 scripts/bootstrap.sh
```

Activate the environment:

```bash
conda activate nlp_game
```

## Download Models

Make sure Ollama is running:

```bash
systemctl status ollama
```

Then pull the default local LLMs and Hugging Face models:

```bash
scripts/download_models.sh
```

The default Ollama set is chosen for a 16 GB RTX 4060 Ti:

- `qwen3:8b`
- `qwen3:14b`
- `gemma3:12b-it-qat`
- `mistral-nemo:12b`

Override it with:

```bash
OLLAMA_MODELS="qwen3:8b gpt-oss:20b" scripts/download_models.sh
```

The default Hugging Face downloads are:

- `albert-base-v2`
- `sentence-transformers/all-MiniLM-L6-v2`

Override them with:

```bash
HF_MODELS="albert-base-v2" scripts/download_models.sh
```

## Data

Place the raw CEAS dataset at:

```text
CEAS_08.csv
```

That file is intentionally ignored because it is large/raw data. If we later want
reproducible dataset versioning, use DVC or Git LFS instead of regular Git.

## Git Hygiene

Tracked files should be source, notebooks, small seed data, scripts, and paper
artifacts. Keep these out of Git:

- raw datasets
- local environments and caches
- model checkpoints and trained model binaries
- generated experiment folders such as `albert_adversarial_game_model_v*`
  and `albert_adversarial_val_sets_v*`
- machine-specific notes or unrelated documents

If a generated CSV becomes a paper artifact, move it into a clearly named
tracked folder such as `paper_artifacts/` instead of committing the whole run
directory.
