# Mail-CAG

Mail-CAG is a cyclic adversarial game for robust email threat detection. The
current experiments train an ALBERT-based classifier, generate adversarial email
variants, fold successful attacks back into training, and repeat the loop across
rounds.

## Repository Layout

- `START_HERE.md`: friendly guide for opening the project tomorrow.
- `mail_cag.py`: friendly command-line entrypoint.
- `mail_cag_project/`: clean script-first workspace for the next version of the
  paper experiments. The main config is
  `mail_cag_project/configs/cyclic_llm_both_labels.yaml`.
- `legacy_workspace/`: original notebook-era files and local legacy artifacts.
- `validation_set_0.csv`: small validation seed tracked in Git.
- `docs/approaches.md`: notes on the current Model A/B/C experiment design and
  older v4/v5 references.
- `mail_cag_project/docs/literature_scan.md`: quick notes on nearby LLM
  phishing/adversarial-training work.
- `scripts/bootstrap.sh`: creates the Python environment and installs packages.
- `scripts/download_models.sh`: pulls local Ollama models and Hugging Face base models.

Large raw datasets, local caches, trained checkpoints, and generated experiment
folders are intentionally ignored by Git.

The old root-level `CEAS_08.csv` and `albert_adversarial_*` compatibility
symlinks now live under `legacy_workspace/root_links/`. The actual local data and
outputs live under `legacy_workspace/artifacts/`.

## Bootstrap

Start with:

```bash
conda activate nlp_game
python mail_cag.py describe
```

That describes the main Model C / both-label LLM cyclic setup. Compare against:

```bash
python mail_cag.py describe baseline
python mail_cag.py describe model-b
python mail_cag.py describe model-c
```

Dry-run a training config before spending GPU time:

```bash
python mail_cag.py run model-a --dry-run
python mail_cag.py run model-b --dry-run
python mail_cag.py run model-c --dry-run
```

Run the clean baseline:

```bash
python mail_cag.py run model-a
```

Run one LLM cyclic model:

```bash
python mail_cag.py run model-b
# or
python mail_cag.py run model-c
```

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
OLLAMA_MODELS="qwen3:14b qwen3:8b dolphin3" scripts/download_models.sh
```

`dolphin3` is optional. It can be useful later as a second local rewrite model,
but keep the first serious run on Qwen so the comparison stays simple.

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
legacy_workspace/artifacts/data/raw/CEAS_08.csv
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
