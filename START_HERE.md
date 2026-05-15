# Start Here

Mail-CAG is a research project about cyclic adversarial training for phishing
email detection.

The short version:

- ALBERT is good at detecting clean phishing emails.
- We want to test if ALBERT becomes more robust when trained against
  adversarial phishing emails generated from previous ALBERT rounds.
- The old notebooks/results are preserved in `legacy_workspace/`.
- The cleaner project we will build from now lives in `mail_cag_project/`.

## First Command

From the repository root:

```bash
conda activate nlp_game
python mail_cag.py describe
```

That command does not train anything. It checks the default Model C setup and
tells you which data/results already exist.

Other useful checks:

```bash
python mail_cag.py describe baseline
python mail_cag.py describe model-b
python mail_cag.py describe model-c
```

Before starting a real training run, check the selected config:

```bash
python mail_cag.py run model-a --dry-run
python mail_cag.py run model-b --dry-run
python mail_cag.py run model-c --dry-run
```

Start the clean baseline first:

```bash
python mail_cag.py run model-a
```

Then run one cyclic model:

```bash
python mail_cag.py run model-b
# or
python mail_cag.py run model-c
```

Outputs go under `runs/`, which is intentionally ignored by Git.
Each round keeps its final model in `round_N/model/`. The current round also
keeps one rolling epoch checkpoint in `round_N/checkpoint_current/`; that folder
is overwritten after every epoch.
For Model B/C, each round starts from the previous round's saved ALBERT model.

Each run is isolated inside a timestamped run folder. Use `--run-id` when you
want a memorable name:

```bash
python mail_cag.py run model-b --run-id smoke_001
python mail_cag.py run model-b --resume --run-id smoke_001
```

## Mental Model

Use this map:

```text
mail_cag.py                 friendly command to run things
mail_cag_project/configs/   experiment settings; use cyclic_llm_both_labels.yaml first
mail_cag_project/src/       reusable Python code
mail_cag_project/scripts/   lower-level scripts
legacy_workspace/           old notebooks, old results, local big files
```

## The Path Forward

We are moving forward with **LLM-based budgeted cyclic training**.

There are three main models:

- **Model A**: clean ALBERT baseline.
- **Model B**: cyclic ALBERT with LLM rewrites for phishing emails only.
- **Model C**: cyclic ALBERT with LLM rewrites for both phishing and benign emails.

The main comparison is:

```text
Model A vs Model B vs Model C
```

This answers two questions:

1. Does LLM-generated cyclic adversarial training help compared with clean ALBERT?
2. Does rewriting only phishing emails behave differently from rewriting both labels?

The old v4 config still exists at:

```text
mail_cag_project/configs/legacy/approach_v4_cumulative.yaml
```

Treat it as reference, not the main path.

## What We Build Next

The next missing pieces are:

1. Improve/inspect the first `python mail_cag.py run` outputs.
2. `python mail_cag.py evaluate`
3. notebooks that only analyze saved results, not contain the whole experiment
   engine

Keep this rule in mind: notebooks are for thinking and reporting; scripts are
for experiments we want to trust.

## Model Providers

Use Ollama/Qwen for the first implementation. `qwen3:14b` is the preferred local
model, with `qwen3:8b` and `qwen3:4b` as smaller fallbacks.

`dolphin3` can be added as an optional local model later. Groq's
`llama-3.3-70b-versatile` is listed in the provider example for future cloud
comparisons, but it should not be part of the first run unless we intentionally
add a cloud-provider experiment.
