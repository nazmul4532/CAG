# Mail-CAG

Mail-CAG is a cyclic adversarial game for robust phishing-email detection.

The current project asks:

1. Does LLM-generated cyclic adversarial training improve ALBERT robustness
   compared with clean ALBERT?
2. Does rewriting only phishing emails behave differently from rewriting both
   phishing and benign emails?

## Current Choices

We are using a simple three-model comparison:

- **Model A**: clean ALBERT baseline.
- **Model B**: cyclic ALBERT with LLM rewrites for phishing emails only.
- **Model C**: cyclic ALBERT with LLM rewrites for both phishing and benign
  emails.

For Model B/C, the defender keeps learning across rounds:

```text
round 1 starts from albert-base-v2
round 2 starts from round_1/model
round 3 starts from round_2/model
```

The attacker for the first implementation is local Ollama with `qwen3:8b`.
`qwen3:14b` is kept as a stronger but slower fallback/comparison model, and
`qwen3:4b` is the smallest fallback. TextFooler, PWWS, and DeepWordBug are kept
for later held-out evaluation, not training-data generation.

The configs currently use 6% of CEAS for smoke testing:

```yaml
sample_frac_per_label: 0.06
```

Move this toward `1.0` only after the pipeline is behaving.

## Repository Layout

- `mail_cag.py`: friendly command-line entrypoint.
- `START_HERE.md`: short guide for getting oriented.
- `mail_cag_project/configs/`: Model A/B/C experiment configs.
- `mail_cag_project/src/mail_cag/`: reusable Python code.
- `mail_cag_project/docs/literature_scan.md`: quick related-work notes.
- `docs/approaches.md`: method notes and legacy v4/v5 references.
- `legacy_workspace/`: old notebooks, old outputs, and local artifacts.
- `scripts/bootstrap.sh`: creates/updates the Python environment.
- `scripts/download_models.sh`: downloads Ollama and Hugging Face models.

Large datasets, checkpoints, caches, and generated runs are ignored by Git.

## Setup

From the repository root:

```bash
cd ~/Documents/NLP
conda activate nlp_game
git pull
```

If the environment needs to be rebuilt:

```bash
scripts/bootstrap.sh
conda activate nlp_game
```

Make sure Ollama is running:

```bash
systemctl status ollama
ollama list
```

Pull the recommended local models if needed:

```bash
OLLAMA_MODELS="qwen3:8b qwen3:14b" scripts/download_models.sh
```

The raw CEAS file should be here:

```text
legacy_workspace/artifacts/data/raw/CEAS_08.csv
```

## Running Experiments

Check the configs without training:

```bash
python mail_cag.py describe model-a
python mail_cag.py describe model-b
python mail_cag.py describe model-c
```

Dry-run before spending GPU time:

```bash
python mail_cag.py run model-a --dry-run --run-id baseline_smoke_001
python mail_cag.py run model-b --dry-run --run-id b_smoke_001
python mail_cag.py run model-c --dry-run --run-id c_smoke_001
```

Run the baseline:

```bash
python mail_cag.py run model-a --run-id baseline_smoke_001
```

Run the cyclic models:

```bash
python mail_cag.py run model-b --run-id b_smoke_001
python mail_cag.py run model-c --run-id c_smoke_001
```

Resume after interruption:

```bash
python mail_cag.py run model-b --resume --run-id b_smoke_001
python mail_cag.py run model-c --resume --run-id c_smoke_001
```

Each run gets its own folder:

```text
runs/model_b_llm_phishing_only/b_smoke_001/
```

Each round keeps:

```text
round_N/model/               final model for the round
round_N/checkpoint_current/  latest epoch checkpoint, overwritten each epoch
round_N/training_data.csv    data used for that round
round_N/generated_rewrites.csv
```

During LLM rewriting, progress should print as:

```text
rewritten emails: 1/200
rewritten emails: 2/200
```

## Current Slow Part

Between rounds, Model B/C asks Qwen to rewrite selected emails. Current smoke
settings can still request up to:

```text
200 selected emails x 3 candidates = 600 rewrites per between-round step
```

If this is too slow, reduce these in the Model B/C config:

```yaml
attacks:
  candidates_per_email: 1
  max_examples_per_round: 25
```

## Future Choices

Near-term:

- Add `python mail_cag.py evaluate`.
- Evaluate Model A/B/C with TextFooler, PWWS, and DeepWordBug.
- Add analysis notebooks that read saved `runs/` outputs.
- Add better run summaries and metrics CSV files.

Later:

- Run with `sample_frac_per_label: 1.0`.
- Compare more local LLM attackers, such as `dolphin3`.
- Add cloud attackers through provider slots for Groq, OpenAI, or Gemini.
- Consider DVC or Git LFS if dataset/run artifact versioning becomes important.

## Git Hygiene

Keep these out of Git:

- raw datasets
- local environments and caches
- model checkpoints and trained model binaries
- generated `runs/`
- unrelated local documents

If a generated CSV becomes a paper artifact, move it into a clearly named
tracked folder such as `paper_artifacts/`.
