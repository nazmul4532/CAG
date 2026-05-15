# Mail-CAG

Mail-CAG is a cyclic adversarial game for robust phishing-email detection.

The current project currently uses ALBERT as the defender, but the pipeline is
written around a generic text classifier so we can swap in another defender
later.

The current project asks:

1. Does LLM-generated cyclic adversarial training improve defender robustness
   compared with a clean defender baseline?
2. Does rewriting only phishing emails behave differently from rewriting both
   phishing and benign emails?

## Current Choices

We are using a simple three-model comparison. The defender is currently
`albert-base-v2`, but the code is arranged so another Hugging Face text
classifier can be used later.

- **Model A**: clean defender baseline.
- **Model B**: cyclic defender with LLM rewrites for phishing emails only.
- **Model C**: cyclic defender with LLM rewrites for both phishing and benign
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
The LLM prompt asks for exactly one plain rewritten email, not JSON, while
preserving language, URL behavior, spam/phishing style, and suspicious intent.

The configs currently use 6% of CEAS for smoke testing:

```yaml
sample_frac_per_label: 0.06
```

Move this toward `1.0` only after the pipeline is behaving. The configs use a
prepared CEAS file containing English emails up to 1600 tokens under the current
defender tokenizer. The preparation step also drops rows with obvious
non-English writing systems such as CJK, Cyrillic, Arabic, Hebrew, Devanagari,
Thai, Japanese, and Hangul.

The defender then trains/evaluates with:

```yaml
model:
  base_model: albert-base-v2
  max_length: 512
```

If we later replace ALBERT with another classifier, change `base_model` and
`max_length`, then regenerate the prepared dataset with that classifier's
tokenizer.

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
scripts/download_required_models.sh
```

The raw CEAS file should be here:

```text
legacy_workspace/artifacts/data/raw/CEAS_08.csv
```

Prepare the English/1600-token CEAS file:

```bash
python scripts/prepare_ceas.py
```

That writes:

```text
data/processed/CEAS_08_en_1600.csv
```

Describe the prepared dataset:

```bash
python scripts/describe_prepared_data.py
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
python mail_cag.py run model-a --dry-run --run-id a_smoke_001
python mail_cag.py run model-b --dry-run --run-id b_smoke_qwen8b_quality_001
python mail_cag.py run model-c --dry-run --run-id c_smoke_qwen8b_quality_001
```

Run the three smoke tests from the beginning:

```bash
python mail_cag.py run model-a --run-id a_smoke_001
python mail_cag.py run model-b --run-id b_smoke_qwen8b_quality_001
python mail_cag.py run model-c --run-id c_smoke_qwen8b_quality_001
```

Resume after interruption:

```bash
python mail_cag.py run model-a --resume --run-id a_smoke_001
python mail_cag.py run model-b --resume --run-id b_smoke_qwen8b_quality_001
python mail_cag.py run model-c --resume --run-id c_smoke_qwen8b_quality_001
```

Evaluate the latest completed round on the clean eval split:

```bash
python mail_cag.py evaluate model-a --run-id a_smoke_001
python mail_cag.py evaluate model-b --run-id b_smoke_qwen8b_quality_001
python mail_cag.py evaluate model-c --run-id c_smoke_qwen8b_quality_001
```

Evaluate a specific round:

```bash
python mail_cag.py evaluate model-b --run-id b_smoke_qwen8b_quality_001 --round 1
```

Each run gets its own folder:

```text
runs/model_b_llm_phishing_only/b_smoke_qwen8b_quality_001/
```

Each round keeps:

```text
round_N/model/               final model for the round
round_N/checkpoint_current/  latest epoch checkpoint, overwritten each epoch
round_N/training_data.csv    data used for that round
round_N/generated_rewrites.csv
round_N/rewrite_quality.csv
round_N/rewrite_quality_summary.json
```

During LLM rewriting, progress should print as:

```text
rewriting emails:  12%|████▌              | 24/200
```

## Current Slow Part

Between rounds, Model B/C asks Qwen to rewrite selected emails. Current smoke
settings can still request up to:

```text
200 selected emails x 1 candidate = 200 rewrites per between-round step
```

Qwen only receives the same text window that the current defender can see. With
today's ALBERT config, that is 512 tokens. This keeps rewriting aligned with the
classifier and avoids wasting generation on long email tails that the defender
truncates away.

After a rewrite batch finishes, the runner writes a small quality report beside
the generated rewrites. It checks whether rewrites changed, preserved URL
behavior, avoided non-English scripts/structured output, and how much the
round's defender confidence dropped on the rewritten email. This is diagnostic
only; it does not silently skip or filter rewrites.

If this is too slow, reduce these in the Model B/C config:

```yaml
attacks:
  candidates_per_email: 1
  max_examples_per_round: 25
```

## Future Choices

Near-term:

- Evaluate Model A/B/C with TextFooler, PWWS, and DeepWordBug.
- Add analysis notebooks that read saved `runs/` outputs.
- Add better run summaries and metrics CSV files.
- Compare rewrite-quality summaries between Model B and Model C.

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
