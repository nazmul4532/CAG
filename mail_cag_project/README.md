# Mail-CAG Project Workspace

This folder is the script-first workspace for the cyclic adversarial game paper.
The original notebooks and outputs live under `../legacy_workspace/`. This
workspace keeps the current experiment logic easier to run, resume, and explain.

## Why This Exists

The original notebooks are useful records, but they mix code, outputs, partial
reruns, checkpoints, and analysis. This workspace separates those concerns:

- `configs/`: experiment definitions.
- `src/mail_cag/`: reusable experiment code.
- `notebooks/`: analysis notebooks that should call code from `src/`.
- `scripts/`: command-line entrypoints.
- `docs/`: notes on method, drawbacks, and research decisions.
- `tests/`: lightweight checks for the reusable code.

## Main Configs

Use these first:

- `configs/baseline_clean.yaml`: Model A, clean defender reference.
- `configs/cyclic_llm_phishing_only.yaml`: Model B, LLM rewrites phishing only.
- `configs/cyclic_llm_both_labels.yaml`: Model C, LLM rewrites both labels.
- `configs/providers.example.yaml`: provider slots for Ollama now and
  OpenAI/Gemini later.
- `docs/literature_scan.md`: quick map of nearby papers and the narrow gap we
  are targeting.

The old TextAttack-training and v4 configs are kept under `configs/legacy/` for
reference only.

## Current Legacy Inputs

The raw CEAS file still comes from the legacy workspace:

- raw data: `../legacy_workspace/artifacts/data/raw/CEAS_08.csv`

Prepare the current English/token-capped dataset from the repository root:

```bash
python scripts/prepare_ceas.py
python scripts/describe_prepared_data.py
```

That writes `../data/processed/CEAS_08_en_1600.csv` and matching stats files.
Local data/output folders are intentionally ignored by Git.

## Environment

Use the existing environment:

```bash
conda activate nlp_game
```

Then from the repository root:

```bash
PYTHONPATH=mail_cag_project/src python mail_cag_project/scripts/describe_setup.py
```

Or install the clean workspace in editable mode:

```bash
python -m pip install -e mail_cag_project
python mail_cag_project/scripts/describe_setup.py
```

The friendly root command can dry-run, train, resume, and evaluate:

```bash
python mail_cag.py run model-a --dry-run --run-id a_smoke_001
python mail_cag.py run model-b --dry-run --run-id b_smoke_qwen8b_quality_001
python mail_cag.py run model-c --dry-run --run-id c_smoke_qwen8b_quality_001

python mail_cag.py run model-a --run-id a_smoke_001
python mail_cag.py run model-b --run-id b_smoke_qwen8b_quality_001
python mail_cag.py run model-c --run-id c_smoke_qwen8b_quality_001

python mail_cag.py run model-b --resume --run-id b_smoke_qwen8b_quality_001
python mail_cag.py evaluate model-b --run-id b_smoke_qwen8b_quality_001
```

Training outputs are written to `../runs/`. Model B/C carry defender weights
forward across rounds, so round 2 starts from `round_1/model`. LLM rewrites are
cached in each run folder as `rewrite_cache.csv`.

## Experiment Terms

Clean baseline means: train the defender on the same clean CEAS train split, with no
adversarial examples added. It is not a different dataset. It is the reference
model we compare the cyclic methods against.

Phishing-only LLM training means only label `1` examples are rewritten by the
LLM. Both-label LLM training means labels `0` and `1` are both rewritten, so the
classifier cannot simply learn that "LLM-looking text means phishing."

Held-out attacker evaluation means: evaluate a trained model using attacks that
were not used to generate its training data. You were partly doing this when you
evaluated across validation sets, but the next version should make it explicit
in config: training attackers and evaluation attackers should be separate fields.
