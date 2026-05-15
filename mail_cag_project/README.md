# Mail-CAG Project Workspace

This folder is the clean, script-first workspace for the cyclic adversarial game
paper. The original notebooks and outputs live under `../legacy_workspace/`.
This gives us a safer place to rebuild the experiment logic step by step.

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

- `configs/baseline_clean.yaml`: Model A, clean ALBERT reference.
- `configs/cyclic_llm_phishing_only.yaml`: Model B, LLM rewrites phishing only.
- `configs/cyclic_llm_both_labels.yaml`: Model C, LLM rewrites both labels.
- `configs/providers.example.yaml`: provider slots for Ollama now and
  OpenAI/Gemini later.
- `docs/literature_scan.md`: quick map of nearby papers and the narrow gap we
  are targeting.

The old TextAttack-training and v4 configs are kept under `configs/legacy/` for
reference only.

## Current Legacy Inputs

The workspace references existing local files through `legacy_workspace/`:

- raw data: `../legacy_workspace/artifacts/data/raw/CEAS_08.csv`
- v5 outputs: `../legacy_workspace/artifacts/outputs/experiments/albert_adversarial_game_model_v5`
- v5 debug outputs: `../legacy_workspace/artifacts/outputs/experiments/albert_adversarial_game_model_v5dummy`

Those local data/output folders are intentionally ignored by Git.

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

## Experiment Terms

Clean baseline means: train ALBERT on the same clean CEAS train split, with no
adversarial examples added. It is not a different dataset. It is the reference
model we compare the cyclic methods against.

Phishing-only LLM training means only label `1` examples are rewritten by the
LLM. Both-label LLM training means labels `0` and `1` are both rewritten, so the
classifier cannot simply learn that "LLM-looking text means phishing."

Held-out attacker evaluation means: evaluate a trained model using attacks that
were not used to generate its training data. You were partly doing this when you
evaluated across validation sets, but the next version should make it explicit
in config: training attackers and evaluation attackers should be separate fields.
