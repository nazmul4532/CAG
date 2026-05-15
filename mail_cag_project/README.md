# Mail-CAG Project Workspace

This folder is the clean, script-first workspace for the cyclic adversarial game
paper. It does not replace or delete the original notebooks and outputs in the
repository root. Instead, it gives us a safer place to rebuild the experiment
logic step by step.

## Why This Exists

The original notebooks are useful records, but they mix code, outputs, partial
reruns, checkpoints, and analysis. This workspace separates those concerns:

- `configs/`: experiment definitions.
- `src/mail_cag/`: reusable experiment code.
- `notebooks/`: analysis notebooks that should call code from `src/`.
- `scripts/`: command-line entrypoints.
- `docs/`: notes on method, drawbacks, and research decisions.
- `tests/`: lightweight checks for the reusable code.

## Current Legacy Inputs

The workspace references existing local files through the repository root:

- raw data: `../CEAS_08.csv` from this folder, or `../../CEAS_08.csv` from config files
- v4 outputs: `../albert_adversarial_game_model_v4`
- v5 outputs: `../albert_adversarial_game_model_v5`
- v5 debug outputs: `../albert_adversarial_game_model_v5dummy`

Those are compatibility symlinks to local ignored data/output folders.

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

Held-out attacker evaluation means: evaluate a trained model using attacks that
were not used to generate its training data. You were partly doing this when you
evaluated across validation sets, but the next version should make it explicit
in config: training attackers and evaluation attackers should be separate fields.
