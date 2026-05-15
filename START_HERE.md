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

That command does not train anything. It checks the default v5 setup and tells
you which data/results already exist.

Other useful checks:

```bash
python mail_cag.py describe baseline
python mail_cag.py describe cyclic
```

## Mental Model

Use this map:

```text
mail_cag.py                 friendly command to run things
mail_cag_project/configs/   experiment settings; use cyclic_budgeted.yaml first
mail_cag_project/src/       reusable Python code
mail_cag_project/scripts/   lower-level scripts
legacy_workspace/           old notebooks, old results, local big files
```

## The Path Forward

We are moving forward with **v5 / budgeted cyclic training**.

It trains each round using fresh clean samples plus the current round's
adversarial samples. This is cleaner for controlled experiments than the v4
cumulative approach.

The old v4 config still exists at:

```text
mail_cag_project/configs/legacy/approach_v4_cumulative.yaml
```

Treat it as reference, not the main path.

## What We Build Next

The next missing pieces are:

1. `python mail_cag.py run`
2. `python mail_cag.py evaluate`
3. notebooks that only analyze saved results, not contain the whole experiment
   engine

Keep this rule in mind: notebooks are for thinking and reporting; scripts are
for experiments we want to trust.
