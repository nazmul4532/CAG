# Start Here

Mail-CAG is a cyclic adversarial training project for phishing-email detection.

## Mental Model

We compare three models:

- **Model A**: clean ALBERT baseline.
- **Model B**: ALBERT + cyclic LLM rewrites for phishing emails only.
- **Model C**: ALBERT + cyclic LLM rewrites for phishing and benign emails.

For Model B/C, each round starts from the previous round's saved model:

```text
round 1: albert-base-v2
round 2: round_1/model
round 3: round_2/model
```

So this is a real cyclic defender-attacker loop, not fresh retraining every
round.

## First Commands

```bash
cd ~/Documents/NLP
conda activate nlp_game
git pull
```

Check the project:

```bash
python mail_cag.py describe model-a
python mail_cag.py describe model-b
python mail_cag.py describe model-c
```

Download required local models:

```bash
scripts/download_required_models.sh
```

Dry-run before training:

```bash
python mail_cag.py run model-b --dry-run --run-id b_smoke_001
```

Start or resume Model B:

```bash
python mail_cag.py run model-b --run-id b_smoke_001
python mail_cag.py run model-b --resume --run-id b_smoke_001
```

Model C is the same shape:

```bash
python mail_cag.py run model-c --run-id c_smoke_001
python mail_cag.py run model-c --resume --run-id c_smoke_001
```

Evaluate a completed run:

```bash
python mail_cag.py evaluate model-b --run-id b_smoke_001
python mail_cag.py evaluate model-b --run-id b_smoke_001 --round 1
```

## Where Things Go

Generated runs live under `runs/`, ignored by Git.

```text
runs/model_b_llm_phishing_only/b_smoke_001/
  clean_train.csv
  clean_eval.csv
  round_1/
    model/
    checkpoint_current/
    rewrite_source.csv
    generated_rewrites.csv
  round_2/
  round_3/
```

`model/` is the final model for that round.
`checkpoint_current/` is overwritten after every epoch.

## Current Defaults

- Dataset: CEAS at `sample_frac_per_label: 0.06`, after filtering emails longer
  than 1500 ALBERT tokens.
- Defender: `albert-base-v2` with `max_length: 512`.
- Local LLM attacker: `qwen3:8b` for faster smoke runs.
- Rounds: 3 for Model B/C.
- Between-round rewrite budget: 200 selected emails, 1 candidate each.

If rewriting is too slow, reduce the Model B/C config:

```yaml
attacks:
  candidates_per_email: 1
  max_examples_per_round: 25
```

## File Map

```text
mail_cag.py                         command you run
mail_cag_project/configs/           experiment settings
mail_cag_project/src/mail_cag/      reusable code
legacy_workspace/                   old notebooks and local artifacts
runs/                               generated outputs, ignored by Git
```

Readable code matters here. Keep notebooks for analysis; keep experiment logic
inside scripts/modules.
