# Start Here

Mail-CAG is a cyclic adversarial training project for phishing-email detection.

## Mental Model

We compare three models:

- **Model A**: clean defender baseline. Right now the defender is ALBERT.
- **Model B**: defender + cyclic LLM rewrites for phishing emails only.
- **Model B Improved**: Model B with an improved phishing-specific LLM
  rewrite prompt and separate cache/run folder.
- **Model C**: defender + cyclic LLM rewrites for phishing and benign emails.
- **Model D**: defender + cyclic LLM rewrites for both labels with
  label-aware prompts and balanced random rewrite selection.

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
python mail_cag.py describe b-improved
python mail_cag.py describe model-c
python mail_cag.py describe model-d
```

Download required local models:

```bash
scripts/download_required_models.sh
```

Dry-run before training:

```bash
python mail_cag.py run model-a --dry-run --run-id a_smoke_001
python mail_cag.py run model-b --dry-run --run-id b_smoke_qwen8b_quality_001
python mail_cag.py run b-improved --dry-run --run-id b_improved_qwen8b_001
python mail_cag.py run model-c --dry-run --run-id c_smoke_qwen8b_quality_001
python mail_cag.py run model-d --dry-run --run-id d_smoke_qwen8b_label_aware_001
```

Run smoke tests from the beginning:

```bash
python mail_cag.py run model-a --run-id a_smoke_001
python mail_cag.py run model-b --run-id b_smoke_qwen8b_quality_001
python mail_cag.py run model-c --run-id c_smoke_qwen8b_quality_001
python mail_cag.py run model-d --run-id d_smoke_qwen8b_label_aware_001
```

Resume after interruption:

```bash
python mail_cag.py run model-a --resume --run-id a_smoke_001
python mail_cag.py run model-b --resume --run-id b_smoke_qwen8b_quality_001
python mail_cag.py run model-c --resume --run-id c_smoke_qwen8b_quality_001
python mail_cag.py run model-d --resume --run-id d_smoke_qwen8b_label_aware_001
```

Evaluate a completed run:

```bash
python mail_cag.py evaluate model-a --run-id a_smoke_001
python mail_cag.py evaluate model-b --run-id b_smoke_qwen8b_quality_001
python mail_cag.py evaluate model-c --run-id c_smoke_qwen8b_quality_001
python mail_cag.py evaluate model-d --run-id d_smoke_qwen8b_label_aware_001
python mail_cag.py evaluate model-b --run-id b_smoke_qwen8b_quality_001 --round 1
```

Evaluate every completed cyclic round with held-out TextAttack attacks:

```bash
python mail_cag.py evaluate model-b --run-id b_smoke_qwen8b_quality_001 --all-rounds --generate-adversarial --attacks pwws textfooler deepwordbug
python mail_cag.py evaluate model-c --run-id c_smoke_qwen8b_quality_001 --all-rounds --generate-adversarial --attacks pwws textfooler deepwordbug
python mail_cag.py evaluate model-d --run-id d_smoke_qwen8b_label_aware_001 --all-rounds --generate-adversarial --attacks pwws textfooler deepwordbug
```

This writes per-round adversarial validation sets, attack stats, predictions,
`evaluation/evaluation_matrix.csv`, separate matrices such as
`evaluation/evaluation_matrix_textfooler.csv`, and
legacy-style cross-round matrices/heatmaps such as
`evaluation/cross_eval_textfooler_accuracy_matrix.csv` and
`evaluation/cross_eval_textfooler_accuracy_heatmap.png`, plus adversarial-only
versions such as `evaluation/cross_eval_textfooler_adv_only_accuracy_matrix.csv`.
The evaluation matrices also include bias diagnostics such as benign false
positive rate, phishing false negative rate, per-class recall, predicted
phishing share, and prediction phishing bias.

Run training, evaluation, and image rendering as one automated flow:

```bash
scripts/run_train_eval_report.sh model-b b_smoke_qwen8b_quality_002
scripts/run_train_eval_report.sh b-improved b_improved_qwen8b_001
scripts/run_train_eval_report.sh model-c c_smoke_qwen8b_quality_002
scripts/run_train_eval_report.sh model-d d_smoke_qwen8b_label_aware_001
```

Smoke-test the full flow with tiny TextAttack budgets:

```bash
scripts/run_train_eval_report.sh model-c c_debug_001 --max-examples 10
```

Resume an interrupted automated run:

```bash
scripts/run_train_eval_report.sh model-c c_smoke_qwen8b_quality_002 --resume
```

## Where Things Go

Generated runs live under `runs/`, ignored by Git.

```text
runs/model_b_llm_phishing_only/b_smoke_qwen8b_quality_001/
  clean_train.csv
  clean_eval.csv
  round_1/
    model/
    checkpoint_current/
    rewrite_source.csv
    generated_rewrites.csv
    training_rewrites.csv
    rewrite_quality.csv
    rewrite_quality_summary.json
  round_2/
  round_3/
```

`model/` is the final model for that round.
`checkpoint_current/` is overwritten after every epoch.

## Current Defaults

- Dataset: prepared English CEAS at `sample_frac_per_label: 0.06`.
- Prepared file: `data/processed/CEAS_08_en_1600.csv`.
- Defender: `albert-base-v2` with `max_length: 512`.
- To swap defenders later, change `model.base_model` and `model.max_length`,
  then regenerate the prepared dataset with that tokenizer.
- Local LLM attacker: `qwen3:8b` for faster smoke runs.
- Rounds: 3 for Model B/C.
- Between-round rewrite source: all eligible active-pool rows.
- Model B rewrites phishing-labeled rows; Model C rewrites both labels.

If rewriting is too slow, reduce the Model B/C config:

```yaml
attacks:
  candidates_per_email: 1
  rewrite_selection_rule: lowest_true_label_confidence
  max_examples_per_round: 25
```

## File Map

```text
mail_cag.py                         command you run
mail_cag_project/configs/           experiment settings
mail_cag_project/src/mail_cag/      reusable code
legacy_workspace/                   old notebooks and local artifacts
runs/                               generated outputs, ignored by Git
data/cache/                         shared local LLM cache, ignored by Git
```

Readable code matters here. Keep notebooks for analysis; keep experiment logic
inside scripts/modules.
