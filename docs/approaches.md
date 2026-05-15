# Experiment Approaches

This project is moving forward with LLM-based budgeted cyclic adversarial
training.

## Main Comparison

- **Model A**: clean ALBERT baseline.
- **Model B**: cyclic ALBERT with LLM rewrites for phishing emails only.
- **Model C**: cyclic ALBERT with LLM rewrites for both phishing and benign
  emails.

This comparison asks:

1. Does LLM-generated cyclic training improve robustness over clean ALBERT?
2. Does phishing-only rewriting help differently than balanced both-label
   rewriting?

## Historical Approach v4: Cumulative Adversarial Game

Notebook: `legacy_workspace/notebook_era/approach_v4_cumulative_game.ipynb`

Local outputs:

- `legacy_workspace/artifacts/outputs/experiments/albert_adversarial_game_model_v4`
- `legacy_workspace/artifacts/outputs/experiments/albert_adversarial_val_sets_v4 copy`

Idea:

- Train ALBERT.
- Generate adversarial samples with TextAttack recipes.
- Carry successful adversarial examples forward by merging the previous
  `round_{r-1}_augmented_dataset.csv` back into the next round's training data.
- Each round grows a cumulative augmented dataset.

Keep this as historical/reference material. It is not the main implementation
path now.

## Historical Approach v5: Budgeted Roundwise Game

Legacy notebook: `legacy_workspace/notebook_era/approach_v5_budgeted_roundwise_game.ipynb`

Debug notebook: `legacy_workspace/notebook_era/approach_v5_dummy_debug.ipynb`

Local outputs:

- `legacy_workspace/artifacts/outputs/experiments/albert_adversarial_game_model_v5`
- `legacy_workspace/artifacts/outputs/experiments/albert_adversarial_game_model_v5dummy`
- `legacy_workspace/artifacts/outputs/experiments/albert_adversarial_val_sets_v5dummy`

Idea:

- Separate the LLM rewrite budget from the clean-data budget.
- Generate label-preserving rewrites for the current round.
- Train on a fresh clean subset plus that round's adversarial samples instead of
  blindly accumulating every previous augmented dataset.
- Evaluate later with TextFooler, PWWS, and DeepWordBug as classic held-out
  adversarial attacks.

Use this as the historical base for the new Model B / Model C plan. The useful
part is budget control; the main change is that TextAttack recipes move to
held-out evaluation instead of training-data generation.
