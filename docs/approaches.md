# Experiment Approaches

This project is moving forward with the v5 budgeted cyclic adversarial-game
variant.

## Legacy Approach v4: Cumulative Adversarial Game

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

## Main Approach v5: Budgeted Roundwise Game

Notebook: `legacy_workspace/notebook_era/approach_v5_budgeted_roundwise_game.ipynb`

Debug notebook: `legacy_workspace/notebook_era/approach_v5_dummy_debug.ipynb`

Local outputs:

- `legacy_workspace/artifacts/outputs/experiments/albert_adversarial_game_model_v5`
- `legacy_workspace/artifacts/outputs/experiments/albert_adversarial_game_model_v5dummy`
- `legacy_workspace/artifacts/outputs/experiments/albert_adversarial_val_sets_v5dummy`

Idea:

- Separate the attack budget from the clean-data budget with
  `adv_samples_per_round` and `orig_samples_per_round`.
- Generate adversarial samples for the current round.
- Train on a fresh clean subset plus that round's adversarial samples instead of
  blindly accumulating every previous augmented dataset.
- Checkpoint TextAttack progress per recipe so interrupted attacks can resume.

Use this as the main project direction. It gives us a cleaner controlled game
with explicit budgets and less uncontrolled dataset growth.
