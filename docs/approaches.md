# Experiment Approaches

This project currently has two active cyclic adversarial-game variants.

## Approach v4: Cumulative Adversarial Game

Notebook: `approach_v4_cumulative_game.ipynb`

Local outputs:

- `albert_adversarial_game_model_v4`
- `albert_adversarial_val_sets_v4`

Idea:

- Train ALBERT.
- Generate adversarial samples with TextAttack recipes.
- Carry successful adversarial examples forward by merging the previous
  `round_{r-1}_augmented_dataset.csv` back into the next round's training data.
- Each round grows a cumulative augmented dataset.

Use this when the paper needs the "memory-based" cyclic game where the defender
keeps prior attacks in its training pool.

## Approach v5: Budgeted Roundwise Game

Notebook: `approach_v5_budgeted_roundwise_game.ipynb`

Debug notebook: `approach_v5_dummy_debug.ipynb`

Local outputs:

- `albert_adversarial_game_model_v5`
- `albert_adversarial_game_model_v5dummy`
- `albert_adversarial_val_sets_v5dummy`

Idea:

- Separate the attack budget from the clean-data budget with
  `adv_samples_per_round` and `orig_samples_per_round`.
- Generate adversarial samples for the current round.
- Train on a fresh clean subset plus that round's adversarial samples instead of
  blindly accumulating every previous augmented dataset.
- Checkpoint TextAttack progress per recipe so interrupted attacks can resume.

Use this when the paper needs a cleaner controlled game with explicit budgets
and less uncontrolled dataset growth.
