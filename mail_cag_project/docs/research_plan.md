# Research Plan Notes

## Current Hypothesis

ALBERT is strong at phishing detection on clean email data. We want to test
whether cyclic adversarial training makes it more robust to adversarial phishing
emails.

## Main Risk

The cyclic game can overfit to the attacker family. If the defender sees many
TextFooler/PWWS/DeepWordBug examples, it may learn their artifacts rather than
general phishing semantics.

## Cleaner Comparison

Use the same clean CEAS subset policy for all methods:

1. Clean ALBERT baseline.
2. v4 cumulative adversarial replay.
3. v5 budgeted roundwise adversarial training.
4. Held-out attacker evaluation.

The CEAS percentage is part of the experiment config. We do not need to train on
the entire CEAS dataset to make the comparison meaningful, but the percentage
must be fixed and reported.

## Held-Out Attacker Evaluation

The next setup should distinguish:

- attackers used to generate training examples
- attackers used only for evaluation

If an attacker is used in both places, call that "seen-attacker robustness."
If it is used only in evaluation, call that "held-out attacker robustness."

## TextAttack Speed

TextFooler, PWWS, and DeepWordBug are slow because the attack search is mostly
CPU-bound and calls the model many times with small candidate batches. GPU helps
the model forward pass, but the transformations, constraints, search control,
and Python overhead remain expensive.

Practical choices:

- attack fewer examples per round
- checkpoint per recipe
- parallelize across recipes or data shards
- evaluate with fixed attack budgets
- add LLM-based semantic attacks as a separate evaluation arm
