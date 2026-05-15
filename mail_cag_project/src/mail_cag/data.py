from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ceas_subset(
    raw_path: str | Path,
    sample_frac_per_label: float,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load the same percentage from each CEAS label.

    The old notebooks used a percentage of CEAS rather than the whole dataset.
    Keeping that choice in one helper makes baseline/v4/v5 comparisons fairer:
    each config can say exactly which fraction it used.
    """

    df = pd.read_csv(raw_path)
    parts = []
    for label in sorted(df["label"].dropna().unique()):
        part = df[df["label"] == label].sample(
            frac=sample_frac_per_label,
            random_state=random_seed,
        )
        parts.append(part)
    return pd.concat(parts).sample(frac=1, random_state=random_seed).reset_index(drop=True)


def describe_labels(df: pd.DataFrame) -> dict[int, int]:
    """Return label counts in a compact, print-friendly form."""

    counts = df["label"].value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}
