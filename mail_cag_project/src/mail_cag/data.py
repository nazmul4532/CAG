from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ceas_subset(
    raw_path: str | Path,
    sample_frac_per_label: float,
    random_seed: int = 42,
) -> pd.DataFrame:
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
    counts = df["label"].value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}
