from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


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


def add_email_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create one text field from the useful CEAS email columns."""

    result = df.copy()
    subject = result["subject"].fillna("").astype(str)
    body = result["body"].fillna("").astype(str)
    result["text"] = ("Subject: " + subject + "\n\n" + body).str.strip()
    return result


def split_train_eval(
    df: pd.DataFrame,
    random_seed: int,
    eval_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make a stable stratified train/eval split."""

    train_df, eval_df = train_test_split(
        df,
        test_size=eval_size,
        random_state=random_seed,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)
