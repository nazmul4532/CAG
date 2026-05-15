from __future__ import annotations

from pathlib import Path

from langdetect import DetectorFactory, LangDetectException, detect
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

DetectorFactory.seed = 42


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
    df = add_email_text(df)

    parts = []
    for label in sorted(df["label"].dropna().unique()):
        part = df[df["label"] == label].sample(
            frac=sample_frac_per_label,
            random_state=random_seed,
        )
        parts.append(part)
    return pd.concat(parts).sample(frac=1, random_state=random_seed).reset_index(drop=True)


def filter_by_language(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """Keep emails detected as the requested language."""

    keep = [detect_language(text) == language for text in df["text"].astype(str)]
    return df[keep].reset_index(drop=True)


def detect_language(text: str) -> str | None:
    try:
        return detect(text)
    except LangDetectException:
        return None


def filter_by_token_length(
    df: pd.DataFrame,
    tokenizer_name: str,
    max_token_length: int,
) -> pd.DataFrame:
    """Keep emails within the configured token-length cap."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    encoded = tokenizer(
        df["text"].astype(str).tolist(),
        add_special_tokens=True,
        truncation=False,
        padding=False,
        verbose=False,
    )
    keep = [len(tokens) <= max_token_length for tokens in encoded["input_ids"]]
    return df[keep].reset_index(drop=True)


def describe_labels(df: pd.DataFrame) -> dict[int, int]:
    """Return label counts in a compact, print-friendly form."""

    counts = df["label"].value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def add_email_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create one text field from the useful CEAS email columns."""

    if "text" in df.columns:
        return df

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
