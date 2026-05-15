#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "mail_cag_project" / "src"
sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from transformers import AutoTokenizer

from mail_cag.data import describe_labels


SCRIPT_PATTERNS = {
    "cjk": r"[\u3400-\u9fff]",
    "cyrillic": r"[\u0400-\u04ff]",
    "arabic": r"[\u0600-\u06ff]",
    "hebrew": r"[\u0590-\u05ff]",
    "japanese": r"[\u3040-\u30ff]",
    "hangul": r"[\uac00-\ud7af]",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe a prepared CEAS CSV.")
    parser.add_argument("path", nargs="?", default="data/processed/CEAS_08_en_1600.csv")
    parser.add_argument("--tokenizer", default="albert-base-v2")
    args = parser.parse_args()

    path = (PROJECT_ROOT / args.path).resolve()
    df = pd.read_csv(path)
    token_lengths = measure_token_lengths(df["text"].fillna("").astype(str), args.tokenizer)

    print(f"file: {path}")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")
    print(f"label counts: {describe_labels(df)}")
    print("label percent:", label_percent(df))
    print_token_stats(token_lengths)
    print_text_stats(df)
    print_script_counts(df)
    print_by_label(df, token_lengths)


def measure_token_lengths(texts: pd.Series, tokenizer_name: str) -> pd.Series:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    encoded = tokenizer(
        texts.tolist(),
        add_special_tokens=True,
        truncation=False,
        padding=False,
        verbose=False,
    )
    return pd.Series([len(tokens) for tokens in encoded["input_ids"]])


def label_percent(df: pd.DataFrame) -> dict[int, float]:
    values = df["label"].value_counts(normalize=True).sort_index() * 100
    return {int(label): round(float(percent), 2) for label, percent in values.items()}


def print_token_stats(lengths: pd.Series) -> None:
    print(f"token mean: {lengths.mean():.1f}")
    print(f"token median: {int(lengths.median())}")
    for q in [0.75, 0.90, 0.95, 0.99]:
        print(f"token p{int(q * 100)}: {int(lengths.quantile(q))}")
    print(f"token max: {int(lengths.max())}")
    for cutoff in [128, 256, 384, 512, 1024, 1600]:
        coverage = (lengths <= cutoff).mean() * 100
        print(f"coverage <= {cutoff}: {coverage:.2f}%")


def print_text_stats(df: pd.DataFrame) -> None:
    text_lengths = df["text"].fillna("").astype(str).str.len()
    print(f"char mean: {text_lengths.mean():.1f}")
    print(f"char median: {int(text_lengths.median())}")
    if "urls" in df:
        has_url = df["urls"].fillna(0).astype(float) > 0
        print(f"url rate: {has_url.mean() * 100:.2f}%")


def print_script_counts(df: pd.DataFrame) -> None:
    text = df["text"].fillna("").astype(str)
    for name, pattern in SCRIPT_PATTERNS.items():
        count = int(text.str.contains(pattern, regex=True).sum())
        print(f"{name} rows: {count}")


def print_by_label(df: pd.DataFrame, token_lengths: pd.Series) -> None:
    frame = pd.DataFrame({"label": df["label"], "tokens": token_lengths})
    if "urls" in df:
        frame["has_url"] = df["urls"].fillna(0).astype(float) > 0

    print("by label:")
    for label, part in frame.groupby("label"):
        line = {
            "rows": len(part),
            "median": int(part["tokens"].median()),
            "p90": int(part["tokens"].quantile(0.90)),
            "p95": int(part["tokens"].quantile(0.95)),
            "coverage_512": round(float((part["tokens"] <= 512).mean() * 100), 2),
        }
        if "has_url" in part:
            line["url_rate"] = round(float(part["has_url"].mean() * 100), 2)
        print(f"  {int(label)}: {line}")


if __name__ == "__main__":
    main()
