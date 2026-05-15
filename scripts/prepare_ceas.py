#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "mail_cag_project" / "src"
sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from mail_cag.data import (
    add_email_text,
    describe_labels,
    filter_by_language,
    filter_by_token_length,
    filter_non_english_scripts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the filtered CEAS dataset.")
    parser.add_argument(
        "--raw",
        default="legacy_workspace/artifacts/data/raw/CEAS_08.csv",
        help="Raw CEAS CSV path, relative to the repo root.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/CEAS_08_en_1600.csv",
        help="Prepared output CSV path, relative to the repo root.",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--keep-non-english-scripts",
        action="store_true",
        help="Keep rows with CJK/Cyrillic/Arabic/etc. characters.",
    )
    parser.add_argument("--tokenizer", default="albert-base-v2")
    parser.add_argument("--max-token-length", type=int, default=1600)
    args = parser.parse_args()

    raw_path = (PROJECT_ROOT / args.raw).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()

    df = pd.read_csv(raw_path)
    df = add_email_text(df)
    print(f"raw rows: {len(df)}")
    print(f"raw labels: {describe_labels(df)}")

    df = filter_by_language(df, args.language)
    print(f"after language={args.language}: {len(df)}")
    print(f"labels: {describe_labels(df)}")

    if not args.keep_non_english_scripts:
        df = filter_non_english_scripts(df)
        print("after non-English script filter:", len(df))
        print(f"labels: {describe_labels(df)}")

    df = filter_by_token_length(df, args.tokenizer, args.max_token_length)
    print(f"after max_token_length={args.max_token_length}: {len(df)}")
    print(f"labels: {describe_labels(df)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
