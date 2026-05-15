#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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
    stats = build_stats(path, df, token_lengths, args.tokenizer)

    print_stats(stats)
    write_stats_files(path, stats)


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


def build_stats(
    path: Path,
    df: pd.DataFrame,
    token_lengths: pd.Series,
    tokenizer_name: str,
) -> dict:
    return {
        "file": str(path),
        "description": (
            "Prepared CEAS dataset: English-only, no obvious non-English "
            "scripts, within the configured tokenizer length cap."
        ),
        "preparation": {
            "source": "CEAS_08.csv",
            "language": "en",
            "blocked_scripts": list(SCRIPT_PATTERNS),
            "tokenizer": tokenizer_name,
            "max_token_length": 1600,
        },
        "rows": len(df),
        "columns": list(df.columns),
        "label_counts": describe_labels(df),
        "label_percent": label_percent(df),
        "token_stats": token_stats(token_lengths),
        "text_stats": text_stats(df),
        "script_counts": script_counts(df),
        "by_label": by_label_stats(df, token_lengths),
    }


def token_stats(lengths: pd.Series) -> dict:
    stats = {
        "mean": round(float(lengths.mean()), 1),
        "median": int(lengths.median()),
        "p75": int(lengths.quantile(0.75)),
        "p90": int(lengths.quantile(0.90)),
        "p95": int(lengths.quantile(0.95)),
        "p99": int(lengths.quantile(0.99)),
        "max": int(lengths.max()),
    }
    stats["coverage_percent"] = {
        str(cutoff): round(float((lengths <= cutoff).mean() * 100), 2)
        for cutoff in [128, 256, 384, 512, 1024, 1600]
    }
    return stats


def text_stats(df: pd.DataFrame) -> dict:
    text_lengths = df["text"].fillna("").astype(str).str.len()
    stats = {
        "char_mean": round(float(text_lengths.mean()), 1),
        "char_median": int(text_lengths.median()),
    }
    if "urls" in df:
        has_url = df["urls"].fillna(0).astype(float) > 0
        stats["url_rate_percent"] = round(float(has_url.mean() * 100), 2)
    return stats


def script_counts(df: pd.DataFrame) -> dict[str, int]:
    text = df["text"].fillna("").astype(str)
    counts = {}
    for name, pattern in SCRIPT_PATTERNS.items():
        counts[name] = int(text.str.contains(pattern, regex=True).sum())
    return counts


def by_label_stats(df: pd.DataFrame, token_lengths: pd.Series) -> dict[str, dict]:
    frame = pd.DataFrame({"label": df["label"], "tokens": token_lengths})
    if "urls" in df:
        frame["has_url"] = df["urls"].fillna(0).astype(float) > 0

    stats = {}
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
        stats[str(int(label))] = line
    return stats


def print_stats(stats: dict) -> None:
    print(f"file: {stats['file']}")
    print(stats["description"])
    print(f"rows: {stats['rows']}")
    print(f"columns: {stats['columns']}")
    print(f"label counts: {stats['label_counts']}")
    print(f"label percent: {stats['label_percent']}")
    print(f"token stats: {stats['token_stats']}")
    print(f"text stats: {stats['text_stats']}")
    print(f"script counts: {stats['script_counts']}")
    print(f"by label: {stats['by_label']}")


def write_stats_files(path: Path, stats: dict) -> None:
    json_path = path.with_name(path.stem + "_stats.json")
    md_path = path.with_name(path.stem + "_stats.md")

    json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(stats), encoding="utf-8")
    print(f"saved: {json_path}")
    print(f"saved: {md_path}")


def render_markdown(stats: dict) -> str:
    token = stats["token_stats"]
    text = stats["text_stats"]
    lines = [
        "# Prepared CEAS Dataset Stats",
        "",
        stats["description"],
        "",
        f"- File: `{stats['file']}`",
        f"- Rows: {stats['rows']}",
        f"- Columns: {', '.join(stats['columns'])}",
        f"- Label counts: {stats['label_counts']}",
        f"- Label percent: {stats['label_percent']}",
        "",
        "## Preparation",
        "",
        f"- Source: {stats['preparation']['source']}",
        f"- Language: {stats['preparation']['language']}",
        f"- Blocked scripts: {', '.join(stats['preparation']['blocked_scripts'])}",
        f"- Tokenizer: {stats['preparation']['tokenizer']}",
        f"- Max token length: {stats['preparation']['max_token_length']}",
        "",
        "## Token Lengths",
        "",
        f"- Mean: {token['mean']}",
        f"- Median: {token['median']}",
        f"- p75: {token['p75']}",
        f"- p90: {token['p90']}",
        f"- p95: {token['p95']}",
        f"- p99: {token['p99']}",
        f"- Max: {token['max']}",
        f"- Coverage percent: {token['coverage_percent']}",
        "",
        "## Text And URL Stats",
        "",
        f"- Character mean: {text['char_mean']}",
        f"- Character median: {text['char_median']}",
        f"- URL rate percent: {text.get('url_rate_percent', 'n/a')}",
        "",
        "## Script Checks",
        "",
        f"- Script counts: {stats['script_counts']}",
        "",
        "## By Label",
        "",
        f"- Label 0: {stats['by_label'].get('0')}",
        f"- Label 1: {stats['by_label'].get('1')}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
