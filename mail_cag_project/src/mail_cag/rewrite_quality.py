from __future__ import annotations

import json
import re
import hashlib
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from mail_cag.data import NON_ENGLISH_SCRIPT_PATTERN
from mail_cag.training import add_true_label_confidence

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def write_rewrite_quality_report(
    *,
    rewrites_df: pd.DataFrame,
    model_dir: Path,
    round_dir: Path,
    max_length: int,
    batch_size: int,
) -> None:
    """Write simple diagnostics for generated rewrites."""

    if rewrites_df.empty:
        return

    quality_df = score_rewrite_quality(
        rewrites_df=rewrites_df,
        model_dir=model_dir,
        max_length=max_length,
        batch_size=batch_size,
    )
    quality_df.to_csv(round_dir / "rewrite_quality.csv", index=False)
    (round_dir / "rewrite_quality_summary.json").write_text(
        json.dumps(summarize_quality(quality_df), indent=2),
        encoding="utf-8",
    )


def score_rewrite_quality(
    *,
    rewrites_df: pd.DataFrame,
    model_dir: Path,
    max_length: int,
    batch_size: int,
) -> pd.DataFrame:
    """Return per-rewrite diagnostics, including defender confidence drop."""

    scored_rewrites = add_true_label_confidence(
        df=rewrites_df,
        model_dir=model_dir,
        max_length=max_length,
        batch_size=batch_size,
    )

    rows = []
    for _, row in scored_rewrites.iterrows():
        source_text = str(row.get("source_text", ""))
        rewrite_text = str(row["text"])
        source_confidence = float(
            row.get("source_true_label_confidence", row.get("true_label_confidence", 0.0))
        )
        rewrite_confidence = float(row["true_label_confidence"])

        rows.append(
            {
                "generated_index": int(row.get("generated_index", len(rows))),
                "label": int(row["label"]),
                "generated_by": row.get("generated_by", ""),
                "source_true_label_confidence": source_confidence,
                "rewrite_true_label_confidence": rewrite_confidence,
                "confidence_drop": source_confidence - rewrite_confidence,
                "changed": normalize(source_text) != normalize(rewrite_text),
                "char_similarity": char_similarity(source_text, rewrite_text),
                "word_jaccard": word_jaccard(source_text, rewrite_text),
                "length_ratio": length_ratio(source_text, rewrite_text),
                "source_url_count": len(find_urls(source_text)),
                "rewrite_url_count": len(find_urls(rewrite_text)),
                "url_behavior_preserved": find_urls(source_text) == find_urls(rewrite_text),
                "has_non_english_script": bool(
                    NON_ENGLISH_SCRIPT_PATTERN.search(rewrite_text)
                ),
                "looks_like_structured_output": looks_like_structured_output(
                    rewrite_text
                ),
                "rewrite_text_hash": normalized_text_hash(rewrite_text),
            }
        )

    return pd.DataFrame(rows)


def summarize_quality(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "changed_rate": 0.0,
            "url_behavior_preserved_rate": 0.0,
            "non_english_script_rate": 0.0,
            "structured_output_rate": 0.0,
            "mean_confidence_drop": 0.0,
            "mean_char_similarity": 0.0,
            "mean_word_jaccard": 0.0,
            "mean_length_ratio": 0.0,
        }

    return {
        "rows": int(len(df)),
        "changed_rate": mean_bool(df["changed"]),
        "url_behavior_preserved_rate": mean_bool(df["url_behavior_preserved"]),
        "non_english_script_rate": mean_bool(df["has_non_english_script"]),
        "structured_output_rate": mean_bool(df["looks_like_structured_output"]),
        "mean_confidence_drop": mean_float(df["confidence_drop"]),
        "mean_char_similarity": mean_float(df["char_similarity"]),
        "mean_word_jaccard": mean_float(df["word_jaccard"]),
        "mean_length_ratio": mean_float(df["length_ratio"]),
    }


def find_urls(text: str) -> list[str]:
    return URL_PATTERN.findall(text)


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def normalized_text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def char_similarity(left: str, right: str) -> float:
    return float(SequenceMatcher(None, normalize(left), normalize(right)).ratio())


def word_jaccard(left: str, right: str) -> float:
    left_words = set(WORD_PATTERN.findall(left.lower()))
    right_words = set(WORD_PATTERN.findall(right.lower()))
    if not left_words and not right_words:
        return 1.0
    return len(left_words & right_words) / max(len(left_words | right_words), 1)


def length_ratio(source_text: str, rewrite_text: str) -> float:
    return len(rewrite_text) / max(len(source_text), 1)


def looks_like_structured_output(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith(("{", "[", "```")):
        return True
    return bool(re.search(r"^\s*['\"]?(text|email|rewrite)['\"]?\s*:", stripped))


def mean_bool(series: pd.Series) -> float:
    return float(series.astype(bool).mean()) if len(series) else 0.0


def mean_float(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0
