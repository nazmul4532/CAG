from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mail_cag.config import load_config, resolve_from_config
from mail_cag.data import add_email_text, load_ceas_subset, split_train_eval
from mail_cag.llm_rewriter import choose_ollama_model, rewrite_email
from mail_cag.run_storage import (
    choose_run_root,
    load_existing_rewrites,
    round_complete,
    write_latest_pointer,
)
from mail_cag.training import add_true_label_confidence, train_albert


def run_config(
    config_path: str | Path,
    dry_run: bool = False,
    run_id: str | None = None,
    resume: bool = False,
) -> None:
    """Run one configured experiment."""

    config_path = Path(config_path)
    config = load_config(config_path)
    experiment_root = resolve_from_config(config_path, config["paths"]["run_root"])
    run_root = choose_run_root(experiment_root, run_id, resume)
    raw_path = resolve_from_config(config_path, config["data"]["raw_path"])

    df = load_ceas_subset(
        raw_path,
        float(config["data"]["sample_frac_per_label"]),
        int(config["data"].get("random_seed", 42)),
    )
    df = add_email_text(df)
    train_df, eval_df = split_train_eval(df, int(config["data"].get("random_seed", 42)))

    print(f"experiment: {config['name']}")
    print(f"experiment root: {experiment_root}")
    print(f"run id: {run_root.name}")
    print(f"run root: {run_root}")
    print(f"train rows: {len(train_df)}")
    print(f"eval rows: {len(eval_df)}", flush=True)
    if dry_run:
        print("dry run: no training or LLM calls were made")
        return

    run_root.mkdir(parents=True, exist_ok=True)
    write_latest_pointer(experiment_root, run_root)
    save_json(run_root / "config_snapshot.json", config)
    write_csv_if_missing(train_df, run_root / "clean_train.csv")
    write_csv_if_missing(eval_df, run_root / "clean_eval.csv")

    if config["training"]["strategy"] == "clean_only":
        run_clean_round(config, run_root, train_df, eval_df, resume=resume)
        return

    run_llm_cyclic_rounds(config, run_root, train_df, eval_df, resume=resume)


def run_clean_round(
    config: dict[str, Any],
    run_root: Path,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    resume: bool,
) -> None:
    round_dir = run_root / "round_1"
    if resume and round_complete(round_dir):
        print("round 1 already complete; skipping")
        return
    round_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(round_dir / "training_data.csv", index=False)
    result = train_one_round(config, round_dir, train_df, eval_df)
    print(f"round 1 eval accuracy: {result.eval_accuracy:.4f}")


def run_llm_cyclic_rounds(
    config: dict[str, Any],
    run_root: Path,
    clean_train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    resume: bool,
) -> None:
    rounds = int(config["training"]["rounds"])
    rewrite_pool = load_existing_rewrites(run_root)
    ollama_model = choose_ollama_model(config)
    print(f"ollama model: {ollama_model}")

    for round_number in range(1, rounds + 1):
        round_dir = run_root / f"round_{round_number}"
        if resume and round_complete(round_dir):
            print(f"round {round_number} already complete; skipping training")
            continue

        round_dir.mkdir(parents=True, exist_ok=True)
        train_df = pd.concat([clean_train_df, rewrite_pool], ignore_index=True)
        train_df.to_csv(round_dir / "training_data.csv", index=False)

        result = train_one_round(config, round_dir, train_df, eval_df)
        print(f"round {round_number} eval accuracy: {result.eval_accuracy:.4f}")

        if round_number == rounds:
            continue

        rewrites_path = round_dir / "generated_rewrites.csv"
        if resume and rewrites_path.exists():
            print(f"round {round_number} rewrites already exist; reusing")
            rewrites = pd.read_csv(rewrites_path)
        else:
            rewrites = generate_round_rewrites(
                config=config,
                source_df=clean_train_df,
                model_dir=result.model_dir,
                round_dir=round_dir,
                ollama_model=ollama_model,
            )
        rewrite_pool = pd.concat([rewrite_pool, rewrites], ignore_index=True)


def train_one_round(
    config: dict[str, Any],
    round_dir: Path,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
):
    training = config["training"]
    model = config["model"]
    return train_albert(
        train_df=train_df,
        eval_df=eval_df,
        model_name=model["base_model"],
        output_dir=round_dir / "model",
        max_length=int(model["max_length"]),
        learning_rate=float(training["learning_rate"]),
        epochs=int(training["num_train_epochs"]),
        train_batch_size=int(training["train_batch_size"]),
        eval_batch_size=int(training["eval_batch_size"]),
        num_labels=int(model["num_labels"]),
        checkpoint_dir=round_dir / "checkpoint_current",
    )


def generate_round_rewrites(
    *,
    config: dict[str, Any],
    source_df: pd.DataFrame,
    model_dir: Path,
    round_dir: Path,
    ollama_model: str,
) -> pd.DataFrame:
    attacks = config["attacks"]
    llm = config["llm"]
    target_labels = {int(label) for label in attacks["target_labels"]}
    max_examples = int(attacks["max_examples_per_round"])
    candidates = int(attacks["candidates_per_email"])

    candidates_df = source_df[source_df["label"].isin(target_labels)].copy()
    scored = add_true_label_confidence(
        df=candidates_df,
        model_dir=model_dir,
        max_length=int(config["model"]["max_length"]),
        batch_size=int(config["training"]["eval_batch_size"]),
    )
    selected = scored.sort_values("true_label_confidence").head(max_examples)
    selected.to_csv(round_dir / "rewrite_source.csv", index=False)

    rows = []
    for _, row in selected.iterrows():
        rewrites = rewrite_email(
            base_url=llm.get("base_url", "http://127.0.0.1:11434"),
            model=ollama_model,
            label=int(row["label"]),
            text=str(row["text"]),
            candidates=candidates,
            temperature=float(llm.get("temperature", 0.6)),
            top_p=float(llm.get("top_p", 0.9)),
        )
        for rewrite in rewrites:
            item = row.to_dict()
            item["text"] = rewrite
            item["subject"] = ""
            item["body"] = rewrite
            item["generated_by"] = ollama_model
            rows.append(item)

    rewrites_df = pd.DataFrame(rows)
    rewrites_df.to_csv(round_dir / "generated_rewrites.csv", index=False)
    print(f"round rewrites: {len(rewrites_df)}")
    return rewrites_df


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv_if_missing(df: pd.DataFrame, path: Path) -> None:
    if not path.exists():
        df.to_csv(path, index=False)
