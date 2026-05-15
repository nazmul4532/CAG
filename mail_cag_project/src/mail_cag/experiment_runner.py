from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from mail_cag.config import load_config, resolve_from_config
from mail_cag.data import add_email_text, load_ceas_subset, split_train_eval
from mail_cag.llm_rewriter import choose_ollama_model, rewrite_cache_key, rewrite_email
from mail_cag.rewrite_cache import RewriteCache
from mail_cag.rewrite_quality import write_rewrite_quality_report
from mail_cag.run_storage import (
    choose_run_root,
    load_existing_rewrites,
    round_complete,
    write_latest_pointer,
)
from mail_cag.training import add_true_label_confidence, train_transformer_classifier


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
    start_model = str(config["model"]["base_model"])
    print(f"ollama model: {ollama_model}")

    for round_number in range(1, rounds + 1):
        round_dir = run_root / f"round_{round_number}"
        if resume and round_complete(round_dir):
            print(f"round {round_number} already complete; skipping training")
            start_model = str(round_dir / "model")
            if round_number < rounds:
                rewrites_path = round_dir / "generated_rewrites.csv"
                if not rewrites_path.exists():
                    rewrites = generate_round_rewrites(
                        config=config,
                        source_df=clean_train_df,
                        model_dir=round_dir / "model",
                        round_dir=round_dir,
                        ollama_model=ollama_model,
                    )
                    rewrite_pool = pd.concat([rewrite_pool, rewrites], ignore_index=True)
            continue

        round_dir.mkdir(parents=True, exist_ok=True)
        train_df = pd.concat([clean_train_df, rewrite_pool], ignore_index=True)
        train_df.to_csv(round_dir / "training_data.csv", index=False)

        print(f"round {round_number} starts from: {start_model}")
        result = train_one_round(config, round_dir, train_df, eval_df, start_model)
        start_model = str(result.model_dir)
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
    start_model: str | Path | None = None,
):
    training = config["training"]
    model = config["model"]
    return train_transformer_classifier(
        train_df=train_df,
        eval_df=eval_df,
        model_name=start_model or model["base_model"],
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
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base_model"],
        local_files_only=True,
    )

    candidates_df = source_df[source_df["label"].isin(target_labels)].copy()
    scored = add_true_label_confidence(
        df=candidates_df,
        model_dir=model_dir,
        max_length=int(config["model"]["max_length"]),
        batch_size=int(config["training"]["eval_batch_size"]),
    )
    selected = scored.sort_values("true_label_confidence").head(max_examples)
    selected.to_csv(round_dir / "rewrite_source.csv", index=False)

    output_path = round_dir / "generated_rewrites.csv"
    rewrite_cache = RewriteCache(round_dir.parent / "rewrite_cache.csv")
    rows = []
    save_every = max(1, int(attacks.get("save_every_rewrites", 50)))
    print(f"selected emails to rewrite: {len(selected)}", flush=True)
    progress = tqdm(
        selected.iterrows(),
        total=len(selected),
        desc="rewriting emails",
    )
    for _, row in progress:
        visible_text = visible_defender_text(
            text=str(row["text"]),
            tokenizer=tokenizer,
            max_length=int(config["model"]["max_length"]),
        )
        temperature = float(llm.get("temperature", 0.6))
        top_p = float(llm.get("top_p", 0.9))
        cache_key = rewrite_cache_key(
            model=ollama_model,
            label=int(row["label"]),
            text=visible_text,
            candidates=candidates,
            temperature=temperature,
            top_p=top_p,
        )
        rewrites = rewrite_cache.get(cache_key)
        if rewrites is None:
            rewrites = rewrite_email(
                base_url=llm.get("base_url", "http://127.0.0.1:11434"),
                model=ollama_model,
                label=int(row["label"]),
                text=visible_text,
                candidates=candidates,
                temperature=temperature,
                top_p=top_p,
            )
            rewrite_cache.put(cache_key, rewrites)
        else:
            progress.write("cache hit")

        for rewrite in rewrites:
            item = row.to_dict()
            item["source_text"] = item["text"]
            item["source_true_label_confidence"] = item["true_label_confidence"]
            item["text"] = rewrite
            item["subject"] = ""
            item["body"] = rewrite
            item["generated_by"] = ollama_model
            rows.append(item)
        if len(rows) % save_every == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False)
            progress.write(f"saved rewrites: {len(rows)}")
        progress.set_postfix(generated=len(rows))

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"saved rewrites: {len(rows)}")
    print(f"round rewrites: {len(rows)}")
    rewrites_df = pd.DataFrame(rows)
    write_rewrite_quality_report(
        rewrites_df=rewrites_df,
        model_dir=model_dir,
        round_dir=round_dir,
        max_length=int(config["model"]["max_length"]),
        batch_size=int(config["training"]["eval_batch_size"]),
    )
    print(f"rewrite quality: {round_dir / 'rewrite_quality_summary.json'}")
    return rewrites_df


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv_if_missing(df: pd.DataFrame, path: Path) -> None:
    if not path.exists():
        df.to_csv(path, index=False)


def visible_defender_text(text: str, tokenizer, max_length: int) -> str:
    """Return the same text window the current defender can see."""

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
