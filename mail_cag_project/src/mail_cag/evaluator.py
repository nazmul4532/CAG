from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm

from mail_cag.config import load_config, resolve_from_config
from mail_cag.run_storage import choose_run_root
from mail_cag.training import predict_saved_model


DEFAULT_ATTACKS = ("textfooler", "pwws", "deepwordbug")


def evaluate_config(
    config_path: str | Path,
    run_id: str | None = None,
    round_number: int | None = None,
    all_rounds: bool = False,
    attacks: list[str] | None = None,
    generate_adversarial: bool = False,
    max_examples: int | None = None,
    overwrite: bool = False,
) -> None:
    """Evaluate saved rounds and optionally generate TextAttack held-out sets."""

    config_path = Path(config_path)
    config = load_config(config_path)
    experiment_root = resolve_from_config(config_path, config["paths"]["run_root"])
    run_root = choose_run_root(experiment_root, run_id, resume=True)
    eval_path = run_root / "clean_eval.csv"

    if not eval_path.exists():
        raise RuntimeError(f"No clean eval split found at {eval_path}")

    eval_df = pd.read_csv(eval_path)
    attack_names = normalize_attacks(
        attacks
        if attacks is not None
        else config.get("evaluation", {}).get("textattack_attackers")
    )
    round_dirs = choose_round_dirs(run_root, round_number, all_rounds)
    output_dir = run_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for round_dir in round_dirs:
        rows.extend(
            evaluate_round(
                config=config,
                run_root=run_root,
                round_dir=round_dir,
                eval_df=eval_df,
                output_dir=output_dir,
                attacks=attack_names,
                generate_adversarial=generate_adversarial,
                max_examples=max_examples,
                overwrite=overwrite,
            )
        )

    matrix_df = pd.DataFrame(rows)
    matrix_path = output_dir / "evaluation_matrix.csv"
    matrix_df.to_csv(matrix_path, index=False)
    write_attack_matrices(matrix_df, output_dir, attack_names)
    write_cross_evaluation_matrices(
        config=config,
        run_root=run_root,
        eval_df=eval_df,
        round_dirs=round_dirs,
        output_dir=output_dir,
        attacks=attack_names,
    )
    report_path = output_dir / "evaluation_report.md"
    report_path.write_text(
        build_markdown_report(config["name"], run_root.name, matrix_df),
        encoding="utf-8",
    )

    print(f"experiment: {config['name']}")
    print(f"run id: {run_root.name}")
    print(f"rounds evaluated: {', '.join(path.name for path in round_dirs)}")
    print(f"matrix: {matrix_path}")
    print(f"report: {report_path}")


def evaluate_round(
    *,
    config: dict[str, Any],
    run_root: Path,
    round_dir: Path,
    eval_df: pd.DataFrame,
    output_dir: Path,
    attacks: list[str],
    generate_adversarial: bool,
    max_examples: int | None,
    overwrite: bool,
) -> list[dict[str, Any]]:
    model_dir = round_dir / "model"
    if not model_dir.exists():
        raise RuntimeError(f"No saved model found at {model_dir}")

    round_output_dir = output_dir / round_dir.name
    round_output_dir.mkdir(parents=True, exist_ok=True)
    max_length = int(config["model"]["max_length"])
    batch_size = int(config["training"]["eval_batch_size"])

    rows: list[dict[str, Any]] = []
    clean_metrics = evaluate_and_save_dataset(
        model_dir=model_dir,
        df=eval_df,
        max_length=max_length,
        batch_size=batch_size,
        output_prefix=round_output_dir / "clean_eval",
    )
    rows.append(
        matrix_row(
            config=config,
            run_root=run_root,
            round_dir=round_dir,
            dataset="clean_eval",
            attack="clean",
            metrics=clean_metrics,
            dataset_path=run_root / "clean_eval.csv",
            stats={},
        )
    )

    if not generate_adversarial:
        print(f"{round_dir.name}: clean accuracy {clean_metrics['accuracy']:.4f}")
        return rows

    adv_dfs: list[pd.DataFrame] = []
    stats_rows: list[dict[str, Any]] = []
    for attack in attacks:
        adv_path = round_output_dir / f"{attack}_adversarial.csv"
        checkpoint_path = round_output_dir / f"{attack}_checkpoint.csv"
        stats_path = round_output_dir / f"{attack}_stats.json"

        if adv_path.exists() and stats_path.exists() and not overwrite:
            adv_df = pd.read_csv(adv_path)
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            print(f"{round_dir.name}/{attack}: using existing adversarial set")
        else:
            adv_df, stats = generate_textattack_adversarial_set(
                model_dir=model_dir,
                eval_df=eval_df,
                attack_name=attack,
                max_examples=max_examples,
                checkpoint_path=checkpoint_path,
                stats_path=stats_path,
            )
            adv_df.to_csv(adv_path, index=False)
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        stats_rows.append({"attack": attack, **stats})
        adv_dfs.append(with_dataset_columns(adv_df, attack))

        adv_metrics = evaluate_and_save_dataset(
            model_dir=model_dir,
            df=adv_df,
            max_length=max_length,
            batch_size=batch_size,
            output_prefix=round_output_dir / f"{attack}_adversarial_eval",
        )
        rows.append(
            matrix_row(
                config=config,
                run_root=run_root,
                round_dir=round_dir,
                dataset=f"{attack}_adversarial",
                attack=attack,
                metrics=adv_metrics,
                dataset_path=adv_path,
                stats=stats,
            )
        )

    if adv_dfs:
        combined = pd.concat(
            [with_dataset_columns(eval_df, "clean"), *adv_dfs],
            ignore_index=True,
        )
        combined_path = round_output_dir / "augmented_eval_all_attacks.csv"
        combined.to_csv(combined_path, index=False)
        combined_metrics = evaluate_and_save_dataset(
            model_dir=model_dir,
            df=combined,
            max_length=max_length,
            batch_size=batch_size,
            output_prefix=round_output_dir / "augmented_eval_all_attacks",
        )
        rows.append(
            matrix_row(
                config=config,
                run_root=run_root,
                round_dir=round_dir,
                dataset="augmented_eval_all_attacks",
                attack="all",
                metrics=combined_metrics,
                dataset_path=combined_path,
                stats={},
            )
        )

    pd.DataFrame(stats_rows).to_csv(round_output_dir / "attack_stats.csv", index=False)
    return rows


def evaluate_and_save_dataset(
    *,
    model_dir: Path,
    df: pd.DataFrame,
    max_length: int,
    batch_size: int,
    output_prefix: Path,
) -> dict[str, Any]:
    if df.empty:
        metrics = empty_dataset_metrics()
        output_prefix.with_name(output_prefix.name + "_metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(columns=["label", "prediction"]).to_csv(
            output_prefix.with_name(output_prefix.name + "_predictions.csv"),
            index=False,
        )
        return metrics

    labels, predictions = predict_saved_model(
        model_dir=model_dir,
        eval_df=training_eval_columns(df),
        max_length=max_length,
        batch_size=batch_size,
    )
    metrics = dataset_metrics(labels, predictions)
    output_prefix.with_name(output_prefix.name + "_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame({"label": labels, "prediction": predictions}).to_csv(
        output_prefix.with_name(output_prefix.name + "_predictions.csv"),
        index=False,
    )
    return metrics


def dataset_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    rows = len(labels)
    true_benign = int(tn + fp)
    true_phishing = int(fn + tp)
    predicted_benign = int(tn + fn)
    predicted_phishing = int(fp + tp)
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["benign", "phishing"],
        zero_division=0,
        output_dict=True,
    )
    return {
        "rows": rows,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": report,
        "benign_precision": float(report["benign"]["precision"]),
        "benign_recall": float(report["benign"]["recall"]),
        "benign_f1": float(report["benign"]["f1-score"]),
        "phishing_precision": float(report["phishing"]["precision"]),
        "phishing_recall": float(report["phishing"]["recall"]),
        "phishing_f1": float(report["phishing"]["f1-score"]),
        "benign_false_positive_rate": fp / true_benign if true_benign else 0.0,
        "phishing_false_negative_rate": fn / true_phishing if true_phishing else 0.0,
        "true_benign_count": true_benign,
        "true_phishing_count": true_phishing,
        "predicted_benign_count": predicted_benign,
        "predicted_phishing_count": predicted_phishing,
        "true_benign_share": true_benign / rows if rows else 0.0,
        "true_phishing_share": true_phishing / rows if rows else 0.0,
        "predicted_benign_share": predicted_benign / rows if rows else 0.0,
        "predicted_phishing_share": predicted_phishing / rows if rows else 0.0,
        "prediction_phishing_bias": (
            predicted_phishing / rows - true_phishing / rows if rows else 0.0
        ),
    }


def empty_dataset_metrics() -> dict[str, Any]:
    return {
        "rows": 0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "confusion_matrix": [[0, 0], [0, 0]],
        "classification_report": {},
        "benign_precision": 0.0,
        "benign_recall": 0.0,
        "benign_f1": 0.0,
        "phishing_precision": 0.0,
        "phishing_recall": 0.0,
        "phishing_f1": 0.0,
        "benign_false_positive_rate": 0.0,
        "phishing_false_negative_rate": 0.0,
        "true_benign_count": 0,
        "true_phishing_count": 0,
        "predicted_benign_count": 0,
        "predicted_phishing_count": 0,
        "true_benign_share": 0.0,
        "true_phishing_share": 0.0,
        "predicted_benign_share": 0.0,
        "predicted_phishing_share": 0.0,
        "prediction_phishing_bias": 0.0,
    }


def generate_textattack_adversarial_set(
    *,
    model_dir: Path,
    eval_df: pd.DataFrame,
    attack_name: str,
    max_examples: int | None,
    checkpoint_path: Path,
    stats_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import torch
        from textattack.attack_recipes import (
            DeepWordBugGao2018,
            PWWSRen2019,
            TextFoolerJin2019,
        )
        from textattack.attack_results import (
            FailedAttackResult,
            SkippedAttackResult,
            SuccessfulAttackResult,
        )
        from textattack.models.wrappers import HuggingFaceModelWrapper
        from textattack.shared import AttackedText
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "TextAttack evaluation requires the textattack, torch, and transformers packages."
        ) from exc

    attack_classes = {
        "textfooler": TextFoolerJin2019,
        "pwws": PWWSRen2019,
        "deepwordbug": DeepWordBugGao2018,
    }
    if attack_name not in attack_classes:
        raise ValueError(f"Unsupported attack: {attack_name}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_df = training_eval_columns(eval_df).reset_index(drop=True)
    if max_examples is not None:
        dataset_df = dataset_df.head(max(0, int(max_examples))).copy()

    if checkpoint_path.exists():
        checkpoint_df = pd.read_csv(checkpoint_path)
        processed_indices = set(checkpoint_df["dataset_idx"].astype(int).tolist())
    else:
        checkpoint_df = pd.DataFrame(
            columns=[
                "dataset_idx",
                "input",
                "adv_input",
                "status",
                "label",
                "num_queries",
                "num_words_changed",
                "perturbed_word_pct",
            ]
        )
        processed_indices = set()

    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        stats = {
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "original_correct": 0,
            "total_queries_success": 0,
            "total_queries_failed": 0,
            "total_perturbed_word_pct": 0.0,
            "total_original_words": 0,
        }

    rows_to_attack = [
        (idx, str(row["text"]), int(row["label"]))
        for idx, row in dataset_df.iterrows()
        if idx not in processed_indices
    ]
    if not rows_to_attack:
        return adversarial_rows_from_checkpoint(checkpoint_df), finalize_attack_stats(stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    ).to(device)
    model.eval()

    class DeviceHuggingFaceWrapper(HuggingFaceModelWrapper):
        def __call__(self, text_input_list):
            inputs = self.tokenizer(
                text_input_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                return self.model(**inputs).logits

    wrapper = DeviceHuggingFaceWrapper(model, tokenizer)
    attack = attack_classes[attack_name].build(wrapper)
    if torch.cuda.is_available():
        attack.cuda_()

    for dataset_idx, text, label in tqdm(rows_to_attack, desc=f"{attack_name} attacks"):
        adv_text = ""
        status = "skipped"
        num_queries = 0
        num_words_changed = 0
        perturbed_word_pct = 0.0

        try:
            result = attack.attack(AttackedText(text), label)
        except Exception as exc:
            stats["skipped"] += 1
            checkpoint_df = append_checkpoint_row(
                checkpoint_df,
                checkpoint_path,
                {
                    "dataset_idx": dataset_idx,
                    "input": text,
                    "adv_input": "",
                    "status": "skipped",
                    "label": label,
                    "num_queries": 0,
                    "num_words_changed": 0,
                    "perturbed_word_pct": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            continue

        original_text = result.original_result.attacked_text
        num_queries = int(getattr(result, "num_queries", 0) or 0)
        stats["total_original_words"] += len(original_text.words)
        if result.original_result.output == result.original_result.ground_truth_output:
            stats["original_correct"] += 1

        if isinstance(result, SuccessfulAttackResult):
            perturbed_text = result.perturbed_result.attacked_text
            adv_text = perturbed_text.text
            status = "success"
            stats["successful"] += 1
            stats["total_queries_success"] += num_queries
            num_words_changed = len(original_text.all_words_diff(perturbed_text))
            perturbed_word_pct = (
                num_words_changed / len(original_text.words) * 100
                if original_text.words
                else 0.0
            )
            stats["total_perturbed_word_pct"] += perturbed_word_pct
        elif isinstance(result, SkippedAttackResult):
            status = "skipped"
            stats["skipped"] += 1
        elif isinstance(result, FailedAttackResult):
            status = "fail"
            stats["failed"] += 1
            stats["total_queries_failed"] += num_queries
        else:
            stats["failed"] += 1
            stats["total_queries_failed"] += num_queries

        checkpoint_df = append_checkpoint_row(
            checkpoint_df,
            checkpoint_path,
            {
                "dataset_idx": dataset_idx,
                "input": original_text.text,
                "adv_input": adv_text,
                "status": status,
                "label": label,
                "num_queries": num_queries,
                "num_words_changed": num_words_changed,
                "perturbed_word_pct": perturbed_word_pct,
            },
        )
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return adversarial_rows_from_checkpoint(checkpoint_df), finalize_attack_stats(stats)


def append_checkpoint_row(
    checkpoint_df: pd.DataFrame,
    checkpoint_path: Path,
    row: dict[str, Any],
) -> pd.DataFrame:
    updated = pd.concat([checkpoint_df, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(checkpoint_path, index=False)
    return updated


def adversarial_rows_from_checkpoint(checkpoint_df: pd.DataFrame) -> pd.DataFrame:
    success_df = checkpoint_df[checkpoint_df["status"] == "success"].copy()
    if success_df.empty:
        return pd.DataFrame(columns=["parent_id", "text", "label", "data_source"])
    return pd.DataFrame(
        {
            "parent_id": [f"adv_eval:{idx}" for idx in success_df["dataset_idx"]],
            "text": success_df["adv_input"].astype(str),
            "label": success_df["label"].astype(int),
            "data_source": "textattack_eval",
        }
    )


def finalize_attack_stats(stats: dict[str, Any]) -> dict[str, Any]:
    successful = int(stats.get("successful", 0))
    failed = int(stats.get("failed", 0))
    skipped = int(stats.get("skipped", 0))
    attempted = successful + failed
    total = attempted + skipped
    total_queries = int(stats.get("total_queries_success", 0)) + int(
        stats.get("total_queries_failed", 0)
    )
    return {
        **stats,
        "total": total,
        "attempted": attempted,
        "original_accuracy": stats.get("original_correct", 0) / total if total else 0.0,
        "accuracy_under_attack": failed / total if total else 0.0,
        "attack_success_rate": successful / attempted if attempted else 0.0,
        "avg_perturbed_word_pct": (
            stats.get("total_perturbed_word_pct", 0.0) / successful if successful else 0.0
        ),
        "avg_queries": total_queries / attempted if attempted else 0.0,
    }


def matrix_row(
    *,
    config: dict[str, Any],
    run_root: Path,
    round_dir: Path,
    dataset: str,
    attack: str,
    metrics: dict[str, Any],
    dataset_path: Path,
    stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": config["name"],
        "run_id": run_root.name,
        "round": round_sort_key(round_dir),
        "round_name": round_dir.name,
        "dataset": dataset,
        "attack": attack,
        "rows": metrics["rows"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tn": metrics["confusion_matrix"][0][0],
        "fp": metrics["confusion_matrix"][0][1],
        "fn": metrics["confusion_matrix"][1][0],
        "tp": metrics["confusion_matrix"][1][1],
        "benign_recall": metrics["benign_recall"],
        "phishing_recall": metrics["phishing_recall"],
        "benign_false_positive_rate": metrics["benign_false_positive_rate"],
        "phishing_false_negative_rate": metrics["phishing_false_negative_rate"],
        "true_benign_count": metrics["true_benign_count"],
        "true_phishing_count": metrics["true_phishing_count"],
        "predicted_benign_count": metrics["predicted_benign_count"],
        "predicted_phishing_count": metrics["predicted_phishing_count"],
        "true_phishing_share": metrics["true_phishing_share"],
        "predicted_phishing_share": metrics["predicted_phishing_share"],
        "prediction_phishing_bias": metrics["prediction_phishing_bias"],
        "attack_success_rate": stats.get("attack_success_rate"),
        "successful_attacks": stats.get("successful"),
        "failed_attacks": stats.get("failed"),
        "skipped_attacks": stats.get("skipped"),
        "avg_queries": stats.get("avg_queries"),
        "dataset_path": str(dataset_path),
    }


def training_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "text" not in result and "body" in result:
        result["text"] = result["body"]
    if "parent_id" not in result:
        result["parent_id"] = [f"eval:{index}" for index in result.index]
    if "data_source" not in result:
        result["data_source"] = "eval"
    return result[["parent_id", "text", "label", "data_source"]].copy()


def with_dataset_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    result = training_eval_columns(df)
    result["eval_source"] = source
    return result


def normalize_attacks(attacks: list[str] | tuple[str, ...] | None) -> list[str]:
    if not attacks:
        return list(DEFAULT_ATTACKS)
    normalized = [
        str(attack).strip().lower()
        for attack in attacks
        if str(attack).strip()
    ]
    unknown = sorted(set(normalized) - set(DEFAULT_ATTACKS))
    if unknown:
        raise ValueError(f"Unsupported attacks: {', '.join(unknown)}")
    return normalized


def build_markdown_report(experiment: str, run_id: str, matrix_df: pd.DataFrame) -> str:
    table_columns = [
        "round",
        "dataset",
        "attack",
        "rows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "attack_success_rate",
        "benign_recall",
        "phishing_recall",
        "benign_false_positive_rate",
        "phishing_false_negative_rate",
        "predicted_phishing_share",
        "prediction_phishing_bias",
    ]
    lines = [
        f"# Evaluation Report: {experiment}",
        "",
        f"Run: `{run_id}`",
        "",
        "## Matrix",
        "",
        markdown_table(matrix_df[table_columns]),
        "",
        "## Confusion Matrix Columns",
        "",
        "`tn`, `fp`, `fn`, and `tp` are saved in `evaluation_matrix.csv`.",
        "Separate round-by-round matrices are saved as `evaluation_matrix_<attack>.csv`.",
        "Legacy-style cross-round matrices are saved as `cross_eval_<attack>_accuracy_matrix.csv`.",
        "",
    ]
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                values.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_attack_matrices(
    matrix_df: pd.DataFrame,
    output_dir: Path,
    attacks: list[str],
) -> None:
    columns = [
        "round",
        "round_name",
        "rows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "benign_recall",
        "phishing_recall",
        "benign_false_positive_rate",
        "phishing_false_negative_rate",
        "predicted_phishing_share",
        "prediction_phishing_bias",
        "attack_success_rate",
        "successful_attacks",
        "failed_attacks",
        "skipped_attacks",
        "avg_queries",
        "dataset_path",
    ]
    for attack in attacks:
        attack_df = matrix_df[matrix_df["attack"] == attack].copy()
        attack_df = attack_df.sort_values("round")
        attack_df[columns].to_csv(
            output_dir / f"evaluation_matrix_{attack}.csv",
            index=False,
        )


def write_cross_evaluation_matrices(
    *,
    config: dict[str, Any],
    run_root: Path,
    eval_df: pd.DataFrame,
    round_dirs: list[Path],
    output_dir: Path,
    attacks: list[str],
) -> None:
    """Legacy-style grids: model rounds against adversarial validation rounds.

    The original notebook's main validation set for round N was:
    clean eval + adversarial examples generated by round N.

    We also write an adversarial-only grid because that is useful for seeing
    the direct attack transfer signal without clean eval rows smoothing it.
    """

    max_length = int(config["model"]["max_length"])
    batch_size = int(config["training"]["eval_batch_size"])
    clean_df = training_eval_columns(eval_df)
    all_rows: list[dict[str, Any]] = []

    for attack in attacks:
        eval_plus_adv_datasets: list[tuple[int, str, Path | None, pd.DataFrame]] = [
            (0, "clean_eval", run_root / "clean_eval.csv", clean_df)
        ]
        adv_only_datasets: list[tuple[int, str, Path | None, pd.DataFrame]] = []
        for source_round_dir in round_dirs:
            source_round = round_sort_key(source_round_dir)
            adv_path = output_dir / source_round_dir.name / f"{attack}_adversarial.csv"
            if not adv_path.exists():
                continue
            adv_df = training_eval_columns(pd.read_csv(adv_path))
            eval_plus_adv = pd.concat([clean_df, adv_df], ignore_index=True)
            eval_plus_adv_datasets.append(
                (
                    source_round,
                    f"clean_eval_plus_{attack}_from_round_{source_round}",
                    adv_path,
                    eval_plus_adv,
                )
            )
            adv_only_datasets.append(
                (
                    source_round,
                    f"{attack}_from_round_{source_round}",
                    adv_path,
                    adv_df,
                )
            )

        if len(eval_plus_adv_datasets) == 1 and not adv_only_datasets:
            continue

        eval_plus_adv_rows = cross_evaluate_dataset_group(
            config=config,
            run_root=run_root,
            round_dirs=round_dirs,
            attack=attack,
            dataset_kind="eval_plus_adv",
            datasets=eval_plus_adv_datasets,
            max_length=max_length,
            batch_size=batch_size,
        )
        all_rows.extend(eval_plus_adv_rows)
        write_cross_outputs(
            rows=eval_plus_adv_rows,
            output_dir=output_dir,
            attack=attack,
            dataset_kind="eval_plus_adv",
            filename_stem=f"cross_eval_{attack}",
            title=f"{config['name']} {attack}: Clean Eval + Adversarial Sets",
        )

        if adv_only_datasets:
            adv_only_rows = cross_evaluate_dataset_group(
                config=config,
                run_root=run_root,
                round_dirs=round_dirs,
                attack=attack,
                dataset_kind="adv_only",
                datasets=adv_only_datasets,
                max_length=max_length,
                batch_size=batch_size,
            )
            all_rows.extend(adv_only_rows)
            write_cross_outputs(
                rows=adv_only_rows,
                output_dir=output_dir,
                attack=attack,
                dataset_kind="adv_only",
                filename_stem=f"cross_eval_{attack}_adv_only",
                title=f"{config['name']} {attack}: Adversarial Sets Only",
            )

    if all_rows:
        pd.DataFrame(all_rows).to_csv(output_dir / "cross_eval_all_attacks_long.csv", index=False)


def cross_evaluate_dataset_group(
    *,
    config: dict[str, Any],
    run_root: Path,
    round_dirs: list[Path],
    attack: str,
    dataset_kind: str,
    datasets: list[tuple[int, str, Path | None, pd.DataFrame]],
    max_length: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_round_dir in round_dirs:
        model_round = round_sort_key(model_round_dir)
        model_dir = model_round_dir / "model"
        for source_round, dataset_name, dataset_path, dataset_df in datasets:
            metrics = evaluate_cross_dataset(
                model_dir=model_dir,
                df=dataset_df,
                max_length=max_length,
                batch_size=batch_size,
            )
            rows.append(
                {
                    "experiment": config["name"],
                    "run_id": run_root.name,
                    "attack": attack,
                    "dataset_kind": dataset_kind,
                    "model_round": model_round,
                    "validation_round": source_round,
                    "dataset": dataset_name,
                    "rows": metrics["rows"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "tn": metrics["confusion_matrix"][0][0],
                    "fp": metrics["confusion_matrix"][0][1],
                    "fn": metrics["confusion_matrix"][1][0],
                    "tp": metrics["confusion_matrix"][1][1],
                    "benign_recall": metrics["benign_recall"],
                    "phishing_recall": metrics["phishing_recall"],
                    "benign_false_positive_rate": metrics["benign_false_positive_rate"],
                    "phishing_false_negative_rate": metrics["phishing_false_negative_rate"],
                    "true_benign_count": metrics["true_benign_count"],
                    "true_phishing_count": metrics["true_phishing_count"],
                    "predicted_benign_count": metrics["predicted_benign_count"],
                    "predicted_phishing_count": metrics["predicted_phishing_count"],
                    "true_phishing_share": metrics["true_phishing_share"],
                    "predicted_phishing_share": metrics["predicted_phishing_share"],
                    "prediction_phishing_bias": metrics["prediction_phishing_bias"],
                    "dataset_path": "" if dataset_path is None else str(dataset_path),
                }
            )
    return rows


def write_cross_outputs(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    attack: str,
    dataset_kind: str,
    filename_stem: str,
    title: str,
) -> None:
    cross_df = pd.DataFrame(rows)
    cross_df.to_csv(output_dir / f"{filename_stem}_long.csv", index=False)
    matrix_df = cross_df.pivot(
        index="model_round",
        columns="validation_round",
        values="accuracy",
    )
    matrix_df.to_csv(output_dir / f"{filename_stem}_accuracy_matrix.csv")
    write_cross_heatmap(
        matrix_df=matrix_df,
        output_path=output_dir / f"{filename_stem}_accuracy_heatmap.png",
        title=title,
    )


def evaluate_cross_dataset(
    *,
    model_dir: Path,
    df: pd.DataFrame,
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
    if df.empty:
        return empty_dataset_metrics()
    labels, predictions = predict_saved_model(
        model_dir=model_dir,
        eval_df=training_eval_columns(df),
        max_length=max_length,
        batch_size=batch_size,
    )
    return dataset_metrics(labels, predictions)


def write_cross_heatmap(
    *,
    matrix_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    width = max(7, 1.2 * len(matrix_df.columns) + 2)
    height = max(5, 0.7 * len(matrix_df.index) + 2)
    plt.figure(figsize=(width, height))
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Accuracy"},
    )
    plt.title(title)
    plt.xlabel("Validation / Adversarial Dataset Round")
    plt.ylabel("Model Round")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def choose_round_dirs(
    run_root: Path,
    round_number: int | None,
    all_rounds: bool,
) -> list[Path]:
    if all_rounds:
        rounds = [
            path
            for path in run_root.glob("round_*")
            if (path / "model" / "model.safetensors").exists()
        ]
        if not rounds:
            raise RuntimeError(f"No completed rounds found under {run_root}")
        return sorted(rounds, key=round_sort_key)
    return [choose_round_dir(run_root, round_number)]


def choose_round_dir(run_root: Path, round_number: int | None) -> Path:
    if round_number is not None:
        return run_root / f"round_{round_number}"

    rounds = []
    for path in run_root.glob("round_*"):
        if (path / "model" / "model.safetensors").exists():
            rounds.append(path)
    if not rounds:
        raise RuntimeError(f"No completed rounds found under {run_root}")
    return sorted(rounds, key=round_sort_key)[-1]


def round_sort_key(path: Path) -> int:
    suffix = path.name.removeprefix("round_")
    return int(suffix) if suffix.isdigit() else -1
