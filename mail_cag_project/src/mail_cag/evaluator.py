from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mail_cag.config import load_config, resolve_from_config
from mail_cag.run_storage import choose_run_root
from mail_cag.training import evaluate_saved_model


def evaluate_config(
    config_path: str | Path,
    run_id: str | None = None,
    round_number: int | None = None,
) -> None:
    """Evaluate a saved run on its clean eval split."""

    config_path = Path(config_path)
    config = load_config(config_path)
    experiment_root = resolve_from_config(config_path, config["paths"]["run_root"])
    run_root = choose_run_root(experiment_root, run_id, resume=True)
    round_dir = choose_round_dir(run_root, round_number)
    model_dir = round_dir / "model"
    eval_path = run_root / "clean_eval.csv"

    if not model_dir.exists():
        raise RuntimeError(f"No saved model found at {model_dir}")
    if not eval_path.exists():
        raise RuntimeError(f"No clean eval split found at {eval_path}")

    eval_df = pd.read_csv(eval_path)
    metrics = evaluate_saved_model(
        model_dir=model_dir,
        eval_df=eval_df,
        max_length=int(config["model"]["max_length"]),
        batch_size=int(config["training"]["eval_batch_size"]),
    )
    metrics.update(
        {
            "experiment": config["name"],
            "run_id": run_root.name,
            "round": round_dir.name,
            "model_dir": str(model_dir),
        }
    )

    output_dir = run_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{round_dir.name}_clean_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"experiment: {metrics['experiment']}")
    print(f"run id: {metrics['run_id']}")
    print(f"round: {metrics['round']}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"f1: {metrics['f1']:.4f}")
    print(f"saved: {output_path}")


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
