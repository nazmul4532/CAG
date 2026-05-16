from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def choose_run_root(experiment_root: Path, run_id: str | None, resume: bool) -> Path:
    """Pick an isolated folder for this run."""

    if resume:
        return run_to_resume(experiment_root, run_id)
    return experiment_root / (run_id or make_run_id())


def run_to_resume(experiment_root: Path, run_id: str | None) -> Path:
    if run_id:
        return experiment_root / run_id

    latest = experiment_root / "latest"
    if latest.exists():
        return latest.resolve()

    raise RuntimeError(
        f"No latest run found under {experiment_root}. "
        "Pass `--run-id <id>` or start a new run first."
    )


def make_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def write_latest_pointer(experiment_root: Path, run_root: Path) -> None:
    """Point experiment_root/latest at the newest run."""

    latest = experiment_root / "latest"
    if latest.exists() or latest.is_symlink():
        remove_path(latest)
    try:
        latest.symlink_to(run_root, target_is_directory=True)
    except OSError:
        latest.write_text(str(run_root), encoding="utf-8")


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def round_complete(round_dir: Path) -> bool:
    return (round_dir / "model" / "model.safetensors").exists()


def load_existing_rewrites(run_root: Path) -> pd.DataFrame:
    parts = []
    for round_dir in sorted(run_root.glob("round_*")):
        path = round_dir / "training_rewrites.csv"
        if not path.exists():
            path = round_dir / "generated_rewrites.csv"
        if not path.exists():
            continue
        parts.append(training_columns(pd.read_csv(path)))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def training_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "data_source" not in result:
        result["data_source"] = "llm_rewrite"
    if "parent_id" not in result:
        result["parent_id"] = [
            f"{data_source}:{index}"
            for index, data_source in zip(result.index, result["data_source"])
        ]
    return result[["parent_id", "text", "label", "data_source"]]
