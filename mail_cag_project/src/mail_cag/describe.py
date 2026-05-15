from __future__ import annotations

from pathlib import Path

from mail_cag.config import load_config, resolve_from_config
from mail_cag.data import describe_labels, load_ceas_subset
from mail_cag.experiment_summary import (
    augmented_dataset_files,
    existing_rounds,
    generated_dataset_files,
)


def describe_config(config_path: str | Path) -> None:
    """Print a small, readable summary of one experiment config.

    This is the safest first thing to run. It does not train models or generate
    attacks. It only answers: which data would this config use, and what legacy
    outputs already exist?
    """

    config_path = Path(config_path)
    config = load_config(config_path)

    raw_path = resolve_from_config(config_path, config["data"]["raw_path"])
    sample_frac = float(config["data"]["sample_frac_per_label"])
    seed = int(config["data"].get("random_seed", 42))
    max_token_length = config["data"].get("max_token_length")
    df = load_ceas_subset(
        raw_path,
        sample_frac,
        seed,
        tokenizer_name=config["model"]["base_model"],
        max_token_length=max_token_length,
    )

    print(f"experiment: {config['name']}")
    print(f"raw data: {raw_path}")
    if max_token_length:
        print(f"max token length filter: {max_token_length}")
    print(f"sample frac per label: {sample_frac}")
    print(f"subset rows: {len(df)}")
    print(f"label counts: {describe_labels(df)}")

    paths = config.get("paths", {})
    for key, value in paths.items():
        if not key.startswith("legacy"):
            continue
        path = resolve_from_config(config_path, value)
        print(f"{key}: {path}")
        print(f"  exists: {path.exists()}")
        print(f"  rounds: {existing_rounds(path)}")
        print(f"  generated files: {len(generated_dataset_files(path))}")
        print(f"  augmented files: {len(augmented_dataset_files(path))}")
