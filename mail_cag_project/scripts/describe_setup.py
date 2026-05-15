from __future__ import annotations

import argparse
from pathlib import Path

from mail_cag.config import load_config, resolve_from_config
from mail_cag.data import describe_labels, load_ceas_subset
from mail_cag.experiment_summary import (
    augmented_dataset_files,
    existing_rounds,
    generated_dataset_files,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="mail_cag_project/configs/approach_v5_budgeted.yaml",
        help="Path to an experiment config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    raw_path = resolve_from_config(config_path, config["data"]["raw_path"])
    sample_frac = float(config["data"]["sample_frac_per_label"])
    seed = int(config["data"].get("random_seed", 42))
    df = load_ceas_subset(raw_path, sample_frac, seed)

    print(f"experiment: {config['name']}")
    print(f"raw data: {raw_path}")
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


if __name__ == "__main__":
    main()
