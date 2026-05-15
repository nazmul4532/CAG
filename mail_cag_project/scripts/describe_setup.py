from __future__ import annotations

import argparse
from pathlib import Path

from mail_cag.describe import describe_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="mail_cag_project/configs/cyclic_llm_both_labels.yaml",
        help="Path to an experiment config.",
    )
    args = parser.parse_args()

    describe_config(Path(args.config))


if __name__ == "__main__":
    main()
