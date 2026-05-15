#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "mail_cag_project" / "src"
DEFAULT_CONFIGS = {
    "baseline": PROJECT_ROOT / "mail_cag_project" / "configs" / "baseline_clean.yaml",
    "v4": PROJECT_ROOT / "mail_cag_project" / "configs" / "approach_v4_cumulative.yaml",
    "v5": PROJECT_ROOT / "mail_cag_project" / "configs" / "approach_v5_budgeted.yaml",
}


def ensure_import_path() -> None:
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)


def resolve_config(name_or_path: str) -> Path:
    if name_or_path in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[name_or_path]
    return Path(name_or_path).expanduser().resolve()


def describe(config_name: str) -> None:
    ensure_import_path()

    from mail_cag.describe import describe_config

    describe_config(resolve_config(config_name))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Friendly entrypoint for the Mail-CAG project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe_parser = subparsers.add_parser(
        "describe",
        help="Show the dataset split and existing legacy outputs for a config.",
    )
    describe_parser.add_argument(
        "config",
        nargs="?",
        default="v5",
        help="One of baseline, v4, v5, or a config path. Default: v5.",
    )

    subparsers.add_parser(
        "run",
        help="Planned next step: run a configured experiment.",
    )
    subparsers.add_parser(
        "evaluate",
        help="Planned next step: evaluate a trained experiment.",
    )

    args = parser.parse_args()

    if args.command == "describe":
        describe(args.config)
        return

    if args.command in {"run", "evaluate"}:
        print(
            f"`python mail_cag.py {args.command}` is not implemented yet. "
            "Next we will build this around the config files."
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
