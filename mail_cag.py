#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "mail_cag_project" / "src"
DEFAULT_CONFIGS = {
    "baseline": PROJECT_ROOT / "mail_cag_project" / "configs" / "baseline_clean.yaml",
    "model-a": PROJECT_ROOT / "mail_cag_project" / "configs" / "baseline_clean.yaml",
    "model-b": PROJECT_ROOT
    / "mail_cag_project"
    / "configs"
    / "cyclic_llm_phishing_only.yaml",
    "phishing-only": PROJECT_ROOT
    / "mail_cag_project"
    / "configs"
    / "cyclic_llm_phishing_only.yaml",
    "model-c": PROJECT_ROOT
    / "mail_cag_project"
    / "configs"
    / "cyclic_llm_both_labels.yaml",
    "both-labels": PROJECT_ROOT
    / "mail_cag_project"
    / "configs"
    / "cyclic_llm_both_labels.yaml",
    "cyclic": PROJECT_ROOT
    / "mail_cag_project"
    / "configs"
    / "cyclic_llm_both_labels.yaml",
    "v5": PROJECT_ROOT / "mail_cag_project" / "configs" / "cyclic_llm_both_labels.yaml",
    "v4-legacy": PROJECT_ROOT
    / "mail_cag_project"
    / "configs"
    / "legacy"
    / "approach_v4_cumulative.yaml",
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


def run(
    config_name: str,
    dry_run: bool,
    run_id: str | None,
    resume: bool,
) -> None:
    ensure_import_path()

    from mail_cag.experiment_runner import run_config

    run_config(
        resolve_config(config_name),
        dry_run=dry_run,
        run_id=run_id,
        resume=resume,
    )


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
        default="cyclic",
        help=(
            "One of baseline/model-a, model-b/phishing-only, "
            "model-c/both-labels, cyclic, v5, v4-legacy, or a config path. "
            "Default: cyclic."
        ),
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run a configured experiment.",
    )
    run_parser.add_argument(
        "config",
        nargs="?",
        default="cyclic",
        help=(
            "One of baseline/model-a, model-b/phishing-only, "
            "model-c/both-labels, cyclic, v5, or a config path. Default: cyclic."
        ),
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the config and data split, then stop before training or LLM calls.",
    )
    run_parser.add_argument(
        "--run-id",
        help="Name this run folder, or choose which run to resume.",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a run by skipping completed rounds and reusing saved rewrites.",
    )
    subparsers.add_parser(
        "evaluate",
        help="Planned next step: evaluate a trained experiment.",
    )

    args = parser.parse_args()

    if args.command == "describe":
        describe(args.config)
        return

    if args.command == "run":
        run(
            args.config,
            dry_run=args.dry_run,
            run_id=args.run_id,
            resume=args.resume,
        )
        return

    if args.command == "evaluate":
        print(
            "`python mail_cag.py evaluate` is not implemented yet. "
            "Next we will build this around the config files."
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
