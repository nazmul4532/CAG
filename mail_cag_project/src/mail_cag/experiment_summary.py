from __future__ import annotations

from pathlib import Path


def existing_rounds(output_dir: str | Path) -> list[int]:
    root = Path(output_dir)
    rounds = []
    if not root.exists():
        return rounds
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("round_"):
            suffix = child.name.removeprefix("round_")
            if suffix.isdigit():
                rounds.append(int(suffix))
    return sorted(rounds)


def generated_dataset_files(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []
    return sorted(root.glob("round_*_generated_adversarial_dataset.csv"))


def augmented_dataset_files(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []
    return sorted(root.glob("round_*_augmented_dataset.csv"))
