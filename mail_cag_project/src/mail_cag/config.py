from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def resolve_from_config(config_path: str | Path, value: str | Path) -> Path:
    base_dir = Path(config_path).resolve().parent
    return (base_dir / Path(value)).resolve()
