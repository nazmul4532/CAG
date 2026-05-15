from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one YAML experiment config.

    Config files answer "what are we trying to run?" They should contain paths,
    sample sizes, model settings, attacker choices, and evaluation choices.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def resolve_from_config(config_path: str | Path, value: str | Path) -> Path:
    """Resolve a path written inside a config file.

    Relative paths in config files are interpreted relative to the config file's
    own folder, not whichever terminal directory you happen to be in.
    """

    base_dir = Path(config_path).resolve().parent
    return (base_dir / Path(value)).resolve()
