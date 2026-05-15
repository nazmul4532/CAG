from __future__ import annotations

from pathlib import Path

import pandas as pd


class RewriteCache:
    """Small CSV cache for expensive LLM rewrites."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.items = self._load(path)

    def get(self, key: str) -> list[str] | None:
        rewrites = self.items.get(key)
        if rewrites is None:
            return None
        return list(rewrites)

    def put(self, key: str, rewrites: list[str]) -> None:
        self.items[key] = list(rewrites)
        self._append(key, rewrites)

    @staticmethod
    def _load(path: Path) -> dict[str, list[str]]:
        if not path.exists():
            return {}

        df = pd.read_csv(path)
        items: dict[str, list[str]] = {}
        for key, group in df.groupby("cache_key", sort=False):
            ordered = group.sort_values("rewrite_index")
            items[str(key)] = ordered["rewrite"].astype(str).tolist()
        return items

    def _append(self, key: str, rewrites: list[str]) -> None:
        if not rewrites:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"cache_key": key, "rewrite_index": index, "rewrite": rewrite}
            for index, rewrite in enumerate(rewrites)
        ]
        pd.DataFrame(rows).to_csv(
            self.path,
            mode="a",
            header=not self.path.exists(),
            index=False,
        )
